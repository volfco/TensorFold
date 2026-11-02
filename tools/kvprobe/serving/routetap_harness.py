"""CPU harness for patches/0710's routetap: fake forward / sparse / lean modules, real routetap.py."""
import os, sys, types, importlib, shutil, tempfile
import torch

tmp = tempfile.mkdtemp()
pkg = os.path.join(tmp, "fakepkg")
os.makedirs(pkg)
shutil.copy(sys.argv[1], os.path.join(pkg, "routetap.py"))
open(os.path.join(pkg, "__init__.py"), "w").write("")
open(os.path.join(pkg, "lean.py"), "w").write("import os\ndef enabled():\n    return os.environ.get('GLM53_TF_LEAN_PREFILL','0')=='1'\n")
open(os.path.join(pkg, "sparse.py"), "w").write('''
import torch
TOPK_POOLS = 512
POOL = 4
def _tokens(pools, pos, R, dev):
    q = pos + torch.arange(R, device=dev, dtype=torch.int64)
    npool = (q + 1) // POOL
    sparse_row = npool > TOPK_POOLS
    body = (pools.to(torch.int64)[:, :, None] * POOL + torch.arange(POOL, device=dev)).reshape(R, -1)
    lane = torch.arange(POOL - 1, device=dev)
    tail = npool[:, None] * POOL + lane
    tail_len = q + 1 - npool * POOL
    tail = torch.where(lane[None, :] < tail_len[:, None], tail, torch.full_like(tail, -1))
    tokens = torch.cat([body, tail], dim=1).to(torch.int32)
    tokens = torch.where(sparse_row[:, None], tokens, torch.full_like(tokens, -1))
    counts = torch.where(sparse_row, TOPK_POOLS * POOL + tail_len, torch.zeros_like(tail_len)).to(torch.int32)
    return tokens.contiguous(), counts
def select_tokens(qi, wts, pk, pos, R, np_max, pos_dev):
    g = torch.Generator().manual_seed(pos)
    pools = torch.stack([torch.randperm(np_max, generator=g)[:TOPK_POOLS].sort().values for _ in range(R)])
    LAST.append(pools)
    return _tokens(pools, pos, R, 'cpu')
def select_tokens_dev(qi, wts, pk, pos_dev, R, np_, sc):
    return select_tokens(qi, wts, pk, int(pos_dev), R, np_, None)
LAST = []
''')
open(os.path.join(pkg, "forward.py"), "w").write('''
import torch
from . import sparse
def moe_block(layer, w, b, R):
    b.pick[:R] = torch.randint(0, 288, (R, 9), dtype=torch.int32)
    b.wts[:R] = torch.rand(R, 9)
    return None
def kda_block(layer, w, st, b, R):
    b.ka[:R] = torch.randn(R, 4 * 128).to(torch.bfloat16)
    return None
def dsa_block(layer, w, pos, R):
    sparse.select_tokens(None, None, None, pos, R, 3000, None)
def layer_forward(layer, w, st, b, R, pos):
    if layer.kind == "kda":
        kda_block(layer, w, st, b, R)
    else:
        dsa_block(layer, w, pos, R)
    if layer.index >= 3:
        moe_block(layer, w, b, R)
''')
sys.path.insert(0, tmp)
os.environ.update(GLM53_TF_DRAFT_DUMP_ROUTE="1", GLM53_TF_DRAFT_DUMP_DSA_SEL="1", GLM53_TF_DRAFT_DUMP_KDA_GATES="1",
                  GLM53_TF_DRAFT_DUMP_DSA_STRIDE="8")
rt = importlib.import_module("fakepkg.routetap")
fw = importlib.import_module("fakepkg.forward")
sp = importlib.import_module("fakepkg.sparse")
L = 8
rt.check(0, L)
NS = types.SimpleNamespace
layers = [NS(index=i, kind="dsa" if i % 4 == 3 else "kda", kda=NS(a_log=torch.zeros(4), dt_bias=torch.zeros(4 * 128))) for i in range(L)]
mtp = NS(index=L, kind="dsa", kda=None)
w = NS(cfg=NS(top_k=8, lower=-5.0))
R, pos = 64, 2048
b = NS(pick=torch.zeros(128, 9, dtype=torch.int32), wts=torch.zeros(128, 9), ka=torch.zeros(128, 512, dtype=torch.bfloat16))
rt.begin(R, pos, False)
picks = {}
for l in layers:
    fw.layer_forward(l, w, None, b, R, pos)
    if l.index >= 3:
        picks[l.index] = b.pick[:R, :8].clone()
fw.moe_block(mtp, w, b, R)                  # the MTP layer: never recorded
arrays, extra = rt.collect()
assert extra["route_layers"] == [3, 4, 5, 6, 7], extra
assert arrays["route_ids"].shape == (R, 5, 8) and arrays["route_ids"].dtype == torch.int16
for j, l in enumerate(extra["route_layers"]):
    assert torch.equal(arrays["route_ids"][:, j].int(), picks[l]), l
assert extra["kda_layers"] == [0, 1, 2, 4, 5, 6] and arrays["kda_lg"].shape == (R, 6, 4)
assert (arrays["kda_lg"] <= 0).all() and (arrays["kda_lg"] >= -5).all()
# selections: DSA layers 3 and 7, rows with pos % 8 == 0 that are sparse (pos 2048.. -> npool > 512 from pos 2051)
sel, sp_pos, sl = arrays["dsa_sel"], arrays["dsa_pos"], arrays["dsa_layer"]
expect_rows = [p for p in range(pos, pos + R) if p % 8 == 0 and (p + 1) // 4 > 512]
assert sp_pos[: len(expect_rows)].tolist() == expect_rows, sp_pos.tolist()
assert sorted(set(sl.tolist())) == [3, 7]
first = sp.LAST[0]
for i, p in enumerate(expect_rows):
    assert torch.equal(sel[i].long(), first[p - pos]), i
# disarmed: nothing is recorded outside a chunk, and lean chunks record nothing
fw.layer_forward(layers[3], w, None, b, R, pos)
assert rt.collect() == ({}, {})
rt.begin(R, pos, True)
fw.layer_forward(layers[3], w, None, b, R, pos)
assert rt.collect() == ({}, {})
# lean on is refused by name
os.environ["GLM53_TF_LEAN_PREFILL"] = "1"
try:
    rt.check(0, L)
    raise SystemExit("lean prefill was not refused")
except RuntimeError as e:
    assert "GLM53_TF_LEAN_PREFILL=1" in str(e)
print("routetap harness: all checks passed")
