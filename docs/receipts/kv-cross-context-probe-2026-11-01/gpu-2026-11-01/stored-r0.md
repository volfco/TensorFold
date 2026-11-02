# Stored-state cross-context comparison (rank 0)

## boilerplate

### A+S vs B+S: 1739 rows of S, keeps at [8192, 8192], [4241, 4241] tokens past S

| layer | latent rel L2 p50 | p90 | head p50 | body p50 | cos p50 | other-token p50 | raw | xor | delta | ring keys p50 | pooled keys p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.094 | 0.140 | 0.174 | 0.093 | 0.996 | 1.229 | 0.668 | 0.529 | 0.539 | n/a | 0.080 |
| 7 | 0.141 | 0.212 | 0.257 | 0.139 | 0.990 | 1.082 | 0.669 | 0.563 | 0.570 | n/a | 0.057 |
| 11 | 0.200 | 0.269 | 0.271 | 0.198 | 0.980 | 1.201 | 0.665 | 0.588 | 0.593 | n/a | 0.132 |
| 15 | 0.383 | 0.610 | 0.502 | 0.380 | 0.927 | 1.221 | 0.661 | 0.631 | 0.636 | n/a | 0.218 |
| 19 | 0.637 | 1.062 | 0.692 | 0.634 | 0.799 | 1.268 | 0.662 | 0.663 | 0.667 | n/a | 0.350 |
| 23 | 0.714 | 1.144 | 0.637 | 0.716 | 0.747 | 1.250 | 0.662 | 0.668 | 0.672 | n/a | 0.387 |
| 27 | 0.708 | 1.161 | 0.635 | 0.716 | 0.749 | 1.246 | 0.661 | 0.666 | 0.673 | n/a | 0.391 |
| 31 | 0.476 | 1.002 | 0.433 | 0.477 | 0.887 | 1.221 | 0.661 | 0.650 | 0.656 | n/a | 0.322 |
| 35 | 0.506 | 1.085 | 0.443 | 0.510 | 0.872 | 1.238 | 0.661 | 0.654 | 0.659 | n/a | 0.297 |
| 39 | 0.548 | 1.139 | 0.524 | 0.550 | 0.850 | 1.237 | 0.663 | 0.657 | 0.663 | n/a | 0.294 |
| 43 | 0.529 | 1.131 | 0.496 | 0.530 | 0.861 | 1.236 | 0.662 | 0.656 | 0.662 | n/a | 0.342 |

| KDA layer | state rel p50 over heads | p90 | eff. rank p50 | rank90 p50 | diff eff. rank p50 | diff energy in rank 8 p50 | rank 32 | conv rel L2 p50 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.010 | 2.009 | 2.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0.009 | 0.029 | 3.188 | 3.500 | 4.290 | 0.952 | 0.999 | 0.005 |
| 2 | 0.037 | 0.079 | 2.469 | 3.000 | 2.651 | 0.989 | 1.000 | 0.016 |
| 4 | 0.057 | 0.072 | 3.203 | 3.000 | 5.431 | 0.972 | 1.000 | 0.020 |
| 5 | 0.063 | 0.093 | 2.230 | 2.000 | 4.774 | 0.973 | 1.000 | 0.041 |
| 6 | 0.065 | 0.117 | 4.489 | 5.500 | 8.391 | 0.884 | 0.993 | 0.064 |
| 8 | 0.097 | 0.155 | 3.325 | 4.000 | 7.145 | 0.906 | 0.998 | 0.071 |
| 9 | 0.073 | 0.112 | 2.432 | 3.000 | 5.348 | 0.946 | 1.000 | 0.063 |
| 10 | 0.077 | 0.162 | 3.877 | 4.000 | 7.509 | 0.899 | 0.998 | 0.054 |
| 12 | 0.136 | 0.220 | 3.596 | 5.000 | 8.120 | 0.856 | 0.992 | 0.099 |
| 13 | 0.152 | 0.255 | 5.863 | 6.500 | 13.032 | 0.797 | 0.978 | 0.114 |
| 14 | 0.214 | 0.354 | 7.694 | 11.500 | 17.399 | 0.714 | 0.955 | 0.107 |
| 16 | 0.191 | 0.326 | 4.671 | 7.000 | 10.022 | 0.833 | 0.983 | 0.117 |
| 17 | 0.231 | 0.296 | 5.142 | 7.000 | 15.415 | 0.762 | 0.974 | 0.179 |
| 18 | 0.305 | 0.378 | 9.717 | 13.500 | 21.038 | 0.666 | 0.947 | 0.249 |
| 20 | 0.362 | 0.446 | 8.512 | 10.500 | 20.288 | 0.688 | 0.949 | 0.305 |
| 21 | 0.356 | 0.470 | 6.923 | 9.000 | 18.426 | 0.714 | 0.966 | 0.293 |
| 22 | 0.363 | 0.481 | 8.829 | 10.500 | 18.361 | 0.694 | 0.952 | 0.284 |
| 24 | 0.339 | 0.479 | 5.160 | 7.000 | 17.464 | 0.743 | 0.961 | 0.293 |
| 25 | 0.349 | 0.485 | 8.873 | 11.500 | 20.592 | 0.675 | 0.956 | 0.295 |
| 26 | 0.355 | 0.489 | 5.524 | 7.000 | 13.907 | 0.784 | 0.978 | 0.278 |
| 28 | 0.340 | 0.575 | 4.241 | 5.000 | 12.069 | 0.821 | 0.987 | 0.239 |
| 29 | 0.294 | 0.484 | 5.095 | 6.000 | 14.219 | 0.776 | 0.987 | 0.261 |
| 30 | 0.242 | 0.386 | 2.995 | 3.000 | 9.788 | 0.848 | 0.986 | 0.193 |
| 32 | 0.254 | 0.381 | 4.279 | 4.000 | 9.631 | 0.844 | 0.992 | 0.206 |
| 33 | 0.324 | 0.491 | 6.540 | 10.000 | 22.512 | 0.655 | 0.952 | 0.261 |
| 34 | 0.238 | 0.349 | 2.590 | 3.000 | 9.009 | 0.868 | 0.991 | 0.154 |
| 36 | 0.211 | 0.395 | 5.808 | 8.500 | 13.476 | 0.792 | 0.979 | 0.164 |
| 37 | 0.292 | 0.374 | 6.871 | 8.000 | 17.993 | 0.686 | 0.961 | 0.234 |
| 38 | 0.286 | 0.562 | 4.559 | 7.500 | 14.903 | 0.772 | 0.973 | 0.231 |
| 40 | 0.300 | 0.457 | 6.607 | 7.500 | 19.951 | 0.692 | 0.967 | 0.234 |
| 41 | 0.322 | 0.404 | 6.287 | 9.000 | 19.709 | 0.673 | 0.954 | 0.209 |
| 42 | 0.337 | 0.538 | 7.384 | 11.000 | 24.208 | 0.610 | 0.949 | 0.203 |
| 44 | 0.286 | 0.360 | 3.359 | 4.500 | 14.126 | 0.770 | 0.973 | 0.188 |

### S vs A+S: 1739 rows of S, keeps at [6144, 8192], [4241, 4241] tokens past S

| layer | latent rel L2 p50 | p90 | head p50 | body p50 | cos p50 | other-token p50 | raw | xor | delta | ring keys p50 | pooled keys p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.119 | 0.178 | 0.235 | 0.118 | 0.993 | 1.227 | 0.668 | 0.550 | 0.559 | n/a | 0.121 |
| 7 | 0.172 | 0.266 | 0.358 | 0.170 | 0.985 | 1.094 | 0.669 | 0.581 | 0.586 | n/a | 0.083 |
| 11 | 0.209 | 0.310 | 0.335 | 0.206 | 0.978 | 1.202 | 0.665 | 0.593 | 0.597 | n/a | 0.138 |
| 15 | 0.384 | 0.619 | 0.511 | 0.378 | 0.927 | 1.225 | 0.661 | 0.632 | 0.636 | n/a | 0.205 |
| 19 | 0.614 | 1.081 | 0.651 | 0.612 | 0.814 | 1.266 | 0.662 | 0.661 | 0.665 | n/a | 0.321 |
| 23 | 0.654 | 1.133 | 0.616 | 0.658 | 0.783 | 1.252 | 0.662 | 0.665 | 0.668 | n/a | 0.359 |
| 27 | 0.663 | 1.153 | 0.572 | 0.665 | 0.779 | 1.252 | 0.661 | 0.662 | 0.668 | n/a | 0.358 |
| 31 | 0.440 | 0.954 | 0.408 | 0.442 | 0.903 | 1.218 | 0.661 | 0.644 | 0.649 | n/a | 0.266 |
| 35 | 0.466 | 1.033 | 0.413 | 0.468 | 0.891 | 1.237 | 0.661 | 0.647 | 0.653 | n/a | 0.258 |
| 39 | 0.511 | 1.089 | 0.498 | 0.512 | 0.868 | 1.237 | 0.663 | 0.652 | 0.658 | n/a | 0.262 |
| 43 | 0.496 | 1.112 | 0.522 | 0.493 | 0.877 | 1.235 | 0.662 | 0.652 | 0.658 | n/a | 0.312 |

| KDA layer | state rel p50 over heads | p90 | eff. rank p50 | rank90 p50 | diff eff. rank p50 | diff energy in rank 8 p50 | rank 32 | conv rel L2 p50 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.008 | 2.011 | 2.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0.008 | 0.033 | 3.194 | 3.500 | 4.229 | 0.952 | 0.999 | 0.003 |
| 2 | 0.041 | 0.085 | 2.459 | 3.000 | 2.406 | 0.990 | 1.000 | 0.016 |
| 4 | 0.046 | 0.071 | 3.299 | 3.000 | 5.263 | 0.975 | 1.000 | 0.019 |
| 5 | 0.043 | 0.085 | 2.226 | 2.000 | 4.974 | 0.960 | 1.000 | 0.029 |
| 6 | 0.063 | 0.123 | 4.517 | 5.500 | 7.437 | 0.884 | 0.994 | 0.049 |
| 8 | 0.076 | 0.138 | 3.522 | 4.000 | 7.754 | 0.882 | 0.998 | 0.057 |
| 9 | 0.066 | 0.106 | 2.501 | 3.000 | 5.743 | 0.943 | 1.000 | 0.048 |
| 10 | 0.075 | 0.169 | 3.817 | 4.000 | 6.106 | 0.907 | 0.998 | 0.048 |
| 12 | 0.107 | 0.175 | 3.620 | 4.500 | 8.759 | 0.858 | 0.990 | 0.075 |
| 13 | 0.135 | 0.216 | 5.601 | 6.500 | 12.961 | 0.792 | 0.980 | 0.082 |
| 14 | 0.184 | 0.308 | 7.262 | 11.500 | 19.771 | 0.673 | 0.949 | 0.080 |
| 16 | 0.162 | 0.280 | 4.538 | 6.500 | 11.254 | 0.815 | 0.982 | 0.108 |
| 17 | 0.205 | 0.254 | 4.871 | 7.000 | 16.022 | 0.750 | 0.975 | 0.151 |
| 18 | 0.267 | 0.338 | 9.602 | 13.000 | 21.370 | 0.666 | 0.945 | 0.215 |
| 20 | 0.332 | 0.426 | 8.049 | 10.000 | 18.447 | 0.702 | 0.951 | 0.231 |
| 21 | 0.335 | 0.413 | 6.693 | 8.500 | 17.047 | 0.732 | 0.970 | 0.235 |
| 22 | 0.336 | 0.452 | 8.145 | 10.000 | 20.369 | 0.680 | 0.955 | 0.243 |
| 24 | 0.312 | 0.435 | 5.094 | 7.000 | 16.609 | 0.739 | 0.963 | 0.238 |
| 25 | 0.353 | 0.434 | 8.430 | 11.000 | 19.261 | 0.701 | 0.961 | 0.255 |
| 26 | 0.325 | 0.478 | 5.338 | 7.000 | 14.154 | 0.790 | 0.977 | 0.234 |
| 28 | 0.324 | 0.558 | 4.244 | 5.000 | 11.738 | 0.820 | 0.988 | 0.210 |
| 29 | 0.280 | 0.487 | 4.920 | 5.500 | 15.500 | 0.760 | 0.986 | 0.232 |
| 30 | 0.256 | 0.399 | 2.846 | 3.000 | 8.377 | 0.878 | 0.991 | 0.159 |
| 32 | 0.281 | 0.620 | 4.093 | 4.000 | 7.594 | 0.880 | 0.995 | 0.165 |
| 33 | 0.330 | 0.529 | 6.292 | 9.000 | 20.296 | 0.676 | 0.958 | 0.212 |
| 34 | 0.295 | 0.628 | 2.484 | 3.000 | 4.532 | 0.936 | 0.997 | 0.127 |
| 36 | 0.200 | 0.369 | 5.837 | 7.500 | 12.776 | 0.791 | 0.980 | 0.145 |
| 37 | 0.275 | 0.369 | 6.516 | 8.000 | 17.314 | 0.717 | 0.965 | 0.188 |
| 38 | 0.307 | 0.654 | 4.074 | 7.000 | 10.091 | 0.825 | 0.981 | 0.193 |
| 40 | 0.344 | 0.474 | 6.476 | 7.500 | 17.768 | 0.721 | 0.971 | 0.189 |
| 41 | 0.334 | 0.451 | 6.624 | 9.000 | 16.454 | 0.729 | 0.957 | 0.177 |
| 42 | 0.413 | 0.581 | 7.185 | 10.500 | 14.895 | 0.742 | 0.966 | 0.183 |
| 44 | 0.318 | 0.635 | 3.517 | 4.500 | 6.409 | 0.892 | 0.990 | 0.154 |

### L+S vs A+S: 1739 rows of S, keeps at [14336, 8192], [4241, 4241] tokens past S

| layer | latent rel L2 p50 | p90 | head p50 | body p50 | cos p50 | other-token p50 | raw | xor | delta | ring keys p50 | pooled keys p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.077 | 0.107 | 0.132 | 0.076 | 0.997 | 1.227 | 0.668 | 0.511 | 0.523 | n/a | 0.071 |
| 7 | 0.114 | 0.173 | 0.173 | 0.112 | 0.994 | 1.094 | 0.669 | 0.547 | 0.556 | n/a | 0.053 |
| 11 | 0.167 | 0.219 | 0.189 | 0.166 | 0.986 | 1.202 | 0.665 | 0.574 | 0.580 | n/a | 0.123 |
| 15 | 0.321 | 0.515 | 0.419 | 0.318 | 0.948 | 1.225 | 0.661 | 0.621 | 0.625 | n/a | 0.186 |
| 19 | 0.570 | 0.956 | 0.571 | 0.570 | 0.837 | 1.266 | 0.662 | 0.657 | 0.661 | n/a | 0.313 |
| 23 | 0.652 | 1.106 | 0.571 | 0.661 | 0.785 | 1.252 | 0.662 | 0.663 | 0.667 | n/a | 0.353 |
| 27 | 0.659 | 1.155 | 0.570 | 0.665 | 0.782 | 1.252 | 0.661 | 0.662 | 0.669 | n/a | 0.357 |
| 31 | 0.422 | 0.970 | 0.361 | 0.426 | 0.911 | 1.218 | 0.661 | 0.645 | 0.650 | n/a | 0.277 |
| 35 | 0.451 | 1.070 | 0.367 | 0.462 | 0.898 | 1.237 | 0.661 | 0.649 | 0.655 | n/a | 0.268 |
| 39 | 0.511 | 1.123 | 0.450 | 0.517 | 0.869 | 1.237 | 0.663 | 0.654 | 0.660 | n/a | 0.279 |
| 43 | 0.500 | 1.120 | 0.460 | 0.503 | 0.875 | 1.235 | 0.662 | 0.652 | 0.658 | n/a | 0.316 |

| KDA layer | state rel p50 over heads | p90 | eff. rank p50 | rank90 p50 | diff eff. rank p50 | diff energy in rank 8 p50 | rank 32 | conv rel L2 p50 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.007 | 2.011 | 2.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0.009 | 0.047 | 3.194 | 3.500 | 4.185 | 0.960 | 1.000 | 0.005 |
| 2 | 0.033 | 0.070 | 2.459 | 3.000 | 2.679 | 0.987 | 0.999 | 0.018 |
| 4 | 0.043 | 0.081 | 3.299 | 3.000 | 6.212 | 0.947 | 1.000 | 0.020 |
| 5 | 0.053 | 0.116 | 2.226 | 2.000 | 5.426 | 0.958 | 1.000 | 0.031 |
| 6 | 0.066 | 0.135 | 4.517 | 5.500 | 8.867 | 0.864 | 0.992 | 0.051 |
| 8 | 0.110 | 0.167 | 3.522 | 4.000 | 7.589 | 0.895 | 0.997 | 0.085 |
| 9 | 0.085 | 0.124 | 2.501 | 3.000 | 5.172 | 0.944 | 1.000 | 0.058 |
| 10 | 0.087 | 0.256 | 3.817 | 4.000 | 6.683 | 0.902 | 0.998 | 0.052 |
| 12 | 0.157 | 0.256 | 3.620 | 4.500 | 9.641 | 0.842 | 0.988 | 0.118 |
| 13 | 0.185 | 0.318 | 5.601 | 6.500 | 13.573 | 0.788 | 0.980 | 0.118 |
| 14 | 0.249 | 0.371 | 7.262 | 11.500 | 20.235 | 0.675 | 0.948 | 0.122 |
| 16 | 0.212 | 0.366 | 4.538 | 6.500 | 11.330 | 0.800 | 0.983 | 0.142 |
| 17 | 0.265 | 0.325 | 4.871 | 7.000 | 13.962 | 0.763 | 0.976 | 0.157 |
| 18 | 0.331 | 0.395 | 9.602 | 13.000 | 20.127 | 0.670 | 0.947 | 0.209 |
| 20 | 0.361 | 0.458 | 8.049 | 10.000 | 20.699 | 0.680 | 0.950 | 0.246 |
| 21 | 0.344 | 0.458 | 6.693 | 8.500 | 18.754 | 0.705 | 0.968 | 0.244 |
| 22 | 0.373 | 0.491 | 8.145 | 10.000 | 16.896 | 0.732 | 0.959 | 0.266 |
| 24 | 0.332 | 0.514 | 5.094 | 7.000 | 16.792 | 0.746 | 0.961 | 0.241 |
| 25 | 0.372 | 0.491 | 8.430 | 11.000 | 19.854 | 0.678 | 0.956 | 0.248 |
| 26 | 0.364 | 0.487 | 5.338 | 7.000 | 13.659 | 0.791 | 0.976 | 0.218 |
| 28 | 0.322 | 0.478 | 4.244 | 5.000 | 11.160 | 0.812 | 0.985 | 0.199 |
| 29 | 0.290 | 0.435 | 4.920 | 5.500 | 16.260 | 0.751 | 0.987 | 0.215 |
| 30 | 0.240 | 0.377 | 2.846 | 3.000 | 11.007 | 0.832 | 0.990 | 0.153 |
| 32 | 0.235 | 0.331 | 4.093 | 4.000 | 10.665 | 0.823 | 0.991 | 0.172 |
| 33 | 0.307 | 0.401 | 6.292 | 9.000 | 23.642 | 0.629 | 0.949 | 0.234 |
| 34 | 0.226 | 0.318 | 2.484 | 3.000 | 8.849 | 0.888 | 0.994 | 0.135 |
| 36 | 0.227 | 0.404 | 5.837 | 7.500 | 13.310 | 0.788 | 0.983 | 0.147 |
| 37 | 0.269 | 0.350 | 6.516 | 8.000 | 19.285 | 0.672 | 0.957 | 0.244 |
| 38 | 0.274 | 0.425 | 4.074 | 7.000 | 14.356 | 0.772 | 0.974 | 0.206 |
| 40 | 0.280 | 0.462 | 6.476 | 7.500 | 20.642 | 0.671 | 0.967 | 0.232 |
| 41 | 0.295 | 0.401 | 6.624 | 9.000 | 19.094 | 0.684 | 0.958 | 0.202 |
| 42 | 0.280 | 0.530 | 7.185 | 10.500 | 21.653 | 0.639 | 0.958 | 0.176 |
| 44 | 0.237 | 0.342 | 3.517 | 4.500 | 13.004 | 0.768 | 0.979 | 0.183 |

bench S: exit 0
    KVBENCH config mib=64 threads=8 reps=3 block=1048576
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep13.bin codec=memcpy threads=1 raw=259744768 stored=259744768 ratio=1.0000 encode_gbs=8.067 decode_gbs=8.067 keep256M_ms=33.3 restore256M_ms=33.3 keep1024M_ms=133.1 restore1024M_ms=133.1 keep4096M_ms=532.4 restore4096M_ms=532.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep13.bin codec=deflate1 threads=1 raw=259744768 stored=213296713 ratio=0.8212 encode_gbs=0.077 decode_gbs=0.180 keep256M_ms=3495.6 restore256M_ms=1488.6 keep1024M_ms=13982.6 restore1024M_ms=5954.4 keep4096M_ms=55930.3 restore4096M_ms=23817.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep13.bin codec=deflate6 threads=1 raw=259744768 stored=213275936 ratio=0.8211 encode_gbs=0.065 decode_gbs=0.180 keep256M_ms=4111.0 restore256M_ms=1494.1 keep1024M_ms=16443.8 restore1024M_ms=5976.5 keep4096M_ms=65775.3 restore4096M_ms=23906.0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep13.bin codec=plain threads=1 raw=259744768 stored=187858175 ratio=0.7232 encode_gbs=0.318 decode_gbs=1.248 keep256M_ms=845.4 restore256M_ms=215.0 keep1024M_ms=3381.5 restore1024M_ms=860.1 keep4096M_ms=13526.2 restore4096M_ms=3440.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep13.bin codec=plain engagement raw=0 planes=256 planes_xor=0 planes_delta=0 planes_delta2=0 t64=0 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep13.bin codec=plain threads=8 raw=259744768 stored=187858175 ratio=0.7232 encode_gbs=0.927 decode_gbs=3.629 keep256M_ms=289.4 restore256M_ms=74.0 keep1024M_ms=1157.7 restore1024M_ms=295.9 keep4096M_ms=4630.8 restore4096M_ms=1183.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep13.bin codec=atlas threads=1 raw=259744768 stored=187634658 ratio=0.7224 encode_gbs=0.098 decode_gbs=1.216 keep256M_ms=2737.9 restore256M_ms=220.8 keep1024M_ms=10951.5 restore1024M_ms=883.0 keep4096M_ms=43806.1 restore4096M_ms=3532.1
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep13.bin codec=atlas engagement raw=0 planes=233 planes_xor=0 planes_delta=22 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep13.bin codec=atlas threads=8 raw=259744768 stored=187634658 ratio=0.7224 encode_gbs=0.295 decode_gbs=3.468 keep256M_ms=908.8 restore256M_ms=77.4 keep1024M_ms=3635.2 restore1024M_ms=309.6 keep4096M_ms=14540.6 restore4096M_ms=1238.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep13.bin codec=atlas-exhaustive threads=1 raw=259744768 stored=187634323 ratio=0.7224 encode_gbs=0.027 decode_gbs=1.212 keep256M_ms=9890.7 restore256M_ms=221.5 keep1024M_ms=39562.7 restore1024M_ms=885.9 keep4096M_ms=158251.0 restore4096M_ms=3543.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep13.bin codec=atlas-exhaustive engagement raw=0 planes=232 planes_xor=0 planes_delta=23 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep13.bin codec=atlas-exhaustive threads=8 raw=259744768 stored=187634323 ratio=0.7224 encode_gbs=0.090 decode_gbs=3.096 keep256M_ms=2982.7 restore256M_ms=86.7 keep1024M_ms=11930.7 restore1024M_ms=346.8 keep4096M_ms=47722.9 restore4096M_ms=1387.2
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep13.bin codec=ch-doubledelta threads=1 raw=259744768 stored=242503669 ratio=0.9336 encode_gbs=0.520 decode_gbs=7.460 keep256M_ms=516.1 restore256M_ms=36.0 keep1024M_ms=2064.2 restore1024M_ms=143.9 keep4096M_ms=8256.9 restore4096M_ms=575.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep13.bin codec=ch-doubledelta width=2 raw_blocks=233/256
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep13.bin codec=ch-gorilla threads=1 raw=259744768 stored=242904833 ratio=0.9352 encode_gbs=0.864 decode_gbs=15.271 keep256M_ms=310.8 restore256M_ms=17.6 keep1024M_ms=1243.2 restore1024M_ms=70.3 keep4096M_ms=4972.6 restore4096M_ms=281.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep13.bin codec=ch-gorilla width=2 raw_blocks=238/256
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep13.bin codec=ch-fpc12 threads=1 raw=259744768 stored=243910783 ratio=0.9390 encode_gbs=1.032 decode_gbs=8.722 keep256M_ms=260.2 restore256M_ms=30.8 keep1024M_ms=1040.8 restore1024M_ms=123.1 keep4096M_ms=4163.2 restore4096M_ms=492.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep13.bin codec=ch-fpc12 width=2 raw_blocks=235/256
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep13.bin codec=ch-alp-rd threads=1 raw=259744768 stored=218220226 ratio=0.8401 encode_gbs=0.180 decode_gbs=2.269 keep256M_ms=1488.1 restore256M_ms=118.3 keep1024M_ms=5952.3 restore1024M_ms=473.2 keep4096M_ms=23809.1 restore4096M_ms=1893.0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep13.bin codec=ch-alp-rd width=2 raw_blocks=137/256
bench A+S: exit 0
    KVBENCH config mib=64 threads=8 reps=3 block=1048576
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep14.bin codec=memcpy threads=1 raw=284255232 stored=284255232 ratio=1.0000 encode_gbs=11.760 decode_gbs=11.760 keep256M_ms=22.8 restore256M_ms=22.8 keep1024M_ms=91.3 restore1024M_ms=91.3 keep4096M_ms=365.2 restore4096M_ms=365.2
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep14.bin codec=deflate1 threads=1 raw=284255232 stored=232842005 ratio=0.8191 encode_gbs=0.077 decode_gbs=0.178 keep256M_ms=3477.7 restore256M_ms=1507.3 keep1024M_ms=13910.8 restore1024M_ms=6029.4 keep4096M_ms=55643.1 restore4096M_ms=24117.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep14.bin codec=deflate6 threads=1 raw=284255232 stored=232819316 ratio=0.8191 encode_gbs=0.064 decode_gbs=0.178 keep256M_ms=4171.0 restore256M_ms=1511.1 keep1024M_ms=16683.9 restore1024M_ms=6044.3 keep4096M_ms=66735.7 restore4096M_ms=24177.1
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep14.bin codec=plain threads=1 raw=284255232 stored=204172300 ratio=0.7183 encode_gbs=0.317 decode_gbs=1.200 keep256M_ms=847.1 restore256M_ms=223.8 keep1024M_ms=3388.5 restore1024M_ms=895.0 keep4096M_ms=13554.2 restore4096M_ms=3580.1
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep14.bin codec=plain engagement raw=0 planes=278 planes_xor=0 planes_delta=0 planes_delta2=0 t64=0 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep14.bin codec=plain threads=8 raw=284255232 stored=204172300 ratio=0.7183 encode_gbs=1.759 decode_gbs=6.394 keep256M_ms=152.6 restore256M_ms=42.0 keep1024M_ms=610.4 restore1024M_ms=167.9 keep4096M_ms=2441.6 restore4096M_ms=671.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep14.bin codec=atlas threads=1 raw=284255232 stored=203911299 ratio=0.7174 encode_gbs=0.096 decode_gbs=1.168 keep256M_ms=2798.7 restore256M_ms=229.8 keep1024M_ms=11194.8 restore1024M_ms=919.2 keep4096M_ms=44779.4 restore4096M_ms=3676.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep14.bin codec=atlas engagement raw=0 planes=253 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep14.bin codec=atlas threads=8 raw=284255232 stored=203911299 ratio=0.7174 encode_gbs=0.436 decode_gbs=5.168 keep256M_ms=615.6 restore256M_ms=51.9 keep1024M_ms=2462.3 restore1024M_ms=207.8 keep4096M_ms=9849.1 restore4096M_ms=831.1
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep14.bin codec=atlas-exhaustive threads=1 raw=284255232 stored=203911299 ratio=0.7174 encode_gbs=0.028 decode_gbs=1.167 keep256M_ms=9574.3 restore256M_ms=230.0 keep1024M_ms=38297.0 restore1024M_ms=920.0 keep4096M_ms=153188.1 restore4096M_ms=3680.2
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep14.bin codec=atlas-exhaustive engagement raw=0 planes=253 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep14.bin codec=atlas-exhaustive threads=8 raw=284255232 stored=203911299 ratio=0.7174 encode_gbs=0.154 decode_gbs=6.107 keep256M_ms=1739.8 restore256M_ms=44.0 keep1024M_ms=6959.1 restore1024M_ms=175.8 keep4096M_ms=27836.5 restore4096M_ms=703.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep14.bin codec=ch-doubledelta threads=1 raw=284255232 stored=266957361 ratio=0.9391 encode_gbs=0.500 decode_gbs=8.063 keep256M_ms=537.4 restore256M_ms=33.3 keep1024M_ms=2149.6 restore1024M_ms=133.2 keep4096M_ms=8598.3 restore4096M_ms=532.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep14.bin codec=ch-doubledelta width=2 raw_blocks=255/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep14.bin codec=ch-gorilla threads=1 raw=284255232 stored=267415293 ratio=0.9408 encode_gbs=0.864 decode_gbs=17.657 keep256M_ms=310.8 restore256M_ms=15.2 keep1024M_ms=1243.1 restore1024M_ms=60.8 keep4096M_ms=4972.5 restore4096M_ms=243.2
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep14.bin codec=ch-gorilla width=2 raw_blocks=260/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep14.bin codec=ch-fpc12 threads=1 raw=284255232 stored=268411875 ratio=0.9443 encode_gbs=1.028 decode_gbs=9.265 keep256M_ms=261.1 restore256M_ms=29.0 keep1024M_ms=1044.3 restore1024M_ms=115.9 keep4096M_ms=4177.1 restore4096M_ms=463.6
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep14.bin codec=ch-fpc12 width=2 raw_blocks=257/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep14.bin codec=ch-alp-rd threads=1 raw=284255232 stored=236501384 ratio=0.8320 encode_gbs=0.185 decode_gbs=2.038 keep256M_ms=1452.8 restore256M_ms=131.7 keep1024M_ms=5811.1 restore1024M_ms=526.9 keep4096M_ms=23244.2 restore4096M_ms=2107.6
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep14.bin codec=ch-alp-rd width=2 raw_blocks=137/278
bench B+S: exit 0
    KVBENCH config mib=64 threads=8 reps=3 block=1048576
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep15.bin codec=memcpy threads=1 raw=284255232 stored=284255232 ratio=1.0000 encode_gbs=24.130 decode_gbs=24.130 keep256M_ms=11.1 restore256M_ms=11.1 keep1024M_ms=44.5 restore1024M_ms=44.5 keep4096M_ms=178.0 restore4096M_ms=178.0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep15.bin codec=deflate1 threads=1 raw=284255232 stored=232843012 ratio=0.8191 encode_gbs=0.077 decode_gbs=0.178 keep256M_ms=3479.1 restore256M_ms=1508.3 keep1024M_ms=13916.4 restore1024M_ms=6033.4 keep4096M_ms=55665.7 restore4096M_ms=24133.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep15.bin codec=deflate6 threads=1 raw=284255232 stored=232820350 ratio=0.8191 encode_gbs=0.064 decode_gbs=0.177 keep256M_ms=4169.7 restore256M_ms=1512.4 keep1024M_ms=16678.8 restore1024M_ms=6049.7 keep4096M_ms=66715.1 restore4096M_ms=24198.9
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep15.bin codec=plain threads=1 raw=284255232 stored=204182788 ratio=0.7183 encode_gbs=0.316 decode_gbs=1.215 keep256M_ms=848.4 restore256M_ms=221.0 keep1024M_ms=3393.7 restore1024M_ms=883.9 keep4096M_ms=13574.9 restore4096M_ms=3535.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep15.bin codec=plain engagement raw=0 planes=278 planes_xor=0 planes_delta=0 planes_delta2=0 t64=0 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep15.bin codec=plain threads=8 raw=284255232 stored=204182788 ratio=0.7183 encode_gbs=1.751 decode_gbs=6.473 keep256M_ms=153.3 restore256M_ms=41.5 keep1024M_ms=613.4 restore1024M_ms=165.9 keep4096M_ms=2453.5 restore4096M_ms=663.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep15.bin codec=atlas threads=1 raw=284255232 stored=203927644 ratio=0.7174 encode_gbs=0.096 decode_gbs=1.179 keep256M_ms=2791.0 restore256M_ms=227.6 keep1024M_ms=11163.8 restore1024M_ms=910.4 keep4096M_ms=44655.3 restore4096M_ms=3641.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep15.bin codec=atlas engagement raw=0 planes=253 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep15.bin codec=atlas threads=8 raw=284255232 stored=203927644 ratio=0.7174 encode_gbs=0.530 decode_gbs=6.153 keep256M_ms=506.7 restore256M_ms=43.6 keep1024M_ms=2026.7 restore1024M_ms=174.5 keep4096M_ms=8106.8 restore4096M_ms=698.1
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep15.bin codec=atlas-exhaustive threads=1 raw=284255232 stored=203927644 ratio=0.7174 encode_gbs=0.028 decode_gbs=1.178 keep256M_ms=9567.1 restore256M_ms=227.8 keep1024M_ms=38268.3 restore1024M_ms=911.4 keep4096M_ms=153073.0 restore4096M_ms=3645.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep15.bin codec=atlas-exhaustive engagement raw=0 planes=253 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep15.bin codec=atlas-exhaustive threads=8 raw=284255232 stored=203927644 ratio=0.7174 encode_gbs=0.156 decode_gbs=6.170 keep256M_ms=1725.6 restore256M_ms=43.5 keep1024M_ms=6902.4 restore1024M_ms=174.0 keep4096M_ms=27609.5 restore4096M_ms=696.1
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep15.bin codec=ch-doubledelta threads=1 raw=284255232 stored=266952723 ratio=0.9391 encode_gbs=0.500 decode_gbs=8.097 keep256M_ms=537.0 restore256M_ms=33.2 keep1024M_ms=2147.8 restore1024M_ms=132.6 keep4096M_ms=8591.3 restore4096M_ms=530.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep15.bin codec=ch-doubledelta width=2 raw_blocks=255/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep15.bin codec=ch-gorilla threads=1 raw=284255232 stored=267415298 ratio=0.9408 encode_gbs=0.859 decode_gbs=15.943 keep256M_ms=312.6 restore256M_ms=16.8 keep1024M_ms=1250.5 restore1024M_ms=67.3 keep4096M_ms=5001.8 restore4096M_ms=269.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep15.bin codec=ch-gorilla width=2 raw_blocks=260/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep15.bin codec=ch-fpc12 threads=1 raw=284255232 stored=268406492 ratio=0.9442 encode_gbs=1.031 decode_gbs=9.321 keep256M_ms=260.2 restore256M_ms=28.8 keep1024M_ms=1041.0 restore1024M_ms=115.2 keep4096M_ms=4163.8 restore4096M_ms=460.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep15.bin codec=ch-fpc12 width=2 raw_blocks=257/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep15.bin codec=ch-alp-rd threads=1 raw=284255232 stored=236593654 ratio=0.8323 encode_gbs=0.184 decode_gbs=2.038 keep256M_ms=1459.4 restore256M_ms=131.7 keep1024M_ms=5837.5 restore1024M_ms=526.9 keep4096M_ms=23350.1 restore4096M_ms=2107.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep15.bin codec=ch-alp-rd width=2 raw_blocks=137/278
bench L+S: exit 0
    KVBENCH config mib=64 threads=8 reps=3 block=1048576
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep16.bin codec=memcpy threads=1 raw=357786624 stored=357786624 ratio=1.0000 encode_gbs=24.702 decode_gbs=24.702 keep256M_ms=10.9 restore256M_ms=10.9 keep1024M_ms=43.5 restore1024M_ms=43.5 keep4096M_ms=173.9 restore4096M_ms=173.9
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep16.bin codec=deflate1 threads=1 raw=357786624 stored=291521040 ratio=0.8148 encode_gbs=0.078 decode_gbs=0.161 keep256M_ms=3454.9 restore256M_ms=1666.6 keep1024M_ms=13819.8 restore1024M_ms=6666.3 keep4096M_ms=55279.2 restore4096M_ms=26665.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep16.bin codec=deflate6 threads=1 raw=357786624 stored=291493865 ratio=0.8147 encode_gbs=0.062 decode_gbs=0.160 keep256M_ms=4303.1 restore256M_ms=1672.8 keep1024M_ms=17212.3 restore1024M_ms=6691.1 keep4096M_ms=68849.0 restore4096M_ms=26764.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep16.bin codec=plain threads=1 raw=357786624 stored=253155970 ratio=0.7076 encode_gbs=0.312 decode_gbs=1.118 keep256M_ms=860.5 restore256M_ms=240.1 keep1024M_ms=3442.1 restore1024M_ms=960.3 keep4096M_ms=13768.5 restore4096M_ms=3841.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep16.bin codec=plain engagement raw=0 planes=344 planes_xor=0 planes_delta=0 planes_delta2=0 t64=0 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep16.bin codec=plain threads=8 raw=357786624 stored=253155970 ratio=0.7076 encode_gbs=1.701 decode_gbs=5.953 keep256M_ms=157.8 restore256M_ms=45.1 keep1024M_ms=631.2 restore1024M_ms=180.4 keep4096M_ms=2524.7 restore4096M_ms=721.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep16.bin codec=atlas threads=1 raw=357786624 stored=252801727 ratio=0.7066 encode_gbs=0.091 decode_gbs=1.093 keep256M_ms=2952.2 restore256M_ms=245.5 keep1024M_ms=11808.8 restore1024M_ms=982.0 keep4096M_ms=47235.2 restore4096M_ms=3928.0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep16.bin codec=atlas engagement raw=0 planes=319 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep16.bin codec=atlas threads=8 raw=357786624 stored=252801727 ratio=0.7066 encode_gbs=0.495 decode_gbs=5.769 keep256M_ms=541.9 restore256M_ms=46.5 keep1024M_ms=2167.6 restore1024M_ms=186.1 keep4096M_ms=8670.5 restore4096M_ms=744.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep16.bin codec=atlas-exhaustive threads=1 raw=357786624 stored=252801727 ratio=0.7066 encode_gbs=0.026 decode_gbs=1.093 keep256M_ms=10419.2 restore256M_ms=245.6 keep1024M_ms=41676.9 restore1024M_ms=982.3 keep4096M_ms=166707.5 restore4096M_ms=3929.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep16.bin codec=atlas-exhaustive engagement raw=0 planes=319 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep16.bin codec=atlas-exhaustive threads=8 raw=357786624 stored=252801727 ratio=0.7066 encode_gbs=0.141 decode_gbs=5.824 keep256M_ms=1909.7 restore256M_ms=46.1 keep1024M_ms=7638.7 restore1024M_ms=184.4 keep4096M_ms=30554.9 restore4096M_ms=737.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep16.bin codec=ch-doubledelta threads=1 raw=357786624 stored=340484882 ratio=0.9516 encode_gbs=0.460 decode_gbs=9.377 keep256M_ms=584.2 restore256M_ms=28.6 keep1024M_ms=2336.7 restore1024M_ms=114.5 keep4096M_ms=9347.0 restore4096M_ms=458.0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep16.bin codec=ch-doubledelta width=2 raw_blocks=321/344
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep16.bin codec=ch-gorilla threads=1 raw=357786624 stored=340946694 ratio=0.9529 encode_gbs=0.855 decode_gbs=18.847 keep256M_ms=314.0 restore256M_ms=14.2 keep1024M_ms=1256.2 restore1024M_ms=57.0 keep4096M_ms=5024.7 restore4096M_ms=227.9
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep16.bin codec=ch-gorilla width=2 raw_blocks=326/344
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep16.bin codec=ch-fpc12 threads=1 raw=357786624 stored=341941759 ratio=0.9557 encode_gbs=1.003 decode_gbs=10.680 keep256M_ms=267.8 restore256M_ms=25.1 keep1024M_ms=1071.0 restore1024M_ms=100.5 keep4096M_ms=4284.1 restore4096M_ms=402.1
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep16.bin codec=ch-fpc12 width=2 raw_blocks=323/344
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep16.bin codec=ch-alp-rd threads=1 raw=357786624 stored=291345940 ratio=0.8143 encode_gbs=0.192 decode_gbs=1.664 keep256M_ms=1399.0 restore256M_ms=161.4 keep1024M_ms=5595.9 restore1024M_ms=645.4 keep4096M_ms=22383.7 restore4096M_ms=2581.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep16.bin codec=ch-alp-rd width=2 raw_blocks=137/344

## code

### A+S vs B+S: 2055 rows of S, keeps at [8192, 8192], [4069, 4069] tokens past S

| layer | latent rel L2 p50 | p90 | head p50 | body p50 | cos p50 | other-token p50 | raw | xor | delta | ring keys p50 | pooled keys p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.114 | 0.207 | 0.238 | 0.112 | 0.994 | 1.350 | 0.668 | 0.548 | 0.557 | 0.064 | 0.116 |
| 7 | 0.134 | 0.286 | 0.342 | 0.131 | 0.991 | 1.329 | 0.669 | 0.565 | 0.572 | 0.035 | 0.073 |
| 11 | 0.148 | 0.256 | 0.321 | 0.146 | 0.989 | 1.346 | 0.665 | 0.570 | 0.576 | 0.084 | 0.104 |
| 15 | 0.258 | 0.404 | 0.394 | 0.254 | 0.967 | 1.338 | 0.661 | 0.606 | 0.611 | 0.188 | 0.183 |
| 19 | 0.402 | 0.756 | 0.551 | 0.396 | 0.919 | 1.324 | 0.662 | 0.639 | 0.642 | 0.282 | 0.289 |
| 23 | 0.450 | 0.934 | 0.572 | 0.443 | 0.898 | 1.317 | 0.662 | 0.646 | 0.649 | 0.369 | 0.309 |
| 27 | 0.458 | 0.946 | 0.557 | 0.452 | 0.894 | 1.311 | 0.661 | 0.646 | 0.651 | 0.350 | 0.338 |
| 31 | 0.353 | 0.708 | 0.448 | 0.348 | 0.938 | 1.320 | 0.661 | 0.630 | 0.634 | 0.406 | 0.271 |
| 35 | 0.369 | 0.841 | 0.427 | 0.363 | 0.932 | 1.302 | 0.661 | 0.634 | 0.639 | 0.325 | 0.245 |
| 39 | 0.398 | 0.957 | 0.457 | 0.395 | 0.921 | 1.292 | 0.663 | 0.641 | 0.646 | 0.422 | 0.229 |
| 43 | 0.421 | 1.031 | 0.510 | 0.416 | 0.911 | 1.315 | 0.662 | 0.645 | 0.650 | 0.576 | 0.286 |

| KDA layer | state rel p50 over heads | p90 | eff. rank p50 | rank90 p50 | diff eff. rank p50 | diff energy in rank 8 p50 | rank 32 | conv rel L2 p50 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.009 | 2.113 | 2.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0.009 | 0.031 | 3.151 | 3.000 | 4.452 | 0.963 | 1.000 | 0.006 |
| 2 | 0.047 | 0.108 | 2.369 | 3.000 | 2.447 | 0.990 | 1.000 | 0.021 |
| 4 | 0.047 | 0.100 | 2.234 | 2.000 | 3.604 | 0.990 | 1.000 | 0.040 |
| 5 | 0.080 | 0.128 | 1.884 | 2.000 | 4.252 | 0.983 | 1.000 | 0.046 |
| 6 | 0.089 | 0.148 | 3.800 | 4.500 | 7.915 | 0.887 | 0.993 | 0.049 |
| 8 | 0.134 | 0.166 | 3.088 | 3.000 | 6.985 | 0.932 | 0.998 | 0.131 |
| 9 | 0.133 | 0.162 | 2.477 | 3.000 | 4.595 | 0.963 | 1.000 | 0.125 |
| 10 | 0.108 | 0.165 | 3.295 | 3.500 | 5.780 | 0.932 | 0.999 | 0.079 |
| 12 | 0.164 | 0.208 | 3.499 | 4.000 | 8.776 | 0.873 | 0.995 | 0.112 |
| 13 | 0.183 | 0.248 | 5.003 | 6.500 | 14.324 | 0.776 | 0.992 | 0.155 |
| 14 | 0.221 | 0.321 | 7.836 | 11.000 | 19.286 | 0.685 | 0.951 | 0.183 |
| 16 | 0.203 | 0.302 | 4.320 | 6.500 | 11.495 | 0.811 | 0.981 | 0.169 |
| 17 | 0.229 | 0.275 | 4.428 | 5.500 | 13.374 | 0.781 | 0.983 | 0.224 |
| 18 | 0.268 | 0.338 | 8.953 | 11.000 | 17.349 | 0.711 | 0.960 | 0.224 |
| 20 | 0.302 | 0.389 | 7.559 | 10.000 | 20.656 | 0.666 | 0.954 | 0.248 |
| 21 | 0.309 | 0.366 | 6.209 | 8.000 | 19.601 | 0.681 | 0.971 | 0.250 |
| 22 | 0.297 | 0.433 | 7.884 | 9.000 | 18.422 | 0.704 | 0.968 | 0.247 |
| 24 | 0.275 | 0.456 | 4.686 | 6.000 | 16.878 | 0.733 | 0.973 | 0.251 |
| 25 | 0.267 | 0.337 | 7.984 | 9.500 | 21.101 | 0.670 | 0.960 | 0.206 |
| 26 | 0.245 | 0.319 | 4.960 | 6.000 | 13.615 | 0.779 | 0.982 | 0.208 |
| 28 | 0.243 | 0.346 | 4.089 | 4.500 | 13.372 | 0.798 | 0.991 | 0.184 |
| 29 | 0.216 | 0.317 | 4.289 | 5.000 | 14.767 | 0.767 | 0.987 | 0.192 |
| 30 | 0.171 | 0.285 | 2.597 | 3.000 | 10.400 | 0.848 | 0.991 | 0.172 |
| 32 | 0.173 | 0.243 | 3.881 | 3.500 | 9.772 | 0.866 | 0.996 | 0.151 |
| 33 | 0.237 | 0.350 | 5.788 | 9.000 | 21.727 | 0.653 | 0.958 | 0.198 |
| 34 | 0.150 | 0.239 | 2.301 | 3.000 | 7.468 | 0.879 | 0.995 | 0.115 |
| 36 | 0.188 | 0.364 | 5.725 | 8.000 | 12.516 | 0.804 | 0.983 | 0.132 |
| 37 | 0.247 | 0.324 | 6.397 | 8.000 | 19.438 | 0.695 | 0.968 | 0.176 |
| 38 | 0.203 | 0.359 | 4.488 | 7.000 | 15.195 | 0.763 | 0.973 | 0.162 |
| 40 | 0.249 | 0.415 | 5.936 | 7.500 | 16.425 | 0.726 | 0.979 | 0.194 |
| 41 | 0.249 | 0.359 | 6.664 | 9.500 | 20.035 | 0.681 | 0.960 | 0.167 |
| 42 | 0.255 | 0.526 | 7.341 | 10.500 | 21.413 | 0.647 | 0.963 | 0.167 |
| 44 | 0.222 | 0.329 | 3.231 | 4.000 | 12.933 | 0.767 | 0.980 | 0.175 |

### S vs A+S: 2055 rows of S, keeps at [6144, 8192], [4069, 4069] tokens past S

| layer | latent rel L2 p50 | p90 | head p50 | body p50 | cos p50 | other-token p50 | raw | xor | delta | ring keys p50 | pooled keys p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.144 | 0.240 | 0.331 | 0.143 | 0.990 | 1.339 | 0.668 | 0.566 | 0.573 | n/a | 0.164 |
| 7 | 0.140 | 0.248 | 0.319 | 0.137 | 0.990 | 1.329 | 0.669 | 0.564 | 0.571 | n/a | 0.092 |
| 11 | 0.153 | 0.237 | 0.272 | 0.150 | 0.988 | 1.339 | 0.665 | 0.568 | 0.575 | n/a | 0.110 |
| 15 | 0.249 | 0.404 | 0.356 | 0.246 | 0.969 | 1.328 | 0.661 | 0.603 | 0.608 | n/a | 0.178 |
| 19 | 0.374 | 0.770 | 0.460 | 0.370 | 0.930 | 1.325 | 0.662 | 0.635 | 0.638 | n/a | 0.264 |
| 23 | 0.400 | 0.904 | 0.456 | 0.397 | 0.920 | 1.328 | 0.662 | 0.641 | 0.644 | n/a | 0.264 |
| 27 | 0.392 | 0.926 | 0.430 | 0.391 | 0.923 | 1.319 | 0.661 | 0.640 | 0.644 | n/a | 0.265 |
| 31 | 0.291 | 0.714 | 0.309 | 0.291 | 0.957 | 1.326 | 0.661 | 0.621 | 0.625 | n/a | 0.208 |
| 35 | 0.296 | 0.842 | 0.300 | 0.296 | 0.956 | 1.314 | 0.661 | 0.625 | 0.630 | n/a | 0.186 |
| 39 | 0.328 | 0.942 | 0.320 | 0.328 | 0.946 | 1.302 | 0.663 | 0.633 | 0.638 | n/a | 0.188 |
| 43 | 0.369 | 1.024 | 0.398 | 0.368 | 0.932 | 1.315 | 0.662 | 0.639 | 0.644 | n/a | 0.237 |

| KDA layer | state rel p50 over heads | p90 | eff. rank p50 | rank90 p50 | diff eff. rank p50 | diff energy in rank 8 p50 | rank 32 | conv rel L2 p50 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.010 | 2.115 | 2.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0.009 | 0.039 | 3.157 | 3.000 | 4.007 | 0.972 | 1.000 | 0.004 |
| 2 | 0.043 | 0.100 | 2.348 | 3.000 | 2.509 | 0.989 | 1.000 | 0.018 |
| 4 | 0.045 | 0.077 | 2.218 | 2.000 | 3.711 | 0.992 | 1.000 | 0.037 |
| 5 | 0.074 | 0.128 | 1.890 | 2.000 | 4.234 | 0.985 | 1.000 | 0.045 |
| 6 | 0.082 | 0.139 | 3.881 | 4.500 | 7.734 | 0.890 | 0.994 | 0.045 |
| 8 | 0.129 | 0.164 | 3.236 | 3.000 | 6.228 | 0.932 | 0.999 | 0.128 |
| 9 | 0.130 | 0.162 | 2.530 | 3.000 | 4.630 | 0.968 | 1.000 | 0.126 |
| 10 | 0.115 | 0.169 | 3.314 | 4.000 | 5.712 | 0.937 | 0.999 | 0.088 |
| 12 | 0.152 | 0.203 | 3.466 | 4.000 | 8.406 | 0.882 | 0.995 | 0.108 |
| 13 | 0.176 | 0.222 | 5.066 | 6.000 | 13.255 | 0.773 | 0.993 | 0.106 |
| 14 | 0.206 | 0.302 | 7.436 | 11.000 | 21.264 | 0.657 | 0.948 | 0.160 |
| 16 | 0.187 | 0.275 | 4.049 | 6.000 | 13.748 | 0.772 | 0.980 | 0.155 |
| 17 | 0.223 | 0.257 | 4.492 | 5.500 | 14.550 | 0.772 | 0.985 | 0.203 |
| 18 | 0.253 | 0.306 | 8.959 | 11.000 | 18.142 | 0.695 | 0.958 | 0.209 |
| 20 | 0.303 | 0.366 | 7.123 | 10.000 | 19.468 | 0.679 | 0.959 | 0.223 |
| 21 | 0.305 | 0.350 | 5.864 | 7.000 | 19.155 | 0.696 | 0.973 | 0.224 |
| 22 | 0.296 | 0.422 | 7.777 | 9.000 | 16.918 | 0.719 | 0.973 | 0.223 |
| 24 | 0.270 | 0.452 | 4.657 | 6.000 | 14.326 | 0.764 | 0.977 | 0.247 |
| 25 | 0.284 | 0.338 | 7.679 | 9.500 | 19.361 | 0.693 | 0.965 | 0.201 |
| 26 | 0.240 | 0.344 | 4.711 | 5.500 | 12.963 | 0.784 | 0.984 | 0.203 |
| 28 | 0.243 | 0.343 | 3.922 | 4.000 | 12.028 | 0.808 | 0.992 | 0.185 |
| 29 | 0.219 | 0.314 | 4.396 | 5.000 | 14.334 | 0.772 | 0.988 | 0.195 |
| 30 | 0.183 | 0.270 | 2.635 | 3.000 | 9.943 | 0.865 | 0.991 | 0.176 |
| 32 | 0.175 | 0.275 | 3.756 | 3.500 | 11.044 | 0.849 | 0.996 | 0.142 |
| 33 | 0.232 | 0.351 | 6.088 | 9.000 | 21.454 | 0.658 | 0.958 | 0.180 |
| 34 | 0.149 | 0.218 | 2.309 | 3.000 | 7.492 | 0.884 | 0.996 | 0.116 |
| 36 | 0.163 | 0.306 | 5.444 | 7.500 | 13.925 | 0.775 | 0.980 | 0.124 |
| 37 | 0.228 | 0.316 | 6.098 | 8.000 | 19.115 | 0.693 | 0.966 | 0.192 |
| 38 | 0.215 | 0.385 | 4.500 | 7.500 | 14.908 | 0.761 | 0.973 | 0.160 |
| 40 | 0.235 | 0.419 | 6.247 | 7.500 | 18.597 | 0.687 | 0.974 | 0.196 |
| 41 | 0.248 | 0.352 | 6.399 | 8.500 | 18.345 | 0.693 | 0.961 | 0.171 |
| 42 | 0.242 | 0.524 | 7.346 | 10.500 | 22.613 | 0.626 | 0.957 | 0.168 |
| 44 | 0.212 | 0.308 | 3.383 | 4.000 | 10.831 | 0.795 | 0.980 | 0.173 |

### L+S vs A+S: 2055 rows of S, keeps at [14336, 8192], [4069, 4069] tokens past S

| layer | latent rel L2 p50 | p90 | head p50 | body p50 | cos p50 | other-token p50 | raw | xor | delta | ring keys p50 | pooled keys p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.090 | 0.133 | 0.139 | 0.089 | 0.996 | 1.339 | 0.668 | 0.523 | 0.534 | n/a | 0.095 |
| 7 | 0.099 | 0.164 | 0.166 | 0.098 | 0.995 | 1.329 | 0.669 | 0.536 | 0.546 | n/a | 0.056 |
| 11 | 0.137 | 0.188 | 0.191 | 0.136 | 0.991 | 1.339 | 0.665 | 0.559 | 0.567 | n/a | 0.119 |
| 15 | 0.224 | 0.349 | 0.294 | 0.223 | 0.975 | 1.328 | 0.661 | 0.596 | 0.601 | n/a | 0.166 |
| 19 | 0.330 | 0.661 | 0.428 | 0.325 | 0.946 | 1.325 | 0.662 | 0.628 | 0.631 | n/a | 0.230 |
| 23 | 0.357 | 0.837 | 0.410 | 0.354 | 0.937 | 1.328 | 0.662 | 0.635 | 0.638 | n/a | 0.218 |
| 27 | 0.360 | 0.870 | 0.389 | 0.358 | 0.936 | 1.319 | 0.661 | 0.634 | 0.639 | n/a | 0.228 |
| 31 | 0.257 | 0.643 | 0.276 | 0.256 | 0.967 | 1.326 | 0.661 | 0.614 | 0.619 | n/a | 0.182 |
| 35 | 0.267 | 0.787 | 0.279 | 0.266 | 0.965 | 1.314 | 0.661 | 0.619 | 0.624 | n/a | 0.170 |
| 39 | 0.298 | 0.915 | 0.299 | 0.298 | 0.956 | 1.302 | 0.663 | 0.627 | 0.632 | n/a | 0.164 |
| 43 | 0.332 | 0.983 | 0.360 | 0.331 | 0.945 | 1.315 | 0.662 | 0.633 | 0.638 | n/a | 0.192 |

| KDA layer | state rel p50 over heads | p90 | eff. rank p50 | rank90 p50 | diff eff. rank p50 | diff energy in rank 8 p50 | rank 32 | conv rel L2 p50 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.007 | 2.115 | 2.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0.011 | 0.050 | 3.157 | 3.000 | 4.137 | 0.971 | 1.000 | 0.006 |
| 2 | 0.044 | 0.104 | 2.348 | 3.000 | 2.510 | 0.990 | 1.000 | 0.016 |
| 4 | 0.054 | 0.106 | 2.218 | 2.000 | 4.160 | 0.983 | 1.000 | 0.040 |
| 5 | 0.080 | 0.143 | 1.890 | 2.000 | 4.546 | 0.974 | 1.000 | 0.050 |
| 6 | 0.094 | 0.157 | 3.881 | 4.500 | 8.883 | 0.879 | 0.993 | 0.049 |
| 8 | 0.152 | 0.183 | 3.236 | 3.000 | 7.428 | 0.906 | 0.998 | 0.112 |
| 9 | 0.119 | 0.147 | 2.530 | 3.000 | 5.584 | 0.939 | 1.000 | 0.074 |
| 10 | 0.113 | 0.250 | 3.314 | 4.000 | 6.188 | 0.932 | 0.999 | 0.081 |
| 12 | 0.182 | 0.269 | 3.466 | 4.000 | 9.851 | 0.839 | 0.993 | 0.109 |
| 13 | 0.210 | 0.302 | 5.066 | 6.000 | 14.548 | 0.767 | 0.993 | 0.139 |
| 14 | 0.252 | 0.338 | 7.436 | 11.000 | 21.469 | 0.662 | 0.950 | 0.168 |
| 16 | 0.230 | 0.332 | 4.049 | 6.000 | 11.039 | 0.815 | 0.982 | 0.190 |
| 17 | 0.253 | 0.300 | 4.492 | 5.500 | 13.430 | 0.771 | 0.982 | 0.234 |
| 18 | 0.296 | 0.361 | 8.959 | 11.000 | 19.537 | 0.686 | 0.958 | 0.231 |
| 20 | 0.336 | 0.426 | 7.123 | 10.000 | 21.589 | 0.661 | 0.951 | 0.269 |
| 21 | 0.334 | 0.408 | 5.864 | 7.000 | 18.711 | 0.708 | 0.971 | 0.287 |
| 22 | 0.318 | 0.479 | 7.777 | 9.000 | 15.599 | 0.744 | 0.971 | 0.247 |
| 24 | 0.290 | 0.499 | 4.657 | 6.000 | 14.453 | 0.754 | 0.974 | 0.254 |
| 25 | 0.292 | 0.361 | 7.679 | 9.500 | 20.516 | 0.666 | 0.963 | 0.198 |
| 26 | 0.288 | 0.372 | 4.711 | 5.500 | 12.746 | 0.793 | 0.983 | 0.207 |
| 28 | 0.246 | 0.349 | 3.922 | 4.000 | 14.216 | 0.776 | 0.990 | 0.176 |
| 29 | 0.221 | 0.320 | 4.396 | 5.000 | 16.033 | 0.738 | 0.985 | 0.182 |
| 30 | 0.189 | 0.298 | 2.635 | 3.000 | 8.091 | 0.886 | 0.993 | 0.176 |
| 32 | 0.181 | 0.263 | 3.756 | 3.500 | 10.293 | 0.839 | 0.995 | 0.139 |
| 33 | 0.235 | 0.347 | 6.088 | 9.000 | 21.044 | 0.659 | 0.962 | 0.190 |
| 34 | 0.166 | 0.278 | 2.309 | 3.000 | 8.552 | 0.873 | 0.996 | 0.112 |
| 36 | 0.195 | 0.353 | 5.444 | 7.500 | 14.256 | 0.773 | 0.982 | 0.126 |
| 37 | 0.242 | 0.331 | 6.098 | 8.000 | 18.360 | 0.692 | 0.966 | 0.175 |
| 38 | 0.226 | 0.363 | 4.500 | 7.500 | 15.727 | 0.754 | 0.971 | 0.159 |
| 40 | 0.262 | 0.467 | 6.247 | 7.500 | 17.913 | 0.709 | 0.974 | 0.172 |
| 41 | 0.270 | 0.408 | 6.399 | 8.500 | 19.177 | 0.681 | 0.962 | 0.174 |
| 42 | 0.264 | 0.529 | 7.346 | 10.500 | 19.769 | 0.670 | 0.960 | 0.150 |
| 44 | 0.219 | 0.335 | 3.383 | 4.000 | 13.464 | 0.767 | 0.983 | 0.174 |

bench S: exit 0
    KVBENCH config mib=64 threads=8 reps=3 block=1048576
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep5.bin codec=memcpy threads=1 raw=259744768 stored=259744768 ratio=1.0000 encode_gbs=24.693 decode_gbs=24.693 keep256M_ms=10.9 restore256M_ms=10.9 keep1024M_ms=43.5 restore1024M_ms=43.5 keep4096M_ms=173.9 restore4096M_ms=173.9
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep5.bin codec=deflate1 threads=1 raw=259744768 stored=213243447 ratio=0.8210 encode_gbs=0.077 decode_gbs=0.180 keep256M_ms=3484.5 restore256M_ms=1487.2 keep1024M_ms=13938.0 restore1024M_ms=5948.9 keep4096M_ms=55751.9 restore4096M_ms=23795.6
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep5.bin codec=deflate6 threads=1 raw=259744768 stored=213221521 ratio=0.8209 encode_gbs=0.065 decode_gbs=0.180 keep256M_ms=4099.0 restore256M_ms=1489.1 keep1024M_ms=16396.1 restore1024M_ms=5956.3 keep4096M_ms=65584.5 restore4096M_ms=23825.0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep5.bin codec=plain threads=1 raw=259744768 stored=187790757 ratio=0.7230 encode_gbs=0.319 decode_gbs=1.252 keep256M_ms=841.7 restore256M_ms=214.4 keep1024M_ms=3366.9 restore1024M_ms=857.7 keep4096M_ms=13467.6 restore4096M_ms=3430.6
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep5.bin codec=plain engagement raw=0 planes=256 planes_xor=0 planes_delta=0 planes_delta2=0 t64=0 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep5.bin codec=plain threads=8 raw=259744768 stored=187790757 ratio=0.7230 encode_gbs=1.743 decode_gbs=6.446 keep256M_ms=154.0 restore256M_ms=41.6 keep1024M_ms=616.0 restore1024M_ms=166.6 keep4096M_ms=2464.2 restore4096M_ms=666.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep5.bin codec=atlas threads=1 raw=259744768 stored=187578849 ratio=0.7222 encode_gbs=0.099 decode_gbs=1.212 keep256M_ms=2719.4 restore256M_ms=221.4 keep1024M_ms=10877.4 restore1024M_ms=885.7 keep4096M_ms=43509.7 restore4096M_ms=3542.9
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep5.bin codec=atlas engagement raw=0 planes=232 planes_xor=0 planes_delta=23 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep5.bin codec=atlas threads=8 raw=259744768 stored=187578849 ratio=0.7222 encode_gbs=0.532 decode_gbs=6.331 keep256M_ms=504.4 restore256M_ms=42.4 keep1024M_ms=2017.4 restore1024M_ms=169.6 keep4096M_ms=8069.6 restore4096M_ms=678.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep5.bin codec=atlas-exhaustive threads=1 raw=259744768 stored=187578727 ratio=0.7222 encode_gbs=0.029 decode_gbs=1.211 keep256M_ms=9172.4 restore256M_ms=221.7 keep1024M_ms=36689.6 restore1024M_ms=886.9 keep4096M_ms=146758.3 restore4096M_ms=3547.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep5.bin codec=atlas-exhaustive engagement raw=0 planes=231 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep5.bin codec=atlas-exhaustive threads=8 raw=259744768 stored=187578727 ratio=0.7222 encode_gbs=0.159 decode_gbs=6.322 keep256M_ms=1692.4 restore256M_ms=42.5 keep1024M_ms=6769.7 restore1024M_ms=169.9 keep4096M_ms=27078.9 restore4096M_ms=679.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep5.bin codec=ch-doubledelta threads=1 raw=259744768 stored=242491309 ratio=0.9336 encode_gbs=0.521 decode_gbs=7.607 keep256M_ms=515.2 restore256M_ms=35.3 keep1024M_ms=2060.7 restore1024M_ms=141.1 keep4096M_ms=8242.8 restore4096M_ms=564.6
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep5.bin codec=ch-doubledelta width=2 raw_blocks=233/256
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep5.bin codec=ch-gorilla threads=1 raw=259744768 stored=242904831 ratio=0.9352 encode_gbs=0.868 decode_gbs=17.366 keep256M_ms=309.2 restore256M_ms=15.5 keep1024M_ms=1236.9 restore1024M_ms=61.8 keep4096M_ms=4947.6 restore4096M_ms=247.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep5.bin codec=ch-gorilla width=2 raw_blocks=238/256
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep5.bin codec=ch-fpc12 threads=1 raw=259744768 stored=243904172 ratio=0.9390 encode_gbs=1.036 decode_gbs=8.810 keep256M_ms=259.0 restore256M_ms=30.5 keep1024M_ms=1036.2 restore1024M_ms=121.9 keep4096M_ms=4144.8 restore4096M_ms=487.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep5.bin codec=ch-fpc12 width=2 raw_blocks=235/256
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep5.bin codec=ch-alp-rd threads=1 raw=259744768 stored=218366710 ratio=0.8407 encode_gbs=0.181 decode_gbs=2.275 keep256M_ms=1481.9 restore256M_ms=118.0 keep1024M_ms=5927.6 restore1024M_ms=472.1 keep4096M_ms=23710.3 restore4096M_ms=1888.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep5.bin codec=ch-alp-rd width=2 raw_blocks=137/256
bench A+S: exit 0
    KVBENCH config mib=64 threads=8 reps=3 block=1048576
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep6.bin codec=memcpy threads=1 raw=284255232 stored=284255232 ratio=1.0000 encode_gbs=24.339 decode_gbs=24.339 keep256M_ms=11.0 restore256M_ms=11.0 keep1024M_ms=44.1 restore1024M_ms=44.1 keep4096M_ms=176.5 restore4096M_ms=176.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep6.bin codec=deflate1 threads=1 raw=284255232 stored=232793597 ratio=0.8190 encode_gbs=0.077 decode_gbs=0.178 keep256M_ms=3484.8 restore256M_ms=1508.5 keep1024M_ms=13939.3 restore1024M_ms=6034.0 keep4096M_ms=55757.4 restore4096M_ms=24136.1
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep6.bin codec=deflate6 threads=1 raw=284255232 stored=232770448 ratio=0.8189 encode_gbs=0.064 decode_gbs=0.177 keep256M_ms=4178.8 restore256M_ms=1512.6 keep1024M_ms=16715.2 restore1024M_ms=6050.4 keep4096M_ms=66860.7 restore4096M_ms=24201.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep6.bin codec=plain threads=1 raw=284255232 stored=204106246 ratio=0.7180 encode_gbs=0.317 decode_gbs=1.219 keep256M_ms=846.8 restore256M_ms=220.1 keep1024M_ms=3387.2 restore1024M_ms=880.6 keep4096M_ms=13549.0 restore4096M_ms=3522.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep6.bin codec=plain engagement raw=0 planes=278 planes_xor=0 planes_delta=0 planes_delta2=0 t64=0 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep6.bin codec=plain threads=8 raw=284255232 stored=204106246 ratio=0.7180 encode_gbs=1.728 decode_gbs=6.367 keep256M_ms=155.3 restore256M_ms=42.2 keep1024M_ms=621.2 restore1024M_ms=168.6 keep4096M_ms=2484.9 restore4096M_ms=674.6
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep6.bin codec=atlas threads=1 raw=284255232 stored=203865071 ratio=0.7172 encode_gbs=0.096 decode_gbs=1.188 keep256M_ms=2804.0 restore256M_ms=225.9 keep1024M_ms=11216.1 restore1024M_ms=903.5 keep4096M_ms=44864.4 restore4096M_ms=3613.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep6.bin codec=atlas engagement raw=0 planes=254 planes_xor=0 planes_delta=23 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep6.bin codec=atlas threads=8 raw=284255232 stored=203865071 ratio=0.7172 encode_gbs=0.528 decode_gbs=6.294 keep256M_ms=508.7 restore256M_ms=42.6 keep1024M_ms=2035.0 restore1024M_ms=170.6 keep4096M_ms=8140.0 restore4096M_ms=682.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep6.bin codec=atlas-exhaustive threads=1 raw=284255232 stored=203865071 ratio=0.7172 encode_gbs=0.028 decode_gbs=1.189 keep256M_ms=9575.4 restore256M_ms=225.7 keep1024M_ms=38301.7 restore1024M_ms=902.9 keep4096M_ms=153206.7 restore4096M_ms=3611.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep6.bin codec=atlas-exhaustive engagement raw=0 planes=254 planes_xor=0 planes_delta=23 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep6.bin codec=atlas-exhaustive threads=8 raw=284255232 stored=203865071 ratio=0.7172 encode_gbs=0.155 decode_gbs=6.289 keep256M_ms=1731.1 restore256M_ms=42.7 keep1024M_ms=6924.4 restore1024M_ms=170.7 keep4096M_ms=27697.6 restore4096M_ms=682.9
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep6.bin codec=ch-doubledelta threads=1 raw=284255232 stored=266945037 ratio=0.9391 encode_gbs=0.500 decode_gbs=8.064 keep256M_ms=536.9 restore256M_ms=33.3 keep1024M_ms=2147.7 restore1024M_ms=133.1 keep4096M_ms=8590.7 restore4096M_ms=532.6
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep6.bin codec=ch-doubledelta width=2 raw_blocks=255/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep6.bin codec=ch-gorilla threads=1 raw=284255232 stored=267415299 ratio=0.9408 encode_gbs=0.859 decode_gbs=16.551 keep256M_ms=312.6 restore256M_ms=16.2 keep1024M_ms=1250.3 restore1024M_ms=64.9 keep4096M_ms=5001.4 restore4096M_ms=259.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep6.bin codec=ch-gorilla width=2 raw_blocks=260/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep6.bin codec=ch-fpc12 threads=1 raw=284255232 stored=268405514 ratio=0.9442 encode_gbs=1.029 decode_gbs=9.311 keep256M_ms=261.0 restore256M_ms=28.8 keep1024M_ms=1043.9 restore1024M_ms=115.3 keep4096M_ms=4175.4 restore4096M_ms=461.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep6.bin codec=ch-fpc12 width=2 raw_blocks=257/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep6.bin codec=ch-alp-rd threads=1 raw=284255232 stored=236632334 ratio=0.8325 encode_gbs=0.186 decode_gbs=2.040 keep256M_ms=1446.1 restore256M_ms=131.6 keep1024M_ms=5784.3 restore1024M_ms=526.4 keep4096M_ms=23137.2 restore4096M_ms=2105.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep6.bin codec=ch-alp-rd width=2 raw_blocks=137/278
bench B+S: exit 0
    KVBENCH config mib=64 threads=8 reps=3 block=1048576
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep7.bin codec=memcpy threads=1 raw=284255232 stored=284255232 ratio=1.0000 encode_gbs=24.493 decode_gbs=24.493 keep256M_ms=11.0 restore256M_ms=11.0 keep1024M_ms=43.8 restore1024M_ms=43.8 keep4096M_ms=175.4 restore4096M_ms=175.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep7.bin codec=deflate1 threads=1 raw=284255232 stored=232794158 ratio=0.8190 encode_gbs=0.077 decode_gbs=0.178 keep256M_ms=3477.8 restore256M_ms=1508.4 keep1024M_ms=13911.0 restore1024M_ms=6033.5 keep4096M_ms=55644.1 restore4096M_ms=24134.0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep7.bin codec=deflate6 threads=1 raw=284255232 stored=232771210 ratio=0.8189 encode_gbs=0.064 decode_gbs=0.178 keep256M_ms=4175.3 restore256M_ms=1511.8 keep1024M_ms=16701.1 restore1024M_ms=6047.2 keep4096M_ms=66804.5 restore4096M_ms=24189.0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep7.bin codec=plain threads=1 raw=284255232 stored=204109383 ratio=0.7180 encode_gbs=0.317 decode_gbs=1.208 keep256M_ms=847.9 restore256M_ms=222.3 keep1024M_ms=3391.5 restore1024M_ms=889.2 keep4096M_ms=13565.9 restore4096M_ms=3556.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep7.bin codec=plain engagement raw=0 planes=278 planes_xor=0 planes_delta=0 planes_delta2=0 t64=0 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep7.bin codec=plain threads=8 raw=284255232 stored=204109383 ratio=0.7180 encode_gbs=1.745 decode_gbs=6.445 keep256M_ms=153.8 restore256M_ms=41.6 keep1024M_ms=615.2 restore1024M_ms=166.6 keep4096M_ms=2460.7 restore4096M_ms=666.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep7.bin codec=atlas threads=1 raw=284255232 stored=203865392 ratio=0.7172 encode_gbs=0.096 decode_gbs=1.175 keep256M_ms=2792.4 restore256M_ms=228.5 keep1024M_ms=11169.7 restore1024M_ms=914.1 keep4096M_ms=44678.8 restore4096M_ms=3656.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep7.bin codec=atlas engagement raw=0 planes=253 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep7.bin codec=atlas threads=8 raw=284255232 stored=203865392 ratio=0.7172 encode_gbs=0.528 decode_gbs=6.239 keep256M_ms=508.5 restore256M_ms=43.0 keep1024M_ms=2033.8 restore1024M_ms=172.1 keep4096M_ms=8135.2 restore4096M_ms=688.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep7.bin codec=atlas-exhaustive threads=1 raw=284255232 stored=203865392 ratio=0.7172 encode_gbs=0.028 decode_gbs=1.037 keep256M_ms=9577.1 restore256M_ms=258.7 keep1024M_ms=38308.4 restore1024M_ms=1034.9 keep4096M_ms=153233.6 restore4096M_ms=4139.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep7.bin codec=atlas-exhaustive engagement raw=0 planes=253 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep7.bin codec=atlas-exhaustive threads=8 raw=284255232 stored=203865392 ratio=0.7172 encode_gbs=0.150 decode_gbs=6.218 keep256M_ms=1794.0 restore256M_ms=43.2 keep1024M_ms=7176.0 restore1024M_ms=172.7 keep4096M_ms=28704.1 restore4096M_ms=690.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep7.bin codec=ch-doubledelta threads=1 raw=284255232 stored=266949455 ratio=0.9391 encode_gbs=0.498 decode_gbs=8.108 keep256M_ms=538.7 restore256M_ms=33.1 keep1024M_ms=2154.7 restore1024M_ms=132.4 keep4096M_ms=8618.8 restore4096M_ms=529.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep7.bin codec=ch-doubledelta width=2 raw_blocks=255/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep7.bin codec=ch-gorilla threads=1 raw=284255232 stored=267415284 ratio=0.9408 encode_gbs=0.864 decode_gbs=17.991 keep256M_ms=310.9 restore256M_ms=14.9 keep1024M_ms=1243.5 restore1024M_ms=59.7 keep4096M_ms=4973.8 restore4096M_ms=238.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep7.bin codec=ch-gorilla width=2 raw_blocks=260/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep7.bin codec=ch-fpc12 threads=1 raw=284255232 stored=268402755 ratio=0.9442 encode_gbs=1.029 decode_gbs=9.389 keep256M_ms=260.8 restore256M_ms=28.6 keep1024M_ms=1043.1 restore1024M_ms=114.4 keep4096M_ms=4172.4 restore4096M_ms=457.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep7.bin codec=ch-fpc12 width=2 raw_blocks=257/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep7.bin codec=ch-alp-rd threads=1 raw=284255232 stored=236641640 ratio=0.8325 encode_gbs=0.184 decode_gbs=2.034 keep256M_ms=1458.3 restore256M_ms=132.0 keep1024M_ms=5833.3 restore1024M_ms=527.9 keep4096M_ms=23333.1 restore4096M_ms=2111.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep7.bin codec=ch-alp-rd width=2 raw_blocks=137/278
bench L+S: exit 0
    KVBENCH config mib=64 threads=8 reps=3 block=1048576
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep8.bin codec=memcpy threads=1 raw=357786624 stored=357786624 ratio=1.0000 encode_gbs=24.867 decode_gbs=24.867 keep256M_ms=10.8 restore256M_ms=10.8 keep1024M_ms=43.2 restore1024M_ms=43.2 keep4096M_ms=172.7 restore4096M_ms=172.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep8.bin codec=deflate1 threads=1 raw=357786624 stored=291469105 ratio=0.8146 encode_gbs=0.078 decode_gbs=0.171 keep256M_ms=3454.8 restore256M_ms=1569.5 keep1024M_ms=13819.0 restore1024M_ms=6278.1 keep4096M_ms=55276.1 restore4096M_ms=25112.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep8.bin codec=deflate6 threads=1 raw=357786624 stored=291441462 ratio=0.8146 encode_gbs=0.062 decode_gbs=0.171 keep256M_ms=4308.1 restore256M_ms=1571.0 keep1024M_ms=17232.3 restore1024M_ms=6284.0 keep4096M_ms=68929.3 restore4096M_ms=25135.9
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep8.bin codec=plain threads=1 raw=357786624 stored=253076311 ratio=0.7073 encode_gbs=0.313 decode_gbs=1.121 keep256M_ms=857.2 restore256M_ms=239.5 keep1024M_ms=3429.0 restore1024M_ms=958.0 keep4096M_ms=13716.0 restore4096M_ms=3832.1
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep8.bin codec=plain engagement raw=0 planes=344 planes_xor=0 planes_delta=0 planes_delta2=0 t64=0 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep8.bin codec=plain threads=8 raw=357786624 stored=253076311 ratio=0.7073 encode_gbs=1.734 decode_gbs=5.945 keep256M_ms=154.8 restore256M_ms=45.2 keep1024M_ms=619.4 restore1024M_ms=180.6 keep4096M_ms=2477.4 restore4096M_ms=722.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep8.bin codec=atlas threads=1 raw=357786624 stored=252736688 ratio=0.7064 encode_gbs=0.091 decode_gbs=1.098 keep256M_ms=2957.2 restore256M_ms=244.4 keep1024M_ms=11828.8 restore1024M_ms=977.5 keep4096M_ms=47315.4 restore4096M_ms=3910.1
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep8.bin codec=atlas engagement raw=0 planes=319 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep8.bin codec=atlas threads=8 raw=357786624 stored=252736688 ratio=0.7064 encode_gbs=0.499 decode_gbs=5.881 keep256M_ms=538.4 restore256M_ms=45.6 keep1024M_ms=2153.8 restore1024M_ms=182.6 keep4096M_ms=8615.1 restore4096M_ms=730.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep8.bin codec=atlas-exhaustive threads=1 raw=357786624 stored=252736688 ratio=0.7064 encode_gbs=0.026 decode_gbs=1.099 keep256M_ms=10428.7 restore256M_ms=244.2 keep1024M_ms=41714.9 restore1024M_ms=976.8 keep4096M_ms=166859.8 restore4096M_ms=3907.2
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep8.bin codec=atlas-exhaustive engagement raw=0 planes=319 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep8.bin codec=atlas-exhaustive threads=8 raw=357786624 stored=252736688 ratio=0.7064 encode_gbs=0.140 decode_gbs=5.809 keep256M_ms=1910.9 restore256M_ms=46.2 keep1024M_ms=7643.5 restore1024M_ms=184.8 keep4096M_ms=30574.1 restore4096M_ms=739.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep8.bin codec=ch-doubledelta threads=1 raw=357786624 stored=340473930 ratio=0.9516 encode_gbs=0.460 decode_gbs=9.442 keep256M_ms=583.7 restore256M_ms=28.4 keep1024M_ms=2334.9 restore1024M_ms=113.7 keep4096M_ms=9339.6 restore4096M_ms=454.9
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep8.bin codec=ch-doubledelta width=2 raw_blocks=321/344
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep8.bin codec=ch-gorilla threads=1 raw=357786624 stored=340946691 ratio=0.9529 encode_gbs=0.850 decode_gbs=16.685 keep256M_ms=315.7 restore256M_ms=16.1 keep1024M_ms=1262.9 restore1024M_ms=64.4 keep4096M_ms=5051.4 restore4096M_ms=257.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep8.bin codec=ch-gorilla width=2 raw_blocks=326/344
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep8.bin codec=ch-fpc12 threads=1 raw=357786624 stored=341934102 ratio=0.9557 encode_gbs=1.009 decode_gbs=10.728 keep256M_ms=266.2 restore256M_ms=25.0 keep1024M_ms=1064.7 restore1024M_ms=100.1 keep4096M_ms=4258.6 restore4096M_ms=400.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep8.bin codec=ch-fpc12 width=2 raw_blocks=323/344
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep8.bin codec=ch-alp-rd threads=1 raw=357786624 stored=291288034 ratio=0.8141 encode_gbs=0.193 decode_gbs=1.663 keep256M_ms=1389.2 restore256M_ms=161.4 keep1024M_ms=5556.8 restore1024M_ms=645.6 keep4096M_ms=22227.2 restore4096M_ms=2582.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep8.bin codec=ch-alp-rd width=2 raw_blocks=137/344

## json

### A+S vs B+S: 2075 rows of S, keeps at [8192, 8192], [3906, 3906] tokens past S

| layer | latent rel L2 p50 | p90 | head p50 | body p50 | cos p50 | other-token p50 | raw | xor | delta | ring keys p50 | pooled keys p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.119 | 0.175 | 0.189 | 0.117 | 0.993 | 1.240 | 0.668 | 0.548 | 0.557 | 0.082 | 0.113 |
| 7 | 0.139 | 0.215 | 0.240 | 0.138 | 0.990 | 1.130 | 0.668 | 0.562 | 0.569 | 0.057 | 0.061 |
| 11 | 0.207 | 0.261 | 0.237 | 0.206 | 0.979 | 1.199 | 0.665 | 0.589 | 0.594 | 0.134 | 0.142 |
| 15 | 0.383 | 0.613 | 0.405 | 0.382 | 0.927 | 1.234 | 0.661 | 0.632 | 0.636 | 0.238 | 0.230 |
| 19 | 0.649 | 1.123 | 0.588 | 0.649 | 0.790 | 1.267 | 0.662 | 0.664 | 0.667 | 0.360 | 0.367 |
| 23 | 0.712 | 1.189 | 0.615 | 0.721 | 0.745 | 1.241 | 0.662 | 0.668 | 0.672 | 0.399 | 0.402 |
| 27 | 0.719 | 1.208 | 0.560 | 0.735 | 0.740 | 1.246 | 0.661 | 0.666 | 0.673 | 0.413 | 0.419 |
| 31 | 0.509 | 1.064 | 0.423 | 0.516 | 0.870 | 1.239 | 0.661 | 0.654 | 0.660 | 0.322 | 0.352 |
| 35 | 0.535 | 1.151 | 0.405 | 0.547 | 0.858 | 1.252 | 0.661 | 0.656 | 0.662 | 0.305 | 0.303 |
| 39 | 0.580 | 1.179 | 0.443 | 0.587 | 0.833 | 1.246 | 0.663 | 0.658 | 0.665 | 0.295 | 0.314 |
| 43 | 0.661 | 1.195 | 0.544 | 0.673 | 0.781 | 1.237 | 0.662 | 0.661 | 0.668 | 0.337 | 0.412 |

| KDA layer | state rel p50 over heads | p90 | eff. rank p50 | rank90 p50 | diff eff. rank p50 | diff energy in rank 8 p50 | rank 32 | conv rel L2 p50 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.013 | 2.023 | 2.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0.009 | 0.036 | 2.893 | 3.000 | 4.322 | 0.960 | 1.000 | 0.004 |
| 2 | 0.036 | 0.078 | 2.453 | 3.000 | 2.763 | 0.988 | 1.000 | 0.017 |
| 4 | 0.039 | 0.062 | 2.461 | 2.000 | 4.566 | 0.983 | 1.000 | 0.018 |
| 5 | 0.039 | 0.069 | 2.450 | 2.000 | 4.971 | 0.968 | 1.000 | 0.024 |
| 6 | 0.055 | 0.115 | 3.654 | 5.000 | 6.887 | 0.908 | 0.996 | 0.026 |
| 8 | 0.079 | 0.182 | 3.323 | 4.000 | 6.084 | 0.924 | 0.998 | 0.056 |
| 9 | 0.058 | 0.114 | 2.510 | 3.000 | 5.973 | 0.949 | 0.999 | 0.039 |
| 10 | 0.071 | 0.163 | 3.365 | 4.000 | 6.521 | 0.914 | 0.998 | 0.046 |
| 12 | 0.123 | 0.208 | 3.786 | 4.000 | 7.167 | 0.883 | 0.995 | 0.079 |
| 13 | 0.131 | 0.271 | 4.962 | 6.000 | 12.203 | 0.813 | 0.987 | 0.078 |
| 14 | 0.223 | 0.371 | 7.976 | 11.500 | 16.581 | 0.731 | 0.959 | 0.119 |
| 16 | 0.178 | 0.325 | 4.382 | 7.000 | 9.740 | 0.854 | 0.988 | 0.129 |
| 17 | 0.197 | 0.279 | 4.151 | 5.000 | 11.981 | 0.811 | 0.983 | 0.157 |
| 18 | 0.268 | 0.350 | 9.364 | 12.500 | 18.832 | 0.699 | 0.958 | 0.247 |
| 20 | 0.299 | 0.397 | 8.119 | 10.000 | 20.384 | 0.686 | 0.955 | 0.305 |
| 21 | 0.275 | 0.351 | 6.825 | 8.000 | 19.671 | 0.686 | 0.966 | 0.330 |
| 22 | 0.276 | 0.408 | 7.983 | 10.000 | 18.885 | 0.701 | 0.963 | 0.300 |
| 24 | 0.259 | 0.457 | 4.894 | 6.000 | 16.520 | 0.731 | 0.962 | 0.287 |
| 25 | 0.272 | 0.370 | 8.108 | 11.000 | 21.009 | 0.671 | 0.951 | 0.310 |
| 26 | 0.235 | 0.310 | 4.588 | 5.500 | 14.974 | 0.762 | 0.978 | 0.272 |
| 28 | 0.222 | 0.335 | 4.044 | 4.000 | 12.162 | 0.799 | 0.977 | 0.196 |
| 29 | 0.198 | 0.326 | 4.257 | 5.000 | 14.196 | 0.761 | 0.979 | 0.198 |
| 30 | 0.167 | 0.268 | 2.720 | 3.000 | 10.197 | 0.845 | 0.989 | 0.161 |
| 32 | 0.161 | 0.253 | 3.359 | 3.500 | 9.366 | 0.853 | 0.992 | 0.138 |
| 33 | 0.218 | 0.391 | 6.421 | 9.500 | 20.082 | 0.675 | 0.959 | 0.181 |
| 34 | 0.143 | 0.231 | 2.320 | 2.000 | 8.727 | 0.861 | 0.994 | 0.104 |
| 36 | 0.189 | 0.397 | 4.949 | 8.000 | 10.987 | 0.821 | 0.985 | 0.118 |
| 37 | 0.213 | 0.338 | 6.680 | 9.000 | 18.183 | 0.711 | 0.969 | 0.147 |
| 38 | 0.201 | 0.391 | 4.985 | 7.500 | 15.112 | 0.758 | 0.974 | 0.152 |
| 40 | 0.237 | 0.449 | 6.724 | 8.000 | 18.067 | 0.714 | 0.971 | 0.165 |
| 41 | 0.253 | 0.374 | 6.497 | 8.500 | 19.401 | 0.680 | 0.966 | 0.143 |
| 42 | 0.252 | 0.558 | 8.073 | 11.000 | 20.479 | 0.654 | 0.961 | 0.141 |
| 44 | 0.212 | 0.338 | 3.200 | 4.500 | 13.705 | 0.768 | 0.980 | 0.132 |

### S vs A+S: 2075 rows of S, keeps at [6144, 8192], [3906, 3906] tokens past S

| layer | latent rel L2 p50 | p90 | head p50 | body p50 | cos p50 | other-token p50 | raw | xor | delta | ring keys p50 | pooled keys p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.140 | 0.193 | 0.236 | 0.139 | 0.990 | 1.247 | 0.668 | 0.562 | 0.570 | n/a | 0.165 |
| 7 | 0.160 | 0.231 | 0.309 | 0.158 | 0.987 | 1.139 | 0.668 | 0.572 | 0.578 | n/a | 0.089 |
| 11 | 0.181 | 0.266 | 0.296 | 0.179 | 0.984 | 1.209 | 0.665 | 0.582 | 0.587 | n/a | 0.130 |
| 15 | 0.353 | 0.610 | 0.423 | 0.350 | 0.938 | 1.243 | 0.661 | 0.627 | 0.631 | n/a | 0.196 |
| 19 | 0.588 | 1.094 | 0.527 | 0.589 | 0.828 | 1.268 | 0.662 | 0.659 | 0.663 | n/a | 0.328 |
| 23 | 0.647 | 1.156 | 0.562 | 0.655 | 0.791 | 1.251 | 0.662 | 0.664 | 0.667 | n/a | 0.346 |
| 27 | 0.647 | 1.163 | 0.522 | 0.653 | 0.791 | 1.257 | 0.661 | 0.661 | 0.667 | n/a | 0.356 |
| 31 | 0.452 | 1.020 | 0.401 | 0.456 | 0.897 | 1.242 | 0.661 | 0.647 | 0.652 | n/a | 0.289 |
| 35 | 0.479 | 1.098 | 0.391 | 0.485 | 0.885 | 1.262 | 0.661 | 0.649 | 0.655 | n/a | 0.251 |
| 39 | 0.509 | 1.133 | 0.421 | 0.516 | 0.871 | 1.249 | 0.663 | 0.652 | 0.658 | n/a | 0.261 |
| 43 | 0.583 | 1.169 | 0.550 | 0.587 | 0.831 | 1.239 | 0.662 | 0.657 | 0.663 | n/a | 0.364 |

| KDA layer | state rel p50 over heads | p90 | eff. rank p50 | rank90 p50 | diff eff. rank p50 | diff energy in rank 8 p50 | rank 32 | conv rel L2 p50 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.012 | 2.025 | 2.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0.009 | 0.037 | 2.898 | 3.000 | 3.812 | 0.968 | 1.000 | 0.003 |
| 2 | 0.033 | 0.082 | 2.389 | 3.000 | 2.737 | 0.989 | 1.000 | 0.016 |
| 4 | 0.034 | 0.059 | 2.492 | 2.000 | 4.614 | 0.980 | 1.000 | 0.017 |
| 5 | 0.040 | 0.069 | 2.471 | 2.500 | 5.309 | 0.964 | 1.000 | 0.028 |
| 6 | 0.051 | 0.120 | 3.685 | 5.000 | 7.715 | 0.895 | 0.994 | 0.023 |
| 8 | 0.080 | 0.120 | 3.331 | 4.000 | 6.815 | 0.910 | 0.998 | 0.060 |
| 9 | 0.060 | 0.105 | 2.549 | 3.000 | 6.043 | 0.952 | 1.000 | 0.050 |
| 10 | 0.073 | 0.170 | 3.412 | 4.000 | 5.313 | 0.920 | 0.998 | 0.054 |
| 12 | 0.098 | 0.161 | 3.784 | 4.000 | 8.684 | 0.853 | 0.992 | 0.059 |
| 13 | 0.118 | 0.214 | 4.895 | 6.000 | 13.103 | 0.782 | 0.984 | 0.064 |
| 14 | 0.182 | 0.303 | 7.519 | 11.000 | 19.308 | 0.705 | 0.956 | 0.086 |
| 16 | 0.135 | 0.265 | 4.118 | 6.000 | 10.813 | 0.820 | 0.985 | 0.095 |
| 17 | 0.160 | 0.215 | 4.204 | 5.000 | 14.115 | 0.764 | 0.981 | 0.122 |
| 18 | 0.212 | 0.289 | 8.966 | 12.000 | 19.537 | 0.689 | 0.955 | 0.170 |
| 20 | 0.261 | 0.342 | 8.063 | 10.000 | 20.633 | 0.675 | 0.952 | 0.297 |
| 21 | 0.254 | 0.318 | 6.770 | 8.000 | 19.633 | 0.689 | 0.968 | 0.259 |
| 22 | 0.258 | 0.372 | 7.469 | 10.000 | 18.646 | 0.694 | 0.961 | 0.265 |
| 24 | 0.249 | 0.433 | 4.894 | 6.000 | 15.602 | 0.742 | 0.964 | 0.228 |
| 25 | 0.269 | 0.330 | 8.010 | 10.500 | 20.328 | 0.679 | 0.951 | 0.222 |
| 26 | 0.222 | 0.275 | 4.600 | 5.500 | 15.451 | 0.742 | 0.974 | 0.215 |
| 28 | 0.203 | 0.319 | 3.870 | 4.000 | 13.722 | 0.782 | 0.978 | 0.167 |
| 29 | 0.191 | 0.284 | 4.202 | 5.000 | 16.189 | 0.751 | 0.978 | 0.171 |
| 30 | 0.150 | 0.237 | 2.665 | 3.000 | 9.954 | 0.861 | 0.989 | 0.136 |
| 32 | 0.137 | 0.218 | 3.317 | 3.000 | 9.883 | 0.851 | 0.991 | 0.114 |
| 33 | 0.221 | 0.361 | 6.246 | 9.500 | 21.458 | 0.658 | 0.955 | 0.191 |
| 34 | 0.123 | 0.203 | 2.235 | 2.000 | 8.186 | 0.860 | 0.994 | 0.085 |
| 36 | 0.166 | 0.333 | 4.784 | 7.000 | 11.411 | 0.817 | 0.983 | 0.098 |
| 37 | 0.204 | 0.319 | 6.594 | 8.000 | 20.189 | 0.673 | 0.965 | 0.167 |
| 38 | 0.196 | 0.362 | 4.669 | 7.500 | 14.297 | 0.768 | 0.975 | 0.151 |
| 40 | 0.246 | 0.439 | 6.639 | 8.000 | 18.620 | 0.716 | 0.971 | 0.173 |
| 41 | 0.249 | 0.355 | 6.107 | 8.500 | 17.677 | 0.717 | 0.968 | 0.158 |
| 42 | 0.232 | 0.510 | 8.110 | 11.000 | 23.644 | 0.639 | 0.950 | 0.132 |
| 44 | 0.183 | 0.299 | 3.300 | 4.500 | 14.840 | 0.748 | 0.979 | 0.142 |

### L+S vs A+S: 2075 rows of S, keeps at [14336, 8192], [3906, 3906] tokens past S

| layer | latent rel L2 p50 | p90 | head p50 | body p50 | cos p50 | other-token p50 | raw | xor | delta | ring keys p50 | pooled keys p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.090 | 0.121 | 0.125 | 0.089 | 0.996 | 1.247 | 0.668 | 0.523 | 0.534 | n/a | 0.094 |
| 7 | 0.099 | 0.147 | 0.165 | 0.098 | 0.995 | 1.139 | 0.668 | 0.534 | 0.544 | n/a | 0.050 |
| 11 | 0.157 | 0.188 | 0.167 | 0.157 | 0.988 | 1.209 | 0.665 | 0.569 | 0.575 | n/a | 0.122 |
| 15 | 0.299 | 0.472 | 0.336 | 0.298 | 0.955 | 1.243 | 0.662 | 0.617 | 0.621 | n/a | 0.183 |
| 19 | 0.546 | 0.984 | 0.483 | 0.553 | 0.848 | 1.268 | 0.663 | 0.655 | 0.659 | n/a | 0.310 |
| 23 | 0.641 | 1.125 | 0.488 | 0.649 | 0.794 | 1.251 | 0.662 | 0.663 | 0.666 | n/a | 0.355 |
| 27 | 0.647 | 1.165 | 0.483 | 0.655 | 0.791 | 1.257 | 0.661 | 0.661 | 0.667 | n/a | 0.370 |
| 31 | 0.458 | 1.037 | 0.343 | 0.469 | 0.895 | 1.242 | 0.661 | 0.648 | 0.654 | n/a | 0.304 |
| 35 | 0.479 | 1.132 | 0.345 | 0.488 | 0.885 | 1.262 | 0.661 | 0.651 | 0.657 | n/a | 0.267 |
| 39 | 0.512 | 1.156 | 0.363 | 0.527 | 0.869 | 1.249 | 0.663 | 0.654 | 0.660 | n/a | 0.284 |
| 43 | 0.598 | 1.176 | 0.418 | 0.614 | 0.820 | 1.239 | 0.662 | 0.658 | 0.664 | n/a | 0.387 |

| KDA layer | state rel p50 over heads | p90 | eff. rank p50 | rank90 p50 | diff eff. rank p50 | diff energy in rank 8 p50 | rank 32 | conv rel L2 p50 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.010 | 2.025 | 2.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0.011 | 0.056 | 2.898 | 3.000 | 3.554 | 0.975 | 1.000 | 0.005 |
| 2 | 0.035 | 0.090 | 2.389 | 3.000 | 2.756 | 0.987 | 1.000 | 0.015 |
| 4 | 0.034 | 0.084 | 2.492 | 2.000 | 4.220 | 0.986 | 1.000 | 0.022 |
| 5 | 0.046 | 0.076 | 2.471 | 2.500 | 4.487 | 0.975 | 1.000 | 0.029 |
| 6 | 0.060 | 0.145 | 3.685 | 5.000 | 8.327 | 0.878 | 0.994 | 0.027 |
| 8 | 0.108 | 0.167 | 3.331 | 4.000 | 6.473 | 0.928 | 0.998 | 0.073 |
| 9 | 0.081 | 0.118 | 2.549 | 3.000 | 5.165 | 0.962 | 1.000 | 0.053 |
| 10 | 0.085 | 0.267 | 3.412 | 4.000 | 6.793 | 0.905 | 0.999 | 0.059 |
| 12 | 0.140 | 0.225 | 3.784 | 4.000 | 8.039 | 0.866 | 0.994 | 0.104 |
| 13 | 0.155 | 0.321 | 4.895 | 6.000 | 12.587 | 0.817 | 0.989 | 0.099 |
| 14 | 0.244 | 0.374 | 7.519 | 11.000 | 17.911 | 0.711 | 0.956 | 0.121 |
| 16 | 0.204 | 0.333 | 4.118 | 6.000 | 9.030 | 0.853 | 0.990 | 0.152 |
| 17 | 0.222 | 0.301 | 4.204 | 5.000 | 11.677 | 0.807 | 0.984 | 0.200 |
| 18 | 0.269 | 0.384 | 8.966 | 12.000 | 19.793 | 0.686 | 0.960 | 0.239 |
| 20 | 0.290 | 0.405 | 8.063 | 10.000 | 20.533 | 0.684 | 0.953 | 0.315 |
| 21 | 0.278 | 0.370 | 6.770 | 8.000 | 17.401 | 0.721 | 0.968 | 0.306 |
| 22 | 0.298 | 0.456 | 7.469 | 10.000 | 15.796 | 0.737 | 0.967 | 0.278 |
| 24 | 0.270 | 0.515 | 4.894 | 6.000 | 15.825 | 0.745 | 0.964 | 0.234 |
| 25 | 0.296 | 0.374 | 8.010 | 10.500 | 20.617 | 0.670 | 0.954 | 0.232 |
| 26 | 0.246 | 0.369 | 4.600 | 5.500 | 14.260 | 0.766 | 0.977 | 0.223 |
| 28 | 0.229 | 0.328 | 3.870 | 4.000 | 12.767 | 0.795 | 0.980 | 0.186 |
| 29 | 0.220 | 0.303 | 4.202 | 5.000 | 13.872 | 0.763 | 0.977 | 0.210 |
| 30 | 0.180 | 0.321 | 2.665 | 3.000 | 7.879 | 0.880 | 0.990 | 0.163 |
| 32 | 0.170 | 0.245 | 3.317 | 3.000 | 10.047 | 0.850 | 0.992 | 0.164 |
| 33 | 0.260 | 0.433 | 6.246 | 9.500 | 19.670 | 0.692 | 0.964 | 0.243 |
| 34 | 0.167 | 0.282 | 2.235 | 2.000 | 7.513 | 0.893 | 0.995 | 0.114 |
| 36 | 0.214 | 0.418 | 4.784 | 7.000 | 11.606 | 0.819 | 0.987 | 0.128 |
| 37 | 0.261 | 0.341 | 6.594 | 8.000 | 16.642 | 0.719 | 0.969 | 0.198 |
| 38 | 0.236 | 0.386 | 4.669 | 7.500 | 13.755 | 0.790 | 0.978 | 0.180 |
| 40 | 0.260 | 0.472 | 6.639 | 8.000 | 17.797 | 0.729 | 0.974 | 0.215 |
| 41 | 0.277 | 0.404 | 6.107 | 8.500 | 18.088 | 0.697 | 0.966 | 0.195 |
| 42 | 0.253 | 0.558 | 8.110 | 11.000 | 19.999 | 0.656 | 0.967 | 0.157 |
| 44 | 0.224 | 0.359 | 3.300 | 4.500 | 13.734 | 0.765 | 0.987 | 0.144 |

bench S: exit 0
    KVBENCH config mib=64 threads=8 reps=3 block=1048576
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep9.bin codec=memcpy threads=1 raw=259744768 stored=259744768 ratio=1.0000 encode_gbs=24.238 decode_gbs=24.238 keep256M_ms=11.1 restore256M_ms=11.1 keep1024M_ms=44.3 restore1024M_ms=44.3 keep4096M_ms=177.2 restore4096M_ms=177.2
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep9.bin codec=deflate1 threads=1 raw=259744768 stored=213330003 ratio=0.8213 encode_gbs=0.077 decode_gbs=0.181 keep256M_ms=3492.4 restore256M_ms=1482.3 keep1024M_ms=13969.7 restore1024M_ms=5929.1 keep4096M_ms=55878.7 restore4096M_ms=23716.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep9.bin codec=deflate6 threads=1 raw=259744768 stored=213307490 ratio=0.8212 encode_gbs=0.065 decode_gbs=0.181 keep256M_ms=4109.1 restore256M_ms=1485.3 keep1024M_ms=16436.3 restore1024M_ms=5941.3 keep4096M_ms=65745.2 restore4096M_ms=23765.2
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep9.bin codec=plain threads=1 raw=259744768 stored=187938265 ratio=0.7235 encode_gbs=0.319 decode_gbs=1.266 keep256M_ms=842.8 restore256M_ms=212.0 keep1024M_ms=3371.1 restore1024M_ms=848.1 keep4096M_ms=13484.2 restore4096M_ms=3392.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep9.bin codec=plain engagement raw=0 planes=256 planes_xor=0 planes_delta=0 planes_delta2=0 t64=0 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep9.bin codec=plain threads=8 raw=259744768 stored=187938265 ratio=0.7235 encode_gbs=1.448 decode_gbs=5.511 keep256M_ms=185.4 restore256M_ms=48.7 keep1024M_ms=741.5 restore1024M_ms=194.8 keep4096M_ms=2965.9 restore4096M_ms=779.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep9.bin codec=atlas threads=1 raw=259744768 stored=187717613 ratio=0.7227 encode_gbs=0.098 decode_gbs=1.227 keep256M_ms=2734.7 restore256M_ms=218.8 keep1024M_ms=10938.6 restore1024M_ms=875.3 keep4096M_ms=43754.4 restore4096M_ms=3501.2
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep9.bin codec=atlas engagement raw=0 planes=232 planes_xor=0 planes_delta=23 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep9.bin codec=atlas threads=8 raw=259744768 stored=187717613 ratio=0.7227 encode_gbs=0.445 decode_gbs=5.346 keep256M_ms=603.2 restore256M_ms=50.2 keep1024M_ms=2412.8 restore1024M_ms=200.9 keep4096M_ms=9651.3 restore4096M_ms=803.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep9.bin codec=atlas-exhaustive threads=1 raw=259744768 stored=187717173 ratio=0.7227 encode_gbs=0.029 decode_gbs=1.224 keep256M_ms=9190.5 restore256M_ms=219.4 keep1024M_ms=36761.8 restore1024M_ms=877.4 keep4096M_ms=147047.3 restore4096M_ms=3509.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep9.bin codec=atlas-exhaustive engagement raw=0 planes=231 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep9.bin codec=atlas-exhaustive threads=8 raw=259744768 stored=187717173 ratio=0.7227 encode_gbs=0.161 decode_gbs=6.410 keep256M_ms=1671.4 restore256M_ms=41.9 keep1024M_ms=6685.8 restore1024M_ms=167.5 keep4096M_ms=26743.1 restore4096M_ms=670.1
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep9.bin codec=ch-doubledelta threads=1 raw=259744768 stored=242480986 ratio=0.9335 encode_gbs=0.521 decode_gbs=7.604 keep256M_ms=515.7 restore256M_ms=35.3 keep1024M_ms=2062.8 restore1024M_ms=141.2 keep4096M_ms=8251.2 restore4096M_ms=564.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep9.bin codec=ch-doubledelta width=2 raw_blocks=233/256
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep9.bin codec=ch-gorilla threads=1 raw=259744768 stored=242904835 ratio=0.9352 encode_gbs=0.867 decode_gbs=17.244 keep256M_ms=309.5 restore256M_ms=15.6 keep1024M_ms=1237.8 restore1024M_ms=62.3 keep4096M_ms=4951.3 restore4096M_ms=249.1
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep9.bin codec=ch-gorilla width=2 raw_blocks=238/256
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep9.bin codec=ch-fpc12 threads=1 raw=259744768 stored=243907727 ratio=0.9390 encode_gbs=1.037 decode_gbs=8.815 keep256M_ms=258.8 restore256M_ms=30.5 keep1024M_ms=1035.0 restore1024M_ms=121.8 keep4096M_ms=4140.1 restore4096M_ms=487.2
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep9.bin codec=ch-fpc12 width=2 raw_blocks=235/256
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep9.bin codec=ch-alp-rd threads=1 raw=259744768 stored=218374834 ratio=0.8407 encode_gbs=0.182 decode_gbs=2.276 keep256M_ms=1473.9 restore256M_ms=117.9 keep1024M_ms=5895.7 restore1024M_ms=471.8 keep4096M_ms=23582.8 restore4096M_ms=1887.0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep9.bin codec=ch-alp-rd width=2 raw_blocks=137/256
bench A+S: exit 0
    KVBENCH config mib=64 threads=8 reps=3 block=1048576
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep10.bin codec=memcpy threads=1 raw=284255232 stored=284255232 ratio=1.0000 encode_gbs=24.712 decode_gbs=24.712 keep256M_ms=10.9 restore256M_ms=10.9 keep1024M_ms=43.5 restore1024M_ms=43.5 keep4096M_ms=173.8 restore4096M_ms=173.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep10.bin codec=deflate1 threads=1 raw=284255232 stored=232880242 ratio=0.8193 encode_gbs=0.077 decode_gbs=0.167 keep256M_ms=3473.9 restore256M_ms=1607.6 keep1024M_ms=13895.6 restore1024M_ms=6430.6 keep4096M_ms=55582.2 restore4096M_ms=25722.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep10.bin codec=deflate6 threads=1 raw=284255232 stored=232857180 ratio=0.8192 encode_gbs=0.064 decode_gbs=0.166 keep256M_ms=4174.2 restore256M_ms=1612.9 keep1024M_ms=16696.8 restore1024M_ms=6451.7 keep4096M_ms=66787.3 restore4096M_ms=25806.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep10.bin codec=plain threads=1 raw=284255232 stored=204250726 ratio=0.7185 encode_gbs=0.316 decode_gbs=1.213 keep256M_ms=850.8 restore256M_ms=221.3 keep1024M_ms=3403.0 restore1024M_ms=885.2 keep4096M_ms=13612.2 restore4096M_ms=3540.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep10.bin codec=plain engagement raw=0 planes=278 planes_xor=0 planes_delta=0 planes_delta2=0 t64=0 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep10.bin codec=plain threads=8 raw=284255232 stored=204250726 ratio=0.7185 encode_gbs=1.727 decode_gbs=6.423 keep256M_ms=155.4 restore256M_ms=41.8 keep1024M_ms=621.6 restore1024M_ms=167.2 keep4096M_ms=2486.4 restore4096M_ms=668.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep10.bin codec=atlas threads=1 raw=284255232 stored=203997513 ratio=0.7177 encode_gbs=0.096 decode_gbs=1.180 keep256M_ms=2797.2 restore256M_ms=227.5 keep1024M_ms=11188.9 restore1024M_ms=910.0 keep4096M_ms=44755.5 restore4096M_ms=3640.1
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep10.bin codec=atlas engagement raw=0 planes=253 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep10.bin codec=atlas threads=8 raw=284255232 stored=203997513 ratio=0.7177 encode_gbs=0.521 decode_gbs=6.106 keep256M_ms=514.8 restore256M_ms=44.0 keep1024M_ms=2059.1 restore1024M_ms=175.9 keep4096M_ms=8236.5 restore4096M_ms=703.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep10.bin codec=atlas-exhaustive threads=1 raw=284255232 stored=203997513 ratio=0.7177 encode_gbs=0.028 decode_gbs=1.179 keep256M_ms=9571.8 restore256M_ms=227.6 keep1024M_ms=38287.1 restore1024M_ms=910.4 keep4096M_ms=153148.4 restore4096M_ms=3641.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep10.bin codec=atlas-exhaustive engagement raw=0 planes=253 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep10.bin codec=atlas-exhaustive threads=8 raw=284255232 stored=203997513 ratio=0.7177 encode_gbs=0.155 decode_gbs=6.277 keep256M_ms=1731.7 restore256M_ms=42.8 keep1024M_ms=6926.7 restore1024M_ms=171.0 keep4096M_ms=27706.7 restore4096M_ms=684.2
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep10.bin codec=ch-doubledelta threads=1 raw=284255232 stored=266944413 ratio=0.9391 encode_gbs=0.500 decode_gbs=8.127 keep256M_ms=537.0 restore256M_ms=33.0 keep1024M_ms=2148.2 restore1024M_ms=132.1 keep4096M_ms=8592.6 restore4096M_ms=528.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep10.bin codec=ch-doubledelta width=2 raw_blocks=255/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep10.bin codec=ch-gorilla threads=1 raw=284255232 stored=267415295 ratio=0.9408 encode_gbs=0.864 decode_gbs=17.837 keep256M_ms=310.6 restore256M_ms=15.0 keep1024M_ms=1242.5 restore1024M_ms=60.2 keep4096M_ms=4970.0 restore4096M_ms=240.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep10.bin codec=ch-gorilla width=2 raw_blocks=260/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep10.bin codec=ch-fpc12 threads=1 raw=284255232 stored=268407088 ratio=0.9442 encode_gbs=1.010 decode_gbs=9.350 keep256M_ms=265.8 restore256M_ms=28.7 keep1024M_ms=1063.0 restore1024M_ms=114.8 keep4096M_ms=4252.1 restore4096M_ms=459.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep10.bin codec=ch-fpc12 width=2 raw_blocks=257/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep10.bin codec=ch-alp-rd threads=1 raw=284255232 stored=236730350 ratio=0.8328 encode_gbs=0.185 decode_gbs=2.036 keep256M_ms=1449.3 restore256M_ms=131.8 keep1024M_ms=5797.4 restore1024M_ms=527.4 keep4096M_ms=23189.4 restore4096M_ms=2109.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep10.bin codec=ch-alp-rd width=2 raw_blocks=137/278
bench B+S: exit 0
    KVBENCH config mib=64 threads=8 reps=3 block=1048576
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep11.bin codec=memcpy threads=1 raw=284255232 stored=284255232 ratio=1.0000 encode_gbs=24.771 decode_gbs=24.771 keep256M_ms=10.8 restore256M_ms=10.8 keep1024M_ms=43.3 restore1024M_ms=43.3 keep4096M_ms=173.4 restore4096M_ms=173.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep11.bin codec=deflate1 threads=1 raw=284255232 stored=232881204 ratio=0.8193 encode_gbs=0.077 decode_gbs=0.178 keep256M_ms=3482.1 restore256M_ms=1508.2 keep1024M_ms=13928.5 restore1024M_ms=6032.7 keep4096M_ms=55714.0 restore4096M_ms=24130.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep11.bin codec=deflate6 threads=1 raw=284255232 stored=232857545 ratio=0.8192 encode_gbs=0.064 decode_gbs=0.177 keep256M_ms=4162.1 restore256M_ms=1512.7 keep1024M_ms=16648.5 restore1024M_ms=6050.9 keep4096M_ms=66594.2 restore4096M_ms=24203.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep11.bin codec=plain threads=1 raw=284255232 stored=204256947 ratio=0.7186 encode_gbs=0.316 decode_gbs=1.209 keep256M_ms=848.4 restore256M_ms=222.0 keep1024M_ms=3393.4 restore1024M_ms=887.8 keep4096M_ms=13573.6 restore4096M_ms=3551.2
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep11.bin codec=plain engagement raw=0 planes=278 planes_xor=0 planes_delta=0 planes_delta2=0 t64=0 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep11.bin codec=plain threads=8 raw=284255232 stored=204256947 ratio=0.7186 encode_gbs=1.748 decode_gbs=6.342 keep256M_ms=153.6 restore256M_ms=42.3 keep1024M_ms=614.3 restore1024M_ms=169.3 keep4096M_ms=2457.4 restore4096M_ms=677.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep11.bin codec=atlas threads=1 raw=284255232 stored=204011432 ratio=0.7177 encode_gbs=0.096 decode_gbs=1.177 keep256M_ms=2808.3 restore256M_ms=228.1 keep1024M_ms=11233.3 restore1024M_ms=912.3 keep4096M_ms=44933.3 restore4096M_ms=3649.1
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep11.bin codec=atlas engagement raw=0 planes=253 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep11.bin codec=atlas threads=8 raw=284255232 stored=204011432 ratio=0.7177 encode_gbs=0.526 decode_gbs=6.266 keep256M_ms=510.4 restore256M_ms=42.8 keep1024M_ms=2041.6 restore1024M_ms=171.3 keep4096M_ms=8166.5 restore4096M_ms=685.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep11.bin codec=atlas-exhaustive threads=1 raw=284255232 stored=204011432 ratio=0.7177 encode_gbs=0.028 decode_gbs=1.176 keep256M_ms=9581.2 restore256M_ms=228.2 keep1024M_ms=38324.8 restore1024M_ms=912.7 keep4096M_ms=153299.0 restore4096M_ms=3650.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep11.bin codec=atlas-exhaustive engagement raw=0 planes=253 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep11.bin codec=atlas-exhaustive threads=8 raw=284255232 stored=204011432 ratio=0.7177 encode_gbs=0.155 decode_gbs=6.161 keep256M_ms=1737.1 restore256M_ms=43.6 keep1024M_ms=6948.5 restore1024M_ms=174.3 keep4096M_ms=27793.8 restore4096M_ms=697.1
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep11.bin codec=ch-doubledelta threads=1 raw=284255232 stored=266941075 ratio=0.9391 encode_gbs=0.500 decode_gbs=8.073 keep256M_ms=536.8 restore256M_ms=33.3 keep1024M_ms=2147.1 restore1024M_ms=133.0 keep4096M_ms=8588.2 restore4096M_ms=532.0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep11.bin codec=ch-doubledelta width=2 raw_blocks=255/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep11.bin codec=ch-gorilla threads=1 raw=284255232 stored=267415298 ratio=0.9408 encode_gbs=0.860 decode_gbs=17.262 keep256M_ms=312.0 restore256M_ms=15.6 keep1024M_ms=1248.0 restore1024M_ms=62.2 keep4096M_ms=4992.2 restore4096M_ms=248.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep11.bin codec=ch-gorilla width=2 raw_blocks=260/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep11.bin codec=ch-fpc12 threads=1 raw=284255232 stored=268401306 ratio=0.9442 encode_gbs=1.029 decode_gbs=9.279 keep256M_ms=261.0 restore256M_ms=28.9 keep1024M_ms=1044.0 restore1024M_ms=115.7 keep4096M_ms=4175.8 restore4096M_ms=462.9
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep11.bin codec=ch-fpc12 width=2 raw_blocks=257/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep11.bin codec=ch-alp-rd threads=1 raw=284255232 stored=236864194 ratio=0.8333 encode_gbs=0.184 decode_gbs=2.031 keep256M_ms=1460.7 restore256M_ms=132.2 keep1024M_ms=5842.8 restore1024M_ms=528.7 keep4096M_ms=23371.3 restore4096M_ms=2114.6
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep11.bin codec=ch-alp-rd width=2 raw_blocks=137/278
bench L+S: exit 0
    KVBENCH config mib=64 threads=8 reps=3 block=1048576
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep12.bin codec=memcpy threads=1 raw=357786624 stored=357786624 ratio=1.0000 encode_gbs=24.359 decode_gbs=24.359 keep256M_ms=11.0 restore256M_ms=11.0 keep1024M_ms=44.1 restore1024M_ms=44.1 keep4096M_ms=176.3 restore4096M_ms=176.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep12.bin codec=deflate1 threads=1 raw=357786624 stored=291561249 ratio=0.8149 encode_gbs=0.078 decode_gbs=0.172 keep256M_ms=3460.3 restore256M_ms=1563.1 keep1024M_ms=13841.4 restore1024M_ms=6252.2 keep4096M_ms=55365.5 restore4096M_ms=25008.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep12.bin codec=deflate6 threads=1 raw=357786624 stored=291532516 ratio=0.8148 encode_gbs=0.062 decode_gbs=0.171 keep256M_ms=4311.6 restore256M_ms=1571.2 keep1024M_ms=17246.2 restore1024M_ms=6284.8 keep4096M_ms=68985.0 restore4096M_ms=25139.0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep12.bin codec=plain threads=1 raw=357786624 stored=253241268 ratio=0.7078 encode_gbs=0.313 decode_gbs=1.125 keep256M_ms=857.8 restore256M_ms=238.7 keep1024M_ms=3431.2 restore1024M_ms=954.8 keep4096M_ms=13725.0 restore4096M_ms=3819.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep12.bin codec=plain engagement raw=0 planes=344 planes_xor=0 planes_delta=0 planes_delta2=0 t64=0 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep12.bin codec=plain threads=8 raw=357786624 stored=253241268 ratio=0.7078 encode_gbs=1.732 decode_gbs=5.947 keep256M_ms=155.0 restore256M_ms=45.1 keep1024M_ms=620.0 restore1024M_ms=180.6 keep4096M_ms=2480.2 restore4096M_ms=722.2
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep12.bin codec=atlas threads=1 raw=357786624 stored=252894204 ratio=0.7068 encode_gbs=0.090 decode_gbs=1.100 keep256M_ms=2966.9 restore256M_ms=244.1 keep1024M_ms=11867.5 restore1024M_ms=976.2 keep4096M_ms=47469.9 restore4096M_ms=3904.9
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep12.bin codec=atlas engagement raw=0 planes=319 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep12.bin codec=atlas threads=8 raw=357786624 stored=252894204 ratio=0.7068 encode_gbs=0.499 decode_gbs=5.897 keep256M_ms=537.9 restore256M_ms=45.5 keep1024M_ms=2151.4 restore1024M_ms=182.1 keep4096M_ms=8605.7 restore4096M_ms=728.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep12.bin codec=atlas-exhaustive threads=1 raw=357786624 stored=252894204 ratio=0.7068 encode_gbs=0.026 decode_gbs=1.101 keep256M_ms=10439.2 restore256M_ms=243.8 keep1024M_ms=41757.0 restore1024M_ms=975.4 keep4096M_ms=167027.8 restore4096M_ms=3901.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep12.bin codec=atlas-exhaustive engagement raw=0 planes=319 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep12.bin codec=atlas-exhaustive threads=8 raw=357786624 stored=252894204 ratio=0.7068 encode_gbs=0.142 decode_gbs=5.909 keep256M_ms=1883.8 restore256M_ms=45.4 keep1024M_ms=7535.1 restore1024M_ms=181.7 keep4096M_ms=30140.5 restore4096M_ms=726.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep12.bin codec=ch-doubledelta threads=1 raw=357786624 stored=340441021 ratio=0.9515 encode_gbs=0.460 decode_gbs=9.348 keep256M_ms=584.1 restore256M_ms=28.7 keep1024M_ms=2336.4 restore1024M_ms=114.9 keep4096M_ms=9345.6 restore4096M_ms=459.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep12.bin codec=ch-doubledelta width=2 raw_blocks=321/344
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep12.bin codec=ch-gorilla threads=1 raw=357786624 stored=340946689 ratio=0.9529 encode_gbs=0.854 decode_gbs=18.747 keep256M_ms=314.3 restore256M_ms=14.3 keep1024M_ms=1257.2 restore1024M_ms=57.3 keep4096M_ms=5028.9 restore4096M_ms=229.1
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep12.bin codec=ch-gorilla width=2 raw_blocks=326/344
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep12.bin codec=ch-fpc12 threads=1 raw=357786624 stored=341932523 ratio=0.9557 encode_gbs=1.010 decode_gbs=10.649 keep256M_ms=265.8 restore256M_ms=25.2 keep1024M_ms=1063.1 restore1024M_ms=100.8 keep4096M_ms=4252.2 restore4096M_ms=403.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep12.bin codec=ch-fpc12 width=2 raw_blocks=323/344
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep12.bin codec=ch-alp-rd threads=1 raw=357786624 stored=291260640 ratio=0.8141 encode_gbs=0.193 decode_gbs=1.664 keep256M_ms=1390.2 restore256M_ms=161.3 keep1024M_ms=5560.9 restore1024M_ms=645.4 keep4096M_ms=22243.5 restore4096M_ms=2581.6
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep12.bin codec=ch-alp-rd width=2 raw_blocks=137/344

## prose

### A+S vs B+S: 2055 rows of S, keeps at [8192, 8192], [4068, 4068] tokens past S

| layer | latent rel L2 p50 | p90 | head p50 | body p50 | cos p50 | other-token p50 | raw | xor | delta | ring keys p50 | pooled keys p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.094 | 0.170 | 0.206 | 0.092 | 0.996 | 1.363 | 0.668 | 0.534 | 0.544 | 0.055 | 0.098 |
| 7 | 0.130 | 0.292 | 0.284 | 0.127 | 0.992 | 1.355 | 0.669 | 0.564 | 0.571 | 0.055 | 0.060 |
| 11 | 0.141 | 0.262 | 0.337 | 0.140 | 0.990 | 1.357 | 0.665 | 0.568 | 0.575 | 0.089 | 0.098 |
| 15 | 0.222 | 0.429 | 0.460 | 0.219 | 0.975 | 1.334 | 0.661 | 0.602 | 0.607 | 0.140 | 0.153 |
| 19 | 0.341 | 0.698 | 0.606 | 0.336 | 0.942 | 1.316 | 0.662 | 0.631 | 0.634 | 0.212 | 0.240 |
| 23 | 0.380 | 0.767 | 0.633 | 0.373 | 0.928 | 1.305 | 0.662 | 0.637 | 0.640 | 0.224 | 0.251 |
| 27 | 0.367 | 0.756 | 0.564 | 0.361 | 0.932 | 1.295 | 0.661 | 0.634 | 0.638 | 0.235 | 0.265 |
| 31 | 0.276 | 0.485 | 0.447 | 0.272 | 0.962 | 1.317 | 0.661 | 0.614 | 0.618 | 0.196 | 0.221 |
| 35 | 0.274 | 0.512 | 0.421 | 0.271 | 0.962 | 1.315 | 0.661 | 0.615 | 0.620 | 0.178 | 0.193 |
| 39 | 0.291 | 0.566 | 0.418 | 0.287 | 0.958 | 1.306 | 0.663 | 0.620 | 0.624 | 0.167 | 0.171 |
| 43 | 0.340 | 0.683 | 0.526 | 0.335 | 0.942 | 1.319 | 0.662 | 0.629 | 0.633 | 0.198 | 0.208 |

| KDA layer | state rel p50 over heads | p90 | eff. rank p50 | rank90 p50 | diff eff. rank p50 | diff energy in rank 8 p50 | rank 32 | conv rel L2 p50 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.008 | 1.998 | 2.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0.009 | 0.024 | 2.931 | 3.000 | 4.039 | 0.966 | 1.000 | 0.003 |
| 2 | 0.036 | 0.082 | 2.016 | 2.000 | 2.447 | 0.990 | 1.000 | 0.015 |
| 4 | 0.041 | 0.079 | 2.309 | 2.000 | 3.848 | 0.987 | 1.000 | 0.021 |
| 5 | 0.065 | 0.094 | 1.919 | 2.000 | 4.085 | 0.978 | 1.000 | 0.049 |
| 6 | 0.073 | 0.132 | 3.856 | 4.500 | 7.486 | 0.893 | 0.995 | 0.043 |
| 8 | 0.087 | 0.168 | 3.155 | 3.000 | 6.886 | 0.912 | 0.998 | 0.059 |
| 9 | 0.076 | 0.108 | 2.107 | 2.000 | 4.753 | 0.963 | 1.000 | 0.050 |
| 10 | 0.078 | 0.156 | 3.109 | 4.000 | 6.475 | 0.916 | 0.999 | 0.058 |
| 12 | 0.129 | 0.190 | 3.328 | 4.000 | 9.041 | 0.858 | 0.994 | 0.087 |
| 13 | 0.149 | 0.220 | 4.856 | 5.500 | 14.664 | 0.778 | 0.991 | 0.094 |
| 14 | 0.190 | 0.296 | 7.141 | 11.000 | 18.983 | 0.684 | 0.957 | 0.115 |
| 16 | 0.190 | 0.268 | 4.162 | 6.000 | 10.780 | 0.816 | 0.982 | 0.135 |
| 17 | 0.226 | 0.295 | 4.221 | 5.500 | 11.714 | 0.820 | 0.987 | 0.189 |
| 18 | 0.249 | 0.307 | 8.631 | 10.000 | 16.615 | 0.731 | 0.964 | 0.195 |
| 20 | 0.292 | 0.339 | 6.962 | 9.500 | 18.252 | 0.715 | 0.965 | 0.224 |
| 21 | 0.301 | 0.350 | 5.611 | 7.000 | 16.809 | 0.733 | 0.974 | 0.237 |
| 22 | 0.300 | 0.387 | 7.134 | 8.500 | 14.223 | 0.745 | 0.971 | 0.239 |
| 24 | 0.298 | 0.426 | 4.577 | 5.500 | 12.111 | 0.817 | 0.981 | 0.307 |
| 25 | 0.289 | 0.383 | 6.809 | 9.500 | 15.728 | 0.749 | 0.974 | 0.286 |
| 26 | 0.266 | 0.327 | 4.513 | 5.000 | 10.403 | 0.849 | 0.990 | 0.280 |
| 28 | 0.266 | 0.445 | 3.615 | 3.500 | 7.534 | 0.904 | 0.997 | 0.239 |
| 29 | 0.284 | 0.411 | 4.220 | 4.500 | 8.718 | 0.893 | 0.995 | 0.270 |
| 30 | 0.193 | 0.305 | 2.653 | 3.000 | 8.355 | 0.889 | 0.995 | 0.238 |
| 32 | 0.191 | 0.322 | 3.556 | 4.000 | 8.517 | 0.893 | 0.997 | 0.193 |
| 33 | 0.257 | 0.396 | 6.014 | 8.000 | 18.543 | 0.694 | 0.974 | 0.285 |
| 34 | 0.189 | 0.248 | 2.447 | 2.500 | 8.369 | 0.869 | 0.996 | 0.166 |
| 36 | 0.168 | 0.347 | 5.334 | 7.000 | 12.041 | 0.805 | 0.984 | 0.149 |
| 37 | 0.241 | 0.344 | 6.460 | 8.000 | 18.416 | 0.701 | 0.971 | 0.297 |
| 38 | 0.231 | 0.435 | 4.225 | 7.000 | 13.103 | 0.779 | 0.981 | 0.244 |
| 40 | 0.274 | 0.395 | 6.119 | 7.500 | 16.695 | 0.743 | 0.979 | 0.307 |
| 41 | 0.275 | 0.366 | 6.615 | 8.500 | 18.903 | 0.678 | 0.964 | 0.262 |
| 42 | 0.273 | 0.491 | 7.649 | 10.500 | 19.319 | 0.691 | 0.964 | 0.222 |
| 44 | 0.258 | 0.322 | 3.316 | 4.500 | 11.329 | 0.830 | 0.983 | 0.221 |

### S vs A+S: 2055 rows of S, keeps at [6144, 8192], [4068, 4068] tokens past S

| layer | latent rel L2 p50 | p90 | head p50 | body p50 | cos p50 | other-token p50 | raw | xor | delta | ring keys p50 | pooled keys p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.098 | 0.173 | 0.271 | 0.097 | 0.995 | 1.366 | 0.668 | 0.537 | 0.547 | n/a | 0.121 |
| 7 | 0.129 | 0.289 | 0.315 | 0.127 | 0.992 | 1.348 | 0.669 | 0.565 | 0.572 | n/a | 0.074 |
| 11 | 0.144 | 0.279 | 0.367 | 0.140 | 0.990 | 1.363 | 0.665 | 0.570 | 0.577 | n/a | 0.111 |
| 15 | 0.215 | 0.450 | 0.489 | 0.212 | 0.977 | 1.345 | 0.661 | 0.600 | 0.606 | n/a | 0.142 |
| 19 | 0.329 | 0.706 | 0.685 | 0.324 | 0.946 | 1.329 | 0.662 | 0.630 | 0.633 | n/a | 0.233 |
| 23 | 0.380 | 0.805 | 0.671 | 0.373 | 0.927 | 1.316 | 0.662 | 0.638 | 0.641 | n/a | 0.260 |
| 27 | 0.365 | 0.786 | 0.612 | 0.359 | 0.933 | 1.305 | 0.661 | 0.634 | 0.639 | n/a | 0.278 |
| 31 | 0.280 | 0.499 | 0.474 | 0.276 | 0.961 | 1.327 | 0.661 | 0.614 | 0.619 | n/a | 0.229 |
| 35 | 0.279 | 0.535 | 0.455 | 0.274 | 0.961 | 1.321 | 0.661 | 0.616 | 0.621 | n/a | 0.209 |
| 39 | 0.290 | 0.586 | 0.444 | 0.287 | 0.958 | 1.313 | 0.663 | 0.620 | 0.625 | n/a | 0.180 |
| 43 | 0.337 | 0.701 | 0.536 | 0.332 | 0.943 | 1.319 | 0.662 | 0.630 | 0.634 | n/a | 0.209 |

| KDA layer | state rel p50 over heads | p90 | eff. rank p50 | rank90 p50 | diff eff. rank p50 | diff energy in rank 8 p50 | rank 32 | conv rel L2 p50 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.009 | 2.000 | 2.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0.008 | 0.034 | 2.934 | 3.000 | 4.288 | 0.968 | 1.000 | 0.003 |
| 2 | 0.028 | 0.070 | 2.003 | 2.000 | 2.504 | 0.991 | 1.000 | 0.011 |
| 4 | 0.034 | 0.061 | 2.340 | 2.000 | 3.586 | 0.986 | 1.000 | 0.015 |
| 5 | 0.059 | 0.100 | 1.937 | 2.000 | 4.670 | 0.980 | 1.000 | 0.043 |
| 6 | 0.063 | 0.106 | 3.865 | 5.000 | 7.315 | 0.892 | 0.994 | 0.036 |
| 8 | 0.078 | 0.114 | 3.230 | 3.000 | 6.440 | 0.911 | 0.998 | 0.052 |
| 9 | 0.073 | 0.101 | 2.075 | 2.000 | 4.399 | 0.951 | 1.000 | 0.060 |
| 10 | 0.081 | 0.148 | 3.025 | 3.500 | 5.794 | 0.935 | 0.999 | 0.060 |
| 12 | 0.112 | 0.169 | 3.318 | 4.000 | 9.226 | 0.860 | 0.994 | 0.082 |
| 13 | 0.139 | 0.185 | 4.860 | 5.500 | 14.320 | 0.783 | 0.993 | 0.094 |
| 14 | 0.162 | 0.247 | 6.875 | 10.500 | 21.381 | 0.660 | 0.950 | 0.136 |
| 16 | 0.182 | 0.244 | 4.021 | 5.500 | 12.001 | 0.794 | 0.982 | 0.125 |
| 17 | 0.214 | 0.251 | 4.117 | 5.500 | 13.003 | 0.803 | 0.989 | 0.187 |
| 18 | 0.234 | 0.282 | 8.377 | 10.500 | 19.316 | 0.684 | 0.963 | 0.195 |
| 20 | 0.269 | 0.301 | 7.022 | 9.000 | 19.982 | 0.690 | 0.962 | 0.205 |
| 21 | 0.281 | 0.349 | 5.543 | 7.000 | 16.652 | 0.729 | 0.973 | 0.224 |
| 22 | 0.277 | 0.361 | 6.998 | 8.000 | 16.311 | 0.724 | 0.967 | 0.214 |
| 24 | 0.266 | 0.393 | 4.585 | 6.000 | 13.072 | 0.802 | 0.982 | 0.278 |
| 25 | 0.285 | 0.355 | 6.536 | 9.000 | 16.572 | 0.746 | 0.975 | 0.268 |
| 26 | 0.261 | 0.325 | 4.579 | 5.000 | 11.193 | 0.838 | 0.990 | 0.256 |
| 28 | 0.249 | 0.379 | 3.522 | 4.000 | 7.881 | 0.905 | 0.997 | 0.221 |
| 29 | 0.255 | 0.395 | 4.318 | 4.500 | 8.903 | 0.891 | 0.995 | 0.257 |
| 30 | 0.176 | 0.261 | 2.603 | 3.000 | 9.062 | 0.881 | 0.995 | 0.219 |
| 32 | 0.184 | 0.284 | 3.584 | 4.000 | 9.320 | 0.887 | 0.997 | 0.183 |
| 33 | 0.259 | 0.384 | 5.900 | 8.000 | 18.336 | 0.705 | 0.975 | 0.250 |
| 34 | 0.174 | 0.227 | 2.422 | 3.000 | 8.397 | 0.871 | 0.998 | 0.148 |
| 36 | 0.150 | 0.291 | 5.323 | 7.000 | 12.137 | 0.786 | 0.983 | 0.135 |
| 37 | 0.233 | 0.335 | 6.249 | 8.000 | 17.483 | 0.701 | 0.969 | 0.257 |
| 38 | 0.208 | 0.399 | 4.277 | 6.500 | 11.255 | 0.801 | 0.978 | 0.221 |
| 40 | 0.255 | 0.381 | 6.215 | 7.000 | 18.679 | 0.698 | 0.977 | 0.271 |
| 41 | 0.266 | 0.350 | 6.587 | 8.500 | 18.766 | 0.691 | 0.959 | 0.233 |
| 42 | 0.248 | 0.477 | 7.857 | 10.500 | 21.499 | 0.663 | 0.959 | 0.185 |
| 44 | 0.242 | 0.295 | 3.300 | 4.500 | 11.385 | 0.817 | 0.980 | 0.191 |

### L+S vs A+S: 2055 rows of S, keeps at [14336, 8192], [4068, 4068] tokens past S

| layer | latent rel L2 p50 | p90 | head p50 | body p50 | cos p50 | other-token p50 | raw | xor | delta | ring keys p50 | pooled keys p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.101 | 0.157 | 0.165 | 0.099 | 0.995 | 1.366 | 0.668 | 0.533 | 0.544 | n/a | 0.106 |
| 7 | 0.145 | 0.274 | 0.241 | 0.143 | 0.990 | 1.348 | 0.669 | 0.568 | 0.575 | n/a | 0.066 |
| 11 | 0.165 | 0.247 | 0.242 | 0.163 | 0.986 | 1.363 | 0.665 | 0.576 | 0.582 | n/a | 0.137 |
| 15 | 0.271 | 0.423 | 0.389 | 0.268 | 0.963 | 1.345 | 0.661 | 0.611 | 0.615 | n/a | 0.196 |
| 19 | 0.392 | 0.674 | 0.545 | 0.388 | 0.923 | 1.329 | 0.662 | 0.636 | 0.639 | n/a | 0.289 |
| 23 | 0.425 | 0.798 | 0.565 | 0.421 | 0.910 | 1.316 | 0.662 | 0.643 | 0.646 | n/a | 0.288 |
| 27 | 0.419 | 0.792 | 0.502 | 0.416 | 0.913 | 1.305 | 0.661 | 0.640 | 0.645 | n/a | 0.311 |
| 31 | 0.325 | 0.512 | 0.392 | 0.323 | 0.947 | 1.327 | 0.661 | 0.623 | 0.627 | n/a | 0.275 |
| 35 | 0.326 | 0.535 | 0.380 | 0.324 | 0.947 | 1.321 | 0.661 | 0.624 | 0.629 | n/a | 0.248 |
| 39 | 0.333 | 0.591 | 0.371 | 0.329 | 0.945 | 1.313 | 0.663 | 0.627 | 0.631 | n/a | 0.214 |
| 43 | 0.359 | 0.662 | 0.424 | 0.356 | 0.935 | 1.319 | 0.662 | 0.632 | 0.636 | n/a | 0.234 |

| KDA layer | state rel p50 over heads | p90 | eff. rank p50 | rank90 p50 | diff eff. rank p50 | diff energy in rank 8 p50 | rank 32 | conv rel L2 p50 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.007 | 2.000 | 2.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0.011 | 0.048 | 2.934 | 3.000 | 3.999 | 0.972 | 1.000 | 0.004 |
| 2 | 0.043 | 0.103 | 2.003 | 2.000 | 2.617 | 0.990 | 1.000 | 0.016 |
| 4 | 0.062 | 0.100 | 2.340 | 2.000 | 3.855 | 0.983 | 1.000 | 0.060 |
| 5 | 0.091 | 0.145 | 1.937 | 2.000 | 4.274 | 0.976 | 1.000 | 0.076 |
| 6 | 0.107 | 0.143 | 3.865 | 5.000 | 7.316 | 0.899 | 0.995 | 0.083 |
| 8 | 0.125 | 0.163 | 3.230 | 3.000 | 7.443 | 0.914 | 0.998 | 0.101 |
| 9 | 0.116 | 0.138 | 2.075 | 2.000 | 4.636 | 0.966 | 1.000 | 0.092 |
| 10 | 0.098 | 0.266 | 3.025 | 3.500 | 7.741 | 0.905 | 0.999 | 0.095 |
| 12 | 0.157 | 0.241 | 3.318 | 4.000 | 10.927 | 0.834 | 0.993 | 0.115 |
| 13 | 0.200 | 0.288 | 4.860 | 5.500 | 13.358 | 0.797 | 0.994 | 0.146 |
| 14 | 0.227 | 0.313 | 6.875 | 10.500 | 20.702 | 0.680 | 0.952 | 0.157 |
| 16 | 0.229 | 0.303 | 4.021 | 5.500 | 11.045 | 0.826 | 0.986 | 0.201 |
| 17 | 0.263 | 0.297 | 4.117 | 5.500 | 12.371 | 0.800 | 0.986 | 0.241 |
| 18 | 0.287 | 0.353 | 8.377 | 10.500 | 17.381 | 0.711 | 0.963 | 0.200 |
| 20 | 0.313 | 0.393 | 7.022 | 9.000 | 18.924 | 0.702 | 0.960 | 0.232 |
| 21 | 0.327 | 0.373 | 5.543 | 7.000 | 17.355 | 0.721 | 0.970 | 0.252 |
| 22 | 0.312 | 0.459 | 6.998 | 8.000 | 13.656 | 0.773 | 0.972 | 0.221 |
| 24 | 0.309 | 0.490 | 4.585 | 6.000 | 13.244 | 0.799 | 0.978 | 0.271 |
| 25 | 0.331 | 0.392 | 6.536 | 9.000 | 17.077 | 0.739 | 0.974 | 0.247 |
| 26 | 0.287 | 0.378 | 4.579 | 5.000 | 11.082 | 0.836 | 0.988 | 0.242 |
| 28 | 0.277 | 0.440 | 3.522 | 4.000 | 8.711 | 0.877 | 0.995 | 0.249 |
| 29 | 0.292 | 0.418 | 4.318 | 4.500 | 9.657 | 0.871 | 0.993 | 0.287 |
| 30 | 0.213 | 0.336 | 2.603 | 3.000 | 8.155 | 0.895 | 0.994 | 0.228 |
| 32 | 0.214 | 0.360 | 3.584 | 4.000 | 8.719 | 0.875 | 0.997 | 0.205 |
| 33 | 0.293 | 0.428 | 5.900 | 8.000 | 19.474 | 0.690 | 0.970 | 0.288 |
| 34 | 0.204 | 0.284 | 2.422 | 3.000 | 8.516 | 0.882 | 0.997 | 0.155 |
| 36 | 0.198 | 0.360 | 5.323 | 7.000 | 13.192 | 0.791 | 0.983 | 0.146 |
| 37 | 0.285 | 0.404 | 6.249 | 8.000 | 18.355 | 0.701 | 0.970 | 0.260 |
| 38 | 0.259 | 0.475 | 4.277 | 6.500 | 12.787 | 0.801 | 0.983 | 0.215 |
| 40 | 0.294 | 0.449 | 6.215 | 7.000 | 17.060 | 0.728 | 0.977 | 0.264 |
| 41 | 0.299 | 0.405 | 6.587 | 8.500 | 18.744 | 0.674 | 0.962 | 0.233 |
| 42 | 0.300 | 0.557 | 7.857 | 10.500 | 21.005 | 0.661 | 0.962 | 0.188 |
| 44 | 0.274 | 0.355 | 3.300 | 4.500 | 11.561 | 0.813 | 0.984 | 0.201 |

bench S: exit 0
    KVBENCH config mib=64 threads=8 reps=3 block=1048576
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep1.bin codec=memcpy threads=1 raw=259744768 stored=259744768 ratio=1.0000 encode_gbs=24.329 decode_gbs=24.329 keep256M_ms=11.0 restore256M_ms=11.0 keep1024M_ms=44.1 restore1024M_ms=44.1 keep4096M_ms=176.5 restore4096M_ms=176.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep1.bin codec=deflate1 threads=1 raw=259744768 stored=213271825 ratio=0.8211 encode_gbs=0.077 decode_gbs=0.181 keep256M_ms=3484.4 restore256M_ms=1483.1 keep1024M_ms=13937.7 restore1024M_ms=5932.4 keep4096M_ms=55750.9 restore4096M_ms=23729.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep1.bin codec=deflate6 threads=1 raw=259744768 stored=213249906 ratio=0.8210 encode_gbs=0.065 decode_gbs=0.181 keep256M_ms=4110.1 restore256M_ms=1484.9 keep1024M_ms=16440.6 restore1024M_ms=5939.4 keep4096M_ms=65762.2 restore4096M_ms=23757.6
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep1.bin codec=plain threads=1 raw=259744768 stored=187821971 ratio=0.7231 encode_gbs=0.319 decode_gbs=1.257 keep256M_ms=842.1 restore256M_ms=213.5 keep1024M_ms=3368.5 restore1024M_ms=853.9 keep4096M_ms=13473.8 restore4096M_ms=3415.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep1.bin codec=plain engagement raw=0 planes=256 planes_xor=0 planes_delta=0 planes_delta2=0 t64=0 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep1.bin codec=plain threads=8 raw=259744768 stored=187821971 ratio=0.7231 encode_gbs=1.752 decode_gbs=6.552 keep256M_ms=153.2 restore256M_ms=41.0 keep1024M_ms=613.0 restore1024M_ms=163.9 keep4096M_ms=2452.0 restore4096M_ms=655.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep1.bin codec=atlas threads=1 raw=259744768 stored=187608688 ratio=0.7223 encode_gbs=0.098 decode_gbs=1.225 keep256M_ms=2725.7 restore256M_ms=219.2 keep1024M_ms=10903.0 restore1024M_ms=876.6 keep4096M_ms=43611.9 restore4096M_ms=3506.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep1.bin codec=atlas engagement raw=0 planes=233 planes_xor=0 planes_delta=22 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep1.bin codec=atlas threads=8 raw=259744768 stored=187608688 ratio=0.7223 encode_gbs=0.540 decode_gbs=6.504 keep256M_ms=497.0 restore256M_ms=41.3 keep1024M_ms=1988.0 restore1024M_ms=165.1 keep4096M_ms=7951.8 restore4096M_ms=660.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep1.bin codec=atlas-exhaustive threads=1 raw=259744768 stored=187608185 ratio=0.7223 encode_gbs=0.029 decode_gbs=1.219 keep256M_ms=9177.5 restore256M_ms=220.2 keep1024M_ms=36710.0 restore1024M_ms=880.9 keep4096M_ms=146839.9 restore4096M_ms=3523.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep1.bin codec=atlas-exhaustive engagement raw=0 planes=231 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep1.bin codec=atlas-exhaustive threads=8 raw=259744768 stored=187608185 ratio=0.7223 encode_gbs=0.160 decode_gbs=6.450 keep256M_ms=1674.4 restore256M_ms=41.6 keep1024M_ms=6697.5 restore1024M_ms=166.5 keep4096M_ms=26790.0 restore4096M_ms=665.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep1.bin codec=ch-doubledelta threads=1 raw=259744768 stored=242483985 ratio=0.9335 encode_gbs=0.520 decode_gbs=7.501 keep256M_ms=516.1 restore256M_ms=35.8 keep1024M_ms=2064.6 restore1024M_ms=143.1 keep4096M_ms=8258.3 restore4096M_ms=572.6
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep1.bin codec=ch-doubledelta width=2 raw_blocks=232/256
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep1.bin codec=ch-gorilla threads=1 raw=259744768 stored=242896910 ratio=0.9351 encode_gbs=0.864 decode_gbs=14.951 keep256M_ms=310.7 restore256M_ms=18.0 keep1024M_ms=1242.6 restore1024M_ms=71.8 keep4096M_ms=4970.4 restore4096M_ms=287.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep1.bin codec=ch-gorilla width=2 raw_blocks=237/256
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep1.bin codec=ch-fpc12 threads=1 raw=259744768 stored=243896224 ratio=0.9390 encode_gbs=1.006 decode_gbs=8.759 keep256M_ms=266.9 restore256M_ms=30.6 keep1024M_ms=1067.5 restore1024M_ms=122.6 keep4096M_ms=4270.0 restore4096M_ms=490.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep1.bin codec=ch-fpc12 width=2 raw_blocks=234/256
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep1.bin codec=ch-alp-rd threads=1 raw=259744768 stored=218273674 ratio=0.8403 encode_gbs=0.181 decode_gbs=2.260 keep256M_ms=1479.9 restore256M_ms=118.8 keep1024M_ms=5919.6 restore1024M_ms=475.2 keep4096M_ms=23678.2 restore4096M_ms=1900.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep1.bin codec=ch-alp-rd width=2 raw_blocks=137/256
bench A+S: exit 0
    KVBENCH config mib=64 threads=8 reps=3 block=1048576
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep2.bin codec=memcpy threads=1 raw=284255232 stored=284255232 ratio=1.0000 encode_gbs=24.205 decode_gbs=24.205 keep256M_ms=11.1 restore256M_ms=11.1 keep1024M_ms=44.4 restore1024M_ms=44.4 keep4096M_ms=177.4 restore4096M_ms=177.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep2.bin codec=deflate1 threads=1 raw=284255232 stored=232834190 ratio=0.8191 encode_gbs=0.077 decode_gbs=0.167 keep256M_ms=3480.0 restore256M_ms=1609.3 keep1024M_ms=13919.8 restore1024M_ms=6437.1 keep4096M_ms=55679.3 restore4096M_ms=25748.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep2.bin codec=deflate6 threads=1 raw=284255232 stored=232810650 ratio=0.8190 encode_gbs=0.064 decode_gbs=0.166 keep256M_ms=4180.1 restore256M_ms=1612.4 keep1024M_ms=16720.3 restore1024M_ms=6449.7 keep4096M_ms=66881.2 restore4096M_ms=25799.0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep2.bin codec=plain threads=1 raw=284255232 stored=204149174 ratio=0.7182 encode_gbs=0.316 decode_gbs=1.208 keep256M_ms=849.2 restore256M_ms=222.1 keep1024M_ms=3396.9 restore1024M_ms=888.6 keep4096M_ms=13587.7 restore4096M_ms=3554.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep2.bin codec=plain engagement raw=0 planes=278 planes_xor=0 planes_delta=0 planes_delta2=0 t64=0 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep2.bin codec=plain threads=8 raw=284255232 stored=204149174 ratio=0.7182 encode_gbs=1.750 decode_gbs=6.434 keep256M_ms=153.4 restore256M_ms=41.7 keep1024M_ms=613.6 restore1024M_ms=166.9 keep4096M_ms=2454.4 restore4096M_ms=667.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep2.bin codec=atlas threads=1 raw=284255232 stored=203901834 ratio=0.7173 encode_gbs=0.096 decode_gbs=1.178 keep256M_ms=2797.3 restore256M_ms=228.0 keep1024M_ms=11189.0 restore1024M_ms=911.8 keep4096M_ms=44756.2 restore4096M_ms=3647.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep2.bin codec=atlas engagement raw=0 planes=253 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep2.bin codec=atlas threads=8 raw=284255232 stored=203901834 ratio=0.7173 encode_gbs=0.523 decode_gbs=6.261 keep256M_ms=513.5 restore256M_ms=42.9 keep1024M_ms=2054.2 restore1024M_ms=171.5 keep4096M_ms=8216.7 restore4096M_ms=685.9
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep2.bin codec=atlas-exhaustive threads=1 raw=284255232 stored=203901834 ratio=0.7173 encode_gbs=0.028 decode_gbs=1.177 keep256M_ms=9566.6 restore256M_ms=228.0 keep1024M_ms=38266.4 restore1024M_ms=912.2 keep4096M_ms=153065.8 restore4096M_ms=3648.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep2.bin codec=atlas-exhaustive engagement raw=0 planes=253 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep2.bin codec=atlas-exhaustive threads=8 raw=284255232 stored=203901834 ratio=0.7173 encode_gbs=0.152 decode_gbs=6.157 keep256M_ms=1761.8 restore256M_ms=43.6 keep1024M_ms=7047.3 restore1024M_ms=174.4 keep4096M_ms=28189.3 restore4096M_ms=697.6
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep2.bin codec=ch-doubledelta threads=1 raw=284255232 stored=266975052 ratio=0.9392 encode_gbs=0.499 decode_gbs=8.017 keep256M_ms=537.8 restore256M_ms=33.5 keep1024M_ms=2151.1 restore1024M_ms=133.9 keep4096M_ms=8604.3 restore4096M_ms=535.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep2.bin codec=ch-doubledelta width=2 raw_blocks=255/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep2.bin codec=ch-gorilla threads=1 raw=284255232 stored=267415302 ratio=0.9408 encode_gbs=0.854 decode_gbs=15.760 keep256M_ms=314.3 restore256M_ms=17.0 keep1024M_ms=1257.1 restore1024M_ms=68.1 keep4096M_ms=5028.6 restore4096M_ms=272.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep2.bin codec=ch-gorilla width=2 raw_blocks=260/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep2.bin codec=ch-fpc12 threads=1 raw=284255232 stored=268411214 ratio=0.9443 encode_gbs=1.022 decode_gbs=9.303 keep256M_ms=262.8 restore256M_ms=28.9 keep1024M_ms=1051.0 restore1024M_ms=115.4 keep4096M_ms=4204.0 restore4096M_ms=461.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep2.bin codec=ch-fpc12 width=2 raw_blocks=257/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep2.bin codec=ch-alp-rd threads=1 raw=284255232 stored=236649546 ratio=0.8325 encode_gbs=0.185 decode_gbs=2.029 keep256M_ms=1447.7 restore256M_ms=132.3 keep1024M_ms=5790.8 restore1024M_ms=529.2 keep4096M_ms=23163.2 restore4096M_ms=2117.0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep2.bin codec=ch-alp-rd width=2 raw_blocks=137/278
bench B+S: exit 0
    KVBENCH config mib=64 threads=8 reps=3 block=1048576
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep3.bin codec=memcpy threads=1 raw=284255232 stored=284255232 ratio=1.0000 encode_gbs=24.205 decode_gbs=24.205 keep256M_ms=11.1 restore256M_ms=11.1 keep1024M_ms=44.4 restore1024M_ms=44.4 keep4096M_ms=177.4 restore4096M_ms=177.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep3.bin codec=deflate1 threads=1 raw=284255232 stored=232834280 ratio=0.8191 encode_gbs=0.077 decode_gbs=0.178 keep256M_ms=3472.2 restore256M_ms=1508.4 keep1024M_ms=13889.0 restore1024M_ms=6033.5 keep4096M_ms=55555.9 restore4096M_ms=24133.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep3.bin codec=deflate6 threads=1 raw=284255232 stored=232810941 ratio=0.8190 encode_gbs=0.065 decode_gbs=0.178 keep256M_ms=4157.8 restore256M_ms=1511.6 keep1024M_ms=16631.2 restore1024M_ms=6046.3 keep4096M_ms=66524.7 restore4096M_ms=24185.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep3.bin codec=plain threads=1 raw=284255232 stored=204144597 ratio=0.7182 encode_gbs=0.317 decode_gbs=1.208 keep256M_ms=847.8 restore256M_ms=222.2 keep1024M_ms=3391.2 restore1024M_ms=888.7 keep4096M_ms=13564.8 restore4096M_ms=3554.9
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep3.bin codec=plain engagement raw=0 planes=278 planes_xor=0 planes_delta=0 planes_delta2=0 t64=0 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep3.bin codec=plain threads=8 raw=284255232 stored=204144597 ratio=0.7182 encode_gbs=1.749 decode_gbs=6.347 keep256M_ms=153.5 restore256M_ms=42.3 keep1024M_ms=613.8 restore1024M_ms=169.2 keep4096M_ms=2455.4 restore4096M_ms=676.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep3.bin codec=atlas threads=1 raw=284255232 stored=203900502 ratio=0.7173 encode_gbs=0.096 decode_gbs=1.179 keep256M_ms=2810.6 restore256M_ms=227.7 keep1024M_ms=11242.2 restore1024M_ms=910.7 keep4096M_ms=44968.8 restore4096M_ms=3642.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep3.bin codec=atlas engagement raw=0 planes=253 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep3.bin codec=atlas threads=8 raw=284255232 stored=203900502 ratio=0.7173 encode_gbs=0.519 decode_gbs=5.963 keep256M_ms=517.0 restore256M_ms=45.0 keep1024M_ms=2068.0 restore1024M_ms=180.1 keep4096M_ms=8272.2 restore4096M_ms=720.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep3.bin codec=atlas-exhaustive threads=1 raw=284255232 stored=203900502 ratio=0.7173 encode_gbs=0.028 decode_gbs=1.177 keep256M_ms=9572.3 restore256M_ms=228.1 keep1024M_ms=38289.0 restore1024M_ms=912.3 keep4096M_ms=153156.1 restore4096M_ms=3649.0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep3.bin codec=atlas-exhaustive engagement raw=0 planes=253 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep3.bin codec=atlas-exhaustive threads=8 raw=284255232 stored=203900502 ratio=0.7173 encode_gbs=0.154 decode_gbs=6.311 keep256M_ms=1741.4 restore256M_ms=42.5 keep1024M_ms=6965.5 restore1024M_ms=170.2 keep4096M_ms=27862.0 restore4096M_ms=680.6
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep3.bin codec=ch-doubledelta threads=1 raw=284255232 stored=266975942 ratio=0.9392 encode_gbs=0.499 decode_gbs=7.981 keep256M_ms=537.7 restore256M_ms=33.6 keep1024M_ms=2150.7 restore1024M_ms=134.5 keep4096M_ms=8602.9 restore4096M_ms=538.2
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep3.bin codec=ch-doubledelta width=2 raw_blocks=255/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep3.bin codec=ch-gorilla threads=1 raw=284255232 stored=267415300 ratio=0.9408 encode_gbs=0.859 decode_gbs=17.300 keep256M_ms=312.3 restore256M_ms=15.5 keep1024M_ms=1249.3 restore1024M_ms=62.1 keep4096M_ms=4997.2 restore4096M_ms=248.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep3.bin codec=ch-gorilla width=2 raw_blocks=260/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep3.bin codec=ch-fpc12 threads=1 raw=284255232 stored=268408299 ratio=0.9443 encode_gbs=1.028 decode_gbs=9.220 keep256M_ms=261.0 restore256M_ms=29.1 keep1024M_ms=1044.1 restore1024M_ms=116.5 keep4096M_ms=4176.4 restore4096M_ms=465.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep3.bin codec=ch-fpc12 width=2 raw_blocks=257/278
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep3.bin codec=ch-alp-rd threads=1 raw=284255232 stored=236467976 ratio=0.8319 encode_gbs=0.185 decode_gbs=2.025 keep256M_ms=1454.9 restore256M_ms=132.5 keep1024M_ms=5819.5 restore1024M_ms=530.1 keep4096M_ms=23277.9 restore4096M_ms=2120.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep3.bin codec=ch-alp-rd width=2 raw_blocks=137/278
bench L+S: exit 0
    KVBENCH config mib=64 threads=8 reps=3 block=1048576
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep4.bin codec=memcpy threads=1 raw=357786624 stored=357786624 ratio=1.0000 encode_gbs=23.985 decode_gbs=23.985 keep256M_ms=11.2 restore256M_ms=11.2 keep1024M_ms=44.8 restore1024M_ms=44.8 keep4096M_ms=179.1 restore4096M_ms=179.1
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep4.bin codec=deflate1 threads=1 raw=357786624 stored=291502947 ratio=0.8147 encode_gbs=0.078 decode_gbs=0.161 keep256M_ms=3450.0 restore256M_ms=1665.0 keep1024M_ms=13799.9 restore1024M_ms=6660.1 keep4096M_ms=55199.6 restore4096M_ms=26640.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep4.bin codec=deflate6 threads=1 raw=357786624 stored=291475651 ratio=0.8147 encode_gbs=0.062 decode_gbs=0.158 keep256M_ms=4306.4 restore256M_ms=1701.8 keep1024M_ms=17225.5 restore1024M_ms=6807.3 keep4096M_ms=68902.2 restore4096M_ms=27229.1
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep4.bin codec=plain threads=1 raw=357786624 stored=253121695 ratio=0.7075 encode_gbs=0.274 decode_gbs=0.932 keep256M_ms=981.1 restore256M_ms=288.1 keep1024M_ms=3924.4 restore1024M_ms=1152.4 keep4096M_ms=15697.5 restore4096M_ms=4609.6
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep4.bin codec=plain engagement raw=0 planes=344 planes_xor=0 planes_delta=0 planes_delta2=0 t64=0 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep4.bin codec=plain threads=8 raw=357786624 stored=253121695 ratio=0.7075 encode_gbs=1.421 decode_gbs=5.052 keep256M_ms=188.8 restore256M_ms=53.1 keep1024M_ms=755.4 restore1024M_ms=212.5 keep4096M_ms=3021.5 restore4096M_ms=850.1
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep4.bin codec=atlas threads=1 raw=357786624 stored=252775453 ratio=0.7065 encode_gbs=0.088 decode_gbs=1.097 keep256M_ms=3035.1 restore256M_ms=244.8 keep1024M_ms=12140.2 restore1024M_ms=979.1 keep4096M_ms=48560.9 restore4096M_ms=3916.5
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep4.bin codec=atlas engagement raw=0 planes=319 planes_xor=0 planes_delta=24 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep4.bin codec=atlas threads=8 raw=357786624 stored=252775453 ratio=0.7065 encode_gbs=0.432 decode_gbs=5.898 keep256M_ms=621.4 restore256M_ms=45.5 keep1024M_ms=2485.8 restore1024M_ms=182.1 keep4096M_ms=9943.2 restore4096M_ms=728.3
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep4.bin codec=atlas-exhaustive threads=1 raw=357786624 stored=252775114 ratio=0.7065 encode_gbs=0.026 decode_gbs=1.057 keep256M_ms=10426.0 restore256M_ms=254.1 keep1024M_ms=41703.8 restore1024M_ms=1016.2 keep4096M_ms=166815.4 restore4096M_ms=4064.9
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep4.bin codec=atlas-exhaustive engagement raw=0 planes=318 planes_xor=0 planes_delta=25 planes_delta2=0 t64=1 t64_xor=0 t64_delta=0 t64_delta2=0
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep4.bin codec=atlas-exhaustive threads=8 raw=357786624 stored=252775114 ratio=0.7065 encode_gbs=0.141 decode_gbs=5.865 keep256M_ms=1902.5 restore256M_ms=45.8 keep1024M_ms=7610.0 restore1024M_ms=183.1 keep4096M_ms=30439.8 restore4096M_ms=732.4
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep4.bin codec=ch-doubledelta threads=1 raw=357786624 stored=340477354 ratio=0.9516 encode_gbs=0.460 decode_gbs=9.374 keep256M_ms=584.2 restore256M_ms=28.6 keep1024M_ms=2336.6 restore1024M_ms=114.5 keep4096M_ms=9346.6 restore4096M_ms=458.2
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep4.bin codec=ch-doubledelta width=2 raw_blocks=321/344
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep4.bin codec=ch-gorilla threads=1 raw=357786624 stored=340946688 ratio=0.9529 encode_gbs=0.847 decode_gbs=18.695 keep256M_ms=317.0 restore256M_ms=14.4 keep1024M_ms=1268.1 restore1024M_ms=57.4 keep4096M_ms=5072.5 restore4096M_ms=229.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep4.bin codec=ch-gorilla width=2 raw_blocks=326/344
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep4.bin codec=ch-fpc12 threads=1 raw=357786624 stored=341935128 ratio=0.9557 encode_gbs=1.001 decode_gbs=10.640 keep256M_ms=268.3 restore256M_ms=25.2 keep1024M_ms=1073.1 restore1024M_ms=100.9 keep4096M_ms=4292.4 restore4096M_ms=403.7
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep4.bin codec=ch-fpc12 width=2 raw_blocks=323/344
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep4.bin codec=ch-alp-rd threads=1 raw=357786624 stored=291372966 ratio=0.8144 encode_gbs=0.194 decode_gbs=1.664 keep256M_ms=1384.1 restore256M_ms=161.4 keep1024M_ms=5536.4 restore1024M_ms=645.4 keep4096M_ms=22145.6 restore4096M_ms=2581.8
    KVBENCH stream=/home/colum/kvprobe/native/dump/glm-rank0-keep4.bin codec=ch-alp-rd width=2 raw_blocks=137/344

