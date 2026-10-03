# Benchmarks

Appended by `scripts/benchmark.py`. Empty until Phase 2.

| device | model | batch | amp | peak_mem_MB | s_per_step | notes |
|---|---|---|---|---|---|---|
| local: NVIDIA GeForce RTX 3050 6GB Laptop GPU | t1 | 32 | off | 468 | 0.0324 | base_ch=32 depth=4 z=256 drop=0.1; 9,785,987 params; 20 timed steps after warmup; loader 2330 img/s (workers=4); val pass 7.9 s (2944 rows); train step incl. SSIM loss |
| local: NVIDIA GeForce RTX 3050 6GB Laptop GPU | t1 | 32 | on | 384 | 0.0272 | base_ch=32 depth=4 z=256 drop=0.1; 9,785,987 params; 20 timed steps after warmup; loader 2315 img/s (workers=4); val pass 3.6 s (2944 rows); train step incl. SSIM loss |
| local: NVIDIA GeForce RTX 3050 6GB Laptop GPU | t1 | 64 | off | 897 | 0.0572 | base_ch=32 depth=4 z=256 drop=0.1; 9,785,987 params; 20 timed steps after warmup; loader 2583 img/s (workers=4); val pass 3.5 s (2944 rows); train step incl. SSIM loss |
| local: NVIDIA GeForce RTX 3050 6GB Laptop GPU | t1 | 64 | on | 582 | 0.0458 | base_ch=32 depth=4 z=256 drop=0.1; 9,785,987 params; 20 timed steps after warmup; loader 2596 img/s (workers=4); val pass 3.7 s (2944 rows); train step incl. SSIM loss |
| local: NVIDIA GeForce RTX 3050 6GB Laptop GPU | t1 | 128 | off | 1175 | 0.1069 | base_ch=32 depth=4 z=256 drop=0.1; 9,785,987 params; 20 timed steps after warmup; loader 2684 img/s (workers=4); val pass 3.6 s (2944 rows); train step incl. SSIM loss |
| local: NVIDIA GeForce RTX 3050 6GB Laptop GPU | t1 | 128 | on | 974 | 0.0825 | base_ch=32 depth=4 z=256 drop=0.1; 9,785,987 params; 20 timed steps after warmup; loader 2428 img/s (workers=4); val pass 3.7 s (2944 rows); train step incl. SSIM loss |
| local: NVIDIA GeForce RTX 3050 6GB Laptop GPU | t1 | 256 | off | 3415 | 0.2078 | base_ch=32 depth=4 z=256 drop=0.1; 9,785,987 params; 20 timed steps after warmup; loader 3015 img/s (workers=4); val pass 7.9 s (2944 rows); train step incl. SSIM loss |
| local: NVIDIA GeForce RTX 3050 6GB Laptop GPU | t1 | 256 | on | 1762 | 0.1569 | base_ch=32 depth=4 z=256 drop=0.1; 9,785,987 params; 20 timed steps after warmup; loader 2627 img/s (workers=4); val pass 3.6 s (2944 rows); train step incl. SSIM loss |
