# Data layout

Place MicroLens-100k under this directory (not tracked by git):

```
data/MicroLens-100k/
├── MicroLens-100k_pairs.tsv
└── features/
    ├── all_layers_tensor.pt          # video (VLLM) features
    └── laionclap_fusion_audio_feature.npy
```

You can symlink from another project, e.g.:

```bash
ln -s /opt/data/private/work/KDD26-SHT/data/MicroLens-100k data/MicroLens-100k
```

Note: KDD26-SHT may use `.npy` video features; this codebase defaults to `all_layers_tensor.pt` — align paths via CLI flags if needed.
