# Data

```
data/MicroLens-100k/
├── MicroLens-100k_pairs.tsv
└── features/
    └── all_layers_tensor.pt    # [num_items, num_layers, dim]
```

Example symlink:

```bash
mkdir -p data
ln -s /path/to/MicroLens-100k data/MicroLens-100k
```
