# Frozen LVLMs for Micro-Video Recommendation

Official implementation of **Frozen LVLMs for Micro-Video Recommendation: A Systematic Study of Feature Extraction and Fusion**.

**Accepted at ICMR 2026.**

## Model

- ID embedding + multi-layer video features (weighted sum + projection)
- Gating: `score = gate * id + (1 - gate) * video`
- Training: SASRec next-item loss on fused item embeddings

## Layout

```
DFF/
├── main.py
├── model/
│   ├── model_dff.py      # Model_dff_id_v
│   ├── user_encoders.py  # SASRec transformer
│   └── modules.py
├── utils/
├── run_sh/run_dff_id_v.sh
└── data/                 # see data/README.md
```

## Run

```bash
cd /opt/data/private/work/DFF
pip install -r requirements.txt
bash run_sh/run_dff_id_v.sh
```

## Data

See `data/README.md`. Default video features: `data/MicroLens-100k/features/all_layers_tensor.pt`
