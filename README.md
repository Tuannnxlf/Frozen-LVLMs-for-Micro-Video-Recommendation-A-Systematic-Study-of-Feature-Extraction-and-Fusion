# DFF

**D**ual-path **F**eature **F**usion for sequential recommendation: **ID + video (VLLM)** with learnable gating, SASRec user encoder.

This repo contains only `dff_id_v` (no audio, no diffusion, no PMRL).

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
