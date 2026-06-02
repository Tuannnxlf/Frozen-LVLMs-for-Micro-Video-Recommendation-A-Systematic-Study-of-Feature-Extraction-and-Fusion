# DFF

Conference follow-up codebase for **D**iffusion / multimodal **F**usion recommendation on MicroLens-100k.

Migrated from `vllmembs4rec/sasrec` (essential training code only; no checkpoints, logs, or raw data).

## Methods

| `--method` | Description |
|------------|-------------|
| `only_id` | ID-only SASRec baseline |
| `dff_id_v` | DFF: ID + video fusion |
| `dff_id_a` | DFF: ID + audio |
| `dff_diffusion_id_v_condition_a` | Diffusion, video denoise conditioned on audio |
| `dff_diffusion_id_a_condition_v` | Diffusion, audio denoise conditioned on video |

## Layout

```
DFF/
├── main.py           # train / eval entry
├── model/            # SASRec + DFF + diffusion models
├── utils/            # data, metrics, logging
├── run_sh/           # example launch scripts
├── data/             # put MicroLens-100k here (see data/README.md)
├── requirements.txt
└── requirements-lock.txt   # full pip freeze from original env (reference)
```

## Quick start

```bash
cd /opt/data/private/work/DFF
pip install -r requirements.txt
# prepare data/MicroLens-100k (see data/README.md)
bash run_sh/run_dff_id_v.sh
```

## Provenance

- Source: `/opt/data/private/work/vllmembs4rec/sasrec`
- Excluded: `data/`, `checkpoint/`, `logs/`, `tblogs/`, `result/`, `sasrec copy/`, `MicroLens-master.zip`
