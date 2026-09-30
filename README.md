# ChorusTIC

This repository contains the code for ChorusTIC. It includes the model, data loading, and inference modules required to run UCR/UEA evaluation.

Pretrained checkpoints are available on Hugging Face: [JTF2000/ChorusTIC](https://huggingface.co/JTF2000/ChorusTIC).

## Module Naming

The code is organized according to the terminology used in the paper:

- `chorustic.model.signal_level_chorus`: Signal-level Chorus, including Random Subchannel Slot Concatenation (RSSC).
- `chorustic.model.chorustic`: The top-level `ChorusTIC` model, connecting RSSC representations with the task-level in-context classifier.
- `chorustic.model.loading`: Checkpoint and hparams loading logic, with compatibility for legacy `state_dict` keys from training artifacts.
- `chorustic.evaluation.datasets`: UCR/UEA data preparation, including crop/pad, optional channel selection, and label remapping.
- `chorustic.evaluation.inference`: `classifier_v2` inference, cyclic label permutation, RSSC ensembling, and OOM retry.
- `chorustic.model.task_level_chorus`: Task-level Chorus, decomposed into Column Distribution Modeling (CDM), Row-wise Feature Interaction, and In-Context Learning (ICL).
- `chorustic.model.task_level_chorus.column_distribution_modeling`: CDM, the distribution-aware feature/column representation module.
- `chorustic.model.task_level_chorus.row_wise_feature_interaction`: Row-wise Feature Interaction, the row representation module.
- `chorustic.model.task_level_chorus.in_context_learning`: ICL, the task-level prediction module.
- `chorustic.model.signal_encoder.TSEncoder.architecture`: TSEncoder, the shared dual-axis time-series encoder implementation.

## Dependencies

Use `ticfs_env.yml` to create the recommended environment. The inference path requires at least:

- Python 3.10+
- PyTorch
- NumPy
- scikit-learn
- scipy
- pandas
- einops
- huggingface_hub
- tqdm

UCR/UEA archives are read through `chorustic.evaluation.data_reader.DataReader`. Set `--ucr_path` and `--uea_path` to the corresponding dataset root directories.

## Model Checkpoints

Download the released checkpoint and configuration from Hugging Face:

```bash
pip install -U huggingface_hub
hf download JTF2000/ChorusTIC --local-dir Checkpoints_ChorusTIC
```

The expected local layout is:

```text
Checkpoints_ChorusTIC/
├── ChorusTIC.ckpt
└── model_hparams_latest.json
```

## Inference Command

```bash
cd <chorustic_repo_root>

python -u scripts/evaluate_chorustic_ucr_uea.py \
  --ucr_path <ucr_data_root> \
  --uea_path <uea_data_root> \
  --rssc_ckpt Checkpoints_ChorusTIC/ChorusTIC.ckpt \
  --rssc_hparams_json Checkpoints_ChorusTIC/model_hparams_latest.json \
  --device cuda:0 \
  --suite both \
  --mode classifier_v2 \
  --n_estimators 8 \
  --v2_n_augmentations 1 \
  --v2_crop_rate_lo 0.0 \
  --v2_crop_rate_hi 0.0 \
  --v2_batch_size 32 \
  --rssc_eval_ensembles 4 \
  --task_level_chorus_ensemble_batch_size 512 \
  --output_csv <output_dir>/chorustic_results.csv \
  --output_json <output_dir>/chorustic_results.json \
  --signal_encoder_batch_size 1024 \
  --oom_min_v2_batch_size 1 \
  --oom_min_signal_encoder_batch_size 64 \
  --oom_min_task_level_chorus_ensemble_batch_size 1 \
  --rssc_encoder_ensemble_batch_size 4 \
  --oom_retry
```

## Outputs

- `--output_csv`: Writes one row per dataset, including accuracy, sample counts, class count, and the effective batch sizes used after any OOM retry.
- `--output_json`: Writes the summary and full per-dataset results.
- `<output_csv>.cmd.txt`: Records the exact command line next to the CSV for reproducibility.

## Benchmark Results

ChorusTIC is evaluated in the [TSC-FM time series classification benchmark](https://tsc-fm.dmirlab.com/). See its [model configurations and benchmark results](https://tsc-fm.dmirlab.com/methods/chorustic), compare it on the [time series classification leaderboard](https://tsc-fm.dmirlab.com/leaderboard), and consult the [Standard and few-shot evaluation protocol](https://tsc-fm.dmirlab.com/evaluation).
