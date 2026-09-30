# ChorusTIC

<p align="center">
  <a href="https://arxiv.org/abs/2608.24033">
    <img src="https://img.shields.io/badge/arXiv-2608.24033-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://huggingface.co/DMIRLAB/ChorusTIC">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DMIRLAB%2FChorusTIC-FFD21E" alt="Hugging Face">
  </a>
  <a href="https://github.com/DMIRLAB-Group/ChorusTIC">
    <img src="https://img.shields.io/badge/GitHub-DMIRLAB--Group%2FChorusTIC-181717?logo=github" alt="GitHub">
  </a>
  <a href="https://tsc-fm.dmirlab.com/leaderboard">
    <img src="https://img.shields.io/badge/TSC--FM-%231%20Overall-gold" alt="TSC-FM #1 Overall">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
</p>

<p align="center">
  <b>🏆 Ranked #1 on the TSC-FM Standard Overall Leaderboard</b>
</p>

<p align="center">
  <b>78.56 Overall Average Accuracy across all 198 time series classification datasets</b>
</p>

**ChorusTIC** is a classification-native foundation model for **univariate and multivariate time series classification**. It performs prediction through in-context learning using labeled context examples, without fitting a target-specific classifier or updating model parameters.

📄 **Paper:** [ChorusTIC: Training-Free Multivariate Time Series Classification via Chorus In-Context Learning](https://arxiv.org/abs/2608.24033)  
🤗 **Pretrained Checkpoint:** [DMIRLAB/ChorusTIC](https://huggingface.co/DMIRLAB/ChorusTIC)  
🏆 **Benchmark:** [TSC-FM Leaderboard](https://tsc-fm.dmirlab.com/leaderboard)

## Highlights

- 🏆 **#1 Standard Overall** on the [TSC-FM Time Series Classification Benchmark](https://tsc-fm.dmirlab.com/leaderboard), with an **Overall Average Accuracy of 78.56 across all 198 datasets**.
- 🚀 **Training-free classification:** no target-specific classifier fitting or parameter updates are required at inference time.
- 🔀 **Unified univariate and multivariate classification:** a single pretrained model supports heterogeneous channel configurations.
- 🧩 **In-context learning:** predictions are conditioned directly on labeled context examples.
- 🎼 **Chorus architecture:** combines signal-level modeling with task-level in-context classification.

---

## Model Checkpoints

The pretrained ChorusTIC checkpoint and configuration are available on Hugging Face:

**[DMIRLAB/ChorusTIC](https://huggingface.co/DMIRLAB/ChorusTIC)**

Download them with:

```bash
pip install -U huggingface_hub

hf download DMIRLAB/ChorusTIC \
  --local-dir Checkpoints_ChorusTIC
```

The expected local layout is:

```text
Checkpoints_ChorusTIC/
├── ChorusTIC.ckpt
└── model_hparams_latest.json
```

---

## Module Naming

The code is organized according to the terminology used in the paper:

- `chorustic.model.signal_level_chorus`  
  Signal-level Chorus, including **Random Subchannel Slot Concatenation (RSSC)**.

- `chorustic.model.chorustic`  
  The top-level `ChorusTIC` model, connecting RSSC representations with the task-level in-context classifier.

- `chorustic.model.loading`  
  Checkpoint and hyperparameter loading logic, with compatibility for legacy `state_dict` keys from training artifacts.

- `chorustic.evaluation.datasets`  
  UCR/UEA data preparation, including crop/pad, optional channel selection, and label remapping.

- `chorustic.evaluation.inference`  
  `classifier_v2` inference, cyclic label permutation, RSSC ensembling, and OOM retry.

- `chorustic.model.task_level_chorus`  
  Task-level Chorus, decomposed into **Column Distribution Modeling (CDM)**, **Row-wise Feature Interaction**, and **In-Context Learning (ICL)**.

- `chorustic.model.task_level_chorus.column_distribution_modeling`  
  CDM, the distribution-aware feature/column representation module.

- `chorustic.model.task_level_chorus.row_wise_feature_interaction`  
  Row-wise Feature Interaction, the row representation module.

- `chorustic.model.task_level_chorus.in_context_learning`  
  ICL, the task-level prediction module.

- `chorustic.model.signal_encoder.TSEncoder.architecture`  
  TSEncoder, the shared dual-axis time-series encoder implementation.

---

## Dependencies

Use `ticfs_env.yml` to create the recommended environment.

The inference pipeline requires at least:

- Python 3.10+
- PyTorch
- NumPy
- scikit-learn
- SciPy
- pandas
- einops
- huggingface_hub
- tqdm

UCR/UEA archives are read through:

```text
chorustic.evaluation.data_reader.DataReader
```

Set `--ucr_path` and `--uea_path` to the corresponding dataset root directories.

---

## Inference

From the ChorusTIC repository root, run:

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

---

## Outputs

The evaluation script produces:

- `--output_csv`  
  One row per dataset, including accuracy, sample counts, class count, and the effective batch sizes used after any OOM retry.

- `--output_json`  
  Summary statistics and full per-dataset evaluation results.

- `<output_csv>.cmd.txt`  
  The exact command line used for evaluation, saved next to the CSV for reproducibility.

---

## Benchmark Results

🏆 **ChorusTIC ranks #1 on the TSC-FM Standard Overall Leaderboard**, achieving an **Overall Average Accuracy of 78.56 across all 198 time series classification datasets**.

TSC-FM is a unified time series classification benchmark covering **198 unique datasets**, with equal dataset weighting and evaluations under both Standard and Low-shot settings.

For detailed results:

- 🥇 [TSC-FM Leaderboard](https://tsc-fm.dmirlab.com/leaderboard)
- 📊 [ChorusTIC Model Configuration and Benchmark Results](https://tsc-fm.dmirlab.com/methods/chorustic)
- 📖 [Standard and Low-shot Evaluation Protocol](https://tsc-fm.dmirlab.com/evaluation)
- 🌐 [TSC-FM Benchmark Homepage](https://tsc-fm.dmirlab.com/)

---

## Citation

If you find ChorusTIC useful in your research, please cite:

```bibtex
@article{fang2026chorustic,
  title   = {ChorusTIC: Training-Free Multivariate Time Series Classification via Chorus In-Context Learning},
  author  = {Fang, Juntao and Xie, Shifeng and Cai, Ruichu and Zheng, Shengji and Li, Zijian and Zhang, Keli and Pan, Lujia and Palpanas, Themis and Hao, Zhifeng},
  journal = {arXiv preprint arXiv:2608.24033},
  year    = {2026}
}
```

---

## License

This project is released under the [Apache License 2.0](LICENSE).
