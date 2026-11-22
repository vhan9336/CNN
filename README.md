# Pancreatic mass CT research baseline

This project trains a 3-D PyTorch convolutional U-Net to segment pancreatic mass/tumor voxels in contrast-enhanced CT. A predicted mask is a research signal, **not** a diagnosis or an early-stage pancreatic-cancer detector. Do not use it for patient care, triage, or clinical decisions.

## Public training data

The pipeline targets [Medical Segmentation Decathlon (MSD), Task07 Pancreas](https://medicaldecathlon.com/), an open **CC BY-SA 4.0** dataset. It has 281 labeled portal-venous phase CT training scans; masks distinguish pancreas (`1`) and mass/cyst/tumor (`2`). The labels do not encode cancer stage, histology, screening population, or prospective outcomes. Accordingly, this repository deliberately makes no early-stage, sensitivity, specificity, or clinical-performance claim.

Please cite the MSD dataset/paper and preserve the CC BY-SA 4.0 attribution/share-alike requirements for data or derivatives. The [AWS Open Data registry entry](https://registry.opendata.aws/msd/) documents the source and licence.

## Run

Use Python 3.10+ and install a PyTorch build appropriate for your CUDA version, then:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py --download                 # downloads ~12 GB of public data
python train.py --epochs 100 --batch-size 1 --workers 4
```

The best checkpoint is saved to `runs/pancreas_mass/best.pt`; `split.json` records the patient-level train/validation split. Validation is on randomly sampled tumor-centred patches, so its metrics are only a training sanity check—not whole-volume, external, or clinical validation.

## Performance defaults

The default model now uses a 64³ patch and base width 16 (instead of the original 96³/24 setting), which makes each training step roughly 7× cheaper. The first epoch creates uncompressed, memory-mapped arrays and cached tumor-coordinate lists in `data/Task07_Pancreas/.cache`, avoiding repeated `.nii.gz` decompression and whole-label scans in later epochs. CUDA training automatically uses mixed precision, pinned transfers, optimized convolution selection, and parallel data loading (up to four workers). The initial cached epoch can therefore be slower; subsequent epochs should be materially faster. Use `--workers 0` if Windows worker processes exhaust RAM. `--compile` may speed up later epochs, but has a substantial one-time startup cost. Add `--deterministic` only when exact reproducibility matters, because it disables faster CUDA kernel selection.

For a quick smoke test, use `python train.py --fast --epochs 2`. This uses 48³ patches and base width 12; it is not the recommended configuration for research results. If the startup log prints `device=cpu`, the practical solution is to run on a CUDA-capable GPU with a CUDA-enabled PyTorch build—3-D convolutional training on CPU will remain slow even with these optimizations.

For a CPU-oriented iteration mode, use `python train.py --cpu-fast --epochs 10`. It uses 32³ patches, base width 8, no competing data-loader processes, 64 randomly sampled patches per epoch, and a small validation split. It is dramatically faster but is a development/smoke-test preset, not a configuration for reporting research performance. Use `--cpu-threads N` to match available physical CPU cores if the default oversubscribes the machine.

`python train.py --ultra-fast --epochs 1000` is available solely to check that training, checkpointing, and the viewer work end-to-end. Here an epoch is one 16³ patch and validation occurs every 100 epochs, so the epoch count has no research meaning and the resulting model must not be evaluated or used as a cancer detector.

## Local scan viewer

After training, launch a local pop-up viewer:

```powershell
python viewer.py
```

Choose a NIfTI CT volume (`.nii` or `.nii.gz`), select `runs/pancreas_mass/best.pt` if it is not already found, and click **Run research preview**. Scroll through axial slices to inspect the red/yellow probability overlay. The viewer uses a downsampled whole-volume preview so it stays responsive on CPU; it does not perform validated lesion localization. It accepts medical CT volumes, not photographs, PNGs, or JPEGs. Its output is research-only and must never be used for diagnosis, treatment, triage, or exclusion of cancer.

## Before any research claim

Use an independent, multi-site, stage-annotated cohort; lock preprocessing and thresholds before testing; report confidence intervals, calibration, lesion-level and patient-level false positives/negatives, and subgroup results. Obtain IRB/privacy review and clinician/radiologist oversight for any patient data. A tumor segmentation dataset of patients undergoing mass resection cannot validate population screening or early detection.
