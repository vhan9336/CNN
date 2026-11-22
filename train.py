"""Research-only 3-D CT pancreatic-mass segmentation baseline.

This is not a clinical device and must not be used to diagnose, triage, or rule out
pancreatic cancer.  It has not been validated for early-stage disease.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm

MSD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar"


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Benchmarking picks fast convolution kernels for the fixed patch shape. It
    # is intentionally off only when exact run-to-run reproducibility is needed.
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def download_msd(destination: Path) -> None:
    """Download and extract the public MSD Task07 archive (about 12 GB)."""
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "Task07_Pancreas.tar"
    if not archive.exists():
        print(f"Downloading {MSD_URL} to {archive} ...")
        urllib.request.urlretrieve(MSD_URL, archive)
    print("Extracting archive ...")
    with tarfile.open(archive) as bundle:
        # MSD's archive contains only Task07_Pancreas; reject unexpected paths.
        members = bundle.getmembers()
        if any(Path(m.name).is_absolute() or ".." in Path(m.name).parts for m in members):
            raise RuntimeError("Refusing archive with unsafe paths")
        bundle.extractall(destination, members=members)


def pad_crop(arr: np.ndarray, center: np.ndarray, size: tuple[int, int, int]) -> np.ndarray:
    """Return fixed-size D,H,W crop, padding outside the image with zeros."""
    out = np.zeros(size, dtype=arr.dtype)
    start = center - np.asarray(size) // 2
    end = start + np.asarray(size)
    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, np.asarray(arr.shape))
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)
    out[tuple(slice(a, b) for a, b in zip(dst_start, dst_end))] = arr[
        tuple(slice(a, b) for a, b in zip(src_start, src_end))
    ]
    return out


class PancreasMassDataset(Dataset):
    def __init__(self, image_paths: list[Path], label_dir: Path, patch_size: int, augment: bool, cache_dir: Path | None):
        self.image_paths = image_paths
        self.label_dir = label_dir
        self.patch_size = (patch_size,) * 3
        self.augment = augment
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.image_paths)

    @staticmethod
    def _write_cache(path: Path, array: np.ndarray) -> None:
        """Atomically create a reusable, uncompressed array cache."""
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, array)
        os.replace(temporary, path)

    def _load_case(self, image_path: Path) -> tuple[np.ndarray, np.ndarray]:
        label_path = self.label_dir / image_path.name.replace("_0000", "")
        if self.cache_dir is None:
            image = nib.load(str(image_path)).get_fdata(dtype=np.float32)
            label = nib.load(str(label_path)).get_fdata(dtype=np.float32).astype(np.int64)
            return np.clip(image, -125, 225) / 350.0, label == 2
        stem = image_path.name.removesuffix(".nii.gz")
        image_cache = self.cache_dir / f"{stem}_image.npy"
        target_cache = self.cache_dir / f"{stem}_tumor.npy"
        if not image_cache.exists() or not target_cache.exists():
            image = nib.load(str(image_path)).get_fdata(dtype=np.float32)
            label = nib.load(str(label_path)).get_fdata(dtype=np.float32).astype(np.int64)
            self._write_cache(image_cache, np.clip(image, -125, 225) / 350.0)
            self._write_cache(target_cache, label == 2)
        return np.load(image_cache, mmap_mode="r"), np.load(target_cache, mmap_mode="r")

    def _tumor_coordinates(self, image_path: Path, target: np.ndarray) -> np.ndarray:
        """Cache coordinates so each crop does not scan an entire 3-D label map."""
        if self.cache_dir is None:
            return np.argwhere(target)
        stem = image_path.name.removesuffix(".nii.gz")
        coords_cache = self.cache_dir / f"{stem}_tumor_coords.npy"
        if not coords_cache.exists():
            self._write_cache(coords_cache, np.argwhere(target).astype(np.int32))
        return np.load(coords_cache, mmap_mode="r")

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path = self.image_paths[index]
        image, target = self._load_case(image_path)
        foreground = self._tumor_coordinates(image_path, target)
        center = foreground[np.random.randint(len(foreground))] if len(foreground) else np.asarray(image.shape) // 2
        image = pad_crop(image, center, self.patch_size)
        target = pad_crop(target.astype(np.float32), center, self.patch_size)
        if self.augment:
            for axis in range(3):
                if random.random() < 0.5:
                    image = np.flip(image, axis).copy()
                    target = np.flip(target, axis).copy()
        return torch.from_numpy(image[None]), torch.from_numpy(target[None])


class ConvBlock(nn.Module):
    def __init__(self, a: int, b: int):
        super().__init__()
        groups = min(8, b)
        self.net = nn.Sequential(
            nn.Conv3d(a, b, 3, padding=1), nn.GroupNorm(groups, b), nn.GELU(),
            nn.Conv3d(b, b, 3, padding=1), nn.GroupNorm(groups, b), nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class UNet3D(nn.Module):
    """Compact 3-D convolutional U-Net for a single tumor-mask channel."""
    def __init__(self, width: int = 24):
        super().__init__()
        self.e1, self.e2, self.e3 = ConvBlock(1, width), ConvBlock(width, width * 2), ConvBlock(width * 2, width * 4)
        self.pool = nn.MaxPool3d(2)
        self.mid = ConvBlock(width * 4, width * 8)
        self.u3 = nn.ConvTranspose3d(width * 8, width * 4, 2, 2)
        self.d3 = ConvBlock(width * 8, width * 4)
        self.u2 = nn.ConvTranspose3d(width * 4, width * 2, 2, 2)
        self.d2 = ConvBlock(width * 4, width * 2)
        self.u1 = nn.ConvTranspose3d(width * 2, width, 2, 2)
        self.d1 = ConvBlock(width * 2, width)
        self.out = nn.Conv3d(width, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        m = self.mid(self.pool(e3))
        d3 = self.d3(torch.cat((self.u3(m), e3), dim=1))
        d2 = self.d2(torch.cat((self.u2(d3), e2), dim=1))
        return self.out(self.d1(torch.cat((self.u1(d2), e1), dim=1)))


def dice_loss(logits: Tensor, target: Tensor) -> Tensor:
    prob = logits.sigmoid()
    inter = (prob * target).sum(dim=(1, 2, 3, 4))
    denom = prob.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    return 1 - ((2 * inter + 1) / (denom + 1)).mean()


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval(); dices, detected = [], []
    for image, target in loader:
        image = image.to(device, non_blocking=True); target = target.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            pred = model(image).sigmoid() >= 0.5
        true = target.bool()
        inter = (pred & true).sum().float()
        dice = (2 * inter + 1) / (pred.sum() + true.sum() + 1)
        dices.append(dice.item())
        detected.append(float(pred[true].any()) if true.any() else float(not pred.any()))
    return {"patch_dice": float(np.mean(dices)), "patch_detection_rate": float(np.mean(detected))}


@dataclass
class Config:
    data: Path; out: Path; epochs: int; batch_size: int; patch_size: int; val_fraction: float; seed: int; lr: float
    workers: int; cache_dir: Path | None; deterministic: bool; compile: bool; width: int; samples_per_epoch: int; cpu_threads: int; validate_every: int


def main(cfg: Config) -> None:
    seed_everything(cfg.seed, cfg.deterministic)
    if cfg.cpu_threads:
        torch.set_num_threads(cfg.cpu_threads)
    images = sorted((cfg.data / "imagesTr").glob("*.nii.gz"))
    labels = cfg.data / "labelsTr"
    if not images or not labels.exists():
        raise FileNotFoundError("Expected Task07_Pancreas/imagesTr and labelsTr. Run with --download first.")
    rng = random.Random(cfg.seed); rng.shuffle(images)
    n_val = max(1, round(len(images) * cfg.val_fraction))
    train_paths, val_paths = images[n_val:], images[:n_val]
    (cfg.out / "split.json").parent.mkdir(parents=True, exist_ok=True)
    (cfg.out / "split.json").write_text(json.dumps({"train": [p.name for p in train_paths], "val": [p.name for p in val_paths]}, indent=2))
    loader_options = {"num_workers": cfg.workers, "pin_memory": torch.cuda.is_available()}
    if cfg.workers:
        loader_options.update({"persistent_workers": True, "prefetch_factor": 2})
    train_dataset = PancreasMassDataset(train_paths, labels, cfg.patch_size, True, cfg.cache_dir)
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=cfg.samples_per_epoch) if cfg.samples_per_epoch else None
    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=train_sampler is None, sampler=train_sampler, **loader_options)
    val_loader = DataLoader(PancreasMassDataset(val_paths, labels, cfg.patch_size, False, cfg.cache_dir), 1, **loader_options)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"device={device.type}; workers={cfg.workers}; threads={torch.get_num_threads()}; patch={cfg.patch_size}^3; width={cfg.width}; samples/epoch={cfg.samples_per_epoch or len(train_dataset)}")
    if device.type == "cpu":
        print("WARNING: CUDA is unavailable. 3-D CNN training on CPU is expected to be slow; install a CUDA PyTorch build or use a much smaller fast preset.")
    model = UNet3D(width=cfg.width).to(device)
    if cfg.compile and device.type != "cuda":
        print("Skipping --compile on CPU: PyTorch Inductor needs MSVC and is not a reliable CPU speedup.")
    elif cfg.compile:
        model = torch.compile(model)
    model_for_checkpoint = model._orig_mod if cfg.compile and device.type == "cuda" else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.0
    for epoch in range(1, cfg.epochs + 1):
        model.train(); losses = []
        for image, target in tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}"):
            image = image.to(device, non_blocking=True); target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(image)
                loss = F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(loss.item())
        if epoch % cfg.validate_every == 0 or epoch == cfg.epochs:
            metrics = validate(model, val_loader, device)
            print({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4), **{k: round(v, 4) for k, v in metrics.items()}})
            if metrics["patch_dice"] > best:
                best = metrics["patch_dice"]
                # Keep checkpoints compatible with PyTorch's restricted weights-only loader.
                serializable_config = {key: str(value) if isinstance(value, Path) else value for key, value in cfg.__dict__.items()}
                torch.save({"model": model_for_checkpoint.state_dict(), "config": serializable_config, "metrics": metrics}, cfg.out / "best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/Task07_Pancreas"))
    parser.add_argument("--out", type=Path, default=Path("runs/pancreas_mass"))
    parser.add_argument("--download", action="store_true", help="Download public MSD Task07 data (~12 GB), then exit")
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=64, help="Cube edge; 64 is the fast default (must be divisible by 8)")
    parser.add_argument("--width", type=int, default=16, help="Base model channels; 16 is the fast default")
    parser.add_argument("--fast", action="store_true", help="Use a 48^3, width-12 smoke-test preset")
    parser.add_argument("--cpu-fast", action="store_true", help="Use CPU-focused 32^3, width-8, 64-sample epoch preset")
    parser.add_argument("--ultra-fast", action="store_true", help="Use a tiny 16^3 one-patch development-only loop")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="Parallel data-loader workers")
    parser.add_argument("--cpu-threads", type=int, default=os.cpu_count() or 1, help="PyTorch CPU compute threads; 0 keeps PyTorch's default")
    parser.add_argument("--samples-per-epoch", type=int, default=0, help="Random patches per epoch; 0 visits every training case")
    parser.add_argument("--validate-every", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Uncompressed .npy cache (default: DATA/.cache)")
    parser.add_argument("--deterministic", action="store_true", help="Favor exact reproducibility over fast CUDA kernels")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile; slower startup, potentially faster later epochs")
    args = parser.parse_args()
    if args.download:
        download_msd(args.data.parent); raise SystemExit
    options = vars(args)
    options.pop("download")
    if options.pop("fast"):
        options["patch_size"], options["width"] = 48, 12
    if options.pop("cpu_fast"):
        # CPU cores should compute convolutions, not compete with loader workers.
        options["patch_size"], options["width"], options["samples_per_epoch"], options["workers"], options["val_fraction"] = 32, 8, 64, 0, 0.05
    if options.pop("ultra_fast"):
        # Intended only to verify the training loop or UI. It cannot learn a
        # medically meaningful detector from one 16^3 patch per "epoch".
        options["patch_size"], options["width"], options["samples_per_epoch"], options["workers"], options["val_fraction"], options["validate_every"] = 16, 4, 1, 0, 0.02, 100
    if options["patch_size"] % 8:
        parser.error("--patch-size must be divisible by 8 for this U-Net")
    if options["validate_every"] < 1:
        parser.error("--validate-every must be at least 1")
    if options["cache_dir"] is None:
        options["cache_dir"] = options["data"] / ".cache"
    main(Config(**options))
