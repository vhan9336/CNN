"""Local viewer for research-only pancreatic-mass model outputs.

This viewer is not a clinical diagnostic application.  It accepts NIfTI CT volumes
(.nii or .nii.gz), not photographs or ordinary PNG/JPEG files.
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path, PosixPath, WindowsPath
from tkinter import filedialog, messagebox, ttk

import nibabel as nib
import numpy as np
import torch
from PIL import Image, ImageTk
from torch.nn import functional as F

from train import UNet3D


def window_ct(volume: np.ndarray) -> np.ndarray:
    """Map CT Hounsfield units to a displayable grayscale volume."""
    return np.clip((volume + 125.0) / 350.0, 0.0, 1.0)


def load_local_checkpoint(path: Path, device: torch.device) -> dict:
    """Load project checkpoints without enabling arbitrary pickle execution.

    Older checkpoints saved ``pathlib.WindowsPath`` in their config. That class is
    inert metadata, so allowlisting only it preserves weights-only loading.
    """
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:
        raise RuntimeError("This viewer requires PyTorch 2.2 or newer for safe checkpoint loading.")
    with safe_globals([WindowsPath, PosixPath]):
        return torch.load(str(path), map_location=device, weights_only=True)


@torch.no_grad()
def predict_volume(volume: np.ndarray, checkpoint_path: Path) -> np.ndarray:
    """Run a fast whole-volume preview by resampling into the model patch size.

    This is deliberately a preview, not a lesion-localization or clinical workflow.
    It avoids hundreds of CPU sliding-window calls, which would make the UI unusable.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_local_checkpoint(checkpoint_path, device)
    config = checkpoint.get("config", {})
    width = int(config.get("width", 24))
    patch = int(config.get("patch_size", 64))
    model = UNet3D(width=width).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    source_shape = tuple(int(x) for x in volume.shape)
    normalized = np.clip(volume.astype(np.float32), -125, 225) / 350.0
    image = torch.from_numpy(normalized)[None, None].to(device)
    image = F.interpolate(image, size=(patch, patch, patch), mode="trilinear", align_corners=False)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        probability = model(image).sigmoid()
    probability = F.interpolate(probability.float(), size=source_shape, mode="trilinear", align_corners=False)
    return probability[0, 0].cpu().numpy()


class Viewer(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Pancreatic CT Research Viewer")
        self.minsize(820, 700)
        self.volume: np.ndarray | None = None
        self.probability: np.ndarray | None = None
        self.scan_path: Path | None = None
        self.checkpoint_path = Path("runs/pancreas_mass/best.pt")
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.slice_index = tk.IntVar(value=0)
        self.status = tk.StringVar(value="Open a NIfTI CT volume to begin.")
        self._build()
        self.after(100, self._poll_events)

    def _build(self) -> None:
        controls = ttk.Frame(self, padding=12)
        controls.pack(fill="x")
        ttk.Button(controls, text="Open CT scan (.nii/.nii.gz)", command=self.open_scan).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Choose checkpoint", command=self.choose_checkpoint).pack(side="left", padx=8)
        self.predict_button = ttk.Button(controls, text="Run research preview", command=self.run_prediction, state="disabled")
        self.predict_button.pack(side="left", padx=8)
        ttk.Label(self, text="RESEARCH ONLY — output is not a diagnosis and must not guide care.", foreground="#a30000", padding=(12, 0)).pack(anchor="w")
        ttk.Label(self, textvariable=self.status, padding=(12, 8)).pack(anchor="w")
        self.canvas = ttk.Label(self, anchor="center")
        self.canvas.pack(fill="both", expand=True, padx=12, pady=8)
        self.slider = ttk.Scale(self, from_=0, to=0, orient="horizontal", command=self.change_slice)
        self.slider.pack(fill="x", padx=24, pady=(0, 4))
        self.slice_label = ttk.Label(self, text="Slice —")
        self.slice_label.pack(pady=(0, 12))

    def open_scan(self) -> None:
        filename = filedialog.askopenfilename(filetypes=[("NIfTI CT volume", "*.nii *.nii.gz")])
        if not filename:
            return
        try:
            volume = nib.load(filename).get_fdata(dtype=np.float32)
            if volume.ndim != 3:
                raise ValueError(f"Expected a 3-D volume, received shape {volume.shape}.")
        except Exception as error:
            messagebox.showerror("Could not open scan", str(error)); return
        self.volume, self.probability, self.scan_path = volume, None, Path(filename)
        mid = volume.shape[2] // 2
        self.slider.configure(to=volume.shape[2] - 1); self.slider.set(mid)
        self.slice_index.set(mid); self.predict_button.configure(state="normal")
        self.status.set(f"Loaded {self.scan_path.name} — shape {volume.shape}. Choose a checkpoint, then run preview.")
        self.draw()

    def choose_checkpoint(self) -> None:
        filename = filedialog.askopenfilename(filetypes=[("PyTorch checkpoint", "*.pt *.pth")])
        if filename:
            self.checkpoint_path = Path(filename)
            self.status.set(f"Selected checkpoint: {self.checkpoint_path.name}")

    def run_prediction(self) -> None:
        if self.volume is None:
            return
        if not self.checkpoint_path.exists():
            messagebox.showerror("Checkpoint missing", f"Choose a trained checkpoint.\nExpected: {self.checkpoint_path}")
            return
        self.predict_button.configure(state="disabled")
        self.status.set("Predicting preview… the window remains responsive.")
        threading.Thread(target=self._prediction_worker, args=(self.volume.copy(), self.checkpoint_path), daemon=True).start()

    def _prediction_worker(self, volume: np.ndarray, checkpoint: Path) -> None:
        try:
            self.events.put(("prediction", predict_volume(volume, checkpoint)))
        except Exception as error:  # surface errors safely on the Tk main thread
            self.events.put(("error", str(error)))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "prediction":
                    self.probability = payload  # type: ignore[assignment]
                    self.status.set("Preview complete. Red/yellow overlay indicates model probability, not a diagnosis.")
                    self.draw()
                else:
                    messagebox.showerror("Prediction failed", str(payload)); self.status.set("Prediction failed.")
                self.predict_button.configure(state="normal" if self.volume is not None else "disabled")
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def change_slice(self, value: str) -> None:
        self.slice_index.set(round(float(value))); self.draw()

    def draw(self) -> None:
        if self.volume is None:
            return
        index = max(0, min(self.slice_index.get(), self.volume.shape[2] - 1))
        gray = (window_ct(self.volume[:, :, index]) * 255).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
        if self.probability is not None:
            p = self.probability[:, :, index]
            alpha = np.clip((p - 0.30) / 0.70, 0, 1)[..., None] * 0.70
            heat = np.stack((np.ones_like(p) * 255, np.clip(p * 255, 0, 255), np.zeros_like(p)), axis=2)
            rgb = rgb * (1 - alpha) + heat * alpha
        image = Image.fromarray(rgb.astype(np.uint8)).resize((640, 640), Image.Resampling.NEAREST)
        self._photo = ImageTk.PhotoImage(image)
        self.canvas.configure(image=self._photo)
        self.slice_label.configure(text=f"Slice {index + 1} / {self.volume.shape[2]}")


if __name__ == "__main__":
    Viewer().mainloop()
