"""Step 4: Real range–Doppler + simple 2-D CA-CFAR.
Run with: uv run python 04_rd_realdata.py
"""
from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "data/simerad60.hdf5"
DETECTIONS_TO_RENDER = 3  # number of frames to visualise


def mag_db(z: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(z) + 1e-12)


def find_rd_dataset(h5: h5py.File) -> tuple[h5py.Dataset, str]:
    """Return the most plausible range–Doppler dataset and its path."""
    candidates: list[tuple[int, str, h5py.Dataset]] = []

    def walk(obj: h5py.Group, prefix: str = "") -> None:
        for key, val in obj.items():
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(val, h5py.Dataset):
                shape = val.shape
                score = 0
                name = key.lower()
                if len(shape) in (2, 3):
                    score += 1
                if any(dim > 32 for dim in shape):
                    score += 1
                if "rd" in name:
                    score += 2
                if "range" in name and "doppler" in name:
                    score += 2
                candidates.append((score, path, val))
            elif isinstance(val, h5py.Group):
                walk(val, path)

    walk(h5)
    if not candidates:
        raise RuntimeError("No datasets found in HDF5 file.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_path, best_dataset = candidates[0]
    if best_score <= 0:
        raise RuntimeError("Could not identify a range–Doppler dataset.")
    return best_dataset, best_path


def cfar2d_db(
    mat_db: np.ndarray,
    guard: tuple[int, int] = (2, 2),
    train: tuple[int, int] = (6, 6),
    scale: float = 6.0,
) -> np.ndarray:
    """Rectangular CA-CFAR returning boolean detection mask."""
    doppler_guard, range_guard = guard
    doppler_train, range_train = train

    mat_amp = 10.0 ** (mat_db / 20.0)
    mat_pow = mat_amp**2
    detections = np.zeros_like(mat_amp, dtype=bool)

    for d in range(
        doppler_train + doppler_guard,
        mat_amp.shape[0] - (doppler_train + doppler_guard),
    ):
        for r in range(
            range_train + range_guard,
            mat_amp.shape[1] - (range_train + range_guard),
        ):
            d0 = d - (doppler_train + doppler_guard)
            d1 = d + doppler_train + doppler_guard + 1
            r0 = r - (range_train + range_guard)
            r1 = r + range_train + range_guard + 1
            window = mat_pow[d0:d1, r0:r1].copy()
            gd0 = doppler_train
            gd1 = doppler_train + 2 * doppler_guard + 1
            gr0 = range_train
            gr1 = range_train + 2 * range_guard + 1
            window[gd0:gd1, gr0:gr1] = 0.0  # zero guard + CUT area
            non_zero = window[window > 0.0]
            if non_zero.size == 0:
                continue
            noise = np.mean(non_zero)
            detections[d, r] = mat_pow[d, r] > scale * noise

    return detections


def main() -> None:
    if not Path(DATA_PATH).exists():
        raise SystemExit(
            f"{DATA_PATH} not found. Download it with the curl command in README.md."
        )

    with h5py.File(DATA_PATH, "r") as h5:
        dataset, path = find_rd_dataset(h5)
        frames = dataset[:DETECTIONS_TO_RENDER]

    print(f"Selected dataset: {path} shape={frames.shape}")

    if frames.ndim == 2:
        frames = frames[np.newaxis, ...]
    if np.iscomplexobj(frames):
        frames = np.abs(frames)

    frames_db = mag_db(frames)
    det_total = 0
    vmin = np.percentile(frames_db, 5)
    vmax = np.percentile(frames_db, 99)

    for idx in range(min(DETECTIONS_TO_RENDER, frames_db.shape[0])):
        rd_slice = frames_db[idx]
        detections = cfar2d_db(rd_slice, guard=(2, 2), train=(6, 6), scale=6.0)
        det_total += detections.sum()

        plt.figure(figsize=(6, 4))
        plt.imshow(rd_slice, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        yy, xx = np.where(detections)
        if yy.size:
            plt.scatter(xx, yy, s=12, edgecolors="white", facecolors="none")
        plt.colorbar(label="Magnitude (dB)")
        plt.title(f"Measured RD Frame {idx} (dB) + 2-D CFAR")
        plt.xlabel("Range bin")
        plt.ylabel("Doppler bin")
        plt.tight_layout()
        out_path = f"out_04_rd_frame{idx}.png"
        plt.savefig(out_path, dpi=160)
        plt.close()

    print(f"Wrote out_04_rd_frame*.png | detections across frames: {int(det_total)}")


if __name__ == "__main__":
    main()
