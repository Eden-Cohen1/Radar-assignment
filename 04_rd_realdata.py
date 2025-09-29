"""Step 4 tutorial script: Real range–Doppler maps and 2-D CA-CFAR with guidance.

Run with:
    uv run python 04_rd_realdata.py
    uv run python 04_rd_realdata.py --help
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer(
    add_completion=False,
    help="Load measured range–Doppler heatmaps and practice 2-D CA-CFAR.",
)

DATA_PATH = Path("data/simerad60.hdf5")


def mag_db(z: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(z) + 1e-12)


def describe_plan(path: Path, frames: int) -> None:
    typer.echo("📡 Step 4: Real range–Doppler processing")
    typer.echo(
        "  • We'll open the SiMeRAD60 HDF5 file (TI AWR6843ISK demo) and auto-locate RD datasets."
    )
    typer.echo(
        f"  • The script plots {frames} consecutive heatmaps, applies a simple 2-D CA-CFAR, and overlays detections."
    )
    typer.echo(
        "  • Outputs: out_04_rd_frame0/1/2.png showing magnitude in dB, with detections circled if present."
    )
    typer.echo(
        "  • Remember: columns ≈ range bins, rows ≈ Doppler bins (positive/negative velocities)."
    )
    typer.echo(f"  • Looking for file at {path} (download steps are in the README).")


def find_rd_dataset(h5: h5py.File) -> Tuple[h5py.Dataset, str]:
    candidates = []

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


def gather_frames(
    h5: h5py.File,
    dataset: h5py.Dataset,
    path: str,
    frames_to_render: int,
) -> np.ndarray:
    data = dataset[:]
    if isinstance(data, np.ndarray) and data.ndim == 2:
        collected = [data]
        base_path, ds_name = path.rsplit("/", 1)
        match = re.search(r"(.*?/frame_)(\d+)$", base_path)
        if match:
            prefix, idx_str = match.groups()
            start_idx = int(idx_str)
            for offset in range(1, frames_to_render):
                next_path = f"{prefix}{start_idx + offset}/{ds_name}"
                if next_path in h5:
                    collected.append(h5[next_path][:])
                if len(collected) == frames_to_render:
                    break
        data = np.stack(collected, axis=0)
    else:
        data = np.array(data)

    if data.ndim == 2:
        data = data[np.newaxis, ...]
    if np.iscomplexobj(data):
        data = np.abs(data)
    return data


def cfar2d_db(
    mat_db: np.ndarray,
    guard: Tuple[int, int],
    train: Tuple[int, int],
    scale: float,
) -> np.ndarray:
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
            window[gd0:gd1, gr0:gr1] = 0.0
            non_zero = window[window > 0.0]
            if non_zero.size == 0:
                continue
            noise = np.mean(non_zero)
            detections[d, r] = mat_pow[d, r] > scale * noise
    return detections


@app.command()
def main(
    path: Path = typer.Option(DATA_PATH, help="Path to the SiMeRAD60 HDF5 file."),
    frames: int = typer.Option(3, help="Number of consecutive RD frames to visualise."),
    guard: Tuple[int, int] = typer.Option(
        (2, 2), help="(Doppler, range) guard cells for 2-D CA-CFAR."
    ),
    train: Tuple[int, int] = typer.Option(
        (6, 6), help="(Doppler, range) training cells."
    ),
    scale: float = typer.Option(
        6.0, help="Scaling factor for noise estimate in CA-CFAR."
    ),
) -> None:
    """Load real range–Doppler data, plot, and apply an intuitive 2-D CA-CFAR."""

    if not path.exists():
        raise SystemExit(
            f"{path} not found. Download it with the curl command listed in README.md before running this step."
        )

    describe_plan(path, frames)

    with h5py.File(path, "r") as h5:
        dataset, dataset_path = find_rd_dataset(h5)
        array = gather_frames(h5, dataset, dataset_path, frames)

    typer.echo(f"✅ Selected dataset {dataset_path} shape={array.shape}")

    frames_db = mag_db(array)
    vmin = float(np.percentile(frames_db, 5))
    vmax = float(np.percentile(frames_db, 99))
    detections_total = 0

    for idx in range(min(frames, frames_db.shape[0])):
        rd_slice = frames_db[idx]
        detections = cfar2d_db(rd_slice, guard=guard, train=train, scale=scale)
        detections_total += int(detections.sum())

        plt.figure(figsize=(6.5, 4.5))
        plt.imshow(rd_slice, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        yy, xx = np.where(detections)
        if yy.size:
            plt.scatter(xx, yy, s=18, edgecolors="white", facecolors="none")
        plt.colorbar(label="Magnitude (dB)")
        plt.title(f"Range–Doppler Frame {idx} + 2-D CA-CFAR")
        plt.xlabel("Range bin")
        plt.ylabel("Doppler bin")
        plt.tight_layout()
        out_path = Path(f"out_04_rd_frame{idx}.png")
        plt.savefig(out_path, dpi=160)
        plt.close()
        typer.echo(f"  • Saved {out_path} ({detections.sum()} detections)")

    typer.echo(f"📈 Total detections across frames: {detections_total}")
    if detections_total == 0:
        typer.echo(
            "  • Tip: Reduce the scale factor or increase training cells to capture weaker targets."
        )
    typer.echo(
        "Think about: Do stationary objects cluster near Doppler≈0? How does CFAR balance misses vs false alarms?"
    )


if __name__ == "__main__":
    app()
