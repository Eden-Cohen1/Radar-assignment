"""Step 3 tutorial script: FMCW range FFT and 1-D CA-CFAR with narrative guidance.

Run with:
    uv run python 03_fmcw_range_cfar.py
    uv run python 03_fmcw_range_cfar.py --help
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer(
    add_completion=False,
    help="Simulate an FMCW radar sweep and explain the FFT-to-range journey.",
)

C0 = 3.0e8  # Speed of light (m/s)


def mag_db(z: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(z) + 1e-12)


def describe_setup(
    bandwidth: float, chirp_time: float, fs: float, targets: List[Tuple[float, float]]
) -> None:
    slope = bandwidth / chirp_time
    typer.echo("📡 Step 3: Hello FMCW — mapping beat frequency to range")
    typer.echo(
        f"  • Sweep bandwidth (B): {bandwidth / 1e6:.1f} MHz, chirp time (T_c): {chirp_time * 1e3:.1f} ms, sample rate: {fs / 1e6:.1f} MSa/s"
    )
    typer.echo(
        f"  • Chirp slope S = B / T_c = {slope / 1e12:.2f} THz/s (controls the beat ↔ range mapping)."
    )
    typer.echo(
        "  • We'll inject a couple of simulated targets (range, SNR) and add complex white noise."
    )
    typer.echo(
        "  • Key equations: R = (c · f_b) / (2S) and ΔR = c / (2B). We'll print both."
    )
    typer.echo("  • Deliverables: out_03_range_cfar.png + out_03_detections.csv.")
    for idx, (rng, snr) in enumerate(targets, start=1):
        typer.echo(f"    – Target {idx}: {rng:.1f} m with linear SNR {snr:.1f}")


def synthesize_beat(
    bandwidth: float,
    chirp_time: float,
    fs: float,
    targets: List[Tuple[float, float]],
    noise_std: float,
) -> np.ndarray:
    slope = bandwidth / chirp_time
    n_samp = int(chirp_time * fs)
    t = np.arange(n_samp) / fs
    beat = np.zeros(n_samp, dtype=complex)
    for rng_m, snr_lin in targets:
        beat_freq = 2.0 * slope * rng_m / C0
        tone = np.exp(1j * 2 * np.pi * beat_freq * t)
        beat += tone * np.sqrt(snr_lin)
    beat += noise_std * (np.random.randn(n_samp) + 1j * np.random.randn(n_samp))
    return beat


def cfar_1d(
    spectrum: np.ndarray,
    guard_cells: int,
    train_cells: int,
    scale: float,
    peak_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mag_amp = np.abs(spectrum)
    mag_pow = mag_amp**2
    mag = mag_db(spectrum)
    threshold_amp = np.zeros_like(mag_amp)
    threshold = np.full_like(mag, np.nan)
    detections = np.zeros_like(mag, dtype=bool)

    for idx in range(train_cells + guard_cells, mag.size - (train_cells + guard_cells)):
        start = idx - (train_cells + guard_cells)
        stop = idx + train_cells + guard_cells + 1
        cutout = mag_pow[start:stop].copy()
        guard_start = train_cells
        guard_stop = train_cells + 2 * guard_cells + 1
        cutout[guard_start:guard_stop] = 0.0
        valid = cutout[cutout > 0.0]
        if valid.size == 0:
            continue
        noise_est = np.mean(valid)
        threshold_pow = scale * noise_est
        threshold_amp[idx] = np.sqrt(threshold_pow)
        threshold[idx] = 20.0 * np.log10(threshold_amp[idx] + 1e-12)
        if mag_amp[idx] > threshold_amp[idx] * peak_margin:
            left = mag_amp[idx - 1]
            right = mag_amp[idx + 1]
            detections[idx] = mag_amp[idx] >= left and mag_amp[idx] >= right

    return mag, threshold, detections


@app.command()
def main(
    bandwidth: float = typer.Option(200e6, help="Chirp bandwidth B in Hz."),
    chirp_time: float = typer.Option(1e-3, help="Chirp time T_c in seconds."),
    fs: float = typer.Option(2e6, help="Sample rate in samples/second."),
    noise_std: float = typer.Option(
        0.35, help="Standard deviation of additive complex noise."
    ),
    guard_cells: int = typer.Option(6, help="Guard cells on each side for CA-CFAR."),
    train_cells: int = typer.Option(
        18, help="Training cells on each side for CA-CFAR."
    ),
    scale: float = typer.Option(
        8.0, help="Noise scaling factor for CA-CFAR threshold."
    ),
    peak_margin: float = typer.Option(
        3.0,
        help="Multiplier to insist detections stand above threshold by this factor.",
    ),
    targets: List[float] = typer.Option(
        [35.0, 50.0, 62.0, 25.0],
        help="Pairs of range (m) and linear SNR: e.g., --targets 35 40 60 20",
    ),
) -> None:
    """Simulate beat notes, run a range FFT, and apply CA-CFAR with explanatory prints."""

    if len(targets) % 2 != 0:
        raise typer.BadParameter(
            "Provide an even number of values: range1 snr1 range2 snr2 …"
        )
    paired_targets = [(targets[i], targets[i + 1]) for i in range(0, len(targets), 2)]

    describe_setup(bandwidth, chirp_time, fs, paired_targets)
    np.random.seed(0)

    beat = synthesize_beat(bandwidth, chirp_time, fs, paired_targets, noise_std)
    n_samp = len(beat)
    slope = bandwidth / chirp_time

    window = np.hanning(n_samp)
    spectrum = np.fft.fft(beat * window)[: n_samp // 2]
    beat_freq = np.arange(spectrum.size) * fs / n_samp
    ranges_m = beat_freq * C0 / (2.0 * slope)

    mag, threshold, detections = cfar_1d(
        spectrum,
        guard_cells=guard_cells,
        train_cells=train_cells,
        scale=scale,
        peak_margin=peak_margin,
    )

    det_bins = np.where(detections)[0]
    det_ranges = ranges_m[det_bins]

    plt.figure(figsize=(10, 4.5))
    plt.plot(ranges_m, mag, label="Range spectrum (dB)")
    plt.plot(ranges_m, threshold, label="CA-CFAR threshold (dB)")
    if det_bins.size:
        plt.scatter(det_ranges, mag[det_bins], c="k", s=24, label="Detections")
    plt.xlim(0.0, 100.0)
    plt.xlabel("Range (m)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FMCW range FFT + CA-CFAR")
    plt.legend()
    plt.tight_layout()
    out_plot = Path("out_03_range_cfar.png")
    plt.savefig(out_plot, dpi=160)
    plt.close()

    range_resolution = C0 / (2.0 * bandwidth)
    typer.echo(f"✅ Saved {out_plot}")
    typer.echo(
        f"  • Theoretical range resolution ΔR = c / (2B) = {range_resolution:.2f} m"
    )
    typer.echo(
        "  • Detected range bins (meters): "
        + ", ".join(f"{val:.2f}" for val in det_ranges)
        if det_ranges.size
        else "  • No detections — adjust CFAR parameters or target SNR."
    )

    out_csv = Path("out_03_detections.csv")
    with out_csv.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["range_m"])
        for rng in det_ranges:
            writer.writerow([float(rng)])
    typer.echo(f"📄 Saved {out_csv} with detection ranges.")

    typer.echo("Next questions to ponder:")
    typer.echo("  • How do the simulated target ranges compare to the detected peaks?")
    typer.echo(
        "  • What knob (B, T_c, or S) would you change to improve range resolution without lengthening runtime?"
    )


if __name__ == "__main__":
    app()
