"""Step 3: FMCW range FFT + 1-D CA-CFAR (simulated data).
Run with: uv run python 03_fmcw_range_cfar.py
"""
from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

C0 = 3.0e8  # speed of light (m/s)
BANDWIDTH = 200.0e6  # Hz
CHIRP_TIME = 1.0e-3  # s
FS = 2.0e6  # Hz
SLOPE = BANDWIDTH / CHIRP_TIME

TARGETS = [
    (35.0, 50.0),  # (range m, linear SNR)
    (62.0, 20.0),
]
NOISE_STD = 0.5
GUARD_CELLS = 4
TRAIN_CELLS = 20
CFAR_SCALE = 12.0


np.random.seed(0)


def fractional_delay(sig: np.ndarray, delay_s: float, fs: float) -> np.ndarray:
    """Apply fractional sample delay using frequency-domain phase shift."""
    spectrum = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(sig.size, d=1.0 / fs)
    shifted = spectrum * np.exp(-1j * 2 * np.pi * freqs * delay_s)
    return np.fft.irfft(shifted, n=sig.size)


def mag_db(z: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(z) + 1e-12)


def main() -> None:
    n_samp = int(CHIRP_TIME * FS)
    t = np.arange(n_samp) / FS

    tx = np.exp(1j * np.pi * SLOPE * t**2)
    rx = np.zeros_like(tx)

    for rng_m, snr_lin in TARGETS:
        tau = 2.0 * rng_m / C0
        echo = fractional_delay(tx, tau, FS)
        echo_power = np.mean(np.abs(echo) ** 2)
        rx += echo * np.sqrt(snr_lin / echo_power)

    noise = NOISE_STD * (np.random.randn(n_samp) + 1j * np.random.randn(n_samp))
    rx += noise

    dechirped = rx * np.conj(tx)
    window = np.hanning(n_samp)
    spectrum = np.fft.fft(dechirped * window)[: n_samp // 2]
    beat_freq = np.arange(spectrum.size) * FS / n_samp
    ranges_m = beat_freq * C0 / (2.0 * SLOPE)
    mag = mag_db(spectrum)

    # 1-D CA-CFAR (sliding window)
    threshold = np.full_like(mag, -np.inf)
    detections = np.zeros_like(mag, dtype=bool)
    for idx in range(TRAIN_CELLS + GUARD_CELLS, mag.size - (TRAIN_CELLS + GUARD_CELLS)):
        start = idx - (TRAIN_CELLS + GUARD_CELLS)
        stop = idx + TRAIN_CELLS + GUARD_CELLS + 1
        cutout = mag[start:stop].copy()
        guard_start = TRAIN_CELLS
        guard_stop = TRAIN_CELLS + 2 * GUARD_CELLS + 1
        cutout[guard_start:guard_stop] = -np.inf
        linear_vals = 10.0 ** (cutout[cutout > -np.inf] / 20.0)
        power_vals = linear_vals**2
        noise_est = np.mean(power_vals)
        threshold[idx] = 10.0 * np.log10(CFAR_SCALE * noise_est + 1e-12)
        detections[idx] = mag[idx] > threshold[idx]

    threshold[np.isneginf(threshold)] = np.nan
    det_bins = np.where(detections)[0]
    det_ranges = ranges_m[det_bins]

    plt.figure(figsize=(9, 4))
    plt.plot(ranges_m, mag, label="Range spectrum (dB)")
    plt.plot(ranges_m, threshold, label="CA-CFAR threshold (dB)")
    if det_bins.size:
        plt.scatter(det_ranges, mag[det_bins], c="k", s=16, label="Detections")
    plt.xlim(0.0, 100.0)
    plt.xlabel("Range (m)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FMCW range FFT + CA-CFAR")
    plt.legend()
    plt.tight_layout()
    out_path = "out_03_range_cfar.png"
    plt.savefig(out_path, dpi=160)
    plt.close()

    range_resolution = C0 / (2.0 * BANDWIDTH)
    print(f"Wrote {out_path} | Theoretical range resolution ~ {range_resolution:.2f} m")
    print("Detected ranges (m):", [round(val, 2) for val in det_ranges])

    with open("out_03_detections.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["range_m"])
        for rng in det_ranges:
            writer.writerow([float(rng)])
    print("Wrote out_03_detections.csv")


if __name__ == "__main__":
    main()
