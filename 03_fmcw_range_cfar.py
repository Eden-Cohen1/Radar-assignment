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
    (62.0, 25.0),
]
NOISE_STD = 0.35
GUARD_CELLS = 6
TRAIN_CELLS = 18
CFAR_SCALE = 8.0


np.random.seed(0)


def mag_db(z: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(z) + 1e-12)


def main() -> None:
    n_samp = int(CHIRP_TIME * FS)
    t = np.arange(n_samp) / FS

    beat = np.zeros(n_samp, dtype=complex)

    for rng_m, snr_lin in TARGETS:
        beat_freq = 2.0 * SLOPE * rng_m / C0
        tone = np.exp(1j * 2 * np.pi * beat_freq * t)
        beat += tone * np.sqrt(snr_lin)

    noise = NOISE_STD * (np.random.randn(n_samp) + 1j * np.random.randn(n_samp))
    beat += noise

    window = np.hanning(n_samp)
    spectrum = np.fft.fft(beat * window)[: n_samp // 2]
    beat_freq = np.arange(spectrum.size) * FS / n_samp
    ranges_m = beat_freq * C0 / (2.0 * SLOPE)
    mag_amp = np.abs(spectrum)
    mag_pow = mag_amp**2
    mag = mag_db(spectrum)

    # 1-D CA-CFAR (sliding window)
    threshold_amp = np.zeros_like(mag_amp)
    threshold = np.full_like(mag, np.nan)
    detections = np.zeros_like(mag, dtype=bool)
    for idx in range(TRAIN_CELLS + GUARD_CELLS, mag.size - (TRAIN_CELLS + GUARD_CELLS)):
        start = idx - (TRAIN_CELLS + GUARD_CELLS)
        stop = idx + TRAIN_CELLS + GUARD_CELLS + 1
        cutout = mag_pow[start:stop].copy()
        guard_start = TRAIN_CELLS
        guard_stop = TRAIN_CELLS + 2 * GUARD_CELLS + 1
        cutout[guard_start:guard_stop] = 0.0
        valid = cutout[cutout > 0.0]
        if valid.size == 0:
            continue
        noise_est = np.mean(valid)
        threshold_pow = CFAR_SCALE * noise_est
        threshold_amp[idx] = np.sqrt(threshold_pow)
        threshold[idx] = 20.0 * np.log10(threshold_amp[idx] + 1e-12)
        if mag_amp[idx] > threshold_amp[idx] * 3.0:
            left = mag_amp[idx - 1]
            right = mag_amp[idx + 1]
            detections[idx] = mag_amp[idx] >= left and mag_amp[idx] >= right

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
