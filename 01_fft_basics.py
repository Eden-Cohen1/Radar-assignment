"""Step 1: FFT & Spectrogram intuition demo.
Run with: uv run python 01_fft_basics.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, get_window


def mag_db(z: np.ndarray) -> np.ndarray:
    """Convert complex magnitude to dB scale with a tiny floor."""
    return 20.0 * np.log10(np.abs(z) + 1e-12)


def main() -> None:
    fs = 200_000.0  # sample rate (Sa/s)
    duration = 0.05  # seconds
    t = np.arange(int(fs * duration)) / fs

    # Linear chirp 10 kHz -> 60 kHz across duration.
    f0, f1 = 10_000.0, 60_000.0
    slope = (f1 - f0) / duration
    chirp = np.exp(1j * 2 * np.pi * (f0 * t + 0.5 * slope * t**2))

    # Add a tone slightly off-bin to emphasize leakage without a window.
    leakage_tone = 0.4 * np.exp(1j * 2 * np.pi * 29_500.0 * t)
    x = chirp + leakage_tone

    n_fft = 4096
    window = get_window("hann", n_fft, fftbins=True)

    # Windowed FFT (Hann) vs rectangular (no window).
    x_seg = x[:n_fft]
    fft_rect = np.fft.fft(x_seg, n=n_fft)
    fft_hann = np.fft.fft(x_seg * window, n=n_fft)
    freqs = np.fft.fftfreq(n_fft, d=1.0 / fs)
    half = n_fft // 2

    # Short-time Fourier transform for spectrogram view.
    f_s, t_s, stft_mat = stft(
        x,
        fs=fs,
        window="hann",
        nperseg=1024,
        noverlap=768,
        nfft=1024,
        boundary=None,
    )

    plt.figure(figsize=(11, 4))
    plt.subplot(1, 3, 1)
    plt.plot(freqs[:half] / 1e3, mag_db(fft_rect[:half]))
    plt.title("FFT (rectangular)")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Magnitude (dB)")

    plt.subplot(1, 3, 2)
    plt.plot(freqs[:half] / 1e3, mag_db(fft_hann[:half]))
    plt.title("FFT (Hann window)")
    plt.xlabel("Frequency (kHz)")

    plt.subplot(1, 3, 3)
    plt.pcolormesh(t_s * 1e3, f_s / 1e3, mag_db(stft_mat), shading="auto")
    plt.title("Spectrogram (STFT)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (kHz)")

    plt.tight_layout()
    output_path = "out_01_fft_spectrogram.png"
    plt.savefig(output_path, dpi=160)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
