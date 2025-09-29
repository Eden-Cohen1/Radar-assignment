"""Step 2: IQ mixing and FIR filtering on real FM/RDS IQ data.
Run with: uv run python 02_iq_ops.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, stft, get_window

DATA_PATH = "data/fm_rds_250k_1Msamples.iq"
FS = 250_000.0  # sample rate (Sa/s)
SHIFT_HZ = 57_000.0  # RDS subcarrier
LPF_CUTOFF = 5_000.0  # Hz
LPF_TAPS = 257
FFT_LEN = 32_768


def mag_db(z: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(z) + 1e-12)


def mix_shift(sig: np.ndarray, fs: float, freq_shift: float) -> np.ndarray:
    n = np.arange(sig.size)
    osc = np.exp(-1j * 2 * np.pi * freq_shift * n / fs)
    return sig * osc


def save_fft_plot(sig: np.ndarray, title: str, fname: str) -> None:
    window = get_window("hann", FFT_LEN)
    fft_vals = np.fft.fft(sig[:FFT_LEN] * window, n=FFT_LEN)
    freqs = np.fft.fftfreq(FFT_LEN, d=1.0 / FS)
    order = np.argsort(freqs)

    plt.figure(figsize=(8, 4))
    plt.plot(freqs[order] / 1e3, mag_db(fft_vals[order]))
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.xlim(-125, 125)
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()


def save_spectrogram(sig: np.ndarray, fname: str) -> None:
    f_s, t_s, stft_mat = stft(
        sig[:200_000],
        fs=FS,
        window="hann",
        nperseg=2_048,
        noverlap=1_536,
        nfft=2_048,
        boundary=None,
    )
    plt.figure(figsize=(7, 4))
    plt.pcolormesh(t_s, f_s / 1e3, mag_db(stft_mat), shading="auto")
    plt.title("Spectrogram of filtered baseband")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()


def main() -> None:
    data = np.fromfile(DATA_PATH, dtype=np.complex64)
    if data.size == 0:
        raise RuntimeError(f"Loaded zero samples from {DATA_PATH}. Did the download finish?")

    shifted = mix_shift(data, FS, SHIFT_HZ)
    taps = firwin(numtaps=LPF_TAPS, cutoff=LPF_CUTOFF, fs=FS)
    filtered = lfilter(taps, 1.0, shifted)

    save_fft_plot(data, "Raw FM band", "out_02_raw_psd.png")
    save_fft_plot(shifted, "After shift by -57 kHz", "out_02_shift_psd.png")
    save_fft_plot(filtered, "After LPF (~5 kHz around DC)", "out_02_lpf_psd.png")

    save_spectrogram(filtered, "out_02_lpf_spectrogram.png")

    print("Wrote out_02_raw_psd.png, out_02_shift_psd.png, out_02_lpf_psd.png, out_02_lpf_spectrogram.png")


if __name__ == "__main__":
    main()
