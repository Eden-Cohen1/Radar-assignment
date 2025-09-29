"""Step 2 tutorial script: IQ mixing, shifting, and filtering with detailed narration.

Run with:
    uv run python 02_iq_ops.py
    uv run python 02_iq_ops.py --help  # explore options
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from scipy.signal import firwin, get_window, lfilter, stft

app = typer.Typer(add_completion=False, help="Walk through IQ data, frequency shifting, and FIR filtering.")

DATA_PATH = Path("data/fm_rds_250k_1Msamples.iq")
DEFAULT_FS = 250_000.0
DEFAULT_SHIFT = 57_000.0
DEFAULT_LPF = 5_000.0
FFT_LEN = 32_768


def mag_db(z: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(z) + 1e-12)


def explain_signal(fs: float, shift_hz: float, lpf_cutoff: float) -> None:
    typer.echo("📡 Step 2: IQ, mixing, and filtering on real FM/RDS data")
    typer.echo("  • IQ samples (I + jQ) carry both amplitude and phase, which lets us shift spectra cleanly.")
    typer.echo(f"  • We'll load the capture at {fs/1e3:.0f} kSa/s and frequency-shift the 57 kHz RDS tone to DC.")
    typer.echo(f"  • After the shift, a {lpf_cutoff/1e3:.1f} kHz low-pass isolates the subcarrier for inspection.")
    typer.echo("  • Deliverables: PSD plots before/after shifts + spectrogram of the filtered baseband.")


def ensure_data(path: Path) -> np.ndarray:
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Download it with the curl command from README.md before running this step."
        )
    data = np.fromfile(path, dtype=np.complex64)
    if data.size == 0:
        raise SystemExit(f"Loaded zero samples from {path}. Redownload the file (download may have failed).")
    return data


def mix_shift(sig: np.ndarray, fs: float, freq_shift: float) -> np.ndarray:
    n = np.arange(sig.size)
    osc = np.exp(-1j * 2 * np.pi * freq_shift * n / fs)
    return sig * osc


def save_fft(sig: np.ndarray, title: str, filename: str, fs: float) -> None:
    window = get_window("hann", FFT_LEN)
    fft_vals = np.fft.fft(sig[:FFT_LEN] * window, n=FFT_LEN)
    freqs = np.fft.fftfreq(FFT_LEN, d=1.0 / fs)
    order = np.argsort(freqs)
    plt.figure(figsize=(8, 4))
    plt.plot(freqs[order] / 1e3, mag_db(fft_vals[order]))
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.xlim(-125, 125)
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()


def save_spectrogram(sig: np.ndarray, filename: str, fs: float) -> None:
    f_s, t_s, stft_mat = stft(
        sig[:200_000],
        fs=fs,
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
    plt.savefig(filename, dpi=160)
    plt.close()


@app.command()
def main(
    path: Path = typer.Option(DATA_PATH, help="Path to the complex64 IQ capture."),
    fs: float = typer.Option(DEFAULT_FS, help="Sample rate in samples/second."),
    shift_hz: float = typer.Option(DEFAULT_SHIFT, help="Frequency shift to center the subcarrier (Hz)."),
    lpf_cutoff: float = typer.Option(DEFAULT_LPF, help="Low-pass cutoff after shifting (Hz)."),
) -> None:
    """Explain frequency translation and FIR filtering on a real IQ capture."""

    explain_signal(fs, shift_hz, lpf_cutoff)

    iq = ensure_data(path)
    typer.echo(f"✅ Loaded {iq.size:,} complex samples from {path}")

    typer.echo("🌀 Mixing the spectrum down so the 57 kHz RDS tone lands at 0 Hz…")
    shifted = mix_shift(iq, fs, shift_hz)

    typer.echo("🔧 Designing a linear-phase FIR low-pass filter around DC…")
    taps = firwin(numtaps=257, cutoff=lpf_cutoff, fs=fs)
    filtered = lfilter(taps, 1.0, shifted)

    typer.echo("📊 Saving PSD snapshots (before shift, after shift, after LPF)…")
    save_fft(iq, "Raw FM band", "out_02_raw_psd.png", fs)
    save_fft(shifted, f"Shifted by -{shift_hz/1e3:.0f} kHz", "out_02_shift_psd.png", fs)
    save_fft(filtered, "Low-pass filtered baseband", "out_02_lpf_psd.png", fs)

    typer.echo("🎨 Saving spectrogram of the filtered baseband to show subcarrier dynamics…")
    save_spectrogram(filtered, "out_02_lpf_spectrogram.png", fs)

    typer.echo("Reflection prompts you can answer in report.md:")
    typer.echo("  • What evidence in the PSD confirms the shift worked?")
    typer.echo("  • How does the spectrogram reveal that the RDS tone now sits at baseband?")
    typer.echo("Results written: out_02_raw_psd.png, out_02_shift_psd.png, out_02_lpf_psd.png, out_02_lpf_spectrogram.png")


if __name__ == "__main__":
    app()
