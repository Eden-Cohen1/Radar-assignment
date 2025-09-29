"""Step 1 tutorial script: Chirps, FFTs, and spectrograms explained for beginners.

Run with:
    uv run python 01_fft_basics.py
    uv run python 01_fft_basics.py --help  # to see adjustable knobs
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from scipy.signal import get_window, stft

app = typer.Typer(
    add_completion=False,
    help="Explore chirps, FFT leakage, and spectrograms with helpful narration.",
)


def mag_db(z: np.ndarray) -> np.ndarray:
    """Convert a complex vector to dB magnitude."""
    return 20.0 * np.log10(np.abs(z) + 1e-12)


def print_intro(fs: float, f_start: float, f_stop: float, duration: float) -> None:
    typer.echo("📡 Step 1: Building FFT & spectrogram intuition")
    typer.echo(
        "  • A chirp is a tone whose frequency slides over time. Here it sweeps linearly from"
        f" {f_start / 1e3:.1f} kHz to {f_stop / 1e3:.1f} kHz across {duration * 1e3:.0f} ms."
    )
    typer.echo(
        "  • We'll sample it at {0:.0f} kSa/s, which fixes the FFT bin spacing.".format(
            fs / 1e3
        )
    )
    typer.echo(
        "  • Alongside the chirp we add a small tone that sits between FFT bins to expose leakage."
    )
    typer.echo(
        "  • Outputs: out_01_fft_spectrogram.png (time trace, instantaneous frequency, FFTs, and spectrogram)."
    )


@app.command()
def main(
    fs: float = typer.Option(200_000.0, help="Sample rate in samples/second."),
    duration: float = typer.Option(0.05, help="Signal length in seconds."),
    f_start: float = typer.Option(10_000.0, help="Chirp start frequency in Hz."),
    f_stop: float = typer.Option(60_000.0, help="Chirp stop frequency in Hz."),
    fft_size: int = typer.Option(4096, help="FFT size for the comparison plots."),
    stft_window: int = typer.Option(1024, help="Samples per STFT window."),
    stft_overlap: int = typer.Option(768, help="Overlap between STFT windows."),
) -> None:
    """Generate a narrated chirp demo with FFT vs windowing and a spectrogram."""

    print_intro(fs, f_start, f_stop, duration)

    t = np.arange(int(fs * duration)) / fs
    slope = (f_stop - f_start) / duration
    inst_freq = f_start + slope * t
    chirp = np.exp(1j * 2 * np.pi * (f_start * t + 0.5 * slope * t**2))
    leakage_tone = 0.4 * np.exp(1j * 2 * np.pi * (29_500.0 * t))
    signal = chirp + leakage_tone

    typer.echo("🛠  Generating FFT views…")
    window = get_window("hann", fft_size, fftbins=True)
    sig_slice = signal[:fft_size]
    fft_rect = np.fft.fft(sig_slice, n=fft_size)
    fft_hann = np.fft.fft(sig_slice * window, n=fft_size)
    freqs = np.fft.fftfreq(fft_size, d=1.0 / fs)

    typer.echo("🛠  Computing STFT (think sliding FFT) to create a spectrogram…")
    f_s, t_s, stft_grid = stft(
        signal,
        fs=fs,
        window="hann",
        nperseg=stft_window,
        noverlap=stft_overlap,
        nfft=stft_window,
        boundary=None,
        return_onesided=False,
    )

    typer.echo(
        "📊 Plotting time trace, frequency sweep, FFT magnitudes, and spectrogram…"
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(t * 1e3, signal.real)
    axes[0, 0].set_title("Chirp (real part)")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("Amplitude")

    axes[0, 1].plot(t * 1e3, inst_freq / 1e3)
    axes[0, 1].set_title("Instantaneous frequency")
    axes[0, 1].set_xlabel("Time (ms)")
    axes[0, 1].set_ylabel("kHz")

    half = fft_size // 2
    axes[1, 0].plot(freqs[:half] / 1e3, mag_db(fft_rect[:half]), label="Rectangular")
    axes[1, 0].plot(freqs[:half] / 1e3, mag_db(fft_hann[:half]), label="Hann window")
    axes[1, 0].set_title("FFT magnitude comparison")
    axes[1, 0].set_xlabel("Frequency (kHz)")
    axes[1, 0].set_ylabel("Magnitude (dB)")
    axes[1, 0].legend(loc="upper right")

    mesh = axes[1, 1].pcolormesh(
        t_s * 1e3, f_s / 1e3, mag_db(stft_grid), shading="auto"
    )
    axes[1, 1].set_title("Spectrogram (STFT)")
    axes[1, 1].set_xlabel("Time (ms)")
    axes[1, 1].set_ylabel("Frequency (kHz)")
    fig.colorbar(mesh, ax=axes[1, 1], label="dB")

    plt.tight_layout()
    output_path = Path("out_01_fft_spectrogram.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    typer.echo(f"✅ Saved {output_path}")
    typer.echo("Key takeaways:")
    typer.echo(
        "  • Windowing suppresses leakage from off-bin tones at the expense of broader main lobes."
    )
    typer.echo(
        "  • STFTs reveal how frequency content evolves — nperseg↑ gives sharper frequency bins but blurrier time."
    )
    typer.echo(
        "  • Chirps underpin FMCW radar: the beat note after mixing is proportional to target range."
    )

    typer.echo(
        "Try options like `--stft-window 2048 --stft-overlap 1536` to experiment with time/frequency trade-offs."
    )


if __name__ == "__main__":
    app()
