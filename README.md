# radar-0-to-60-undergrad-cs

> Zero-to-sixty radar processing primer for Python-only undergrads (CLI + VS Code + internet, no hardware, no admin rights).

## Setup with UV (≈15 minutes)

Install [uv](https://docs.astral.sh/uv/) once per workstation.

- **macOS / Linux**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Windows (PowerShell)**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

Project bootstrap:

```bash
mkdir radar-zero-to-60 && cd radar-zero-to-60
uv python install 3.12 --default
uv venv
uv pip install numpy scipy matplotlib h5py sigmf typer
```

Open the folder in VS Code and pick the `.venv` interpreter (`Python: Select Interpreter`).

> `uv run <command>` executes inside the project environment; `uv pip` installs packages at lightning speed.

Every tutorial script uses [Typer](https://typer.tiangolo.com/) for a friendly CLI. Run `uv run python <script>.py --help` to see adjustable parameters; the scripts narrate what each processing stage means so an FFT newcomer can follow along.

---

## Step 1 — FFT & Spectrogram Intuition (Simulated) — ~1.5 h

- **Purpose:** Build intuition for sampling, leakage, windowing, and STFT resolution trade-offs.
- **Tasks:**
  1. Synthesize a linear chirp plus a near-bin complex tone.
  2. Visualise the chirp in time and instantaneous frequency.
  3. Compare FFTs with and without a Hann window (see leakage vs sidelobes).
  4. Generate a spectrogram (STFT) and experiment with `nperseg`/`noverlap` via CLI flags.
- **Run:**
  ```bash
  uv run python 01_fft_basics.py
  uv run python 01_fft_basics.py --help  # optional knobs
  ```
- **Deliverable:** `out_01_fft_spectrogram.png` (time trace, inst. frequency, FFT comparison, spectrogram).
- **Sanity checks (intuition):**
  - Larger `nperseg` narrows bins but blurs time; smaller does the opposite.
  - Hann window tames leakage but slightly widens the main lobe.
  - A tone near a bin edge leaks badly without tapering.

---

## Step 2 — IQ Mixing & FIR Filtering on Real FM/RDS IQ — ~2.5 h

- **Purpose:** Practice loading complex IQ, frequency shifting via mixing, and FIR low-pass filtering.
- **Tasks:**
  1. Download a public FM/RDS capture (non-login).
  2. Visualize PSD before processing.
  3. Mix by `-57 kHz` to center the RDS subcarrier at DC and FIR-LPF to isolate it.
  4. Plot PSDs before/after and a spectrogram of the filtered baseband.
- **Data download:**
  ```bash
  mkdir -p data
  curl -L -o data/fm_rds_250k_1Msamples.iq \
    https://raw.githubusercontent.com/777arc/498x/master/fm_rds_250k_1Msamples.iq
  ```
- **Run:**
  ```bash
  uv run python 02_iq_ops.py
  uv run python 02_iq_ops.py --help  # optional knobs
  ```
- **Deliverables:**
  - `out_02_raw_psd.png`
  - `out_02_shift_psd.png`
  - `out_02_lpf_psd.png`
  - `out_02_lpf_spectrogram.png`
- **Sanity checks (intuition):**
  - After mixing, the RDS spike should sit near 0 Hz; the LPF should leave a narrow baseband.
  - Multiplying by `exp(-j2πf0t)` translates spectra by `f0` (frequency translation).
  - Label axes carefully (Hz vs kHz) to avoid unit mistakes.

---

## Step 3 — Hello FMCW: Range FFT + 1-D CA-CFAR (Simulated) — ~3 h

- **Purpose:** Connect FFT intuition to FMCW ranging and adaptive detection.
- **Tasks:**
  1. Use the FMCW relations to turn two target ranges into beat frequencies.
  2. Synthesize the dechirped beat signal as the sum of those tones plus complex AWGN.
  3. Window and FFT the beat to get a range spectrum, then run 1-D CA-CFAR.
  4. Export both the annotated PNG and a CSV listing the detected range bins.
- **Run:**
  ```bash
  uv run python 03_fmcw_range_cfar.py
  uv run python 03_fmcw_range_cfar.py --help  # explore CFAR/target knobs
  ```
- **Deliverables:** `out_03_range_cfar.png`, `out_03_detections.csv` (per-range CSV); console prints theoretical `ΔR` and detected ranges.
- **Key relations (with intuition):**
  - `R = c * f_b / (2 * S)` where `S = B / Tc`. (Beat frequency grows with round-trip delay.)
  - `ΔR = c / (2 * B)`. (More bandwidth squeezes range bins closer together.)
- **Sanity checks (intuition):**
  - Halving `B` roughly doubles `ΔR`.
  - Detections land near simulated ranges (± one resolution bin).
  - Adjusting CFAR scale/guard/train trades misses vs false alarms.

---

## Step 4 — Real Range–Doppler (SiMeRAD60) + 2-D CA-CFAR — ~2.5 h

- **Purpose:** Work with measured range–Doppler (RD) frames, visualize them, and overlay 2-D CFAR detections.
- **Tasks:**
  1. Download the SiMeRAD60 HDF5 dataset (TI AWR6843ISK, CC BY 4.0).
  2. Auto-discover an RD dataset within the file.
  3. Plot three RD frames in dB.
  4. Run a simple 2-D CA-CFAR (rectangular training ring) and overlay detections.
- **Data download:**
  ```bash
  mkdir -p data
  curl -L -o data/simerad60.hdf5 \
    "https://zenodo.org/records/14916564/files/data.hdf5?download=1"
  ```
- **Run:**
  ```bash
  uv run python 04_rd_realdata.py
  uv run python 04_rd_realdata.py --help  # tweak CA-CFAR windowing
  ```
- **Deliverables:** `out_04_rd_frame0.png`, `out_04_rd_frame1.png`, `out_04_rd_frame2.png`.
- **Sanity checks (intuition):**
  - Stationary clutter clusters near zero Doppler; movers appear at ± Doppler bins.
  - CFAR parameters control the false alarm vs miss balance.
  - Colorbar + shared dB range make it easy to compare frames at a glance.
  - HDF5 behaves like folders (groups) containing NumPy-like datasets.

---

## Submission Checklist & Quick Reflection

Keep your generated outputs handy by copying them into an `artifacts/` folder (already ignored by git). Turn in:

- `out_01_fft_spectrogram.png`
- `out_02_raw_psd.png`, `out_02_shift_psd.png`, `out_02_lpf_psd.png`, `out_02_lpf_spectrogram.png`
- `out_03_range_cfar.png`, `out_03_detections.csv`
- `out_04_rd_frame0.png`, `out_04_rd_frame1.png`, `out_04_rd_frame2.png`
- Copy `report_template.md` to `report.md` and answer the reflections below (1–3 sentences each is plenty).

Reflection prompts:
1. Step 1: What changed when you switched from the rectangular FFT to the Hann-windowed FFT, and why does that matter for radar processing?
2. Step 2: How did you confirm that the 57 kHz RDS subcarrier was shifted to baseband and preserved by the low-pass filter?
3. Step 3: The script prints both the theoretical range resolution and the detected ranges. How are they linked, and what happens to ∆R if you halve the sweep bandwidth?
4. Step 4: If the 2-D CA-CFAR misses a target you can see in the RD map, which parameter would you tweak first and why?

---

## Folder Layout

```
radar-zero-to-60/
  data/
  01_fft_basics.py
  02_iq_ops.py
  03_fmcw_range_cfar.py
  04_rd_realdata.py
  report_template.md
  README.md
```

---

## UV Cheat-Sheet

```bash
# once per machine/project
uv python install 3.12 --default
uv venv
uv pip install numpy scipy matplotlib h5py sigmf typer

# per step
uv run python 01_fft_basics.py
uv run python 02_iq_ops.py
uv run python 03_fmcw_range_cfar.py
uv run python 04_rd_realdata.py
```

## Formatting

Use [ruff](https://docs.astral.sh/ruff/) via uv to keep the Python files tidy:

```bash
uvx ruff format
```

`uvx` will fetch ruff on demand and format every script in-place.

---

## Troubleshooting (frequent snags)

- **Plots look blank:** Confirm axes/units (Hz vs kHz, bins vs meters, Doppler sign convention).
- **Spectrogram smeared:** Lower `nperseg` for better time resolution or raise it for frequency clarity.
- **FFT leakage too high:** Apply a window (Hann) or increase FFT length.
- **CFAR too chatty or too blind:** Tune `scale`, `train`, and `guard` parameters.
- **HDF5 dataset missing:** Use the script’s auto-discovery or inspect the file with `h5ls -r data/simerad60.hdf5`.
- **PowerShell execution policy blocks uv installer:** Copy the provided installer command exactly.

---

## Glossary (15 core terms)

- **Aliasing:** High-frequency content folding into lower frequencies when sampling too slowly.
- **IQ / complex baseband:** Complex samples (I + jQ) after mixing RF to baseband; enables clean shifts and filtering.
- **Window (Hann):** Time-domain taper applied before FFT to reduce leakage at a small resolution cost.
- **FFT bin:** Discrete frequency bucket produced by an FFT.
- **STFT / spectrogram:** Sliding short-time FFTs; `nperseg` trades time vs frequency resolution.
- **Mixing / frequency translation:** Multiplying by `exp(±j2πf0t)` to shift spectra.
- **FIR filter:** Linear-phase filter with finite taps (e.g., designed with `firwin`).
- **FMCW chirp:** Linear frequency sweep; slope `S = B / Tc`.
- **Slope (S):** Chirp frequency slope; maps beat frequency to range.
- **Range mapping:** `R = c * f_b / (2 * S)`.
- **Range resolution (ΔR):** `ΔR = c / (2 * B)`.
- **Range–Doppler map:** 2-D FFT (fast-time → range, slow-time → Doppler/velocity).
- **Doppler:** Frequency shift proportional to radial velocity.
- **CFAR:** Adaptive thresholding using neighboring training cells and guard cells to hold constant false alarm rate.
- **CUT:** Cell under test inside the CFAR stencil.
- **Guard cells:** Samples around the CUT excluded from noise estimation.
- **Training cells:** Samples used to estimate local noise level.
- **HDF5:** Hierarchical data format; groups like folders, datasets like arrays.
- **SigMF:** Open metadata standard (and Python library) for IQ recordings.

---

## 60% Ready Rubric

A learner at “60% ready” can:

- Select appropriate FFT/STFT parameters and explain leakage vs resolution.
- Load IQ data, frequency-shift to baseband, FIR-filter, and annotate spectra correctly.
- Build an FMCW range FFT with 1-D CA-CFAR, citing `ΔR = c/(2B)` and `R = c*f_b/(2S)`.
- Read HDF5 range–Doppler data, generate visuals, and tune a basic 2-D CA-CFAR.
- Reason why increasing `B` improves range resolution and why Doppler corresponds to velocity.

---

## References (verified September 2025)

- uv install & CLI docs — https://docs.astral.sh/uv/getting-started/installation/
- uv scripts guide — https://docs.astral.sh/uv/guides/scripts/
- SciPy STFT — https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html
- TI mmWave fundamentals & FAQ — https://www.ti.com/lit/pdf/sszt906 and https://e2e.ti.com/support/sensors-group/sensors/f/sensors-forum/1050220/faq-computing-maximum-range-velocity-and-resolution-mmwave-system
- FM/RDS IQ capture — https://github.com/777arc/498x/blob/master/fm_rds_250k_1Msamples.iq
- SiMeRAD60 dataset (Zenodo) — https://zenodo.org/records/14916564
- h5py quick-start — https://docs.h5py.org/en/stable/quick.html
- SigMF docs — https://sigmf.readthedocs.io/en/latest/

---

Happy radar hacking! Predict your plots before hitting enter, and tweak one knob at a time.
