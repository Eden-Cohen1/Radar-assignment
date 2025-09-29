# Radar Zero → 60% Reflection

Instructions: copy this template to `report.md` and answer in 1–3 sentences per prompt.

## Step 1 – FFT & Spectrogram Intuition
- What changed when you switched from a rectangular FFT to a Hann-windowed FFT, and why does this matter for radar processing?

## Step 2 – IQ Mixing & FIR Filtering
- How did you verify that the 57 kHz RDS subcarrier moved to baseband and remained after filtering?

## Step 3 – FMCW Range FFT + CA-CFAR
- The script prints both the theoretical range resolution (ΔR) and detected ranges. How are they linked, and what happens to ΔR if you halve the sweep bandwidth?

## Step 4 – Real Range–Doppler + 2-D CA-CFAR
- If the 2-D CA-CFAR misses a target visible in the RD map, which parameter would you tune first and why?

## Bonus (Optional)
- List one concept you would like to explore next (e.g., Doppler FFT, angle estimation, SigMF metadata).
