# Radar Zero → 60% Reflection

## Step 1 – FFT & Spectrogram Intuition

**Question:** What changed when you switched from a rectangular FFT to a Hann-windowed FFT, and why does this matter for radar processing?

**Answer:** The Hann window reduced spectral leakage compared to the rectangular window, producing cleaner peaks with fewer side lobes. This matters for radar processing because cleaner peaks make it easier to distinguish and identify targets, especially when multiple targets are close together in frequency. The trade-off is that the Hann window slightly widens the main lobe, but the reduction in leakage is more valuable for accurate target detection.

---

## Step 2 – IQ Mixing & FIR Filtering

**Question:** How did you verify that the 57 kHz RDS subcarrier moved to baseband and remained after filtering?

**Answer:** After shifting and filtering, a strong peak appeared at 0 kHz in the filtered spectrum, while most other frequencies were suppressed. This confirms that the 57 kHz RDS subcarrier was successfully moved to baseband (0 Hz) and isolated by the low-pass filter. The peak at 0 kHz in the final plot demonstrated that the frequency translation preserved the signal while removing unwanted components.

---

## Step 3 – FMCW Range FFT + CA-CFAR

**Question:** The script prints both the theoretical range resolution (ΔR) and detected ranges. How are they linked, and what happens to ΔR if you halve the sweep bandwidth?

**Answer:** ΔR (range resolution) determines the minimum distance between two targets for them to be detected separately. The detected ranges appear as distinct peaks when spaced farther apart than ΔR. Since ΔR = c/(2B), halving the sweep bandwidth B doubles ΔR (from 0.75m to 1.5m in this case), worsening resolution because targets need to be farther apart to be detected separately. More bandwidth provides better resolution by creating smaller, more distinct range bins.

---

## Step 4 – Real Range–Doppler + 2-D CA-CFAR

**Question:** If the 2-D CA-CFAR misses a target visible in the RD map, which parameter would you tune first and why?

**Answer:** I would decrease the scale factor first. The scale factor controls how much higher the signal must be above the estimated noise level to be detected as a target. Lowering it makes the threshold less strict, allowing weaker targets to be detected. The trade-off is that decreasing the scale factor might also increase false alarms from noise, so finding the right balance is important for reliable detection.

---

## Bonus (Optional)

**Question:** List one concept you would like to explore next (e.g., Doppler FFT, angle estimation, SigMF metadata).

**Answer:** I would like to explore angle estimation (Direction of Arrival) using multiple receive antennas. Understanding how radar can determine not just the range and velocity of a target, but also its angular position, would complete the picture of how radar creates a full 3D awareness of the environment. This would be particularly interesting for applications like autonomous vehicles and tracking systems.

---

## Key Takeaways

Through this assignment, I learned:

1. **FFT and windowing** are fundamental tools for analyzing signals in the frequency domain, with windowing providing cleaner target detection at the cost of slightly wider peaks.

2. **IQ (complex) signals** enable frequency shifting and preserve directional information (positive vs negative frequencies), which is critical for radar applications.

3. **FMCW radar** cleverly converts distance measurements into frequency measurements (beat frequencies), making range detection much easier than measuring tiny time delays directly.

4. **Range resolution** is fundamentally limited by bandwidth—more bandwidth means better ability to distinguish closely-spaced targets.

5. **CA-CFAR** (Constant False Alarm Rate) provides adaptive thresholding that balances target detection against false alarms, with different parameters offering different trade-offs between sensitivity and reliability.

6. **Real radar data** is significantly more challenging than simulations due to clutter, multipath, hardware imperfections, and environmental noise.

The progression from simulated chirps to real range-Doppler maps provided an intuitive understanding of how modern FMCW radar systems work, from the fundamental signal processing techniques to practical detection algorithms.