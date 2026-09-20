# Light pole detector ideas
This is a summary of ideas output by GLM and deepseek. See archive for the raw
outputs.

# ideas notes
## distance-robust algs, ignoring hardware
- Ordered pixel-level → structure-level → temporal, can compose, do hybrid

### DONE: P1. Emitter gating (photometric prior) hue_gate.py
Gate the hue test with brightness and saturation floors instead of hue alone:

- Pixel passes iff hue ∈ {green, pink, blue windows} **and** S ≥ s₀ **and** V ≥
  v₀.
- Adaptive variants when margins tighten: Otsu threshold on V *within* the hue
  mask, or "keep the brightest k% of hue-matched pixels".

Maybe weaknesses: LED color differences, background brightness/reflections,
bloom/desaturation of very near LEDs (a soft score, P2, handles the transition).

### DONE: P2. Soft colour scoring (no hard thresholds) (glm, dspro) soft_gate.py
Replace the _hue_mask check from hue_gate.py with a membership weight w ∈
[0, 1] per pixel — e.g. trapezoid falls in H, S and V around each pole colour —
and compute the centroid as weighted moments `Σx·w / Σw`. No pixel is
hard-zeroed at a boundary, so the detector degrades gracefully instead of
falling off a cliff; subpixel accuracy comes free. Pairs naturally with P1 (the
S/V terms *are* the emitter gate).

Weakness: low-SNR distant poles need their faint weights *accumulated* over a
region (P3) before they can be trusted pixel-by-pixel. Still need tuning of HSV
filter values.

### DONE: P3. Vertical projection (1-D aggregation) column_proj.py
Sum the (gated, soft) response down each column: `h[x] = Σ_y w(x, y)`. A
vertical pole produces a sharp 1-D peak; a 1-px-wide distant pole still
contributes ~8 counts (one per LED) that no erosion can delete. Take the peak
band, then compute centroid x from moments of `h` and centroid y from row
moments within the band. This replaces morphology + contours + 2-D moments with
two 1-D histograms, and is the cheapest way to exploit the vertical prior.

Weakness: other vertical colored structures (bollards, lit signs) also peak —
needs verification (P5).

### SKIPPED P4. Scale-space blob detection (LoG / DoG / MSER)
The textbook answer to "blobs of unknown size". However, too intensive to run
real-time.

### P5. Structural matched filter / constellation matching
Exploit the known structure of the light pole - 8 evenly spaced colored blobs on
a near-vertical line. Applicable techniques:

- run a 'vertical blob stack' filter over the image, output the peaks
    - hardware FFT-based correlation makes this cheap
- Hough-style voting: candidate blob votes for all `(x, spacing)` hypotheses
  consistent with it; peaks in the 2-D accumulator are poles
- Periodicity check: autocorrelation (or FFT) of the row projection inside the
  pole band reveals the LED spacing; a clean periodicity is a strong confidence
  signal.

Weakness: needs ≥ 3–4 visible LEDs and a roughly known aspect (near-vertical).
This is acceptable.

### P6. Learned detector (tiny CNN / landmark regressor)
Worth mentioning, but too many drawbacks:

- labelled images for training
- training per environment
- failure modes are not interpretable
- maybe too computationally intensive to run on target hardware

### P7. Temporal integration and tracking
Kalman/α-β filter to track the ROI. Something to think about later, maybe.

### P8. Scale-aware morphology + contour retention
Discounted idea from dspro. Meh.

### P9. Geometric vertical-line fit, hough, ransac
Look for vertical lines with hough or ransac. Dspro idea. Discounted due to
better vertical ideas already existing, plus these are expensive (?).

### P10. Run-length / scanline state machine

For each column, walk down the binary pixels and record alternating run
lengths: `on₁, off₁, on₂, off₂, …`. A true pole column has ~8 on-runs of
similar width with off-gaps of similar length. Score a column by the number of
alternations whose run lengths fall within tolerance of each other (e.g. all
on-runs within 2× of the median).

Cheap variant: count only 0→1 transitions per column (`mask[y] & !mask[y-1]`
accumulated down each column — one pass, integer adds, SIMD-friendly). Columns
through a pole score ~8; columns through a single blob score 1.

**Pros:** the single cheapest option; pure integer arithmetic, single pass,
zero extra memory beyond a per-column accumulator; naturally gives you the x
extent of the pole; extends trivially to tilted poles (track x drift while
scanning).
**Cons:** sensitive to threshold noise splitting/merging runs — needs a small
morphological open/close first, or hysteresis on run lengths; purely columnar,
so it says nothing about global alignment across columns (a horizontal row of
8 LEDs side by side would score equally — combine with an x-alignment check).

### P11. Morphological alternating-structure filters

Search for the light/dark/light rhythm directly with structuring elements:

- **Hit-or-miss transform** with vertical kernels like
  `[1;0;1]`, `[1;0;0;0;1]`, `[1;0;1;0;1]` at several pitches — an erosion of
  the mask with a sparse vertical kernel keeps only pixels that have lit LEDs
  at exactly that vertical separation. Union the responses across pitches;
  pole regions survive, isolated blobs do not.
- **Shifted-XOR trick:** `score[y,x] = mask[y,x] & ~mask[y-s,x]` for each trial
  pitch `s` highlights stripe boundaries; accumulate per column to get an
  alternation score. Equivalent to a 1-tap bandpass, one AND + one shift per
  pixel.
- **Alternate sequential filtering:** close then open with a vertical line
  element whose length ≈ LED pitch suppresses both single blobs and solid
  vertical bars, leaving only the periodic stack.

**Pros:** fixed-point trivial, maps well to SIMD, output is still a mask you
can centroid.
**Cons:** pitch varies with distance, so you need several kernel sizes (each
is another pass); brittle if the pole is tilted relative to the kernel axis.

### P12. Integral-image Haar-like / box-pair features

Build an integral image of the mask once (O(W·H) adds), then any rectangular
sum is O(1). Define a vertical comb response at scale `s`:

```
C(x, y, s) = Σ(light boxes at y, y+s, y+2s, …) − Σ(dark boxes between)
```

i.e. subtract the inter-LED gaps from the LED rows over a window. Scan `(x, y,
s)` over a small scale range; maxima of |C| are pole candidates. This is the
same family of features Viola–Jones uses, and exactly what PIE SIMD + integral
images are good at.

**Pros:** O(1) per candidate window after one integral pass; multi-scale for
free; response value doubles as a confidence score; very natural in C.
**Cons:** rectangular boxes straddle perspective tilt; needs a coarse-to-fine
scale search; response magnitude depends on blob width, normalise by window
area.

### P13. 1-D Gabor / oriented bandpass filtering

Filter the mask (or the original grayscale) with a Gabor kernel elongated
vertically and modulated at the LED pitch — a matched filter for "vertical
stripes at frequency f". Sweep a few pitches; the filter energy localises the
pole. A cheap approximation: box-filter the mask vertically with length = pitch
and subtract a version box-filtered with length = pitch/2 — a poor-man's
bandpass using only running sums.

**Pros:** optimal linear detector for a known periodic pattern in noise;
smooth response degrades gracefully with partial occlusion.
**Cons:** floating-point MACs per pixel per scale (ESP32-S3 FPU is
single-precision only — use the integer box-filter approximation instead);
more tuning parameters (frequency, bandwidth, orientation).

### P14. Template matching over scale

Correlate the thresholded mask with a binary "comb" template — 8 horizontal
bars at the LED pitch — swept over y and over a small set of pitches (and
optionally small tilts). Normalised cross-correlation or a plain count of
overlapping set pixels both work on binary images (popcount, not multiply).

**Pros:** conceptually simple, directly encodes the known geometry (8 LEDs),
score is interpretable.
**Cons:** naive sweep is O(H · pitches) per column band — fine when the column
band has already narrowed x; template shape must adapt to distance/perspective
or scores smear; more expensive than options 1–3.

### P15. Connected components + geometric grouping

`cv2.connectedComponents` on the mask, then test groups of blobs for the pole
signature: x-centroids aligned within tolerance, ≥ 3 blobs, vertical spacing
consistent (fit `y = y₀ + k·s` robustly, e.g. RANSAC over pairs), monotonic
sizes. Output the fitted line's centroid.

**Pros:** uses the strong prior that the object is a *stack of discrete
blobs*, not just any stripes; immune to the "any vertical stripes" false
positives; labels are also useful for the colour classifier downstream.
**Cons:** union-find labelling on an MCU is more code than the 1-D methods;
merged blobs (close range) and single-pixel blobs (far range) break the
assumptions; needs care with the "≥ 3 of 8 visible" case.

### P16. 2-D FFT / spectral analysis

Take the 2-D FFT of a window; a vertical comb has energy concentrated on the
horizontal-frequency axis at the LED pitch. The ratio of energy at the
expected structure to total energy is a scale/rotation-tolerant score.

**Pros:** fully handles unknown pitch and modest tilt in one transform;
principled confidence metric.
**Cons:** far too heavy for the ESP32-S3 at frame rate, and 2-D spectra are
overkill for a 1-D periodicity problem. Listed mainly for completeness /
offline analysis of your test photos.



## suited to target hardware
### Hardware facts that shape the design
- single precision FPU: floats are fine, avoid doubles
- PIE SIMD (128-bit vectors, 8/16/32-bit lanes, MAC/saturating ops), esp-dsp
  primitives (vector ops, dot products, FIR, FFT, Kalman) are well suited to
  some of the P-ideas above.
- ~512 KB internal SRAM (fast) + 8 MB PSRAM (big, higher latency) — frame
  buffers live in PSRAM; per-pixel passes pay PSRAM bandwidth.
- OV3660 via esp32-camera: sensor-level binning/skipping when requesting smaller
  frame sizes; RGB565/YUV422 output (no JPEG decode needed in the detection
  path); AEC/AWB are controllable.

GLM recommends: The binding constraint is memory bandwidth and per-pixel work,
not arithmetic sophistication. Do exactly one full-frame pass; integer or FP32
only; small working state in SRAM; deterministic latency; no library calls
beyond ESP-IDF/esp-dsp.

### E1: streaming classifier + 1-D accumulators
The embedded embodiment of P1+P3, with no mask buffer at all:

1. **Classify.** Per pixel (RGB565): look up a colour-class byte in a small
   lookup table (LUT) indexed by quantized RGB (12-bit index → 4 KB table in
   flash), where the LUT encodes "green/pink/blue *and* bright *and* saturated".
   SIMD compares on the 8/16-bit lanes are the vectorized alternative; both are
   a few cycles/px.
2. **Accumulate while streaming.** For each matched pixel add to per-column
   SRAM accumulators: `count[x] += 1`, `sumY[x] += y`. Two `W`-entry int32
   arrays (5 KB at VGA) — the whole detector state fits in internal SRAM.
3. **Reduce.** Column score `h[x] = count[x]`; light 1-D smoothing (esp-dsp
   Finite impulse response (FIR)) for tilt tolerance; peak-pick; centroid x from
   moments of `h`, centroid y from `Σ sumY / Σ count` in the peak band.
4. **Verify (E3) and track (E4).**

Cost: one pass over QVGA (77 k px) is ~1–2 ms at 240 MHz; VGA ~5–10 ms —
comfortably inside a 10–30 fps budget, on one core. No morphology, no contours,
no connected components, nothing with a hidden scale parameter: distance
robustness comes from the photometric gate (P1) and 1-D aggregation (P3), both
of which are size-free.

### E2: camera-side conditioning (free performance)
Because the target is an *emitter*, sensor configuration replaces a lot of
algorithm:

- **Reduce exposure** until the background is near-black and only LEDs (and sky)
  survive — the mask becomes the pole, at any distance. The OV3660's AEC is
  controllable from esp32-camera.
- **Lock AWB.** Auto white balance silently shifts LED hues between frames,
  which breaks fixed hue windows; freeze the gains.
- **Fix the working resolution** via sensor binning (e.g. QVGA/VGA stream) and,
  once tracking (E4) has a lock, crop a small ROI around the prediction — a
  digital zoom that puts more pixels on a distant pole.

### E3: structural verification, integer edition
On the 1-D histograms only (cheap): count row-bands inside the pole band and
require ≥ 5 bands with roughly even spacing (integer gap comparison), or run a
256-point FFT/autocorrelation of the row projection (esp-dsp) to detect the LED
spacing. Rejects single blobs, vertical impostors, and yields a confidence
output. All state is a few hundred bytes.

### E4: temporal tracking (esp-dsp Kalman)
Later, maybe

Track `(x, y, scale)` across frames with the esp-dsp Kalman filter; gate
acquisition vs tracking; coast through dropouts; declare loss to re-acquire
full-frame. After lock, E1 runs on a small ROI only, multiplying the effective
frame rate or allowing higher resolution on target.
