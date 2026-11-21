# Theory behind the project

This note explains the signal model, each algorithm, and why the results look
the way they do. Notation: `M` microphones, spacing `d`, speed of sound
`c = 343 m/s`, frequency `f`, STFT frame `t`.

## 1. Signal model

A far-field source at angle `θ` (0° = broadside) reaches microphone `m` with a
delay

```
τ_m(θ) = m · d · sin(θ) / c
```

In the STFT domain a delay becomes a phase shift, so the recording is

```
X(f,t) = a(f,θ_s) S(f,t) + a(f,θ_i) I(f,t) + N(f,t),     a_m(f,θ) = exp(-j 2π f τ_m(θ))
```

`S` is the target talker, `I` the interfering talker, `N` the noise and
`a(f,θ)` the **steering vector**. This is the *narrowband approximation*: it is
valid when the frame is much longer than the largest propagation delay across
the array. Here the array is 15 cm, so the largest delay is 0.44 ms, against a
32 ms frame (512 samples at 16 kHz): the approximation holds in the
anechoic simulation. With reverberation it becomes weaker (see limits).

A beamformer outputs `y(f,t) = w(f)^H X(f,t)`. The **distortionless
constraint** `w^H a(f,θ_s) = 1` keeps the target unchanged.

## 2. The four methods

| Method | Weights | Idea |
|---|---|---|
| Delay-and-sum | `w = a / M` | Align the microphones on the target and average. Noise that is independent across microphones drops by `10·log10(M)` dB (6 dB for 4 mics). |
| MVDR (Capon, 1969) | `w = R_n⁻¹ a / (a^H R_n⁻¹ a)` | Minimise the output power of interference + noise (`R_n` is their spatial covariance) under the distortionless constraint. Places spatial nulls on the interferer. |
| MPDR | same formula with `R_x` | Same as MVDR but `R_x` is estimated from the noisy mixture only, so nothing has to be known about the noise: this is the practical version. |
| Post-filter | gain `G = max(1 - α·N̂/|Y|², g_min)` | Wiener-like spectral gain on the beamformer output; `N̂` is a per-frequency noise floor (20th percentile over time). |

**Derivation of MVDR in one line.** Minimise `w^H R_n w` subject to
`w^H a = 1` with a Lagrange multiplier: `R_n w = λ a`, so `w ∝ R_n⁻¹ a`, and the
constraint fixes the scale.

**Diagonal loading.** `R` is replaced by `R + ε·tr(R)/M · I`. It makes the
inversion stable when `R` is ill-conditioned and trades a little noise
suppression for robustness to steering errors (it increases the white noise
gain).

**Why MPDR works here but is fragile in practice.** `R_x` contains the target
too. If the steering vector is exact (as in this simulation), the constraint
protects the target. If the direction is wrong or the room is reverberant,
MPDR cancels the target ("signal cancellation"). Real systems estimate `R_n`
from noise-only frames (voice activity detection) or from neural time-frequency
masks, which is the main extension of this project.

## 3. Why delay-and-sum gives almost nothing here

The result table shows +0.5 dB for delay-and-sum against +13 dB for MPDR. Two
reasons:

1. **Beam width.** A beam is about `λ / (M·d)` wide. At 500 Hz the wavelength is
   0.69 m but the aperture is only 0.15 m, so the beam covers almost all
   directions: the interferer at 50° is barely attenuated at low frequency,
   where most speech energy lives.
2. **The error is dominated by the interferer.** With SIR = 0 dB, the
   interfering talker is as loud as the target. Delay-and-sum removes the
   uncorrelated noise (that part gains `10·log10(4) = 6 dB`) but not the
   directional interferer.

Adaptive beamformers fix this because they place a null exactly in the
interferer direction at every frequency, instead of relying on aperture.

## 4. Spatial aliasing

With spacing `d`, frequencies above `c / (2d)` (3.4 kHz for 5 cm) are spatially
undersampled and produce grating lobes. This is a reason why speech arrays use
spacings of 3 to 8 cm: wide enough for low-frequency directivity, narrow
enough to avoid aliasing in the speech band.

## 5. Metric: SI-SDR

```
α = <ŝ, s> / <s, s>,        SI-SDR = 10 · log10( ||α s||² / ||ŝ - α s||² )
```

It is the energy of the part of the estimate that is a scaled copy of the
reference, divided by the energy of everything else. It ignores a global gain
(unlike plain SNR) and is the standard metric for separation and enhancement.
The reference is the clean target as received by microphone 0.

## 6. Reading the experiments

- **Number of microphones** (`--sweep`): SI-SDR grows with `M` because the array
  gains degrees of freedom: `M` microphones can place up to `M - 1` nulls.
  With 1 microphone no spatial filtering is possible, so all methods equal the
  input (apart from the post-filter).
- **Oracle MVDR vs MPDR**: the gap (13.5 vs 12.7 dB at 4 mics) is the price of
  not knowing the noise covariance. It is small here because the steering is
  exact.
- **Post-filter**: +2 dB on top of MVDR, at the cost of possible "musical noise"
  and slight speech distortion (controlled by the floor `g_min`).

## References

- J. Capon, "High-resolution frequency-wavenumber spectrum analysis", Proc. IEEE, 1969.
- B. D. Van Veen and K. M. Buckley, "Beamforming: a versatile approach to spatial filtering", IEEE ASSP Magazine, 1988.
- J. Le Roux et al., "SDR - half-baked or well done?", ICASSP 2019 (SI-SDR).
