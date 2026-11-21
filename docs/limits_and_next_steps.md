# Limits, honest caveats and next steps

## What the results do and do not show

- The scene is **anechoic** (no room reverberation) and the **direction of
  arrival is known**. Numbers such as +16 dB are therefore an upper range, not
  what a real room gives. In a reverberant room, a few dB is already good.
- The voices are **synthetic** (pitched source filtered by formants). They have
  speech-like spectra and syllable rhythm but not the variability of real
  speech. The code accepts real `.wav` files (`--target`, `--interferer`).
- The **noise is white and independent** per microphone. Real noise is
  spatially correlated (diffuse field), which makes noise harder to remove at
  low frequency.
- **MVDR (oracle)** uses the true interference + noise covariance: it is an
  upper bound, not a deployable method. MPDR is the deployable one.
- Only **one scene and one seed** are reported in the main table. The sweep
  over the number of microphones shows the trend, but a proper study would
  average over several seeds, angles and SIR values.

## Planned improvements (in order of value)

1. **Real data and rooms.** Use LibriSpeech or VCTK speech and simulate rooms
   with `pyroomacoustics` (image-source method, controllable RT60). Report
   SI-SDR and also PESQ/STOI, which relate to perceived quality and
   intelligibility.
2. **Direction-of-arrival estimation.** Replace the known angle by GCC-PHAT,
   SRP-PHAT or MUSIC, and measure how DOA errors degrade MPDR (signal
   cancellation).
3. **Covariance from masks.** Estimate `R_n` from a time-frequency mask
   (first with an energy-based VAD, then with a small neural network: U-Net or
   Conv-TasNet). This is the standard "mask-based MVDR" pipeline.
4. **Neural post-filter** instead of the percentile noise floor.
5. **Audio-visual extension.** Use the lip region of the target talker to
   decide who to extract, which is the multi-modal part of the internship topic
   (assistive listening: the listener looks at the person they want to hear).
6. **Robustness study.** Vary diagonal loading and steering mismatch, and plot
   the trade-off between noise suppression and target distortion.

## Likely interview questions and short answers

**Why does MVDR beat delay-and-sum with the same microphones?**
Delay-and-sum is data-independent: its directivity is fixed by the aperture. MVDR
adapts to the covariance and places nulls on interferers, so it can exceed the
fixed array's resolution in the interferer direction.

**Why not simply use the mixture covariance (MPDR) in a real room?**
Because the target is in the covariance, and any steering error or reverberation
makes the constraint inexact: the filter treats the target as interference and
cancels part of it. Hence diagonal loading, noise-only covariance estimates, or
mask-based estimation.

**What limits the number of nulls?**
`M - 1`. With 4 microphones, at most 3 directional interferers can be nulled.

**What is the unit of your main metric and why not SNR?**
SI-SDR in dB. It is invariant to the output gain, so a quieter or louder output
is not rewarded or punished, and it counts residual interference and distortion
together.

**Why STFT with 512 samples and 75 % overlap?**
32 ms is long enough for the narrowband approximation (array delays are below
0.5 ms) and short enough for speech to be quasi-stationary; 75 % overlap with a
Hann window satisfies the constant-overlap-add condition for clean
reconstruction.

**How would you handle a moving talker?**
Track the DOA (Kalman or particle filter on SRP-PHAT peaks) and use a
recursive covariance estimate with a forgetting factor instead of a
whole-utterance covariance.

**How would the audio-visual part help?**
The visual stream tells who is speaking and where the target is, even when
audio-only cues are ambiguous (two talkers at similar angles), and it can
provide a mask prior for the beamformer.
