# Multi-microphone speech enhancement

Extract one talker from a noisy multi-talker recording made with a small
microphone array. The project simulates a linear array, mixes a target talker,
an interfering talker and background noise, then compares classical
beamformers and a post-filter.

Context: preparation for a master's internship on multi-microphone and
multi-modal speech enhancement for assistive listening.

## Method

1. **Scene simulation** (`src/simulate.py`): far-field linear array (default 4 mics,
   5 cm spacing), target at 0°, interferer at 50°, SIR 0 dB, SNR 10 dB, 16 kHz.
   Voices are synthetic (pitched source + formants) or your own `.wav` files.
2. **STFT** (512-sample Hann window, 75 % overlap).
3. **Beamformers** (`src/beamformers.py`):
   - Delay-and-sum (steer toward the target and average).
   - MPDR: minimum-power distortionless response, covariance estimated from the
     noisy mixture only (practical, no oracle).
   - MVDR with the covariance of interference + noise (oracle upper bound).
4. **Post-filter**: Wiener-like gain with a per-frequency noise floor.
5. **Metric**: SI-SDR against the clean target (`src/metrics.py`).

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_demo.py --sweep
```

With your own recordings (mono or stereo wav, any sampling rate):

```bash
python src/run_demo.py --target talker.wav --interferer other.wav
```

Useful options: `--mics 4 --spacing 0.05 --theta-target 0 --theta-interf 50 --snr 10 --sir 0`.
Outputs go to `results/` (wav files, `spectrograms.png`, `results.md`, `sweep_mics.*`).

## Results (default scene, 4 microphones)

| Method | SI-SDR (dB) | Gain vs input (dB) |
|---|---|---|
| Mic 0 (input) | -0.51 | +0.00 |
| Delay-and-sum | 0.01 | +0.52 |
| MPDR | 12.73 | +13.25 |
| MVDR (oracle cov.) | 13.49 | +14.01 |
| MVDR + post-filter | 15.48 | +16.00 |

Effect of the number of microphones (SI-SDR, dB):

| Mics | Delay-and-sum | MPDR | MVDR (oracle) | MVDR + post-filter |
|---|---|---|---|---|
| 2 | -0.26 | 4.26 | 7.22 | 9.03 |
| 4 | 0.01 | 12.73 | 13.49 | 15.48 |
| 6 | 0.31 | 16.09 | 16.29 | 18.12 |
| 8 | 0.69 | 17.85 | 18.01 | 19.65 |

Delay-and-sum barely helps with a 15 cm array: its beam is too wide at low
frequencies, while adaptive beamformers place a null on the interferer.

## Documentation

- [docs/theory.md](docs/theory.md): signal model, MVDR derivation, why delay-and-sum fails here, spatial aliasing, SI-SDR.
- [docs/limits_and_next_steps.md](docs/limits_and_next_steps.md): honest caveats, roadmap, and answers to likely technical questions.

## Limits and next steps

- Anechoic simulation (no reverberation) and known directions of arrival.
- Synthetic voices: re-run with real speech (LibriSpeech, VCTK) and a room
  impulse response simulator (e.g. `pyroomacoustics`).
- Replace the post-filter by a neural mask estimator (e.g. a small U-Net or
  Conv-TasNet) and estimate the target direction automatically.
- Add the visual modality (lip region) for audio-visual enhancement.
