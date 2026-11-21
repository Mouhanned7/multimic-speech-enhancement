"""Simulation of a multi-microphone recording (far-field, linear array).

The target talker and an interfering talker arrive from different directions.
Diffuse-like background noise is added independently on each microphone.
"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, resample_poly

SPEED_OF_SOUND = 343.0

# Rough formant frequencies (Hz) of a few vowels, used by the synthetic voice.
_VOWELS = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240),
           (530, 1840, 2480), (570, 840, 2410)]


def _resonator(x, freq, bw, fs):
    """Second-order resonator (one formant)."""
    r = np.exp(-np.pi * bw / fs)
    theta = 2 * np.pi * freq / fs
    b = [1 - r]
    a = [1, -2 * r * np.cos(theta), r * r]
    return lfilter(b, a, x)


def synth_voice(duration, fs=16000, f0=120.0, seed=0):
    """Speech-like signal: pitched source + formants, syllable by syllable."""
    rng = np.random.default_rng(seed)
    n_total = int(duration * fs)
    out = np.zeros(n_total + fs)
    pos = 0
    phase = 0.0
    while pos < n_total:
        n = int(rng.uniform(0.15, 0.30) * fs)
        base = f0 * rng.uniform(0.85, 1.2)
        f0_curve = base * (1 + 0.15 * np.sin(np.linspace(0, np.pi, n)))
        phases = phase + np.cumsum(f0_curve / fs)
        phase = phases[-1] % 1.0
        src = 2 * (phases % 1.0) - 1.0                      # sawtooth glottal source
        formants = _VOWELS[rng.integers(len(_VOWELS))]
        y = sum(_resonator(src, f, 80 + 0.06 * f, fs) for f in formants)
        y = y + 0.02 * rng.standard_normal(n)               # a little breathiness
        out[pos:pos + n] += y * np.hanning(n) ** 0.5
        pos += int(n * 0.85)
        if rng.random() < 0.25:                             # short pause
            pos += int(rng.uniform(0.05, 0.2) * fs)
    out = out[:n_total]
    return out / (np.max(np.abs(out)) + 1e-9)


def load_wav(path, fs=16000, duration=None):
    """Load a wav file as mono float at the target sampling rate."""
    rate, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(np.asarray(x).dtype, np.integer) or np.max(np.abs(x)) > 1:
        x = x / (np.max(np.abs(x)) + 1e-9)
    if rate != fs:
        x = resample_poly(x, fs, rate)
    if duration:
        x = x[: int(duration * fs)]
    return x


def mic_positions(n_mics, spacing):
    """Linear array along the x axis, first microphone at the origin."""
    return np.arange(n_mics) * spacing


def steering_vector(freqs, theta_deg, positions):
    """Far-field steering vector, shape (F, M). Angle 0 = broadside."""
    tau = positions * np.sin(np.deg2rad(theta_deg)) / SPEED_OF_SOUND   # (M,)
    return np.exp(-2j * np.pi * freqs[:, None] * tau[None, :])


def delay_signal(x, fs, theta_deg, positions):
    """Return the signal as received by each microphone, shape (M, N)."""
    n = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    a = steering_vector(freqs, theta_deg, positions)                    # (F, M)
    return np.fft.irfft(X[None, :] * a.T, n=n, axis=1)


def _power(x):
    return float(np.mean(x ** 2)) + 1e-12


def make_scene(target, interferer, fs, n_mics=4, spacing=0.05,
               theta_target=0.0, theta_interf=50.0, snr_db=10.0, sir_db=0.0,
               seed=0):
    """Build the multichannel mixture.

    Returns dict with mixture (M, N), per-mic target, interference and noise.
    SIR/SNR are measured at microphone 0 relative to the target.
    """
    rng = np.random.default_rng(seed)
    n = min(len(target), len(interferer))
    target, interferer = target[:n], interferer[:n]
    pos = mic_positions(n_mics, spacing)

    tgt = delay_signal(target, fs, theta_target, pos)
    itf = delay_signal(interferer, fs, theta_interf, pos)
    itf *= np.sqrt(_power(tgt[0]) / _power(itf[0]) / 10 ** (sir_db / 10))

    noise = rng.standard_normal((n_mics, n))
    noise *= np.sqrt(_power(tgt[0]) / _power(noise[0]) / 10 ** (snr_db / 10))

    return dict(mixture=tgt + itf + noise, target=tgt, interference=itf,
                noise=noise, reference=target, positions=pos, fs=fs)
