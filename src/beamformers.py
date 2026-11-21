"""Frequency-domain beamformers and a simple post-filter."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import istft, stft

from simulate import steering_vector

NPERSEG = 512
NOVERLAP = 384


def to_stft(x, fs):
    """x: (M, N) -> freqs (F,), Z (M, F, T)."""
    f, _, Z = stft(x, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP, axis=-1)
    return f, Z


def from_stft(Y, fs, length):
    """Y: (F, T) -> time signal of the requested length."""
    _, y = istft(Y, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP)
    y = y[:length]
    return np.pad(y, (0, max(0, length - len(y))))


def _apply(w, Z):
    """Beamformer output y(f, t) = w(f)^H z(f, t)."""
    return np.einsum("fm,mft->ft", np.conj(w), Z)


def delay_and_sum(Z, a):
    """Align the microphones on the target direction and average."""
    w = a / a.shape[1]
    return _apply(w, Z)


def _mvdr_weights(R, a, loading):
    F, M = a.shape
    R = R + loading * (np.trace(R, axis1=1, axis2=2).real / M)[:, None, None] * np.eye(M)
    Ra = np.linalg.solve(R, a[..., None])[..., 0]            # R^-1 a
    denom = np.einsum("fm,fm->f", np.conj(a), Ra)
    return Ra / denom[:, None]


def _covariance(Z):
    T = Z.shape[2]
    return np.einsum("mft,nft->fmn", Z, np.conj(Z)) / T


def mpdr(Z, a, loading=1e-2):
    """Minimum-power distortionless response using the mixture covariance.

    Practical (no oracle): the covariance is estimated from the noisy signal.
    """
    return _apply(_mvdr_weights(_covariance(Z), a, loading), Z)


def mvdr_oracle(Z, Z_noise, a, loading=1e-3):
    """MVDR with the covariance of the interference + noise (upper bound)."""
    return _apply(_mvdr_weights(_covariance(Z_noise), a, loading), Z)


def postfilter(Y, alpha=1.0, floor=0.2, percentile=20):
    """Wiener-like spectral gain; noise floor tracked per frequency bin."""
    P = np.abs(Y) ** 2
    noise = np.percentile(P, percentile, axis=1, keepdims=True)
    gain = np.maximum(1.0 - alpha * noise / (P + 1e-12), floor)
    gain = uniform_filter1d(gain, size=3, axis=1)
    return Y * gain


def run_all(scene, theta_target):
    """Apply every method; returns {name: time-domain signal}."""
    fs = scene["fs"]
    n = scene["mixture"].shape[1]
    f, Z = to_stft(scene["mixture"], fs)
    _, Zn = to_stft(scene["interference"] + scene["noise"], fs)
    a = steering_vector(f, theta_target, scene["positions"])

    outputs = {"Mic 0 (input)": scene["mixture"][0]}
    ds = delay_and_sum(Z, a)
    mp = mpdr(Z, a)
    mv = mvdr_oracle(Z, Zn, a)
    outputs["Delay-and-sum"] = from_stft(ds, fs, n)
    outputs["MPDR"] = from_stft(mp, fs, n)
    outputs["MVDR (oracle cov.)"] = from_stft(mv, fs, n)
    outputs["MVDR + post-filter"] = from_stft(postfilter(mv), fs, n)
    return outputs
