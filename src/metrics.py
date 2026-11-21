"""Evaluation metrics."""
import numpy as np


def si_sdr(estimate, reference, eps=1e-12):
    """Scale-invariant SDR in dB (higher is better)."""
    estimate = estimate - np.mean(estimate)
    reference = reference - np.mean(reference)
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
    target = alpha * reference
    noise = estimate - target
    return 10 * np.log10((np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps))
