import numpy as np


def member_bootstrap_rmsd(preds, truth, mask, norm_const, idx):
    """
    preds: (M, T) member predictions; truth: (T,); mask: (T,) bool;
    idx: (B, M) int resampling indices shared across methods (paired bootstrap).
    Returns (point, boot) in percent; point equals the current bar value.
    """
    err2 = ((preds[:, mask] - truth[None, mask]) ** 2).mean(axis=1)  # per-member MSE, (M,)
    point = np.sqrt(err2.mean()) / norm_const * 100.0
    boot = np.sqrt(err2[idx].mean(axis=1)) / norm_const * 100.0
    return point, boot


def paired_difference(boot_method, boot_ref, level=95.0):
    """
    boot_method, boot_ref: (B,) bootstrap RMSDs computed with the same resampling indices.
    Returns (diff_ci_lo, diff_ci_hi, ref_better); ref_better is True when the reference
    has significantly lower RMSD, i.e. the lower bound of (method - ref) is above zero.
    """
    diff = boot_method - boot_ref
    tail = (100.0 - level) / 2.0
    lo, hi = np.percentile(diff, [tail, 100.0 - tail])
    return lo, hi, bool(lo > 0)
