#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extra orthonormal bases for residual sparsification in SIR-CS LFISTA.

Provides:
  - build_wavelet_basis(L, name): L x L orthonormal DWT matrix for
    'haar', 'db4', 'sym4' via pywt + periodization mode.
  - build_fd1_basis(L): L x L orthonormal basis obtained by QR-orthogonalizing
    [DC || forward difference rows]. First column is DC; remaining columns
    span the piecewise-variation subspace (TV-like).
  - orthonormality_error(Psi): max(|Psi.T @ Psi - I|) sanity check.

Convention: y = Psi @ alpha, alpha = Psi.T @ y.  L must be a power of 2 for
the wavelet bases.

ASCII-only.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _check_power_of_two(n: int) -> None:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(
            "Wavelet basis requires L power of 2; got L={}.".format(int(n))
        )


def build_wavelet_basis(length: int, name: str) -> np.ndarray:
    """
    Construct an orthonormal L x L DWT matrix using pywt with periodization.

    Parameters
    ----------
    length : int
        Signal length L. Must be a power of 2 (32, 64, 128, ...).
    name : str
        Wavelet short name. Supported here: 'haar', 'db4', 'sym4'.
        Any orthogonal wavelet with pywt 'periodization' mode is accepted.

    Returns
    -------
    Psi : np.ndarray
        L x L matrix such that y = Psi @ alpha and alpha = Psi.T @ y (numerical).
    """
    import pywt

    n = int(length)
    _check_power_of_two(n)
    key = name.strip().lower()
    wavelet = pywt.Wavelet(key)
    if not wavelet.orthogonal:
        raise ValueError(
            "Wavelet '{}' is not orthogonal; LFISTA expects orthonormal Psi.".format(key)
        )
    coeffs_template = pywt.wavedec(np.zeros(n), wavelet, mode="periodization")
    sizes = [int(c.size) for c in coeffs_template]
    total = int(sum(sizes))
    if total != n:
        raise ValueError(
            "wavedec with periodization returned {} coeffs, expected {}.".format(total, n)
        )
    psi = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        arrs = []
        t = 0
        for sz in sizes:
            arr = np.zeros(sz, dtype=np.float64)
            if t <= j < t + sz:
                arr[j - t] = 1.0
            arrs.append(arr)
            t += sz
        col = pywt.waverec(arrs, wavelet, mode="periodization")
        if col.size != n:
            col = col[:n] if col.size > n else np.concatenate([col, np.zeros(n - col.size)])
        psi[:, j] = col
    return psi


def build_fd1_basis(length: int) -> np.ndarray:
    """
    Orthonormal basis built from [DC || forward-difference rows], QR-orthogonalized.
    First column is 1/sqrt(L) (DC). Remaining columns span piecewise-variation.
    Not identical to Haar, but piecewise-constant signals remain very sparse here.
    """
    n = int(length)
    if n < 2:
        raise ValueError("fd1 basis requires L >= 2.")
    dc = np.ones((n, 1), dtype=np.float64) / np.sqrt(float(n))
    d_rows = np.zeros((n - 1, n), dtype=np.float64)
    for i in range(n - 1):
        d_rows[i, i] = -1.0
        d_rows[i, i + 1] = 1.0
    gen = np.concatenate([dc.T, d_rows], axis=0).T
    q, _ = np.linalg.qr(gen)
    if q.shape[1] < n:
        raise RuntimeError("fd1 generator matrix has rank < L; unexpected.")
    return q[:, :n]


def orthonormality_error(psi: np.ndarray) -> float:
    """Return max |Psi.T Psi - I| (scalar sanity metric)."""
    p = np.asarray(psi, dtype=np.float64)
    gram = p.T @ p
    return float(np.max(np.abs(gram - np.eye(p.shape[1]))))


SUPPORTED_WAVELETS: Tuple[str, ...] = ("haar", "db4", "sym4")
