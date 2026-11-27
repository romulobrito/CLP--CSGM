#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared matplotlib style for CLP-CSGM paper figures (ASCII-only)."""

from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt

PAPER_DPI = 300
PAPER_FONT_FAMILY = "sans-serif"
PAPER_FONT_SANS = ["DejaVu Sans"]


def paper_rcparams() -> Dict[str, Any]:
    """Return rcParams tuned for legible figures at 0.5--1.0 linewidth in LaTeX."""
    return {
        "font.family": PAPER_FONT_FAMILY,
        "font.sans-serif": PAPER_FONT_SANS,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
    }


def apply_paper_style() -> None:
    """Apply paper figure typography globally for the current matplotlib session."""
    plt.rcParams.update(paper_rcparams())


def save_paper_figure(fig: Any, path: str) -> None:
    """Save a figure with paper DPI and tight bounding box."""
    fig.savefig(path, dpi=PAPER_DPI, bbox_inches="tight")
