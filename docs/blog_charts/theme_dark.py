"""Dark (SA-style) matplotlib theme for the Substack blog charts."""
import logging
import matplotlib as mpl

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

BG        = "#1a1a2e"
PANEL     = "#1f1f38"
TEXT      = "#E0E0E0"
MUTED     = "#9CA3AF"
FAINT     = "#6B7280"
GRID      = "#2E2E46"
POSITIVE  = "#4ADE80"
NEGATIVE  = "#F87171"
NEUTRAL   = "#6B7280"

MODEL_COLORS = {
    "deepseek-r1":       "#FF6B6B",
    "qwen3.5-397b-a17b": "#4ECDC4",
    "minimax-m2.7":      "#45B7D1",
    "glm-5.1":           "#96CEB4",
    "kimi-k2.5":         "#FFEAA7",
}

MODEL_DISPLAY = {
    "deepseek-r1":       "DeepSeek R1",
    "qwen3.5-397b-a17b": "Qwen 3.5 397B",
    "minimax-m2.7":      "MiniMax M2.7",
    "glm-5.1":           "GLM-5.1",
    "kimi-k2.5":         "Kimi K2.5",
}

MODEL_ORDER = ["deepseek-r1", "qwen3.5-397b-a17b", "minimax-m2.7", "glm-5.1", "kimi-k2.5"]

FOOTER_DEFAULT = "Source: Cornerstone Cloud — github.com/BlacktraderKhan/b300_benchmark"


def apply_dark():
    mpl.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    BG,
        "savefig.facecolor": BG,
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
        "font.family":       ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size":         11,
        "text.color":        TEXT,
        "axes.labelcolor":   TEXT,
        "axes.edgecolor":    MUTED,
        "axes.linewidth":    0.8,
        "axes.titlecolor":   TEXT,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "grid.color":        GRID,
        "grid.linewidth":    0.45,
        "grid.linestyle":    "-",
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.facecolor":  BG,
        "legend.edgecolor":  MUTED,
        "legend.labelcolor": TEXT,
        "legend.frameon":    False,
        "legend.fontsize":   9,
        "lines.linewidth":   2.2,
        "lines.markersize":  6,
    })


def title_block(fig, title, subtitle=None, footer=FOOTER_DEFAULT):
    fig.text(0.035, 0.965, title, fontsize=14, fontweight="bold",
             color=TEXT, ha="left", va="top")
    if subtitle:
        fig.text(0.035, 0.918, subtitle, fontsize=10, color=MUTED,
                 ha="left", va="top")
    fig.text(0.035, 0.020, footer, fontsize=7.5, color=FAINT,
             ha="left", style="italic")
