"""Light (SemiAnalysis-style) matplotlib theme for the blog charts."""
import logging
import matplotlib as mpl

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

COLORS = {
    "deepseek-r1":       "#D4574E",
    "qwen3.5-397b-a17b": "#E9A23B",
    "minimax-m2.7":      "#2A9D8F",
    "glm-5.1":           "#4E6E81",
    "kimi-k2.5":         "#8E6F9E",
}

MODEL_ORDER = ["deepseek-r1", "qwen3.5-397b-a17b", "minimax-m2.7", "glm-5.1", "kimi-k2.5"]

MODEL_DISPLAY = {
    "deepseek-r1":       "DeepSeek R1",
    "qwen3.5-397b-a17b": "Qwen 3.5 397B",
    "minimax-m2.7":      "MiniMax M2.7",
    "glm-5.1":           "GLM-5.1",
    "kimi-k2.5":         "Kimi K2.5",
}

MODEL_PARAMS = {
    "deepseek-r1":       {"total_b": 685, "active_b": 37},
    "qwen3.5-397b-a17b": {"total_b": 397, "active_b": 17},
    "minimax-m2.7":      {"total_b": 230, "active_b": 10},
    "glm-5.1":           {"total_b": 355, "active_b": 32},
    "kimi-k2.5":         {"total_b": 1000, "active_b": 32},
}

INK       = "#1A1A1A"
MUTED     = "#6B6B6B"
FAINT     = "#AAAAAA"
GRID      = "#E3E3E3"
POSITIVE  = "#2E8B57"
NEGATIVE  = "#C0392B"
HIGHLIGHT = "#1F4E79"
NEUTRAL   = "#9CA3AF"

PROFILE_DESC = {
    "1k1k": "1024 in / 1024 out",
    "1k4k": "1024 in / 4096 out",
    "4k1k": "4096 in / 1024 out",
}

FOOTER_DEFAULT = "Source: CoCloud AI — github.com/kenzhangwangshu/b300_benchmark"


def apply_theme():
    mpl.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
        "font.family":       ["Inter", "IBM Plex Sans", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "axes.titlecolor":   INK,
        "axes.labelsize":    10,
        "axes.labelcolor":   INK,
        "axes.edgecolor":    INK,
        "axes.linewidth":    0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "grid.color":        GRID,
        "grid.linewidth":    0.5,
        "grid.linestyle":    "--",
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "legend.frameon":    False,
        "legend.fontsize":   9,
        "lines.linewidth":   2.0,
        "lines.markersize":  5,
    })


def title_block(fig, title, subtitle=None, footer=FOOTER_DEFAULT):
    fig.text(0.035, 0.965, title, fontsize=15, fontweight="bold",
             color=INK, ha="left", va="top")
    if subtitle:
        fig.text(0.035, 0.918, subtitle, fontsize=10, color=MUTED,
                 ha="left", va="top")
    fig.text(0.035, 0.018, footer, fontsize=7.5, color=MUTED,
             ha="left", style="italic")
