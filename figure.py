"""Generate the architecture figure for the JOSS paper.

Run: python figure.py
Output: figure.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")

# Colours
c_io = "#3498db"
c_solver = "#e74c3c"
c_output = "#2ecc71"
c_python = "#9b59b6"
c_bg = "#ecf0f1"

def box(x, y, w, h, text, color, fontsize=9, bold=False):
    rect = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="white", linewidth=2, alpha=0.9
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="white", fontweight=weight,
            linespacing=1.4)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.15",
                                color="#7f8c8d", lw=2))

# Title
ax.text(5, 6.7, "OpenImpala Architecture", ha="center", va="center",
        fontsize=14, fontweight="bold", color="#2c3e50")

# ---- Python API layer (top) ----
box(1.5, 5.8, 7, 0.7, "Python API:  import openimpala\n"
    "Session  |  tortuosity()  |  volume_fraction()  |  percolation_check()",
    c_python, fontsize=9, bold=True)

# Arrow down
arrow(5, 5.8, 5, 5.4)

# ---- I/O Layer ----
box(0.3, 4.2, 2.4, 1.1,
    "I/O Layer\n\nTIFF | HDF5\nRAW | DAT",
    c_io, fontsize=9, bold=False)

# ---- Solver Layer ----
box(3.1, 4.2, 3.8, 1.1,
    "Physics Solvers\n\nHYPRE (Krylov + AMG)\nAMReX MLMG (matrix-free)",
    c_solver, fontsize=9, bold=False)

# ---- Output Layer ----
box(7.3, 4.2, 2.4, 1.1,
    "Output Layer\n\nJSON (BPX)\nCSV | Plotfiles",
    c_output, fontsize=9, bold=False)

# Arrows between layers
arrow(2.7, 4.75, 3.1, 4.75)
arrow(6.9, 4.75, 7.3, 4.75)

# ---- Transport Properties (middle) ----
box(0.3, 2.6, 4.3, 1.2,
    "Transport Properties\n\n"
    "Tortuosity factor  |  D_eff tensor\n"
    "Multi-phase transport  |  Percolation",
    "#e67e22", fontsize=9)

box(5.0, 2.6, 4.7, 1.2,
    "Microstructural Metrics\n\n"
    "Volume fraction  |  SSA  |  PSD\n"
    "Connected components  |  REV study",
    "#e67e22", fontsize=9)

# Arrows down from solvers
arrow(3.5, 4.2, 2.5, 3.8)
arrow(5.0, 4.2, 5.0, 3.8)
arrow(6.5, 4.2, 7.3, 3.8)

# ---- AMReX / HPC layer (bottom) ----
box(0.3, 1.0, 9.4, 1.1,
    "AMReX Infrastructure:   MPI + OpenMP + CUDA\n"
    "iMultiFab (voxel data)  |  BoxArray (domain decomposition)  |  "
    "Geometry  |  BL_PROFILE timers",
    "#34495e", fontsize=9, bold=True)

# Arrows down to AMReX
arrow(2.5, 2.6, 2.5, 2.1)
arrow(7.3, 2.6, 7.3, 2.1)

# ---- External data (left of I/O) ----
ax.text(0.15, 5.55, "3D voxel\nimages", ha="center", va="center",
        fontsize=8, color="#7f8c8d", style="italic")
arrow(0.15, 5.3, 0.8, 4.9)

# ---- Downstream (right of output) ----
ax.text(9.85, 5.55, "PyBaMM\nBPX", ha="center", va="center",
        fontsize=8, color="#7f8c8d", style="italic")
arrow(9.2, 4.9, 9.85, 5.3)

plt.tight_layout()
plt.savefig("figure.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("figure.pdf", bbox_inches="tight", facecolor="white")
print("Saved figure.png and figure.pdf")
