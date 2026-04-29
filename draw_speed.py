import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter

# =========================================================
# Strict literature-backed inference-efficiency plot
# ---------------------------------------------------------
# Confirmed points:
#   ThinkAct-7B      : 84.4, 7513 ms  (LaRA-VLA)
#   ECoT-7B          : 70.3, 4434 ms  (LaRA-VLA Figure 7)
#                      OR 70.3, 4997 ms/step (Fast ECoT)
#   Fast-ThinkAct-3B : 89.7, 805 ms   (LaRA-VLA)
#   LaRA-VLA-4B      : 97.9, 135 ms   (LaRA-VLA)
#   ICL-VLA          : fill with your own measured numbers
#
# Methods intentionally omitted for now:
#   UniVLA
# because I did not find directly comparable published latency
# suitable for a "guaranteed real" main-paper figure.
# =========================================================

USE_ECOT_LATENCY_FROM_LARA = True  # True -> 4434; False -> 4997

ecot_latency = 4434 if USE_ECOT_LATENCY_FROM_LARA else 4997

# points = [
#     {
#         "name": "ThinkAct-7B",
#         "latency_ms": 7513,
#         "success": 84.4,
#         "family": "Explicit CoT",
#     },
#     {
#         "name": "ECoT-7B",
#         "latency_ms": ecot_latency,
#         "success": 70.3,
#         "family": "Explicit CoT",
#     },
#     {
#         "name": "Fast-ThinkAct-3B",
#         "latency_ms": 805,
#         "success": 89.7,
#         "family": "Implicit CoT",
#     },
#     {
#         "name": "LaRA-VLA-4B",
#         "latency_ms": 135,
#         "success": 97.9,
#         "family": "Implicit CoT",
#     },
#     {
#         "name": "OpenVLA-7B",
#         "latency_ms": 247,
#         "success": 76.5,
#         "family": "Others",
#     },
#     {
#         "name": "pi0",
#         "latency_ms": 230,
#         "success": 94.2,
#         "family": "Others",
#     },
#     {
#         "name": "CoT-VLA",
#         "latency_ms": 3178,
#         "success": 81.1,
#         "family": "Others",
#     },
#     {
#         "name": "ICL-VLA",
#         "latency_ms": 450,   # TODO: replace with your measured latency
#         "success": 98.1,     # TODO: replace with your measured success
#         "family": "Ours",
#     },
# ]

# Optional future points after YOU benchmark them yourself:
# points += [
#     {"name": "UniVLA",     "latency_ms": ..., "success": 95.5, "family": "Others"},
# ]

# Empty-plot mode: keep points commented above and leave this as an empty list.
points = []

colors = {
    "Explicit CoT": "#7A5FB3",   # purple
    "Implicit CoT": "#43B77A",   # green
    "Others": "#E49A3A",         # orange
    "Ours": "#2F8FCE",           # blue
}

sizes = {
    "Explicit CoT": 680,
    "Implicit CoT": 620,
    "Others": 590,
    "Ours": 740,
}

# Per-point label placement to reduce overlaps in crowded regions.
label_styles = {
    "ThinkAct-7B": {"dx": 0, "dy": 0, "ha": "center", "va": "center"},
    "ECoT-7B": {"dx": 10, "dy": 1, "ha": "left", "va": "center"},
    "Fast-ThinkAct-3B": {"dx": 0, "dy": 0, "ha": "center", "va": "center"},
    "LaRA-VLA-4B": {"dx": 10, "dy": 4, "ha": "left", "va": "center"},
    "OpenVLA-7B": {"dx": -8, "dy": 0, "ha": "right", "va": "center"},
    "pi0": {"dx": 8, "dy": 0, "ha": "left", "va": "center"},
    "CoT-VLA": {"dx": 8, "dy": 0, "ha": "center", "va": "center"},
    "ICL-VLA": {"dx": -10, "dy": 1, "ha": "right", "va": "center"},
}

fig, ax = plt.subplots(figsize=(9.2, 6.8))

for p in points:
    ax.scatter(
        p["latency_ms"],
        p["success"],
        s=sizes[p["family"]],
        c=colors[p["family"]],
        edgecolors="black",
        linewidths=1.2,
        alpha=0.92,
        zorder=3,
    )

for p in points:
    style = label_styles.get(
        p["name"],
        {"dx": 0, "dy": 0, "ha": "center", "va": "center"},
    )
    txt = ax.annotate(
        p["name"],
        (p["latency_ms"], p["success"]),
        xytext=(style["dx"], style["dy"]),
        textcoords="offset points",
        ha=style["ha"],
        va=style["va"],
        fontsize=10.5,
        color="black",
        zorder=4,
    )
    txt.set_path_effects([pe.withStroke(linewidth=3.2, foreground="white")])

ax.set_xlabel("Inference Latency (ms)", fontsize=13)
ax.set_ylabel("Success Rate (%)", fontsize=13)
ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))

# Faster on the right
ax.invert_xaxis()

if points:
    latencies = [p["latency_ms"] for p in points]
    successes = [p["success"] for p in points]

    xmin = min(latencies)
    xmax = max(latencies)
    ymin = min(successes)
    ymax = max(successes)

    # Leave extra blank space on the right (small-latency side)
    right_blank = xmin * 0.16
    left_blank = xmax * 1.15
    ax.set_xlim(left_blank, right_blank)

    # Leave extra blank space on both top and bottom
    ax.set_ylim(max(67, ymin - 2.8), min(102, ymax + 3.8))
else:
    # Default canvas limits for empty chart mode.
    ax.set_xlim(8200, 80)
    ax.set_ylim(67, 102)

ax.grid(True, linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

ax.text(
    0.52, 0.05, "Faster →",
    transform=ax.transAxes,
    fontsize=11,
    ha="center",
    va="bottom"
)

legend_handles = [
    Line2D([0], [0], marker='o', color='w', label=family,
           markerfacecolor=colors[family], markeredgecolor='black',
           markersize=10)
    for family in ["Explicit CoT", "Implicit CoT", "Others", "Ours"]
]
ax.legend(
    handles=legend_handles,
    title="Method Family",
    loc="upper left",
    fontsize=10,
    title_fontsize=10.5,
    frameon=True
)

plt.tight_layout()
plt.savefig("inference_efficiency_tradeoff_strict.png", dpi=300, bbox_inches="tight")
plt.savefig("inference_efficiency_tradeoff_strict.pdf", bbox_inches="tight")
plt.show()