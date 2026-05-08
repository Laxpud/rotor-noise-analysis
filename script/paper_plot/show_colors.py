import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "plot_config.json"), "r") as f:
    config = json.load(f)
palettes = config["palettes"]

fig, axes = plt.subplots(2, 1, figsize=(10, 4))

for ax, (name, colors) in zip(axes, palettes.items()):
    n = len(colors)
    x = np.arange(n)
    y = np.ones(n)

    for i, (xi, yi, c) in enumerate(zip(x, y, colors)):
        ax.add_patch(
            mpatches.Rectangle(
                (xi - 0.4, 0), 0.8, 1, facecolor=c, edgecolor="white", linewidth=1
            )
        )
        ax.text(xi, -0.25, c, ha="center", va="top", fontsize=8, rotation=45)

    ax.set_xlim(-1, n)
    ax.set_ylim(-0.6, 1.2)
    ax.set_title(f"{name.capitalize()} ({n} colors)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(os.path.join(here, "color_palettes.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Export plot: {os.path.join(here, 'color_palettes.png')}")
