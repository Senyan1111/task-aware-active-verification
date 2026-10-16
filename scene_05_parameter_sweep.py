from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


SUCCESS_UTILITY = 1.0
COLLISION_UTILITY = -5.0
STOP_UTILITY = -0.2
NO_BUMP_PROBABILITY = 0.2

ACTION_NAMES = ["grasp-left", "grasp-right", "observe", "stop"]
ACTION_COLORS = ["#4C78A8", "#F58518", "#59A14F", "#9D9D9D"]


def action_utilities(probability_left, sensing_cost):
    """Return expected utilities for acting, observing, or stopping."""
    probability_right = 1.0 - NO_BUMP_PROBABILITY - probability_left

    grasp_left = (
        probability_left * COLLISION_UTILITY
        + probability_right * SUCCESS_UTILITY
        + NO_BUMP_PROBABILITY * SUCCESS_UTILITY
    )
    grasp_right = (
        probability_left * SUCCESS_UTILITY
        + probability_right * COLLISION_UTILITY
        + NO_BUMP_PROBABILITY * SUCCESS_UTILITY
    )

    # In this controlled scene, either side view reveals enough information
    # to choose a collision-free grasp, so its task value is 1 minus its cost.
    observe = SUCCESS_UTILITY - sensing_cost
    stop = STOP_UTILITY

    return np.array([grasp_left, grasp_right, observe, stop])


def main(show_figure=True):
    probability_left_values = np.linspace(0.0, 0.8, 161)
    sensing_cost_values = np.linspace(0.0, 1.6, 161)

    decision_grid = np.zeros(
        (len(sensing_cost_values), len(probability_left_values)),
        dtype=int,
    )

    for row, sensing_cost in enumerate(sensing_cost_values):
        for column, probability_left in enumerate(probability_left_values):
            utilities = action_utilities(probability_left, sensing_cost)
            decision_grid[row, column] = int(np.argmax(utilities))

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    color_map = ListedColormap(ACTION_COLORS)
    axes[0].imshow(
        decision_grid,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
        vmin=-0.5,
        vmax=3.5,
        extent=[0.0, 0.8, 0.0, 1.6],
    )
    axes[0].scatter(
        [0.4],
        [0.2],
        marker="*",
        s=220,
        color="black",
        edgecolor="white",
        linewidth=1.2,
        label="previous example",
        zorder=5,
    )
    axes[0].set_title("A. Decision regions")
    axes[0].set_xlabel("belief P(H1: left bump)")
    axes[0].set_ylabel("sensing cost")
    axes[0].text(
        0.02,
        1.54,
        "P(H3: no bump) = 0.2\nP(H2: right bump) = 0.8 - P(H1)",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    legend_handles = [
        Patch(facecolor=color, label=action)
        for color, action in zip(ACTION_COLORS, ACTION_NAMES)
    ]
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="white",
            markerfacecolor="black",
            markersize=12,
            label="previous example",
        )
    )
    axes[0].legend(handles=legend_handles, loc="lower center", ncol=2)

    selected_cost = 0.2
    utility_curves = np.array(
        [
            action_utilities(probability_left, selected_cost)
            for probability_left in probability_left_values
        ]
    )

    for action_index, (action_name, color) in enumerate(
        zip(ACTION_NAMES, ACTION_COLORS)
    ):
        axes[1].plot(
            probability_left_values,
            utility_curves[:, action_index],
            label=action_name,
            color=color,
            linewidth=2.2,
        )

    axes[1].axvline(0.4, color="black", linestyle="--", linewidth=1.0)
    axes[1].scatter(
        [0.4],
        [SUCCESS_UTILITY - selected_cost],
        marker="*",
        s=180,
        color="black",
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
    )
    axes[1].set_title("B. Why the decision changes (sensing cost = 0.2)")
    axes[1].set_xlabel("belief P(H1: left bump)")
    axes[1].set_ylabel("expected utility")
    axes[1].set_ylim(-4.0, 1.2)
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].legend(loc="lower center", ncol=2)

    figure.suptitle(
        "When Should the Robot Act, Observe, or Stop?",
        fontsize=15,
        fontweight="bold",
    )

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "decision_parameter_sweep.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")

    previous_utilities = action_utilities(0.4, 0.2)
    previous_decision = ACTION_NAMES[int(np.argmax(previous_utilities))]

    print("\nPARAMETER SWEEP")
    print("---------------")
    print("Varied P(H1: left bump) from 0.0 to 0.8")
    print("Kept P(H3: no bump) fixed at 0.2")
    print("Varied sensing cost from 0.0 to 1.6")
    print("\nPrevious example:")
    print("  belief = [0.4, 0.4, 0.2]")
    print("  sensing cost = 0.2")
    print(f"  utilities = {dict(zip(ACTION_NAMES, previous_utilities.round(2)))}")
    print(f"  selected decision = {previous_decision}")

    unique, counts = np.unique(decision_grid, return_counts=True)
    print("\nGrid decision counts:")
    for decision_index, count in zip(unique, counts):
        print(f"  {ACTION_NAMES[decision_index]}: {count}")
    print(f"\nSaved figure: {output_path}")

    if show_figure:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening a window.",
    )
    arguments = parser.parse_args()
    main(show_figure=not arguments.no_show)
