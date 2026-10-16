from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np


HYPOTHESES = ["H1: left protrusion", "H2: right protrusion", "H3: no protrusion"]
BELIEF = np.array([0.4, 0.4, 0.2])

SUCCESS_UTILITY = 1.0
COLLISION_UTILITY = -5.0
STOP_UTILITY = -0.2
SENSING_COST = 0.2


def boxes_intersect(box_a, box_b):
    """Return True when two axis-aligned 3D boxes overlap."""
    minimum_a, maximum_a = np.asarray(box_a[0]), np.asarray(box_a[1])
    minimum_b, maximum_b = np.asarray(box_b[0]), np.asarray(box_b[1])
    return bool(np.all(np.maximum(minimum_a, minimum_b) < np.minimum(maximum_a, maximum_b)))


def build_side_grasp_utilities():
    """Compute utilities from geometric collision tests, not a hand-written table."""
    hidden_geometry = {
        "H1: left protrusion": ([-1.25, -0.175, 0.225], [-0.70, 0.175, 0.575]),
        "H2: right protrusion": ([0.70, -0.175, 0.225], [1.25, 0.175, 0.575]),
        "H3: no protrusion": None,
    }

    approach_regions = {
        "grasp-left": ([-1.35, -0.25, 0.15], [-0.72, 0.25, 0.65]),
        "grasp-right": ([0.72, -0.25, 0.15], [1.35, 0.25, 0.65]),
    }

    action_names = ["grasp-left", "grasp-right", "stop"]
    utility_matrix = np.zeros((len(HYPOTHESES), len(action_names)))

    for row, hypothesis in enumerate(HYPOTHESES):
        hidden_box = hidden_geometry[hypothesis]

        for column, action in enumerate(action_names):
            if action == "stop":
                utility_matrix[row, column] = STOP_UTILITY
            elif hidden_box is not None and boxes_intersect(
                hidden_box, approach_regions[action]
            ):
                utility_matrix[row, column] = COLLISION_UTILITY
            else:
                utility_matrix[row, column] = SUCCESS_UTILITY

    return action_names, utility_matrix


def evaluate_task(action_names, utility_matrix):
    """Evaluate current action value and the value of perfectly revealing the shape."""
    expected_utilities = BELIEF @ utility_matrix
    current_action_index = int(np.argmax(expected_utilities))
    current_best_utility = float(expected_utilities[current_action_index])

    perfect_information_utility = float(BELIEF @ np.max(utility_matrix, axis=1))
    evpi = perfect_information_utility - current_best_utility

    # For this first decision experiment, the extra view is assumed to reveal
    # which hypothesis is true. A later experiment will relax this assumption.
    net_view_value = evpi - SENSING_COST

    if net_view_value > 0:
        recommendation = "OBSERVE FIRST"
    else:
        recommendation = f"ACT NOW: {action_names[current_action_index]}"

    return {
        "expected_utilities": expected_utilities,
        "current_action_index": current_action_index,
        "current_best_utility": current_best_utility,
        "perfect_information_utility": perfect_information_utility,
        "evpi": evpi,
        "net_view_value": net_view_value,
        "recommendation": recommendation,
    }


def annotate_heatmap(axis, matrix):
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if value == COLLISION_UTILITY:
                label = f"{value:.1f}\ncollision"
            elif value == SUCCESS_UTILITY:
                label = f"{value:.1f}\nsuccess"
            else:
                label = f"{value:.1f}\nstop"
            axis.text(column, row, label, ha="center", va="center", fontsize=9)


def main(show_figure=True):
    side_actions, side_utilities = build_side_grasp_utilities()
    side_result = evaluate_task(side_actions, side_utilities)

    push_actions = ["push-front", "stop"]
    push_utilities = np.array(
        [
            [SUCCESS_UTILITY, STOP_UTILITY],
            [SUCCESS_UTILITY, STOP_UTILITY],
            [SUCCESS_UTILITY, STOP_UTILITY],
        ]
    )
    push_result = evaluate_task(push_actions, push_utilities)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    heatmap = axes[0].imshow(side_utilities, cmap="RdYlGn", vmin=-5.0, vmax=1.0)
    axes[0].set_title("A. Side-grasp outcomes")
    axes[0].set_xticks(range(len(side_actions)), side_actions, rotation=20)
    axes[0].set_yticks(range(len(HYPOTHESES)), HYPOTHESES)
    axes[0].set_xlabel("candidate action")
    axes[0].set_ylabel("possible true shape")
    annotate_heatmap(axes[0], side_utilities)
    figure.colorbar(heatmap, ax=axes[0], shrink=0.82, label="utility")

    action_colors = ["#4C78A8", "#F58518", "#9D9D9D"]
    axes[1].bar(
        side_actions,
        side_result["expected_utilities"],
        color=action_colors,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylim(-1.7, 0.2)
    axes[1].set_title("B. Decision under current belief")
    axes[1].set_ylabel("expected utility")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].text(
        0.5,
        0.94,
        "Belief: [0.4, 0.4, 0.2]",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
    )
    for index, value in enumerate(side_result["expected_utilities"]):
        axes[1].text(index, value - 0.07, f"{value:.2f}", ha="center", va="top")

    task_names = ["front push", "side grasp"]
    evpi_values = [push_result["evpi"], side_result["evpi"]]
    net_values = [push_result["net_view_value"], side_result["net_view_value"]]
    positions = np.arange(len(task_names))
    width = 0.34

    axes[2].bar(positions - width / 2, evpi_values, width, label="EVPI", color="#54A24B")
    axes[2].bar(
        positions + width / 2,
        net_values,
        width,
        label="value after sensing cost",
        color="#E45756",
    )
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xticks(positions, task_names)
    axes[2].set_ylim(-0.35, 1.4)
    axes[2].set_ylabel("value")
    axes[2].set_title("C. Same uncertainty, different tasks")
    axes[2].legend(loc="upper left")
    axes[2].text(0, -0.27, "ACT NOW", ha="center", fontweight="bold")
    axes[2].text(1, 1.27, "OBSERVE FIRST", ha="center", fontweight="bold")

    figure.suptitle(
        "Hidden Geometry Matters Only When It Changes the Task Decision",
        fontsize=15,
        fontweight="bold",
    )

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "task_dependent_decision.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")

    print("\nTASK 1: FRONT PUSH")
    print("------------------")
    print(f"Current best action: {push_actions[push_result['current_action_index']]}")
    print(f"Current best expected utility: {push_result['current_best_utility']:.2f}")
    print(f"EVPI: {push_result['evpi']:.2f}")
    print(f"Net view value: {push_result['net_view_value']:.2f}")
    print(f"Recommendation: {push_result['recommendation']}")

    print("\nTASK 2: SIDE GRASP")
    print("------------------")
    print(f"Expected utilities: {dict(zip(side_actions, side_result['expected_utilities'].round(2)))}")
    print(f"Current best action: {side_actions[side_result['current_action_index']]}")
    print(f"Current best expected utility: {side_result['current_best_utility']:.2f}")
    print(f"EVPI: {side_result['evpi']:.2f}")
    print(f"Net view value: {side_result['net_view_value']:.2f}")
    print(f"Recommendation: {side_result['recommendation']}")
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
