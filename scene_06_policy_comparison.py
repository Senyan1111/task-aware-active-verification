from pathlib import Path
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


RANDOM_SEED = 20261015
VALIDATION_TRIALS = 2_000
TEST_TRIALS = 10_000

SUCCESS_UTILITY = 1.0
COLLISION_UTILITY = -5.0
STOP_UTILITY = -0.2
MAX_DIRECT_COLLISION_PROBABILITY = 0.05

H1_LEFT_BUMP = 0
H2_RIGHT_BUMP = 1
H3_NO_BUMP = 2

TASK_FRONT_PUSH = 0
TASK_SIDE_GRASP = 1

SIDE_UTILITY = np.array(
    [
        [-5.0, 1.0, -0.2],
        [1.0, -5.0, -0.2],
        [1.0, 1.0, -0.2],
    ]
)

POLICY_NAMES = [
    "MAP only",
    "Always observe",
    "Entropy trigger",
    "Task-aware",
]
POLICY_COLORS = ["#4C78A8", "#F58518", "#B279A2", "#59A14F"]


def generate_trials(number_of_trials, seed):
    rng = np.random.default_rng(seed)

    beliefs = rng.dirichlet([1.2, 1.2, 0.8], size=number_of_trials)
    tasks = rng.integers(0, 2, size=number_of_trials)
    sensing_costs = rng.uniform(0.05, 0.60, size=number_of_trials)

    cumulative = np.cumsum(beliefs, axis=1)
    samples = rng.random(number_of_trials)
    true_shapes = (samples[:, None] > cumulative).sum(axis=1)

    return beliefs, tasks, sensing_costs, true_shapes


def normalized_entropy(belief):
    safe_belief = np.clip(belief, 1e-12, 1.0)
    return float(-np.sum(safe_belief * np.log(safe_belief)) / np.log(3.0))


def map_action(belief):
    most_likely_shape = int(np.argmax(belief))
    if most_likely_shape == H1_LEFT_BUMP:
        return "grasp-right"
    return "grasp-left"


def task_aware_side_action(belief, sensing_cost):
    expected_utilities = belief @ SIDE_UTILITY

    left_collision_probability = float(belief[H1_LEFT_BUMP])
    right_collision_probability = float(belief[H2_RIGHT_BUMP])

    if left_collision_probability > MAX_DIRECT_COLLISION_PROBABILITY:
        expected_utilities[0] = -np.inf
    if right_collision_probability > MAX_DIRECT_COLLISION_PROBABILITY:
        expected_utilities[1] = -np.inf

    candidates = {
        "grasp-left": float(expected_utilities[0]),
        "grasp-right": float(expected_utilities[1]),
        "observe": SUCCESS_UTILITY - sensing_cost,
        "stop": STOP_UTILITY,
    }
    return max(candidates, key=candidates.get)


def choose_action(policy_name, belief, task, sensing_cost, entropy_threshold):
    if policy_name == "Always observe":
        return "observe"

    if policy_name == "MAP only":
        return "push-front" if task == TASK_FRONT_PUSH else map_action(belief)

    if policy_name == "Entropy trigger":
        if normalized_entropy(belief) >= entropy_threshold:
            return "observe"
        return "push-front" if task == TASK_FRONT_PUSH else map_action(belief)

    if policy_name == "Task-aware":
        if task == TASK_FRONT_PUSH:
            return "push-front"
        return task_aware_side_action(belief, sensing_cost)

    raise ValueError(f"Unknown policy: {policy_name}")


def realized_outcome(action, task, true_shape, sensing_cost):
    if action == "observe":
        return SUCCESS_UTILITY - sensing_cost, True, False, True, False

    if action == "stop":
        return STOP_UTILITY, False, False, False, True

    if task == TASK_FRONT_PUSH and action == "push-front":
        return SUCCESS_UTILITY, True, False, False, False

    if action == "grasp-left":
        collision = true_shape == H1_LEFT_BUMP
    elif action == "grasp-right":
        collision = true_shape == H2_RIGHT_BUMP
    else:
        raise ValueError(f"Invalid action {action} for task {task}")

    if collision:
        return COLLISION_UTILITY, False, True, False, False
    return SUCCESS_UTILITY, True, False, False, False


def evaluate_policy(
    policy_name,
    beliefs,
    tasks,
    sensing_costs,
    true_shapes,
    entropy_threshold=0.75,
):
    utilities = []
    successes = []
    collisions = []
    observations = []
    stops = []

    for belief, task, sensing_cost, true_shape in zip(
        beliefs, tasks, sensing_costs, true_shapes
    ):
        action = choose_action(
            policy_name,
            belief,
            int(task),
            float(sensing_cost),
            entropy_threshold,
        )
        utility, success, collision, observed, stopped = realized_outcome(
            action,
            int(task),
            int(true_shape),
            float(sensing_cost),
        )
        utilities.append(utility)
        successes.append(success)
        collisions.append(collision)
        observations.append(observed)
        stops.append(stopped)

    return {
        "mean_utility": float(np.mean(utilities)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "sensing_rate": float(np.mean(observations)),
        "stop_rate": float(np.mean(stops)),
    }


def tune_entropy_threshold(validation_data):
    beliefs, tasks, sensing_costs, true_shapes = validation_data
    thresholds = np.linspace(0.0, 1.0, 51)
    utilities = []

    for threshold in thresholds:
        result = evaluate_policy(
            "Entropy trigger",
            beliefs,
            tasks,
            sensing_costs,
            true_shapes,
            entropy_threshold=float(threshold),
        )
        utilities.append(result["mean_utility"])

    best_index = int(np.argmax(utilities))
    return float(thresholds[best_index]), thresholds, np.asarray(utilities)


def main(show_figure=True):
    validation_data = generate_trials(VALIDATION_TRIALS, RANDOM_SEED)
    test_data = generate_trials(TEST_TRIALS, RANDOM_SEED + 1)

    best_entropy_threshold, thresholds, validation_utilities = tune_entropy_threshold(
        validation_data
    )

    results = {}
    for policy_name in POLICY_NAMES:
        results[policy_name] = evaluate_policy(
            policy_name,
            *test_data,
            entropy_threshold=best_entropy_threshold,
        )

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    metric_names = [
        ("mean_utility", "Mean utility", False),
        ("success_rate", "Task success rate", True),
        ("collision_rate", "Collision rate", True),
        ("sensing_rate", "Extra-observation rate", True),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()

    for axis, (metric_key, title, is_rate) in zip(axes, metric_names):
        values = [results[name][metric_key] for name in POLICY_NAMES]
        bars = axis.bar(POLICY_NAMES, values, color=POLICY_COLORS)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=18)
        if is_rate:
            axis.set_ylim(0.0, 1.08)
            axis.set_ylabel("fraction of trials")
        else:
            lower_limit = min(0.0, min(values) - 0.1)
            axis.set_ylim(lower_limit, max(values) + 0.15)
            axis.set_ylabel("utility")
        axis.axhline(0.0, color="black", linewidth=0.7)

        for bar, value in zip(bars, values):
            label = f"{100 * value:.1f}%" if is_rate else f"{value:.3f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.025 if value >= 0 else -0.025),
                label,
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
                fontweight="bold",
            )

    figure.suptitle(
        f"Policy Comparison on {TEST_TRIALS:,} Reproducible Random Trials",
        fontsize=15,
        fontweight="bold",
    )

    comparison_path = output_dir / "policy_comparison.png"
    figure.savefig(comparison_path, dpi=180, bbox_inches="tight")

    tuning_figure, tuning_axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    tuning_axis.plot(thresholds, validation_utilities, color="#B279A2", linewidth=2.3)
    tuning_axis.axvline(
        best_entropy_threshold,
        color="black",
        linestyle="--",
        label=f"selected threshold = {best_entropy_threshold:.2f}",
    )
    tuning_axis.set_xlabel("normalized entropy threshold")
    tuning_axis.set_ylabel("validation mean utility")
    tuning_axis.set_title("Entropy Baseline Threshold Tuning")
    tuning_axis.legend()
    tuning_path = output_dir / "entropy_threshold_tuning.png"
    tuning_figure.savefig(tuning_path, dpi=180, bbox_inches="tight")

    csv_path = output_dir / "policy_comparison_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["policy", *results[POLICY_NAMES[0]].keys()])
        for policy_name in POLICY_NAMES:
            writer.writerow([policy_name, *results[policy_name].values()])

    print("\nPOLICY COMPARISON")
    print("-----------------")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Validation trials: {VALIDATION_TRIALS}")
    print(f"Test trials: {TEST_TRIALS}")
    print(f"Tuned entropy threshold: {best_entropy_threshold:.2f}")

    for policy_name in POLICY_NAMES:
        result = results[policy_name]
        print(f"\n{policy_name}")
        print(f"  mean utility: {result['mean_utility']:.3f}")
        print(f"  success rate: {100 * result['success_rate']:.1f}%")
        print(f"  collision rate: {100 * result['collision_rate']:.1f}%")
        print(f"  sensing rate: {100 * result['sensing_rate']:.1f}%")
        print(f"  stop rate: {100 * result['stop_rate']:.1f}%")

    print(f"\nSaved comparison: {comparison_path}")
    print(f"Saved tuning curve: {tuning_path}")
    print(f"Saved summary table: {csv_path}")

    if show_figure:
        plt.show()
    else:
        plt.close(figure)
        plt.close(tuning_figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures without opening windows.",
    )
    arguments = parser.parse_args()
    main(show_figure=not arguments.no_show)
