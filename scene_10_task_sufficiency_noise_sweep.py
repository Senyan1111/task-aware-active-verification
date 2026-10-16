from pathlib import Path
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

from scene_09_depth_belief_update import (
    ACTION_NAMES,
    CAMERAS,
    DROPOUT_PROBABILITY,
    HIDDEN_PARTS,
    HYPOTHESIS_NAMES,
    PRIOR,
    SENSING_COST,
    add_sensor_noise,
    best_action,
    posterior_from_depth,
    render_depth_and_semantics,
)


RANDOM_SEEDS = [20261016 + index for index in range(8)]
REPEATS_PER_SHAPE = 100
NOISE_LEVELS = np.array([0.0, 0.01, 0.03, 0.06, 0.10, 0.15])

STRATEGY_NAMES = ["One task-sufficient view", "Two-view full verification"]
STRATEGY_COLORS = ["#59A14F", "#4C78A8"]


def add_sensor_noise_at_level(clean_depth, rng, noise_std):
    available = rng.random(clean_depth.shape) >= DROPOUT_PROBABILITY
    observation = clean_depth.copy()
    finite = np.isfinite(clean_depth) & available
    if noise_std > 0.0:
        observation[finite] += rng.normal(
            0.0, noise_std, size=np.count_nonzero(finite)
        )
    observation[~available] = np.nan
    return observation, available


def realized_outcome(action, true_shape, sensing_cost):
    if action == "stop":
        return -0.2 - sensing_cost, False, False, True
    if action == "grasp-left":
        collision = true_shape == 0
    elif action == "grasp-right":
        collision = true_shape == 1
    else:
        raise ValueError(f"Unknown action: {action}")

    if collision:
        return -5.0 - sensing_cost, False, True, False
    return 1.0 - sensing_cost, True, False, False


def evaluate_strategy(
    strategy_name,
    noise_std,
    seed,
    right_predictions,
    left_predictions,
):
    rng = np.random.default_rng(seed)
    exact_matches = []
    successes = []
    collisions = []
    stops = []
    utilities = []
    posterior_true_probabilities = []

    for true_shape in range(3):
        for _ in range(REPEATS_PER_SHAPE):
            right_observation, right_available = add_sensor_noise_at_level(
                right_predictions[true_shape], rng, noise_std
            )
            posterior, _, _ = posterior_from_depth(
                PRIOR,
                right_observation,
                right_available,
                right_predictions,
            )
            sensing_cost = SENSING_COST

            if strategy_name == "Two-view full verification":
                left_observation, left_available = add_sensor_noise_at_level(
                    left_predictions[true_shape], rng, noise_std
                )
                posterior, _, _ = posterior_from_depth(
                    posterior,
                    left_observation,
                    left_available,
                    left_predictions,
                )
                sensing_cost += SENSING_COST
            elif strategy_name != "One task-sufficient view":
                raise ValueError(f"Unknown strategy: {strategy_name}")

            action, _, _ = best_action(posterior)
            utility, success, collision, stopped = realized_outcome(
                action, true_shape, sensing_cost
            )

            exact_matches.append(int(np.argmax(posterior)) == true_shape)
            successes.append(success)
            collisions.append(collision)
            stops.append(stopped)
            utilities.append(utility)
            posterior_true_probabilities.append(float(posterior[true_shape]))

    return {
        "exact_shape_accuracy": float(np.mean(exact_matches)),
        "task_success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "stop_rate": float(np.mean(stops)),
        "mean_utility": float(np.mean(utilities)),
        "mean_true_shape_probability": float(
            np.mean(posterior_true_probabilities)
        ),
    }


def mean_and_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    ci = float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, ci


def collect_results():
    right_predictions = [
        render_depth_and_semantics(hidden_part, CAMERAS["right-view"])[0]
        for hidden_part in HIDDEN_PARTS
    ]
    left_predictions = [
        render_depth_and_semantics(hidden_part, CAMERAS["left-view"])[0]
        for hidden_part in HIDDEN_PARTS
    ]

    seed_rows = []
    for noise_std in NOISE_LEVELS:
        for seed in RANDOM_SEEDS:
            for strategy_name in STRATEGY_NAMES:
                result = evaluate_strategy(
                    strategy_name,
                    float(noise_std),
                    seed,
                    right_predictions,
                    left_predictions,
                )
                seed_rows.append(
                    {
                        "depth_noise_std": float(noise_std),
                        "seed": seed,
                        "strategy": strategy_name,
                        **result,
                    }
                )
    return seed_rows


def summarize(seed_rows):
    metric_names = [
        "exact_shape_accuracy",
        "task_success_rate",
        "collision_rate",
        "stop_rate",
        "mean_utility",
        "mean_true_shape_probability",
    ]
    summary_rows = []

    for noise_std in NOISE_LEVELS:
        for strategy_name in STRATEGY_NAMES:
            matching = [
                row
                for row in seed_rows
                if np.isclose(row["depth_noise_std"], noise_std)
                and row["strategy"] == strategy_name
            ]
            summary = {
                "depth_noise_std": float(noise_std),
                "strategy": strategy_name,
            }
            for metric_name in metric_names:
                mean, ci = mean_and_ci([row[metric_name] for row in matching])
                summary[metric_name] = mean
                summary[f"{metric_name}_ci95"] = ci
            summary_rows.append(summary)
    return summary_rows


def get_series(summary_rows, strategy_name, metric_name):
    matching = [
        row for row in summary_rows if row["strategy"] == strategy_name
    ]
    matching.sort(key=lambda row: row["depth_noise_std"])
    means = np.array([row[metric_name] for row in matching])
    cis = np.array([row[f"{metric_name}_ci95"] for row in matching])
    return means, cis


def make_figure(summary_rows, output_dir):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metric_specs = [
        (
            "exact_shape_accuracy",
            "A. Exact hidden-shape identification",
            "fraction correct",
            (0.0, 1.05),
        ),
        (
            "task_success_rate",
            "B. Safe task completion",
            "fraction successful",
            (0.0, 1.05),
        ),
        (
            "mean_utility",
            "C. Task value after sensing cost",
            "mean utility",
            None,
        ),
        (
            "collision_rate",
            "D. Collision rate",
            "fraction of trials",
            (0.0, 0.35),
        ),
    ]

    for axis, (metric_name, title, ylabel, ylim) in zip(
        axes.ravel(), metric_specs
    ):
        for strategy_name, color in zip(STRATEGY_NAMES, STRATEGY_COLORS):
            means, cis = get_series(summary_rows, strategy_name, metric_name)
            axis.plot(
                NOISE_LEVELS,
                means,
                marker="o",
                linewidth=2.1,
                color=color,
                label=strategy_name,
            )
            axis.fill_between(
                NOISE_LEVELS,
                means - cis,
                means + cis,
                color=color,
                alpha=0.16,
            )
        axis.set_xlabel("depth-noise standard deviation")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.grid(alpha=0.25)

    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        "Exact Shape Recovery Is Not Always Needed for the Task",
        fontsize=15,
        fontweight="bold",
    )
    path = output_dir / "task_sufficiency_noise_sweep.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    return figure, path


def write_results(seed_rows, summary_rows, output_dir):
    seed_path = output_dir / "task_sufficiency_seed_results.csv"
    with seed_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(seed_rows)

    summary_path = output_dir / "task_sufficiency_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    return seed_path, summary_path


def main(show_figure=True):
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    seed_rows = collect_results()
    summary_rows = summarize(seed_rows)
    figure, figure_path = make_figure(summary_rows, output_dir)
    seed_path, summary_path = write_results(seed_rows, summary_rows, output_dir)

    print("\nTASK SUFFICIENCY UNDER DEPTH NOISE")
    print("----------------------------------")
    print(f"Seeds: {len(RANDOM_SEEDS)}")
    print(f"Repeats per true shape and seed: {REPEATS_PER_SHAPE}")
    print("True shapes are balanced: H1, H2, and H3 each appear equally often.")
    print(f"Pixel dropout probability: {DROPOUT_PROBABILITY:.0%}")

    for noise_std in NOISE_LEVELS:
        print(f"\nDepth noise std = {noise_std:.2f}")
        for strategy_name in STRATEGY_NAMES:
            row = next(
                item
                for item in summary_rows
                if item["strategy"] == strategy_name
                and np.isclose(item["depth_noise_std"], noise_std)
            )
            print(
                f"  {strategy_name}: exact shape="
                f"{row['exact_shape_accuracy']:.1%}, task success="
                f"{row['task_success_rate']:.1%}, collision="
                f"{row['collision_rate']:.1%}, utility="
                f"{row['mean_utility']:.3f}"
            )

    print(f"\nSaved figure: {figure_path}")
    print(f"Saved seed results: {seed_path}")
    print(f"Saved summary: {summary_path}")
    print("This remains a synthetic depth-camera experiment, not a real-robot result.")

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
