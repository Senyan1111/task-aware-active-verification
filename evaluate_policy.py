from pathlib import Path
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


RANDOM_SEEDS = [20261015 + index for index in range(10)]
TRIALS_PER_SEED = 5_000
VALIDATION_SEEDS = [91001, 91002, 91003]
VALIDATION_TRIALS_PER_SEED = 2_500
SENSITIVITY_SEEDS = [73001, 73002, 73003]
SENSITIVITY_TRIALS_PER_SEED = 2_000

VIEW_ACCURACY = 0.85
SUCCESS_UTILITY = 1.0
DEFAULT_COLLISION_UTILITY = -5.0
STOP_UTILITY = -0.2

H1_LEFT_BUMP = 0
H2_RIGHT_BUMP = 1
H3_NO_BUMP = 2

TASK_FRONT_PUSH = 0
TASK_SIDE_GRASP = 1

ACTION_NAMES = ["grasp-left", "grasp-right", "stop"]
POLICY_NAMES = [
    "MAP only",
    "EU no observation",
    "Always observe",
    "Entropy trigger",
    "Task-aware",
]
POLICY_COLORS = ["#4C78A8", "#72B7B2", "#F58518", "#B279A2", "#59A14F"]


def side_utility_matrix(collision_utility):
    return np.array(
        [
            [collision_utility, SUCCESS_UTILITY, STOP_UTILITY],
            [SUCCESS_UTILITY, collision_utility, STOP_UTILITY],
            [SUCCESS_UTILITY, SUCCESS_UTILITY, STOP_UTILITY],
        ],
        dtype=float,
    )


def normalized_entropy(belief):
    safe_belief = np.clip(belief, 1e-12, 1.0)
    return float(-np.sum(safe_belief * np.log(safe_belief)) / np.log(3.0))


def observation_likelihoods(target_shape, accuracy, observation):
    likelihoods = np.full(3, 1.0 - accuracy, dtype=float)
    likelihoods[target_shape] = accuracy
    if observation == 0:
        likelihoods = 1.0 - likelihoods
    return likelihoods


def update_belief(belief, target_shape, accuracy, observation):
    likelihoods = observation_likelihoods(target_shape, accuracy, observation)
    unnormalized = belief * likelihoods
    return unnormalized / unnormalized.sum()


def best_side_action(belief, collision_utility):
    expected_utilities = belief @ side_utility_matrix(collision_utility)
    action_index = int(np.argmax(expected_utilities))
    return ACTION_NAMES[action_index], float(expected_utilities[action_index])


def expected_value_after_view(belief, target_shape, accuracy, collision_utility):
    expected_value = 0.0
    for observation in (0, 1):
        likelihoods = observation_likelihoods(
            target_shape, accuracy, observation
        )
        observation_probability = float(belief @ likelihoods)
        posterior = update_belief(
            belief, target_shape, accuracy, observation
        )
        _, posterior_value = best_side_action(posterior, collision_utility)
        expected_value += observation_probability * posterior_value
    return expected_value


def best_view(belief, accuracy, collision_utility):
    left_value = expected_value_after_view(
        belief, H1_LEFT_BUMP, accuracy, collision_utility
    )
    right_value = expected_value_after_view(
        belief, H2_RIGHT_BUMP, accuracy, collision_utility
    )
    if left_value >= right_value:
        return H1_LEFT_BUMP, left_value
    return H2_RIGHT_BUMP, right_value


def map_action(belief):
    most_likely_shape = int(np.argmax(belief))
    if most_likely_shape == H1_LEFT_BUMP:
        return "grasp-right"
    return "grasp-left"


def sample_observation(true_shape, target_shape, accuracy, random_number):
    bump_really_visible = true_shape == target_shape
    observation_is_correct = random_number < accuracy
    return int(
        bump_really_visible
        if observation_is_correct
        else not bump_really_visible
    )


def realized_side_outcome(action, true_shape, collision_utility):
    if action == "stop":
        return STOP_UTILITY, False, False, True
    if action == "grasp-left":
        collision = true_shape == H1_LEFT_BUMP
    elif action == "grasp-right":
        collision = true_shape == H2_RIGHT_BUMP
    else:
        raise ValueError(f"Unknown action: {action}")

    if collision:
        return collision_utility, False, True, False
    return SUCCESS_UTILITY, True, False, False


def generate_trials(number_of_trials, seed, sensing_cost_scale=1.0):
    rng = np.random.default_rng(seed)
    beliefs = rng.dirichlet([1.2, 1.2, 0.8], size=number_of_trials)
    tasks = rng.integers(0, 2, size=number_of_trials)
    sensing_costs = (
        rng.uniform(0.05, 0.60, size=number_of_trials) * sensing_cost_scale
    )

    cumulative = np.cumsum(beliefs, axis=1)
    shape_samples = rng.random(number_of_trials)
    true_shapes = (shape_samples[:, None] > cumulative).sum(axis=1)
    observation_random_numbers = rng.random(number_of_trials)
    return (
        beliefs,
        tasks,
        sensing_costs,
        true_shapes,
        observation_random_numbers,
    )


def evaluate_policy(
    policy_name,
    trial_data,
    entropy_threshold,
    view_accuracy=VIEW_ACCURACY,
    collision_utility=DEFAULT_COLLISION_UTILITY,
):
    beliefs, tasks, sensing_costs, true_shapes, observation_random_numbers = trial_data
    utilities = []
    successes = []
    collisions = []
    observations = []
    stops = []

    for belief, task, cost, true_shape, observation_random in zip(
        beliefs,
        tasks,
        sensing_costs,
        true_shapes,
        observation_random_numbers,
    ):
        task = int(task)
        true_shape = int(true_shape)
        cost = float(cost)
        observed = False

        if task == TASK_FRONT_PUSH:
            if policy_name == "Always observe":
                observed = True
            elif policy_name == "Entropy trigger":
                observed = normalized_entropy(belief) >= entropy_threshold
            utility = SUCCESS_UTILITY - (cost if observed else 0.0)
            success, collision, stopped = True, False, False
        else:
            current_action, current_value = best_side_action(
                belief, collision_utility
            )
            target_shape = None

            if policy_name == "MAP only":
                action = map_action(belief)
            elif policy_name == "EU no observation":
                action = current_action
            else:
                target_shape, post_view_value = best_view(
                    belief, view_accuracy, collision_utility
                )
                observe_value = post_view_value - cost

                if policy_name == "Always observe":
                    observed = True
                elif policy_name == "Entropy trigger":
                    observed = normalized_entropy(belief) >= entropy_threshold
                elif policy_name == "Task-aware":
                    observed = observe_value > current_value
                else:
                    raise ValueError(f"Unknown policy: {policy_name}")

                if observed:
                    observation = sample_observation(
                        true_shape,
                        target_shape,
                        view_accuracy,
                        float(observation_random),
                    )
                    posterior = update_belief(
                        belief,
                        target_shape,
                        view_accuracy,
                        observation,
                    )
                    action, _ = best_side_action(
                        posterior, collision_utility
                    )
                else:
                    action = current_action

            utility, success, collision, stopped = realized_side_outcome(
                action, true_shape, collision_utility
            )
            if observed:
                utility -= cost

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


def tune_entropy_threshold():
    validation_sets = [
        generate_trials(VALIDATION_TRIALS_PER_SEED, seed)
        for seed in VALIDATION_SEEDS
    ]
    thresholds = np.linspace(0.0, 1.0, 41)
    mean_utilities = []

    for threshold in thresholds:
        seed_utilities = [
            evaluate_policy(
                "Entropy trigger",
                trial_data,
                entropy_threshold=float(threshold),
            )["mean_utility"]
            for trial_data in validation_sets
        ]
        mean_utilities.append(float(np.mean(seed_utilities)))

    best_index = int(np.argmax(mean_utilities))
    return (
        float(thresholds[best_index]),
        thresholds,
        np.asarray(mean_utilities),
    )


def mean_and_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, 1.96 * standard_error


def run_main_comparison(entropy_threshold):
    seed_results = []
    for seed in RANDOM_SEEDS:
        trial_data = generate_trials(TRIALS_PER_SEED, seed)
        for policy_name in POLICY_NAMES:
            result = evaluate_policy(
                policy_name,
                trial_data,
                entropy_threshold=entropy_threshold,
            )
            seed_results.append({"seed": seed, "policy": policy_name, **result})
    return seed_results


def summarize_seed_results(seed_results):
    summary = {}
    metric_names = [
        "mean_utility",
        "success_rate",
        "collision_rate",
        "sensing_rate",
        "stop_rate",
    ]
    for policy_name in POLICY_NAMES:
        matching = [row for row in seed_results if row["policy"] == policy_name]
        summary[policy_name] = {}
        for metric_name in metric_names:
            mean, ci = mean_and_ci([row[metric_name] for row in matching])
            summary[policy_name][metric_name] = mean
            summary[policy_name][f"{metric_name}_ci95"] = ci
    return summary


def run_sensitivity(entropy_threshold):
    collision_penalties = np.array([-2.0, -3.0, -5.0, -8.0, -10.0])
    sensing_cost_scales = np.array([0.50, 0.75, 1.00, 1.25, 1.50])
    advantages = np.zeros((len(collision_penalties), len(sensing_cost_scales)))

    for penalty_index, collision_utility in enumerate(collision_penalties):
        for cost_index, cost_scale in enumerate(sensing_cost_scales):
            seed_advantages = []
            for seed in SENSITIVITY_SEEDS:
                trial_data = generate_trials(
                    SENSITIVITY_TRIALS_PER_SEED,
                    seed,
                    sensing_cost_scale=float(cost_scale),
                )
                eu_result = evaluate_policy(
                    "EU no observation",
                    trial_data,
                    entropy_threshold=entropy_threshold,
                    collision_utility=float(collision_utility),
                )
                task_aware_result = evaluate_policy(
                    "Task-aware",
                    trial_data,
                    entropy_threshold=entropy_threshold,
                    collision_utility=float(collision_utility),
                )
                seed_advantages.append(
                    task_aware_result["mean_utility"]
                    - eu_result["mean_utility"]
                )
            advantages[penalty_index, cost_index] = np.mean(seed_advantages)

    return collision_penalties, sensing_cost_scales, advantages


def save_main_figure(summary, output_dir):
    metrics = [
        ("mean_utility", "Mean utility", False),
        ("success_rate", "Task success rate", True),
        ("collision_rate", "Collision rate", True),
        ("sensing_rate", "Extra-observation rate", True),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    for axis, (metric_name, title, is_rate) in zip(axes.ravel(), metrics):
        means = [summary[name][metric_name] for name in POLICY_NAMES]
        errors = [summary[name][f"{metric_name}_ci95"] for name in POLICY_NAMES]
        bars = axis.bar(
            POLICY_NAMES,
            means,
            yerr=errors,
            capsize=4,
            color=POLICY_COLORS,
            edgecolor="black",
            linewidth=0.4,
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=18, labelsize=8)
        axis.axhline(0.0, color="black", linewidth=0.7)
        if is_rate:
            axis.set_ylim(0.0, 1.08)
            axis.set_ylabel("fraction of trials")
        else:
            axis.set_ylabel("utility")
            axis.set_ylim(min(0.0, min(means) - 0.1), max(means) + 0.16)

        for bar, mean in zip(bars, means):
            label = f"{100 * mean:.1f}%" if is_rate else f"{mean:.3f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                mean + 0.025,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    figure.suptitle(
        "Policy Comparison with 10 Seeds and 95% Confidence Intervals",
        fontsize=15,
        fontweight="bold",
    )
    path = output_dir / "credible_policy_comparison.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    return figure, path


def save_sensitivity_figure(
    collision_penalties, sensing_cost_scales, advantages, output_dir
):
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    maximum = float(np.max(np.abs(advantages)))
    image = axis.imshow(
        advantages,
        cmap="RdYlGn",
        vmin=-maximum,
        vmax=maximum,
        aspect="auto",
    )
    axis.set_xticks(range(len(sensing_cost_scales)))
    axis.set_xticklabels([f"{value:.2f}x" for value in sensing_cost_scales])
    axis.set_yticks(range(len(collision_penalties)))
    axis.set_yticklabels([f"{value:.0f}" for value in collision_penalties])
    axis.set_xlabel("sensing-cost scale")
    axis.set_ylabel("collision utility")
    axis.set_title(
        "Task-aware Utility Advantage over EU without Observation"
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("mean-utility advantage")

    for row in range(advantages.shape[0]):
        for column in range(advantages.shape[1]):
            value = advantages[row, column]
            axis.text(
                column,
                row,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )

    path = output_dir / "parameter_sensitivity.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    return figure, path


def write_csv_files(
    seed_results,
    summary,
    collision_penalties,
    sensing_cost_scales,
    advantages,
    output_dir,
):
    seed_path = output_dir / "credible_policy_seed_results.csv"
    with seed_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(seed_results[0].keys())
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(seed_results)

    summary_path = output_dir / "credible_policy_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["policy", *summary[POLICY_NAMES[0]].keys()]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for policy_name in POLICY_NAMES:
            writer.writerow({"policy": policy_name, **summary[policy_name]})

    sensitivity_path = output_dir / "parameter_sensitivity.csv"
    with sensitivity_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["collision_utility", "sensing_cost_scale", "task_aware_advantage"]
        )
        for row, collision_utility in enumerate(collision_penalties):
            for column, cost_scale in enumerate(sensing_cost_scales):
                writer.writerow(
                    [collision_utility, cost_scale, advantages[row, column]]
                )
    return seed_path, summary_path, sensitivity_path


def main(show_figures=True):
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    entropy_threshold, thresholds, validation_utilities = tune_entropy_threshold()
    seed_results = run_main_comparison(entropy_threshold)
    summary = summarize_seed_results(seed_results)
    collision_penalties, sensing_cost_scales, advantages = run_sensitivity(
        entropy_threshold
    )

    main_figure, main_path = save_main_figure(summary, output_dir)
    sensitivity_figure, sensitivity_path = save_sensitivity_figure(
        collision_penalties,
        sensing_cost_scales,
        advantages,
        output_dir,
    )
    seed_path, summary_path, sensitivity_csv_path = write_csv_files(
        seed_results,
        summary,
        collision_penalties,
        sensing_cost_scales,
        advantages,
        output_dir,
    )

    tuning_figure, tuning_axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    tuning_axis.plot(thresholds, validation_utilities, color="#B279A2", linewidth=2.2)
    tuning_axis.axvline(
        entropy_threshold,
        color="black",
        linestyle="--",
        label=f"frozen threshold = {entropy_threshold:.3f}",
    )
    tuning_axis.set_xlabel("normalized entropy threshold")
    tuning_axis.set_ylabel("validation mean utility")
    tuning_axis.set_title("Entropy Threshold Tuned on Separate Validation Seeds")
    tuning_axis.legend()
    tuning_path = output_dir / "credible_entropy_tuning.png"
    tuning_figure.savefig(tuning_path, dpi=180, bbox_inches="tight")

    print("\nCREDIBLE POLICY EVALUATION")
    print("--------------------------")
    print(f"View accuracy: {VIEW_ACCURACY:.0%}")
    print(f"Independent test seeds: {len(RANDOM_SEEDS)}")
    print(f"Trials per test seed: {TRIALS_PER_SEED:,}")
    print(f"Total independent test scenarios: {len(RANDOM_SEEDS) * TRIALS_PER_SEED:,}")
    print(f"Frozen entropy threshold: {entropy_threshold:.3f}")

    for policy_name in POLICY_NAMES:
        item = summary[policy_name]
        print(f"\n{policy_name}")
        print(
            f"  utility: {item['mean_utility']:.3f} "
            f"+/- {item['mean_utility_ci95']:.3f} (95% CI)"
        )
        print(
            f"  success: {item['success_rate']:.1%} "
            f"+/- {item['success_rate_ci95']:.1%}"
        )
        print(
            f"  collision: {item['collision_rate']:.1%} "
            f"+/- {item['collision_rate_ci95']:.1%}"
        )
        print(
            f"  observe: {item['sensing_rate']:.1%} "
            f"+/- {item['sensing_rate_ci95']:.1%}"
        )

    print(f"\nSaved main figure: {main_path}")
    print(f"Saved sensitivity figure: {sensitivity_path}")
    print(f"Saved tuning figure: {tuning_path}")
    print(f"Saved seed results: {seed_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved sensitivity table: {sensitivity_csv_path}")
    print("Synthetic probabilistic simulation only; no physical-robot claim is made.")

    if show_figures:
        plt.show()
    else:
        plt.close(main_figure)
        plt.close(sensitivity_figure)
        plt.close(tuning_figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures without opening windows.",
    )
    arguments = parser.parse_args()
    main(show_figures=not arguments.no_show)
