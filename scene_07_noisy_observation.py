from pathlib import Path
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


RANDOM_SEED = 20261015
NUMBER_OF_TRIALS = 8_000

SUCCESS_UTILITY = 1.0
COLLISION_UTILITY = -5.0
STOP_UTILITY = -0.2

H1_LEFT_BUMP = 0
H2_RIGHT_BUMP = 1
H3_NO_BUMP = 2

TASK_FRONT_PUSH = 0
TASK_SIDE_GRASP = 1

ACTION_NAMES = ["grasp-left", "grasp-right", "stop"]
SIDE_UTILITY = np.array(
    [
        [-5.0, 1.0, -0.2],
        [1.0, -5.0, -0.2],
        [1.0, 1.0, -0.2],
    ]
)

POLICY_NAMES = ["MAP only", "Always observe", "Noise-aware task-aware"]
POLICY_COLORS = ["#4C78A8", "#F58518", "#59A14F"]


def observation_likelihoods(target_shape, accuracy, observation):
    """Likelihood P(observation | shape) for a noisy binary side view."""
    likelihoods = np.full(3, 1.0 - accuracy, dtype=float)
    likelihoods[target_shape] = accuracy
    if observation == 0:
        likelihoods = 1.0 - likelihoods
    return likelihoods


def observation_probability(belief, target_shape, accuracy, observation):
    likelihoods = observation_likelihoods(target_shape, accuracy, observation)
    return float(belief @ likelihoods)


def update_belief(belief, target_shape, accuracy, observation):
    """Bayes update after 'bump seen' (1) or 'not seen' (0)."""
    likelihoods = observation_likelihoods(target_shape, accuracy, observation)
    unnormalized = belief * likelihoods
    return unnormalized / unnormalized.sum()


def best_side_action(belief):
    expected_utilities = belief @ SIDE_UTILITY
    action_index = int(np.argmax(expected_utilities))
    return ACTION_NAMES[action_index], float(expected_utilities[action_index])


def expected_value_after_view(belief, target_shape, accuracy):
    expected_value = 0.0
    for observation in (0, 1):
        probability = observation_probability(
            belief, target_shape, accuracy, observation
        )
        posterior = update_belief(belief, target_shape, accuracy, observation)
        _, posterior_value = best_side_action(posterior)
        expected_value += probability * posterior_value
    return expected_value


def best_view(belief, accuracy):
    candidates = {
        "left-view": expected_value_after_view(
            belief, H1_LEFT_BUMP, accuracy
        ),
        "right-view": expected_value_after_view(
            belief, H2_RIGHT_BUMP, accuracy
        ),
    }
    view_name = max(candidates, key=candidates.get)
    target_shape = H1_LEFT_BUMP if view_name == "left-view" else H2_RIGHT_BUMP
    return view_name, target_shape, candidates[view_name]


def sample_observation(true_shape, target_shape, accuracy, random_number):
    bump_really_visible = true_shape == target_shape
    correct = random_number < accuracy
    return int(bump_really_visible if correct else not bump_really_visible)


def map_action(belief):
    most_likely_shape = int(np.argmax(belief))
    if most_likely_shape == H1_LEFT_BUMP:
        return "grasp-right"
    return "grasp-left"


def realized_side_utility(action, true_shape):
    if action == "stop":
        return STOP_UTILITY, False, False, True
    if action == "grasp-left":
        collision = true_shape == H1_LEFT_BUMP
    elif action == "grasp-right":
        collision = true_shape == H2_RIGHT_BUMP
    else:
        raise ValueError(f"Unknown action: {action}")

    if collision:
        return COLLISION_UTILITY, False, True, False
    return SUCCESS_UTILITY, True, False, False


def generate_trials(number_of_trials, seed):
    rng = np.random.default_rng(seed)
    beliefs = rng.dirichlet([1.2, 1.2, 0.8], size=number_of_trials)
    tasks = rng.integers(0, 2, size=number_of_trials)
    sensing_costs = rng.uniform(0.05, 0.60, size=number_of_trials)

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


def evaluate_policy(policy_name, trial_data, accuracy):
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
            # Hidden side bumps do not change the front-push action.
            observed = policy_name == "Always observe"
            utility = SUCCESS_UTILITY - (cost if observed else 0.0)
            success, collision, stopped = True, False, False
        else:
            current_action, current_value = best_side_action(belief)

            if policy_name == "MAP only":
                action = map_action(belief)
            else:
                _, target_shape, post_view_value = best_view(belief, accuracy)
                observe_value = post_view_value - cost
                if policy_name == "Always observe":
                    observed = True
                elif policy_name == "Noise-aware task-aware":
                    observed = observe_value > current_value
                else:
                    raise ValueError(f"Unknown policy: {policy_name}")

                if observed:
                    observation = sample_observation(
                        true_shape,
                        target_shape,
                        accuracy,
                        float(observation_random),
                    )
                    posterior = update_belief(
                        belief, target_shape, accuracy, observation
                    )
                    action, _ = best_side_action(posterior)
                else:
                    action = current_action

            utility, success, collision, stopped = realized_side_utility(
                action, true_shape
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


def main(show_figure=True):
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # One transparent example of a Bayesian belief update.
    example_prior = np.array([0.4, 0.4, 0.2])
    example_accuracy = 0.80
    posterior_seen = update_belief(
        example_prior, H1_LEFT_BUMP, example_accuracy, observation=1
    )
    posterior_not_seen = update_belief(
        example_prior, H1_LEFT_BUMP, example_accuracy, observation=0
    )

    accuracies = np.linspace(0.50, 1.00, 11)
    example_cost = 0.20
    _, current_value = best_side_action(example_prior)
    net_view_values = np.array(
        [
            expected_value_after_view(example_prior, H1_LEFT_BUMP, accuracy)
            - current_value
            - example_cost
            for accuracy in accuracies
        ]
    )

    trial_data = generate_trials(NUMBER_OF_TRIALS, RANDOM_SEED)
    results = {
        policy_name: [
            evaluate_policy(policy_name, trial_data, float(accuracy))
            for accuracy in accuracies
        ]
        for policy_name in POLICY_NAMES
    }

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # A. Show the prior and two possible posteriors after a noisy left-side view.
    labels = ["prior", "bump seen", "not seen"]
    belief_matrix = np.vstack([example_prior, posterior_seen, posterior_not_seen])
    bottom = np.zeros(3)
    shape_labels = ["H1: left bump", "H2: right bump", "H3: no bump"]
    colors = ["#4C78A8", "#F58518", "#B279A2"]
    for shape_index, (shape_label, color) in enumerate(zip(shape_labels, colors)):
        axes[0, 0].bar(
            labels,
            belief_matrix[:, shape_index],
            bottom=bottom,
            label=shape_label,
            color=color,
        )
        bottom += belief_matrix[:, shape_index]
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].set_ylabel("belief probability")
    axes[0, 0].set_title("A. Bayes update after a noisy left view (80% accurate)")
    axes[0, 0].legend(fontsize=8, loc="lower left")

    # B. Show when this imperfect view becomes worth its cost.
    axes[0, 1].plot(
        accuracies,
        net_view_values,
        marker="o",
        linewidth=2.2,
        color="#59A14F",
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[0, 1].fill_between(
        accuracies,
        net_view_values,
        0.0,
        where=net_view_values > 0.0,
        color="#59A14F",
        alpha=0.15,
        label="worth observing",
    )
    axes[0, 1].set_xlabel("view accuracy")
    axes[0, 1].set_ylabel("net value of the view")
    axes[0, 1].set_title("B. Information must be useful enough to justify cost")
    axes[0, 1].legend()

    # C-D. Random-trial robustness curves.
    for policy_name, color in zip(POLICY_NAMES, POLICY_COLORS):
        axes[1, 0].plot(
            accuracies,
            [item["mean_utility"] for item in results[policy_name]],
            marker="o",
            linewidth=2.0,
            color=color,
            label=policy_name,
        )
        axes[1, 1].plot(
            accuracies,
            [item["sensing_rate"] for item in results[policy_name]],
            marker="o",
            linewidth=2.0,
            color=color,
            label=policy_name,
        )

    axes[1, 0].set_xlabel("view accuracy")
    axes[1, 0].set_ylabel("mean utility")
    axes[1, 0].set_title(f"C. Performance over {NUMBER_OF_TRIALS:,} noisy trials")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_xlabel("view accuracy")
    axes[1, 1].set_ylabel("extra-observation rate")
    axes[1, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].set_title("D. Task-aware sensing adapts to view reliability")
    axes[1, 1].legend(fontsize=8)

    figure.suptitle(
        "Selective Verification with Imperfect Observations",
        fontsize=15,
        fontweight="bold",
    )
    figure_path = output_dir / "noisy_observation_robustness.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")

    csv_path = output_dir / "noisy_observation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "view_accuracy",
                "policy",
                "mean_utility",
                "success_rate",
                "collision_rate",
                "sensing_rate",
                "stop_rate",
            ]
        )
        for accuracy_index, accuracy in enumerate(accuracies):
            for policy_name in POLICY_NAMES:
                item = results[policy_name][accuracy_index]
                writer.writerow(
                    [
                        f"{accuracy:.2f}",
                        policy_name,
                        item["mean_utility"],
                        item["success_rate"],
                        item["collision_rate"],
                        item["sensing_rate"],
                        item["stop_rate"],
                    ]
                )

    print("\nNOISY OBSERVATION ROBUSTNESS")
    print("----------------------------")
    print(f"Prior belief: {example_prior.tolist()}")
    print(f"Left-view accuracy: {example_accuracy:.0%}")
    print(f"Posterior if bump is seen: {posterior_seen.round(3).tolist()}")
    print(f"Posterior if bump is not seen: {posterior_not_seen.round(3).tolist()}")
    print("\nSelected robustness points:")
    for selected_accuracy in (0.60, 0.70, 0.80, 0.90, 1.00):
        index = int(np.argmin(np.abs(accuracies - selected_accuracy)))
        task_aware = results["Noise-aware task-aware"][index]
        print(
            f"  accuracy={accuracies[index]:.0%}: "
            f"utility={task_aware['mean_utility']:.3f}, "
            f"success={task_aware['success_rate']:.1%}, "
            f"collision={task_aware['collision_rate']:.1%}, "
            f"observe={task_aware['sensing_rate']:.1%}"
        )
    print(f"\nSaved figure: {figure_path}")
    print(f"Saved table: {csv_path}")
    print("Synthetic simulation only: these are not physical-robot results.")

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
