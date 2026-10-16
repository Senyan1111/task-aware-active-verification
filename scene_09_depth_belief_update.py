from pathlib import Path
import argparse
import csv

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import open3d as o3d


IMAGE_WIDTH = 240
IMAGE_HEIGHT = 180
RANDOM_SEED = 20261015

HYPOTHESIS_NAMES = ["H1: left bump", "H2: right bump", "H3: no bump"]
HIDDEN_PARTS = ["left", "right", "none"]
TRUE_SHAPE_INDEX = 0
PRIOR = np.array([0.4, 0.4, 0.2], dtype=float)

ACTION_NAMES = ["grasp-left", "grasp-right", "stop"]
UTILITY_MATRIX = np.array(
    [
        [-5.0, 1.0, -0.2],
        [1.0, -5.0, -0.2],
        [1.0, 1.0, -0.2],
    ],
    dtype=float,
)

SENSING_COST = 0.20
DEPTH_NOISE_STD = 0.01
DROPOUT_PROBABILITY = 0.03
SILHOUETTE_MISMATCH_PENALTY = 0.25
LIKELIHOOD_TEMPERATURE = 0.055
PLANNING_SAMPLES_PER_HYPOTHESIS = 8

BODY_COLOR = np.array([0.30, 0.65, 0.90])
BUMP_COLOR = np.array([1.00, 0.55, 0.10])
OCCLUDER_COLOR = np.array([0.60, 0.60, 0.60])

CAMERAS = {
    "front-view": {
        "eye": [0.0, -4.0, 0.65],
        "center": [0.0, 0.0, 0.4],
        "up": [0.0, 0.0, 1.0],
    },
    "left-view": {
        "eye": [-4.0, -0.1, 0.65],
        "center": [0.0, 0.0, 0.4],
        "up": [0.0, 0.0, 1.0],
    },
    "right-view": {
        "eye": [4.0, -0.1, 0.65],
        "center": [0.0, 0.0, 0.4],
        "up": [0.0, 0.0, 1.0],
    },
}


def create_box(size, position):
    box = o3d.geometry.TriangleMesh.create_box(
        width=size[0],
        height=size[1],
        depth=size[2],
    )
    box.translate(position)
    box.compute_vertex_normals()
    return box


def create_shape_hypothesis(hidden_part):
    meshes = [create_box([1.4, 0.8, 0.8], [-0.7, -0.4, 0.0])]
    if hidden_part == "left":
        meshes.append(create_box([0.55, 0.35, 0.35], [-1.25, -0.175, 0.225]))
    elif hidden_part == "right":
        meshes.append(create_box([0.55, 0.35, 0.35], [0.70, -0.175, 0.225]))
    elif hidden_part != "none":
        raise ValueError(f"Unknown hidden part: {hidden_part}")
    return meshes


def create_occluders():
    return [
        create_box([1.0, 0.10, 1.2], [-1.5, -0.65, -0.1]),
        create_box([1.0, 0.10, 1.2], [0.5, -0.65, -0.1]),
    ]


def render_depth_and_semantics(hidden_part, camera):
    scene = o3d.t.geometry.RaycastingScene()
    geometry_colors = {}

    shape_meshes = create_shape_hypothesis(hidden_part)
    for mesh_index, mesh in enumerate(shape_meshes):
        geometry_id = scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        )
        geometry_colors[geometry_id] = BODY_COLOR if mesh_index == 0 else BUMP_COLOR

    for mesh in create_occluders():
        geometry_id = scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        )
        geometry_colors[geometry_id] = OCCLUDER_COLOR

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=50.0,
        center=camera["center"],
        eye=camera["eye"],
        up=camera["up"],
        width_px=IMAGE_WIDTH,
        height_px=IMAGE_HEIGHT,
    )
    answers = scene.cast_rays(rays)
    depth = answers["t_hit"].numpy()
    geometry_ids = answers["geometry_ids"].numpy()
    depth[~np.isfinite(depth)] = np.nan

    semantic_image = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=float)
    for geometry_id, color in geometry_colors.items():
        semantic_image[geometry_ids == geometry_id] = color
    return depth, semantic_image


def add_sensor_noise(clean_depth, rng):
    """Return a noisy observation plus a mask identifying non-dropped rays."""
    available = rng.random(clean_depth.shape) >= DROPOUT_PROBABILITY
    observation = clean_depth.copy()
    finite = np.isfinite(clean_depth) & available
    observation[finite] += rng.normal(0.0, DEPTH_NOISE_STD, size=np.count_nonzero(finite))
    observation[~available] = np.nan
    return observation, available


def depth_fit_error(observation, available, prediction):
    """Robust geometric discrepancy using depth and foreground silhouette."""
    observation_finite = np.isfinite(observation)
    prediction_finite = np.isfinite(prediction)
    active = available & (observation_finite | prediction_finite)
    if not np.any(active):
        return 0.0

    both_finite = active & observation_finite & prediction_finite
    silhouette_mismatch = active & (observation_finite != prediction_finite)

    squared_error = 0.0
    if np.any(both_finite):
        residual = observation[both_finite] - prediction[both_finite]
        squared_error += float(np.sum(residual**2))
    squared_error += float(
        np.count_nonzero(silhouette_mismatch)
        * SILHOUETTE_MISMATCH_PENALTY**2
    )
    return float(np.sqrt(squared_error / np.count_nonzero(active)))


def posterior_from_depth(prior, observation, available, predictions):
    errors = np.array(
        [
            depth_fit_error(observation, available, prediction)
            for prediction in predictions
        ]
    )
    log_likelihoods = -0.5 * (errors / LIKELIHOOD_TEMPERATURE) ** 2
    log_likelihoods -= np.max(log_likelihoods)
    likelihoods = np.exp(log_likelihoods)
    unnormalized = prior * likelihoods
    posterior = unnormalized / unnormalized.sum()
    return posterior, errors, likelihoods


def best_action(belief):
    expected_utilities = belief @ UTILITY_MATRIX
    index = int(np.argmax(expected_utilities))
    return ACTION_NAMES[index], float(expected_utilities[index]), expected_utilities


def estimate_view_value(prior, predictions, rng):
    """Monte Carlo estimate of task value after a rendered noisy observation."""
    expected_post_observation_value = 0.0
    for true_index, true_probability in enumerate(prior):
        conditional_total = 0.0
        for _ in range(PLANNING_SAMPLES_PER_HYPOTHESIS):
            observation, available = add_sensor_noise(predictions[true_index], rng)
            posterior, _, _ = posterior_from_depth(
                prior, observation, available, predictions
            )
            _, posterior_value, _ = best_action(posterior)
            conditional_total += posterior_value
        expected_post_observation_value += true_probability * (
            conditional_total / PLANNING_SAMPLES_PER_HYPOTHESIS
        )

    _, current_value, _ = best_action(prior)
    net_value = expected_post_observation_value - current_value - SENSING_COST
    return float(expected_post_observation_value), float(net_value)


def finite_range(depth_maps):
    values = np.concatenate(
        [depth[np.isfinite(depth)] for depth in depth_maps if np.any(np.isfinite(depth))]
    )
    return float(values.min()), float(values.max())


def show_depth(axis, depth, title, vmin, vmax):
    image = axis.imshow(
        np.ma.masked_invalid(depth),
        cmap="viridis_r",
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_title(title, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("white")
    return image


def show_semantics(axis, semantic_image, title):
    axis.imshow(semantic_image)
    axis.set_title(title, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("white")


def main(show_figure=True):
    rng_planning = np.random.default_rng(RANDOM_SEED)
    rng_actual = np.random.default_rng(RANDOM_SEED + 1)

    all_predictions = {}
    all_semantics = {}
    for view_name, camera in CAMERAS.items():
        view_depths = []
        view_semantics = []
        for hidden_part in HIDDEN_PARTS:
            depth, semantics = render_depth_and_semantics(hidden_part, camera)
            view_depths.append(depth)
            view_semantics.append(semantics)
        all_predictions[view_name] = view_depths
        all_semantics[view_name] = view_semantics

    current_action, current_value, current_utilities = best_action(PRIOR)
    view_results = {}
    for view_name, predictions in all_predictions.items():
        post_value, net_value = estimate_view_value(
            PRIOR, predictions, rng_planning
        )
        view_results[view_name] = {
            "post_value": post_value,
            "net_value": net_value,
        }

    selected_view = max(
        view_results,
        key=lambda name: view_results[name]["net_value"],
    )
    selected_predictions = all_predictions[selected_view]
    selected_semantics = all_semantics[selected_view]

    true_clean_depth = selected_predictions[TRUE_SHAPE_INDEX]
    noisy_observation, available = add_sensor_noise(true_clean_depth, rng_actual)
    posterior, fit_errors, likelihoods = posterior_from_depth(
        PRIOR,
        noisy_observation,
        available,
        selected_predictions,
    )
    final_action, final_value, final_utilities = best_action(posterior)

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    figure, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)

    show_semantics(
        axes[0, 0],
        all_semantics["front-view"][TRUE_SHAPE_INDEX],
        "A. Current front semantic view\n(H1/H2/H3 are occluded)",
    )
    axes[0, 0].legend(
        handles=[
            Patch(color=BODY_COLOR, label="object body"),
            Patch(color=BUMP_COLOR, label="hidden bump"),
            Patch(color=OCCLUDER_COLOR, label="occluder"),
        ],
        loc="lower center",
        fontsize=7,
    )

    selected_vmin, selected_vmax = finite_range(selected_predictions)
    for column, (name, semantics) in enumerate(
        zip(HYPOTHESIS_NAMES, selected_semantics), start=1
    ):
        show_semantics(
            axes[0, column],
            semantics,
            f"{selected_view} semantic prediction\n{name}",
        )

    noisy_image = show_depth(
        axes[1, 0],
        noisy_observation,
        f"B. Actual noisy depth\ntrue shape = {HYPOTHESIS_NAMES[TRUE_SHAPE_INDEX]}",
        selected_vmin,
        selected_vmax,
    )
    noisy_image.set_cmap("gray_r")
    figure.colorbar(noisy_image, ax=axes[1, 0], fraction=0.046, pad=0.03)

    error_bars = axes[1, 1].bar(
        ["H1", "H2", "H3"],
        fit_errors,
        color=["#4C78A8", "#F58518", "#B279A2"],
    )
    axes[1, 1].set_title("C. Depth fit error\n(lower is a better match)")
    axes[1, 1].set_ylabel("geometric RMSE")
    for bar, value in zip(error_bars, fit_errors):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    x_positions = np.arange(3)
    width = 0.36
    axes[1, 2].bar(
        x_positions - width / 2,
        PRIOR,
        width,
        label="prior",
        color="#9D9D9D",
    )
    axes[1, 2].bar(
        x_positions + width / 2,
        posterior,
        width,
        label="posterior",
        color="#59A14F",
    )
    axes[1, 2].set_xticks(x_positions, ["H1", "H2", "H3"])
    axes[1, 2].set_ylim(0.0, 1.0)
    axes[1, 2].set_ylabel("belief probability")
    axes[1, 2].set_title("D. Belief update from depth")
    axes[1, 2].legend()

    view_names = list(view_results)
    net_values = [view_results[name]["net_value"] for name in view_names]
    colors = [
        "#59A14F" if name == selected_view else "#9D9D9D"
        for name in view_names
    ]
    value_bars = axes[1, 3].bar(view_names, net_values, color=colors)
    axes[1, 3].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 3].tick_params(axis="x", rotation=18)
    axes[1, 3].set_ylabel("EVSI - sensing cost")
    axes[1, 3].set_title(
        f"E. View choice and final action\nselected: {selected_view}; action: {final_action}"
    )
    for bar, value in zip(value_bars, net_values):
        axes[1, 3].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:+.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
            fontweight="bold",
        )

    figure.suptitle(
        "Closed Loop: 3D Depth Observation to Belief Update and Task Action",
        fontsize=15,
        fontweight="bold",
    )
    figure_path = output_dir / "depth_belief_update_closed_loop.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")

    csv_path = output_dir / "depth_belief_update_trace.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "hypothesis",
                "prior",
                "depth_fit_error",
                "relative_likelihood",
                "posterior",
            ]
        )
        for index, name in enumerate(HYPOTHESIS_NAMES):
            writer.writerow(
                [name, PRIOR[index], fit_errors[index], likelihoods[index], posterior[index]]
            )

    print("\n3D DEPTH-BASED BELIEF UPDATE")
    print("----------------------------")
    print(f"True hidden shape: {HYPOTHESIS_NAMES[TRUE_SHAPE_INDEX]}")
    print(f"Prior: {PRIOR.round(3).tolist()}")
    print(f"Current action: {current_action} (utility {current_value:.3f})")
    print(f"Current action utilities: {current_utilities.round(3).tolist()}")
    print("\nCandidate view values:")
    for view_name, result in view_results.items():
        print(
            f"  {view_name}: post-observation utility="
            f"{result['post_value']:.3f}, net value={result['net_value']:+.3f}"
        )
    print(f"Selected view: {selected_view}")
    print(f"Depth fit errors: {fit_errors.round(4).tolist()}")
    print(f"Posterior: {posterior.round(3).tolist()}")
    print(f"Final action utilities: {final_utilities.round(3).tolist()}")
    print(f"Final action: {final_action} (utility {final_value:.3f})")
    print(f"Saved figure: {figure_path}")
    print(f"Saved trace: {csv_path}")

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
