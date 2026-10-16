from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


IMAGE_WIDTH = 240
IMAGE_HEIGHT = 180

HYPOTHESIS_NAMES = ["H1: left bump", "H2: right bump", "H3: no bump"]
HIDDEN_PARTS = ["left", "right", "none"]
BELIEF = np.array([0.4, 0.4, 0.2])

ACTION_NAMES = ["grasp-left", "grasp-right", "stop"]
UTILITY_MATRIX = np.array(
    [
        [-5.0, 1.0, -0.2],  # H1: left bump
        [1.0, -5.0, -0.2],  # H2: right bump
        [1.0, 1.0, -0.2],   # H3: no bump
    ]
)

SENSING_COST = 0.2
OBSERVATION_TOLERANCE = 1e-6

CAMERAS = {
    "front-again": {
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
        raise ValueError(f"Unknown hidden-part type: {hidden_part}")

    return meshes


def create_occluders():
    return [
        create_box([1.0, 0.10, 1.2], [-1.5, -0.65, -0.1]),
        create_box([1.0, 0.10, 1.2], [0.5, -0.65, -0.1]),
    ]


def render_depth(hidden_part, camera):
    scene = o3d.t.geometry.RaycastingScene()

    for mesh in create_shape_hypothesis(hidden_part) + create_occluders():
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=50.0,
        center=camera["center"],
        eye=camera["eye"],
        up=camera["up"],
        width_px=IMAGE_WIDTH,
        height_px=IMAGE_HEIGHT,
    )
    depth = scene.cast_rays(rays)["t_hit"].numpy()
    depth[~np.isfinite(depth)] = np.nan
    return depth


def observations_match(depth_a, depth_b):
    valid_a = np.isfinite(depth_a)
    valid_b = np.isfinite(depth_b)
    if not np.array_equal(valid_a, valid_b):
        return False
    if not np.any(valid_a):
        return True
    return bool(
        np.max(np.abs(depth_a[valid_a] - depth_b[valid_b]))
        <= OBSERVATION_TOLERANCE
    )


def observation_groups(depth_maps):
    """Group hypotheses that produce indistinguishable observations."""
    groups = []
    for hypothesis_index, depth in enumerate(depth_maps):
        placed = False
        for group in groups:
            representative = group[0]
            if observations_match(depth, depth_maps[representative]):
                group.append(hypothesis_index)
                placed = True
                break
        if not placed:
            groups.append([hypothesis_index])
    return groups


def expected_utility_after_view(groups):
    """Compute expected best task utility after seeing the view result."""
    total = 0.0
    group_decisions = []

    for group in groups:
        group_probability = float(BELIEF[group].sum())
        posterior = BELIEF[group] / group_probability
        expected_actions = posterior @ UTILITY_MATRIX[group]
        best_action_index = int(np.argmax(expected_actions))
        best_value = float(expected_actions[best_action_index])
        total += group_probability * best_value
        group_decisions.append(
            {
                "group": group,
                "probability": group_probability,
                "action": ACTION_NAMES[best_action_index],
                "value": best_value,
            }
        )

    return total, group_decisions


def readable_groups(groups):
    return " | ".join(
        "{" + ", ".join(HYPOTHESIS_NAMES[index].split(":")[0] for index in group) + "}"
        for group in groups
    )


def main(show_figure=True):
    current_expected_actions = BELIEF @ UTILITY_MATRIX
    current_best_utility = float(current_expected_actions.max())

    all_depth_maps = {}
    view_results = {}

    for view_name, camera in CAMERAS.items():
        depth_maps = [render_depth(part, camera) for part in HIDDEN_PARTS]
        groups = observation_groups(depth_maps)
        utility_after_view, group_decisions = expected_utility_after_view(groups)
        gross_value = utility_after_view - current_best_utility
        net_value = gross_value - SENSING_COST

        all_depth_maps[view_name] = depth_maps
        view_results[view_name] = {
            "groups": groups,
            "group_decisions": group_decisions,
            "utility_after_view": utility_after_view,
            "gross_value": gross_value,
            "net_value": net_value,
        }

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    observation_figure, axes = plt.subplots(
        len(CAMERAS),
        len(HYPOTHESIS_NAMES),
        figsize=(11, 8),
        constrained_layout=True,
    )

    for row, (view_name, depth_maps) in enumerate(all_depth_maps.items()):
        finite_values = np.concatenate(
            [depth[np.isfinite(depth)] for depth in depth_maps]
        )
        vmin = float(finite_values.min())
        vmax = float(finite_values.max())

        for column, (hypothesis_name, depth) in enumerate(
            zip(HYPOTHESIS_NAMES, depth_maps)
        ):
            axes[row, column].imshow(
                np.ma.masked_invalid(depth),
                cmap="viridis_r",
                vmin=vmin,
                vmax=vmax,
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(hypothesis_name)
            if column == 0:
                groups_text = readable_groups(view_results[view_name]["groups"])
                axes[row, column].set_ylabel(
                    f"{view_name}\nvisible groups: {groups_text}",
                    fontsize=10,
                )

    observation_figure.suptitle(
        "What Each Candidate Camera View Can Distinguish",
        fontsize=15,
        fontweight="bold",
    )
    observation_path = output_dir / "candidate_view_observations.png"
    observation_figure.savefig(observation_path, dpi=180, bbox_inches="tight")

    view_names = list(view_results)
    net_values = [view_results[name]["net_value"] for name in view_names]
    colors = ["#9D9D9D" if value <= 0 else "#59A14F" for value in net_values]

    value_figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bars = axis.bar(view_names, net_values, color=colors)
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set_ylabel("expected task improvement - sensing cost")
    axis.set_title(
        "Select a View by Its Expected Effect on the Task",
        fontweight="bold",
    )
    axis.set_ylim(min(-0.35, min(net_values) - 0.1), max(net_values) + 0.35)

    for bar, value in zip(bars, net_values):
        vertical_alignment = "bottom" if value >= 0 else "top"
        offset = 0.04 if value >= 0 else -0.04
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.2f}",
            ha="center",
            va=vertical_alignment,
            fontweight="bold",
        )

    best_view = max(view_names, key=lambda name: view_results[name]["net_value"])
    best_value = view_results[best_view]["net_value"]
    if best_value > 0:
        decision_text = f"Decision: observe from {best_view}"
    else:
        decision_text = "Decision: do not take another observation"
    axis.text(
        0.5,
        0.94,
        decision_text,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )

    value_path = output_dir / "view_selection_value.png"
    value_figure.savefig(value_path, dpi=180, bbox_inches="tight")

    print("\nCURRENT DECISION")
    print("----------------")
    print(f"Expected utilities: {dict(zip(ACTION_NAMES, current_expected_actions.round(2)))}")
    print(f"Best current utility: {current_best_utility:.2f}")

    print("\nCANDIDATE VIEWS")
    print("---------------")
    for view_name, result in view_results.items():
        print(f"\n{view_name}")
        print(f"  distinguishable groups: {readable_groups(result['groups'])}")
        print(f"  utility after view: {result['utility_after_view']:.2f}")
        print(f"  net view value: {result['net_value']:.2f}")
        for decision in result["group_decisions"]:
            group_name = ", ".join(
                HYPOTHESIS_NAMES[index].split(":")[0]
                for index in decision["group"]
            )
            print(f"    if observation indicates {{{group_name}}}: {decision['action']}")

    print(f"\nSelected view: {best_view}")
    print(f"Saved observations: {observation_path}")
    print(f"Saved view values: {value_path}")

    if show_figure:
        plt.show()
    else:
        plt.close(observation_figure)
        plt.close(value_figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures without opening windows.",
    )
    arguments = parser.parse_args()
    main(show_figure=not arguments.no_show)
