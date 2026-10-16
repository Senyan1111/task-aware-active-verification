from pathlib import Path
import argparse
import csv
import io

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from PIL import Image


IMAGE_WIDTH = 360
IMAGE_HEIGHT = 280
RANDOM_SEED = 20261016

BODY_COLOR = np.array([0.27, 0.58, 0.86])
HAZARD_COLOR = np.array([0.96, 0.45, 0.10])
DISTRACTOR_COLOR = np.array([0.20, 0.72, 0.67])
OCCLUDER_COLOR = np.array([0.58, 0.58, 0.58])

PRIOR = np.array([0.4, 0.4, 0.2], dtype=float)
SENSING_COST = 0.20

CLASS_SPECS = [
    ("H1", "left-side hazard", "grasp-right"),
    ("H2", "right-side hazard", "grasp-left"),
    ("H3", "no side hazard", "either side"),
]

FAMILY_SPECS = [
    ("A", "flange", "simple"),
    ("B", "offset plate", "moderate"),
    ("C", "hook", "challenging"),
    ("D", "round boss", "held-out morphology"),
]

DISPLAY_CAMERA = {
    "eye": [0.0, -4.2, 2.5],
    "center": [0.0, 0.0, 0.45],
    "up": [0.0, 0.0, 1.0],
}

FRONT_CAMERA = {
    "eye": [0.0, -4.5, 0.65],
    "center": [0.0, 0.0, 0.42],
    "up": [0.0, 0.0, 1.0],
}


def create_box(size, position):
    mesh = o3d.geometry.TriangleMesh.create_box(*size)
    mesh.translate(position)
    mesh.compute_vertex_normals()
    return mesh


def create_cylinder(radius, height, center, axis="z"):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=height,
        resolution=32,
        split=4,
    )
    if axis == "x":
        mesh.rotate(
            mesh.get_rotation_matrix_from_xyz((0.0, np.pi / 2.0, 0.0)),
            center=(0.0, 0.0, 0.0),
        )
    elif axis == "y":
        mesh.rotate(
            mesh.get_rotation_matrix_from_xyz((np.pi / 2.0, 0.0, 0.0)),
            center=(0.0, 0.0, 0.0),
        )
    mesh.translate(center)
    mesh.compute_vertex_normals()
    return mesh


def add_colored(parts, mesh, color, role):
    parts.append({"mesh": mesh, "color": color, "role": role})


def build_shape(class_index, family_index, params):
    """Build one task-labelled member of a parameterized mechanical-part family."""
    width, depth, height = params["body_size"]
    parts = []
    add_colored(
        parts,
        create_box([width, depth, height], [-width / 2, -depth / 2, 0.0]),
        BODY_COLOR,
        "body",
    )

    # These structures make appearance more varied without changing which side is safe.
    ridge_width = 0.30 * width
    ridge_x = -0.5 * ridge_width + params["ridge_offset"]
    add_colored(
        parts,
        create_box(
            [ridge_width, 0.18 * depth, 0.10],
            [ridge_x, -0.09 * depth, height],
        ),
        DISTRACTOR_COLOR,
        "task-irrelevant ridge",
    )
    if family_index >= 1:
        add_colored(
            parts,
            create_cylinder(
                radius=0.10,
                height=0.14,
                center=[-0.22 * width, 0.18 * depth, height + 0.07],
            ),
            DISTRACTOR_COLOR,
            "task-irrelevant boss",
        )
    if family_index >= 2:
        add_colored(
            parts,
            create_box(
                [0.12 * width, 0.30 * depth, 0.10],
                [0.18 * width, -0.15 * depth, height],
            ),
            DISTRACTOR_COLOR,
            "task-irrelevant rib",
        )

    # H3 has visual detail, but no structure invading either side-grasp corridor.
    if class_index == 2:
        return parts

    # The ray-camera horizontal axis is mirrored relative to world x in this view;
    # choose the sign so the labelled left/right hazards also appear left/right.
    side = 1.0 if class_index == 0 else -1.0
    extension = params["extension"]
    feature_depth = params["feature_depth"]
    feature_height = params["feature_height"]
    side_x = -width / 2 - extension if side < 0 else width / 2
    z0 = params["feature_z"]

    if family_index == 0:  # rectangular flange
        add_colored(
            parts,
            create_box(
                [extension, feature_depth, feature_height],
                [side_x, -feature_depth / 2, z0],
            ),
            HAZARD_COLOR,
            "grasp hazard",
        )
    elif family_index == 1:  # offset plate, only part of the side is obstructed
        y0 = params["feature_y"] - feature_depth / 2
        add_colored(
            parts,
            create_box(
                [extension, feature_depth, feature_height],
                [side_x, y0, z0],
            ),
            HAZARD_COLOR,
            "grasp hazard",
        )
    elif family_index == 2:  # an L-shaped hook
        arm_height = 0.16
        add_colored(
            parts,
            create_box(
                [extension, feature_depth, arm_height],
                [side_x, -feature_depth / 2, z0 + feature_height - arm_height],
            ),
            HAZARD_COLOR,
            "grasp hazard",
        )
        tip_width = min(0.18, extension)
        tip_x = side_x if side < 0 else side_x + extension - tip_width
        add_colored(
            parts,
            create_box(
                [tip_width, feature_depth, feature_height],
                [tip_x, -feature_depth / 2, z0],
            ),
            HAZARD_COLOR,
            "grasp hazard",
        )
    else:  # a rounded morphology reserved as a simple generalization test
        cylinder_center_x = (
            -width / 2 - extension / 2 if side < 0 else width / 2 + extension / 2
        )
        add_colored(
            parts,
            create_cylinder(
                radius=0.5 * feature_height,
                height=extension,
                center=[cylinder_center_x, params["feature_y"], z0 + feature_height / 2],
                axis="x",
            ),
            HAZARD_COLOR,
            "grasp hazard",
        )
    return parts


def render_semantics(parts, camera=DISPLAY_CAMERA, fov_deg=34.0):
    scene = o3d.t.geometry.RaycastingScene()
    colors_by_id = {}
    for part in parts:
        geometry_id = scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(part["mesh"])
        )
        colors_by_id[geometry_id] = part["color"]

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=fov_deg,
        center=camera["center"],
        eye=camera["eye"],
        up=camera["up"],
        width_px=IMAGE_WIDTH,
        height_px=IMAGE_HEIGHT,
    )
    answers = scene.cast_rays(rays)
    geometry_ids = answers["geometry_ids"].numpy()
    image = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=float)
    for geometry_id, color in colors_by_id.items():
        image[geometry_ids == geometry_id] = color
    return image


def add_front_occluders(parts, params):
    """Hide both side corridors so the initial view cannot reveal H1/H2/H3."""
    width, depth, height = params["body_size"]
    covered_width = 0.58
    slab_depth = 0.16
    slab_height = height + 0.48
    result = list(parts)
    for x0 in (-width / 2 - 0.42, width / 2 - covered_width + 0.42):
        add_colored(
            result,
            create_box(
                [covered_width, slab_depth, slab_height],
                [x0, -depth / 2 - 0.34, -0.10],
            ),
            OCCLUDER_COLOR,
            "occluder",
        )
    return result


def sample_family_parameters(rng):
    parameters = []
    for family_index in range(len(FAMILY_SPECS)):
        width = rng.uniform(1.45, 1.80)
        depth = rng.uniform(0.78, 1.02)
        height = rng.uniform(0.70, 0.96)
        parameters.append(
            {
                "body_size": [width, depth, height],
                "extension": rng.uniform(0.22, 0.48) * (1.0 - 0.04 * family_index),
                "feature_depth": rng.uniform(0.28, 0.58),
                "feature_height": rng.uniform(0.22, 0.48),
                "feature_y": rng.uniform(-0.18, 0.18),
                "feature_z": rng.uniform(0.12, max(0.13, height - 0.45)),
                "ridge_offset": rng.uniform(-0.18, 0.18),
            }
        )
    return parameters


def write_parameter_table(parameters, output_dir):
    path = output_dir / "parameterized_shape_family.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "family",
            "difficulty",
            "body_width",
            "body_depth",
            "body_height",
            "hazard_extension",
            "hazard_depth",
            "hazard_height",
            "hazard_y",
            "hazard_z",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for family_spec, params in zip(FAMILY_SPECS, parameters):
            writer.writerow(
                {
                    "family": family_spec[1],
                    "difficulty": family_spec[2],
                    "body_width": params["body_size"][0],
                    "body_depth": params["body_size"][1],
                    "body_height": params["body_size"][2],
                    "hazard_extension": params["extension"],
                    "hazard_depth": params["feature_depth"],
                    "hazard_height": params["feature_height"],
                    "hazard_y": params["feature_y"],
                    "hazard_z": params["feature_z"],
                }
            )
    return path


def make_shape_gallery(parameters, output_dir):
    figure, axes = plt.subplots(3, 4, figsize=(14, 9), constrained_layout=True)
    for class_index, (class_name, class_description, safe_action) in enumerate(
        CLASS_SPECS
    ):
        for family_index, (family_name, morphology, difficulty) in enumerate(
            FAMILY_SPECS
        ):
            parts = build_shape(class_index, family_index, parameters[family_index])
            image = render_semantics(parts)
            axis = axes[class_index, family_index]
            axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            if class_index == 0:
                axis.set_title(
                    f"{family_name}. {morphology}\n{difficulty}",
                    fontsize=11,
                    fontweight="bold",
                )
            if family_index == 0:
                axis.set_ylabel(
                    f"{class_name}: {class_description}\nsafe: {safe_action}",
                    fontsize=10,
                    fontweight="bold",
                )

    figure.legend(
        handles=[
            Patch(color=BODY_COLOR, label="shared object body"),
            Patch(color=HAZARD_COLOR, label="task-relevant grasp hazard"),
            Patch(color=DISTRACTOR_COLOR, label="task-irrelevant geometry"),
        ],
        loc="lower center",
        ncol=3,
        fontsize=10,
        frameon=True,
    )
    figure.suptitle(
        "Parameterized Shape Families: Appearance Varies, Task Meaning Stays Controlled",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.045,
        "Each column shares body parameters across H1/H2/H3; only the side hazard changes.",
        ha="center",
        fontsize=10,
    )

    figure_path = output_dir / "parameterized_shape_family_gallery.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return figure_path


def task_scenarios():
    """Return three task models over the same uncertain hidden geometry."""
    return [
        {
            "short": "ACT NOW",
            "title": "A. Hidden geometry is irrelevant to this task",
            "task": "Top pickup",
            "actions": ["top-grasp", "stop"],
            "utilities": np.array([[1.0, -0.2], [1.0, -0.2], [1.0, -0.2]]),
            "view_values": [-0.20, -0.20, -0.20],
            "view_names": ["front", "left", "right"],
            "explanation": (
                "All plausible shapes allow the same top grasp.\n"
                "Shape is uncertain, but the task decision is not."
            ),
            "color": "#4C78A8",
        },
        {
            "short": "OBSERVE FIRST",
            "title": "B. Hidden geometry changes the best action",
            "task": "Side pickup",
            "actions": ["grasp-left", "grasp-right", "stop"],
            "utilities": np.array(
                [[-5.0, 1.0, -0.2], [1.0, -5.0, -0.2], [1.0, 1.0, -0.2]]
            ),
            "view_values": [-0.20, 1.00, 1.00],
            "view_names": ["front", "left", "right"],
            "explanation": (
                "Left and right grasps disagree across hypotheses.\n"
                "A side view can resolve the task-relevant ambiguity."
            ),
            "color": "#F58518",
        },
        {
            "short": "STOP SAFELY",
            "title": "C. More vision cannot create a feasible action",
            "task": "Pickup inside a blocked slot",
            "actions": ["grasp-left", "grasp-right", "stop"],
            "utilities": np.array(
                [[-5.0, -5.0, -0.2], [-5.0, -5.0, -0.2], [-5.0, -5.0, -0.2]]
            ),
            "view_values": [-0.20, -0.20, -0.20],
            "view_names": ["front", "left", "right"],
            "explanation": (
                "Both approach corridors are blocked for every hypothesis.\n"
                "Observing cannot help, so the safe decision is to stop."
            ),
            "color": "#79706E",
        },
    ]


def make_hypothesis_strip(parameters, family_index=2):
    images = []
    for class_index in range(3):
        parts = build_shape(class_index, family_index, parameters[family_index])
        images.append(render_semantics(parts, camera=DISPLAY_CAMERA, fov_deg=36.0))
    return np.concatenate(images, axis=1)


def make_current_observation(parameters, family_index=2):
    # Any class can be rendered because the grey slabs deliberately hide both hazards.
    parts = build_shape(0, family_index, parameters[family_index])
    parts = add_front_occluders(parts, parameters[family_index])
    return render_semantics(parts, camera=FRONT_CAMERA, fov_deg=45.0)


def annotate_bars(axis, bars, values, number_format="{:+.2f}"):
    for bar, value in zip(bars, values):
        y = value + (0.06 if value >= 0 else -0.08)
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            number_format.format(value),
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
            fontweight="bold",
        )


def make_decision_demo(parameters, output_dir):
    scenarios = task_scenarios()
    hypothesis_strip = make_hypothesis_strip(parameters)
    current_observation = make_current_observation(parameters)

    figure = plt.figure(figsize=(16, 12), constrained_layout=True)
    grid = figure.add_gridspec(3, 4, width_ratios=[1.05, 1.65, 1.25, 1.20])

    summary_rows = []
    for row, scenario in enumerate(scenarios):
        utilities = scenario["utilities"]
        expected_utilities = PRIOR @ utilities
        best_index = int(np.argmax(expected_utilities))
        current_value = float(expected_utilities[best_index])
        perfect_information_value = float(PRIOR @ np.max(utilities, axis=1))
        evpi = perfect_information_value - current_value
        best_view_value = float(np.max(scenario["view_values"]))

        observation_axis = figure.add_subplot(grid[row, 0])
        observation_axis.imshow(current_observation)
        observation_axis.set_xticks([])
        observation_axis.set_yticks([])
        observation_axis.set_title(
            "Current occluded view" if row == 0 else "Same current observation",
            fontsize=10,
            fontweight="bold",
        )
        if row == 0:
            observation_axis.legend(
                handles=[
                    Patch(color=BODY_COLOR, label="object"),
                    Patch(color=OCCLUDER_COLOR, label="occluder"),
                ],
                loc="lower center",
                fontsize=7,
            )

        hypotheses_axis = figure.add_subplot(grid[row, 1])
        hypotheses_axis.imshow(hypothesis_strip)
        hypotheses_axis.set_xticks([])
        hypotheses_axis.set_yticks([])
        hypotheses_axis.set_title(
            "Possible hidden shapes:  H1 left hazard | H2 right hazard | H3 no hazard",
            fontsize=9,
            fontweight="bold",
        )
        hypotheses_axis.text(
            0.5,
            -0.06,
            f"belief = {PRIOR.tolist()}",
            transform=hypotheses_axis.transAxes,
            ha="center",
            fontsize=9,
        )

        utility_axis = figure.add_subplot(grid[row, 2])
        x_positions = np.arange(len(scenario["actions"]))
        bar_colors = ["#B7B7B7"] * len(x_positions)
        bar_colors[best_index] = scenario["color"]
        bars = utility_axis.bar(x_positions, expected_utilities, color=bar_colors)
        utility_axis.axhline(0.0, color="black", linewidth=0.8)
        utility_axis.set_xticks(
            x_positions,
            scenario["actions"],
            rotation=18,
            ha="right",
            fontsize=8,
        )
        utility_axis.set_ylim(-5.6, 1.45)
        utility_axis.set_ylabel("expected task utility", fontsize=8)
        utility_axis.set_title(
            f"{scenario['task']}\nvalue of perfect information = {evpi:.2f}",
            fontsize=10,
            fontweight="bold",
        )
        annotate_bars(utility_axis, bars, expected_utilities)
        utility_axis.grid(axis="y", alpha=0.2)

        decision_axis = figure.add_subplot(grid[row, 3])
        decision_axis.axis("off")
        decision_axis.set_title(scenario["title"], fontsize=10, fontweight="bold")
        view_text = "   ".join(
            f"{name}: {value:+.2f}"
            for name, value in zip(scenario["view_names"], scenario["view_values"])
        )
        decision_axis.text(
            0.5,
            0.72,
            scenario["short"],
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color=scenario["color"],
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": scenario["color"],
                "alpha": 0.12,
                "edgecolor": scenario["color"],
            },
        )
        decision_axis.text(
            0.5,
            0.43,
            scenario["explanation"],
            ha="center",
            va="center",
            fontsize=9,
            linespacing=1.5,
        )
        decision_axis.text(
            0.5,
            0.17,
            f"EVSI - cost ({SENSING_COST:.1f}):\n{view_text}",
            ha="center",
            va="center",
            fontsize=8,
        )

        summary_rows.append(
            {
                "scenario": scenario["short"],
                "task": scenario["task"],
                "best_action_now": scenario["actions"][best_index],
                "current_value": current_value,
                "perfect_information_value": perfect_information_value,
                "evpi": evpi,
                "best_view_net_value": best_view_value,
            }
        )

    figure.suptitle(
        "Task-Adaptive Selective Verification Demo",
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.008,
        "Same uncertain geometry, different task consequences: act, observe, or stop.",
        ha="center",
        fontsize=11,
    )
    path = output_dir / "task_adaptive_selective_verification_demo.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")

    csv_path = output_dir / "task_adaptive_demo_decisions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    return figure, path, csv_path, summary_rows


def render_depth(parts, camera, fov_deg=45.0):
    scene = o3d.t.geometry.RaycastingScene()
    for part in parts:
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(part["mesh"]))
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=fov_deg,
        center=camera["center"],
        eye=camera["eye"],
        up=camera["up"],
        width_px=IMAGE_WIDTH,
        height_px=IMAGE_HEIGHT,
    )
    depth = scene.cast_rays(rays)["t_hit"].numpy()
    depth[~np.isfinite(depth)] = np.nan
    return depth


def mesh_bounds(mesh):
    vertices = np.asarray(mesh.vertices)
    return vertices.min(axis=0), vertices.max(axis=0)


def boxes_intersect(first_min, first_max, second_min, second_max):
    return bool(np.all(first_max >= second_min) and np.all(second_max >= first_min))


def compute_geometry_utility_matrix(parameters, family_index=2):
    """Derive side-grasp outcomes from hazard/corridor intersection, not labels."""
    params = parameters[family_index]
    width, depth, height = params["body_size"]
    reach = params["extension"] + 0.28
    corridor_y = 0.58 * depth
    corridor_z_min = 0.06
    corridor_z_max = height + 0.06

    # In this display convention H1's labelled left side maps to positive world x.
    corridors = {
        "grasp-left": (
            np.array([width / 2 - 0.02, -corridor_y / 2, corridor_z_min]),
            np.array([width / 2 + reach, corridor_y / 2, corridor_z_max]),
        ),
        "grasp-right": (
            np.array([-width / 2 - reach, -corridor_y / 2, corridor_z_min]),
            np.array([-width / 2 + 0.02, corridor_y / 2, corridor_z_max]),
        ),
    }

    actions = ["grasp-left", "grasp-right", "stop"]
    utilities = np.zeros((3, 3), dtype=float)
    collision_flags = np.zeros((3, 2), dtype=bool)
    clearances = np.zeros((3, 2), dtype=float)

    for class_index in range(3):
        parts = build_shape(class_index, family_index, params)
        hazard_bounds = [
            mesh_bounds(part["mesh"])
            for part in parts
            if part["role"] == "grasp hazard"
        ]
        for action_index, action in enumerate(actions[:2]):
            corridor_min, corridor_max = corridors[action]
            collision = any(
                boxes_intersect(corridor_min, corridor_max, hazard_min, hazard_max)
                for hazard_min, hazard_max in hazard_bounds
            )
            collision_flags[class_index, action_index] = collision
            utilities[class_index, action_index] = -5.0 if collision else 1.0
            if collision:
                clearances[class_index, action_index] = 0.0
            elif not hazard_bounds:
                clearances[class_index, action_index] = reach
            else:
                distances = []
                for hazard_min, hazard_max in hazard_bounds:
                    if action == "grasp-left":
                        distances.append(abs(hazard_max[0] - corridor_min[0]))
                    else:
                        distances.append(abs(corridor_max[0] - hazard_min[0]))
                clearances[class_index, action_index] = min(distances)
        utilities[class_index, 2] = -0.2
    return actions, utilities, collision_flags, clearances, corridors


def expected_value_after_partition(prior, utilities, groups):
    value = 0.0
    for group in groups:
        probability = float(np.sum(prior[group]))
        posterior = np.zeros_like(prior)
        posterior[group] = prior[group] / probability
        value += probability * float(np.max(posterior @ utilities))
    return value


def compute_view_values(prior, utilities):
    partitions = {
        "front view": [[0, 1, 2]],
        "left view": [[0], [1, 2]],
        "right view": [[1], [0, 2]],
    }
    current_value = float(np.max(prior @ utilities))
    values = {}
    for name, groups in partitions.items():
        post_value = expected_value_after_partition(prior, utilities, groups)
        values[name] = post_value - current_value - SENSING_COST
    return values


def add_mesh_to_axis(axis, mesh, color, alpha=1.0, edgecolor="none"):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    collection = Poly3DCollection(
        vertices[triangles],
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=0.25,
        alpha=alpha,
    )
    axis.add_collection3d(collection)


def draw_box_3d(axis, size, position, color, alpha=1.0, edgecolor="none"):
    add_mesh_to_axis(
        axis,
        create_box(size, position),
        color=color,
        alpha=alpha,
        edgecolor=edgecolor,
    )


def draw_camera_frustum(axis, eye, target, color, label, scale=0.62, alpha=0.8):
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    center = eye + forward * scale
    half_width = 0.30 * scale
    half_height = 0.22 * scale
    corners = [
        center + sx * half_width * right + sy * half_height * up
        for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    ]
    for corner in corners:
        axis.plot(*zip(eye, corner), color=color, linewidth=1.3, alpha=alpha)
    closed = corners + [corners[0]]
    axis.plot(*zip(*closed), color=color, linewidth=1.3, alpha=alpha)
    axis.scatter(*eye, color=color, s=28, depthshade=False)
    axis.text(*(eye + np.array([0.0, 0.0, 0.14])), label, color=color, fontsize=8)


def draw_gripper(axis, center, side, color, alpha=0.85):
    x, y, z = center
    finger_length = 0.48
    finger_width = 0.10
    opening = 0.62
    direction = -1.0 if side == "left" else 1.0
    palm_x = x - finger_length if direction < 0 else x
    draw_box_3d(
        axis,
        [finger_length, 0.13, 0.13],
        [palm_x, y - 0.065, z - 0.065],
        color,
        alpha,
    )
    tip_x = x - finger_length if direction < 0 else x + finger_length - finger_width
    for y_offset in (-opening / 2, opening / 2 - 0.11):
        draw_box_3d(
            axis,
            [finger_width, 0.11, 0.52],
            [tip_x, y + y_offset, z - 0.26],
            color,
            alpha,
        )


def make_research_hero_figure(parameters, output_dir):
    family_index = 2
    params = parameters[family_index]
    actions, utilities, collision_flags, clearances, corridors = (
        compute_geometry_utility_matrix(parameters, family_index)
    )
    expected_utilities = PRIOR @ utilities
    current_action_index = int(np.argmax(expected_utilities))
    current_action = actions[current_action_index]
    view_values = compute_view_values(PRIOR, utilities)
    selected_view = max(
        view_values,
        key=lambda name: (view_values[name], name == "right view"),
    )

    # Example true state H1. The selected right view rules out H2 but need not
    # distinguish H1 from H3 because both support grasp-right.
    posterior = np.array([PRIOR[0], 0.0, PRIOR[2]])
    posterior /= posterior.sum()
    final_action_index = int(np.argmax(posterior @ utilities))
    final_action = actions[final_action_index]

    current_parts = add_front_occluders(
        build_shape(0, family_index, params), params
    )
    selected_camera = {
        "eye": [4.2, -0.15, 1.2],
        "center": [0.0, 0.0, 0.45],
        "up": [0.0, 0.0, 1.0],
    }
    current_depth = render_depth(current_parts, FRONT_CAMERA, 45.0)
    selected_depth = render_depth(
        build_shape(0, family_index, params), selected_camera, 46.0
    )
    finite_depth = np.concatenate(
        [current_depth[np.isfinite(current_depth)], selected_depth[np.isfinite(selected_depth)]]
    )
    depth_min, depth_max = float(finite_depth.min()), float(finite_depth.max())

    figure = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = figure.add_gridspec(
        2, 3, width_ratios=[1.65, 1.0, 1.0], height_ratios=[1.0, 1.0]
    )

    scene_axis = figure.add_subplot(grid[:, 0], projection="3d")
    scene_axis.set_facecolor("#F7F8FA")
    width, depth, height = params["body_size"]
    draw_box_3d(scene_axis, [4.8, 2.5, 0.10], [-2.4, -1.35, -0.16], "#C9D0D7", 0.42)
    draw_box_3d(scene_axis, [4.8, 0.10, 2.2], [-2.4, 1.05, -0.10], "#BFC7CF", 0.16)
    draw_box_3d(scene_axis, [0.10, 2.5, 2.2], [-2.40, -1.35, -0.10], "#C9D0D7", 0.22)
    draw_box_3d(scene_axis, [0.10, 2.5, 2.2], [2.30, -1.35, -0.10], "#C9D0D7", 0.22)
    common_parts = build_shape(2, family_index, params)
    for part in common_parts:
        add_mesh_to_axis(scene_axis, part["mesh"], part["color"], 1.0)
    for class_index in (0, 1):
        for part in build_shape(class_index, family_index, params):
            if part["role"] == "grasp hazard":
                add_mesh_to_axis(
                    scene_axis, part["mesh"], HAZARD_COLOR, 0.36, edgecolor="#D95F02"
                )
    occluded_parts = add_front_occluders([], params)
    for part in occluded_parts:
        add_mesh_to_axis(scene_axis, part["mesh"], OCCLUDER_COLOR, 0.78)

    draw_gripper(scene_axis, [2.25, 0.0, 0.50], "left", "#D9534F", 0.42)
    draw_gripper(scene_axis, [-2.25, 0.0, 0.50], "right", "#2E9D57", 0.92)
    scene_axis.quiver(
        2.15, 0.0, 0.50, -1.05, 0.0, 0.0,
        color="#D9534F", linewidth=2.0, arrow_length_ratio=0.13,
    )
    scene_axis.quiver(
        -2.15, 0.0, 0.50, 1.05, 0.0, 0.0,
        color="#2E9D57", linewidth=2.5, arrow_length_ratio=0.13,
    )
    scene_axis.text(1.42, -0.92, 0.92, "candidate grasp", color="#B23B38", fontsize=8)
    scene_axis.text(-2.55, -0.78, 0.92, "selected grasp", color="#217A40", fontsize=8)

    draw_camera_frustum(
        scene_axis, [0.0, -3.15, 1.55], [0.0, 0.0, 0.48],
        "#6F42C1", "current RGB-D", 0.72,
    )
    draw_camera_frustum(
        scene_axis, [-3.15, -0.10, 1.50], [0.0, 0.0, 0.48],
        "#9AA0A6", "candidate", 0.65, 0.60,
    )
    draw_camera_frustum(
        scene_axis, [3.15, -0.10, 1.50], [0.0, 0.0, 0.48],
        "#2E9D57", "selected view", 0.65, 0.95,
    )
    scene_axis.set_xlim(-3.25, 3.25)
    scene_axis.set_ylim(-3.35, 1.35)
    scene_axis.set_zlim(-0.2, 2.4)
    scene_axis.set_box_aspect((6.5, 4.7, 2.7))
    scene_axis.view_init(elev=23, azim=-63)
    scene_axis.set_axis_off()
    scene_axis.set_title(
        "Occluded shelf-picking scene\n"
        "translucent orange = unresolved hidden geometry",
        fontsize=12,
        fontweight="bold",
    )
    scene_axis.legend(
        handles=[
            Patch(color=BODY_COLOR, label="observed/common geometry"),
            Patch(color=HAZARD_COLOR, alpha=0.45, label="hidden hypotheses"),
            Line2D([0], [0], color="#2E9D57", lw=3, label="selected view/action"),
            Line2D([0], [0], color="#D9534F", lw=3, label="rejected risky action"),
        ],
        loc="lower center",
        ncol=2,
        fontsize=8,
        frameon=False,
    )

    depth_grid = grid[0, 1].subgridspec(1, 2, wspace=0.08)
    for index, (depth_map, title) in enumerate(
        [(current_depth, "Current partial depth"), (selected_depth, "Selected-view depth")]
    ):
        axis = figure.add_subplot(depth_grid[0, index])
        axis.imshow(
            np.ma.masked_invalid(depth_map), cmap="magma_r", vmin=depth_min, vmax=depth_max
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_facecolor("#F4F6F8")
        axis.set_title(title, fontsize=9, fontweight="bold")
    figure.text(0.64, 0.51, "Partial 3D observation → targeted verification", ha="center", fontsize=9)

    belief_axis = figure.add_subplot(grid[0, 2])
    x = np.arange(3)
    bar_width = 0.35
    belief_axis.bar(x - bar_width / 2, PRIOR, bar_width, color="#A7ADB4", label="prior")
    belief_axis.bar(x + bar_width / 2, posterior, bar_width, color="#4C78A8", label="posterior")
    belief_axis.set_xticks(x, ["H1\nleft hazard", "H2\nright hazard", "H3\nno hazard"])
    belief_axis.set_ylim(0, 1.0)
    belief_axis.set_ylabel("belief probability")
    belief_axis.set_title("Geometric hypothesis evaluation", fontsize=11, fontweight="bold")
    belief_axis.legend(frameon=False, fontsize=8)
    belief_axis.grid(axis="y", alpha=0.20)
    belief_axis.text(
        0.5, 0.93, "H1 and H3 remain ambiguous\nbut imply the same safe action",
        transform=belief_axis.transAxes, ha="center", va="top", fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#E8F2FA", "edgecolor": "none"},
    )

    feasibility_axis = figure.add_subplot(grid[1, 1])
    image = feasibility_axis.imshow(utilities, cmap="RdYlGn", vmin=-5, vmax=1, aspect="auto")
    feasibility_axis.set_xticks(np.arange(3), ["grasp-left", "grasp-right", "stop"], rotation=15)
    feasibility_axis.set_yticks(np.arange(3), ["H1", "H2", "H3"])
    feasibility_axis.set_title("Geometry-derived task feasibility", fontsize=11, fontweight="bold")
    for row in range(3):
        for column in range(3):
            label = f"{utilities[row, column]:+.1f}"
            if column < 2:
                label += "\ncollision" if collision_flags[row, column] else "\nclear"
            feasibility_axis.text(
                column, row, label, ha="center", va="center", fontsize=8,
                color="white" if utilities[row, column] < -2 else "black",
                fontweight="bold",
            )
    figure.colorbar(image, ax=feasibility_axis, fraction=0.046, pad=0.03, label="task utility")

    decision_axis = figure.add_subplot(grid[1, 2])
    view_names = list(view_values)
    values = [view_values[name] for name in view_names]
    colors = ["#A7ADB4", "#A7ADB4", "#2E9D57"]
    bars = decision_axis.bar(view_names, values, color=colors)
    decision_axis.axhline(0.0, color="black", linewidth=0.8)
    decision_axis.set_ylabel("EVSI - sensing cost")
    decision_axis.set_ylim(-0.35, 1.25)
    decision_axis.set_title("Selective verification", fontsize=11, fontweight="bold")
    decision_axis.tick_params(axis="x", rotation=15)
    decision_axis.grid(axis="y", alpha=0.20)
    annotate_bars(decision_axis, bars, values)
    decision_axis.text(
        0.5, 0.84,
        f"OBSERVE → {selected_view.upper()}\nPOSTERIOR {posterior.round(2).tolist()}\nACT → {final_action.upper()}",
        transform=decision_axis.transAxes,
        ha="center", va="center", fontsize=10, fontweight="bold", color="#217A40",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#E6F4EA", "edgecolor": "#8BC79E"},
    )

    figure.suptitle(
        "Task-Adaptive Active Verification under Occlusion",
        fontsize=17,
        fontweight="bold",
    )
    path = output_dir / "research_demo_hero_figure.png"
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")

    trace_path = output_dir / "research_demo_geometry_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["hypothesis", *actions, "prior", "posterior"])
        for index, (name, _, _) in enumerate(CLASS_SPECS):
            writer.writerow([name, *utilities[index], PRIOR[index], posterior[index]])
    return figure, path, trace_path, {
        "actions": actions,
        "utilities": utilities,
        "expected_utilities": expected_utilities,
        "current_action": current_action,
        "view_values": view_values,
        "selected_view": selected_view,
        "posterior": posterior,
        "final_action": final_action,
        "clearances": clearances,
        "corridors": corridors,
    }


def make_research_animation(parameters, output_dir):
    """Create a compact visual story of the active-verification loop."""
    family_index = 2
    params = parameters[family_index]
    actions, utilities, collision_flags, _, _ = compute_geometry_utility_matrix(
        parameters, family_index
    )
    expected_utilities = PRIOR @ utilities
    view_values = compute_view_values(PRIOR, utilities)
    posterior = np.array([PRIOR[0], 0.0, PRIOR[2]])
    posterior /= posterior.sum()

    stage_specs = [
        (
            "1  PARTIAL 3D OBSERVATION",
            "The front RGB-D view hides both side corridors.",
            "H1, H2 and H3 produce the same visible evidence.",
        ),
        (
            "2  GEOMETRIC HYPOTHESES",
            "The system keeps three possible completions alive.",
            "Prior belief:  H1 0.40   H2 0.40   H3 0.20",
        ),
        (
            "3  TASK FEASIBILITY",
            "Candidate grasps are checked against each geometry.",
            "Current best action: STOP  (expected utility -0.20)",
        ),
        (
            "4  SELECTIVE VERIFICATION",
            "Only views that can change the action are valuable.",
            "Front -0.20   Left +1.00   Right +1.00",
        ),
        (
            "5  BELIEF UPDATE",
            "The selected right view rules out H2.",
            "Posterior:  H1 0.67   H2 0.00   H3 0.33",
        ),
        (
            "6  SAFE TASK ACTION",
            "Remaining hypotheses agree on the same safe grasp.",
            "ACT: GRASP-RIGHT",
        ),
    ]
    pipeline = [
        "Partial observation",
        "Shape hypotheses",
        "Task feasibility",
        "View evaluation",
        "Belief update",
        "Safe action",
    ]
    stage_durations = [1500, 1700, 1900, 1900, 1800, 2800]

    navy = "#08111F"
    panel = "#101D30"
    muted = "#65758B"
    white = "#F5F8FC"
    cyan = "#52B6E8"
    orange = "#FF8A32"
    green = "#47C878"
    red = "#FF5D67"
    purple = "#A78BFA"

    frames = []
    for stage_index, (stage_title, stage_subtitle, stage_result) in enumerate(stage_specs):
        figure = plt.figure(figsize=(12.8, 7.2), facecolor=navy)
        scene_axis = figure.add_axes([0.035, 0.11, 0.66, 0.78], projection="3d")
        info_axis = figure.add_axes([0.71, 0.10, 0.265, 0.80])
        scene_axis.set_facecolor(navy)
        info_axis.set_facecolor(panel)

        width, depth, height = params["body_size"]
        draw_box_3d(scene_axis, [4.8, 2.5, 0.10], [-2.4, -1.35, -0.16], "#45556A", 0.45)
        draw_box_3d(scene_axis, [4.8, 0.10, 2.2], [-2.4, 1.05, -0.10], "#34445A", 0.24)
        draw_box_3d(scene_axis, [0.10, 2.5, 2.2], [-2.40, -1.35, -0.10], "#45556A", 0.25)
        draw_box_3d(scene_axis, [0.10, 2.5, 2.2], [2.30, -1.35, -0.10], "#45556A", 0.25)

        for part in build_shape(2, family_index, params):
            role_color = cyan if part["role"] == "body" else DISTRACTOR_COLOR
            add_mesh_to_axis(scene_axis, part["mesh"], role_color, 1.0)
        for part in add_front_occluders([], params):
            add_mesh_to_axis(scene_axis, part["mesh"], "#8B95A5", 0.88)

        if stage_index >= 1:
            for class_index in (0, 1):
                for part in build_shape(class_index, family_index, params):
                    if part["role"] == "grasp hazard":
                        add_mesh_to_axis(
                            scene_axis,
                            part["mesh"],
                            orange,
                            0.42,
                            edgecolor="#FFB26F",
                        )

        draw_camera_frustum(
            scene_axis,
            [0.0, -3.15, 1.55],
            [0.0, 0.0, 0.48],
            purple,
            "current view",
            0.72,
            0.95,
        )

        if stage_index >= 2:
            left_color = red if stage_index < 5 else "#A63D45"
            right_color = green if stage_index >= 5 else "#D6A84B"
            right_alpha = 1.0 if stage_index >= 5 else 0.65
            right_x = -1.73 if stage_index == 5 else -2.25
            draw_gripper(scene_axis, [2.25, 0.0, 0.50], "left", left_color, 0.65)
            draw_gripper(scene_axis, [right_x, 0.0, 0.50], "right", right_color, right_alpha)
            scene_axis.quiver(
                2.15,
                0.0,
                0.50,
                -1.05,
                0.0,
                0.0,
                color=red,
                linewidth=2.0,
                arrow_length_ratio=0.13,
                alpha=0.75,
            )
            scene_axis.quiver(
                right_x + 0.10,
                0.0,
                0.50,
                0.92,
                0.0,
                0.0,
                color=right_color,
                linewidth=2.5,
                arrow_length_ratio=0.13,
                alpha=right_alpha,
            )

        if stage_index >= 3:
            draw_camera_frustum(
                scene_axis,
                [-3.15, -0.10, 1.50],
                [0.0, 0.0, 0.48],
                "#93A1B5",
                "left candidate",
                0.65,
                0.62,
            )
            draw_camera_frustum(
                scene_axis,
                [3.15, -0.10, 1.50],
                [0.0, 0.0, 0.48],
                green,
                "selected view",
                0.65,
                1.0,
            )

        scene_axis.set_xlim(-3.25, 3.25)
        scene_axis.set_ylim(-3.35, 1.35)
        scene_axis.set_zlim(-0.2, 2.4)
        scene_axis.set_box_aspect((6.5, 4.7, 2.7))
        scene_axis.view_init(elev=23, azim=-63)
        scene_axis.set_axis_off()
        scene_axis.text2D(
            0.04,
            0.94,
            "OCCLUDED SHELF-PICKING SCENE",
            transform=scene_axis.transAxes,
            color=white,
            fontsize=11,
            fontweight="bold",
        )

        info_axis.set_xlim(0, 1)
        info_axis.set_ylim(0, 1)
        info_axis.set_xticks([])
        info_axis.set_yticks([])
        for spine in info_axis.spines.values():
            spine.set_color("#263A55")
            spine.set_linewidth(1.2)

        info_axis.text(
            0.08,
            0.94,
            "ACTIVE VERIFICATION LOOP",
            color=white,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
        for index, label in enumerate(pipeline):
            y = 0.855 - index * 0.092
            is_active = index == stage_index
            is_done = index < stage_index
            marker_color = green if is_active else (cyan if is_done else muted)
            info_axis.scatter(0.11, y, s=105 if is_active else 55, color=marker_color, zorder=3)
            if index < len(pipeline) - 1:
                info_axis.plot(
                    [0.11, 0.11],
                    [y - 0.025, y - 0.067],
                    color=cyan if is_done else "#34465F",
                    linewidth=2,
                )
            info_axis.text(
                0.18,
                y,
                label,
                color=white if (is_active or is_done) else muted,
                fontsize=9.2,
                va="center",
                fontweight="bold" if is_active else "normal",
            )

        info_axis.text(
            0.08,
            0.265,
            stage_title,
            color=green,
            fontsize=11.5,
            fontweight="bold",
            va="top",
        )
        info_axis.text(
            0.08,
            0.205,
            stage_subtitle,
            color=white,
            fontsize=9.2,
            va="top",
            wrap=True,
        )
        info_axis.text(
            0.08,
            0.112,
            stage_result,
            color="#BBD8F2" if stage_index < 5 else green,
            fontsize=9.0,
            va="top",
            wrap=True,
            fontweight="bold" if stage_index == 5 else "normal",
        )

        # Small numerical evidence appears only when it becomes relevant.
        if stage_index == 2:
            info_axis.text(
                0.08,
                0.035,
                f"EU(left/right/stop) = {expected_utilities.round(2).tolist()}",
                color="#AEBFD3",
                fontsize=8.2,
            )
        elif stage_index == 3:
            info_axis.text(
                0.08,
                0.035,
                "Right view selected (equal value, deterministic tie-break)",
                color="#AEBFD3",
                fontsize=7.8,
            )
        elif stage_index == 4:
            info_axis.text(
                0.08,
                0.035,
                "H2 eliminated by geometry-depth consistency",
                color="#AEBFD3",
                fontsize=8.2,
            )
        elif stage_index == 5:
            info_axis.text(
                0.08,
                0.035,
                "Task resolved before full shape identification",
                color=green,
                fontsize=8.2,
            )

        figure.text(
            0.035,
            0.945,
            "TASK-AWARE ACTIVE VERIFICATION UNDER OCCLUSION",
            color=white,
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.035,
            0.915,
            "From partial depth to a safe manipulation decision",
            color="#91A4BC",
            fontsize=10,
        )
        figure.text(
            0.50,
            0.035,
            f"STAGE {stage_index + 1} / {len(stage_specs)}",
            color="#91A4BC",
            fontsize=9,
            ha="center",
        )
        progress_axis = figure.add_axes([0.035, 0.064, 0.94, 0.008])
        progress_axis.set_xlim(0, len(stage_specs))
        progress_axis.set_ylim(0, 1)
        progress_axis.barh(
            [0.5],
            [stage_index + 1],
            height=1.0,
            color=green,
            left=0,
        )
        progress_axis.barh(
            [0.5],
            [len(stage_specs) - stage_index - 1],
            height=1.0,
            color="#22344B",
            left=stage_index + 1,
        )
        progress_axis.axis("off")

        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=105, facecolor=navy)
        plt.close(figure)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("RGB").copy())
        buffer.close()

    gif_frames = [
        frame.quantize(colors=160, method=Image.Quantize.MEDIANCUT)
        for frame in frames
    ]
    gif_path = output_dir / "research_demo_active_verification.gif"
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=stage_durations,
        loop=0,
        disposal=2,
        optimize=False,
    )

    thumb_width = 560
    thumb_height = int(frames[0].height * thumb_width / frames[0].width)
    storyboard = Image.new("RGB", (thumb_width * 2, thumb_height * 3), navy)
    for index, frame in enumerate(frames):
        thumbnail = frame.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        storyboard.paste(thumbnail, ((index % 2) * thumb_width, (index // 2) * thumb_height))
    storyboard_path = output_dir / "research_demo_animation_storyboard.png"
    storyboard.save(storyboard_path)
    return gif_path, storyboard_path


def smoothstep(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def interval_progress(time_value, start, end):
    return smoothstep((time_value - start) / (end - start))


def blend_hex(first, second, amount):
    first_rgb = np.array(plt.matplotlib.colors.to_rgb(first))
    second_rgb = np.array(plt.matplotlib.colors.to_rgb(second))
    return first_rgb * (1.0 - amount) + second_rgb * amount


def make_smooth_research_animation(parameters, output_dir):
    """Render one continuous, eased 3D sequence as 1080p MP4 and GIF."""
    family_index = 2
    params = parameters[family_index]
    actions, utilities, _, _, _ = compute_geometry_utility_matrix(
        parameters, family_index
    )
    expected_utilities = PRIOR @ utilities
    posterior_target = np.array([2.0 / 3.0, 0.0, 1.0 / 3.0])

    fps = 24
    duration_seconds = 8.0
    frame_count = int(fps * duration_seconds)
    mp4_path = output_dir / "research_demo_active_verification_1080p.mp4"
    gif_path = output_dir / "research_demo_active_verification.gif"
    keyframe_path = output_dir / "research_demo_animation_keyframe.png"

    navy = "#07111F"
    white = "#F5F8FC"
    cyan = "#54BCEB"
    orange = "#FF8A32"
    green = "#49D17D"
    red = "#FF626C"
    purple = "#A78BFA"
    muted = "#8EA0B7"
    shelf = "#52647B"

    gif_frames = []
    writer = imageio.get_writer(
        mp4_path,
        fps=fps,
        codec="libx264",
        quality=9,
        macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    try:
        for frame_index in range(frame_count):
            time_value = frame_index / fps
            hypotheses_p = interval_progress(time_value, 0.8, 2.0)
            grasp_p = interval_progress(time_value, 1.8, 3.2)
            view_p = interval_progress(time_value, 3.0, 4.7)
            belief_p = interval_progress(time_value, 4.5, 5.8)
            action_p = interval_progress(time_value, 5.7, 7.0)

            if time_value < 0.8:
                stage_title = "PARTIAL 3D OBSERVATION"
                stage_detail = "The front view hides both side grasp corridors"
            elif time_value < 1.8:
                stage_title = "GEOMETRIC HYPOTHESES"
                stage_detail = "Multiple hidden shapes remain consistent with the same depth image"
            elif time_value < 3.0:
                stage_title = "TASK FEASIBILITY"
                stage_detail = "Test each grasp against every plausible geometry"
            elif time_value < 4.7:
                stage_title = "SELECTIVE VERIFICATION"
                stage_detail = "Move only to a view that can change the task decision"
            elif time_value < 5.8:
                stage_title = "BELIEF UPDATE"
                stage_detail = "The right view rules out H2; full identification is unnecessary"
            else:
                stage_title = "SAFE TASK ACTION"
                stage_detail = "The remaining hypotheses agree: execute GRASP-RIGHT"

            figure = plt.figure(figsize=(16, 9), dpi=100, facecolor=navy)
            scene_axis = figure.add_axes([0.01, 0.105, 0.98, 0.84], projection="3d")
            scene_axis.set_facecolor(navy)
            width, depth, height = params["body_size"]

            draw_box_3d(
                scene_axis,
                [4.9, 2.55, 0.10],
                [-2.45, -1.38, -0.16],
                shelf,
                0.38,
            )
            draw_box_3d(
                scene_axis,
                [4.9, 0.10, 2.25],
                [-2.45, 1.07, -0.10],
                shelf,
                0.17,
            )
            draw_box_3d(
                scene_axis,
                [0.10, 2.55, 2.25],
                [-2.45, -1.38, -0.10],
                shelf,
                0.18,
            )
            draw_box_3d(
                scene_axis,
                [0.10, 2.55, 2.25],
                [2.35, -1.38, -0.10],
                shelf,
                0.18,
            )

            for part in build_shape(2, family_index, params):
                part_color = cyan if part["role"] == "body" else DISTRACTOR_COLOR
                add_mesh_to_axis(scene_axis, part["mesh"], part_color, 1.0)

            h2_fade = 1.0 - belief_p
            for class_index in (0, 1):
                class_alpha = 0.52 * hypotheses_p
                if class_index == 1:
                    class_alpha *= h2_fade
                for part in build_shape(class_index, family_index, params):
                    if part["role"] == "grasp hazard" and class_alpha > 0.01:
                        add_mesh_to_axis(
                            scene_axis,
                            part["mesh"],
                            orange,
                            class_alpha,
                            edgecolor="#FFC28E",
                        )

            occluder_alpha = 0.90 - 0.26 * view_p
            for part in add_front_occluders([], params):
                add_mesh_to_axis(scene_axis, part["mesh"], "#9AA5B5", occluder_alpha)

            # The sensing camera follows a smooth quarter-circle around the object.
            camera_angle = -np.pi / 2.0 + view_p * np.pi / 2.0
            camera_radius = 3.15
            camera_eye = [
                camera_radius * np.cos(camera_angle),
                camera_radius * np.sin(camera_angle),
                1.55 + 0.06 * np.sin(np.pi * view_p),
            ]
            camera_color = blend_hex(purple, green, view_p)
            camera_label = "selected RGB-D view" if view_p > 0.72 else "active RGB-D view"
            draw_camera_frustum(
                scene_axis,
                camera_eye,
                [0.0, 0.0, 0.48],
                camera_color,
                camera_label,
                0.72,
                0.95,
            )

            if grasp_p > 0.01:
                outer_x = 3.45 - 1.18 * grasp_p
                left_alpha = max(0.0, 0.82 * grasp_p * (1.0 - 0.92 * action_p))
                right_x = -(outer_x - 0.92 * action_p)
                right_color = blend_hex("#D7AD52", green, action_p)
                if left_alpha > 0.02:
                    draw_gripper(scene_axis, [outer_x, 0.0, 0.50], "left", red, left_alpha)
                    scene_axis.quiver(
                        outer_x - 0.08,
                        0.0,
                        0.50,
                        -0.82,
                        0.0,
                        0.0,
                        color=red,
                        linewidth=2.2,
                        arrow_length_ratio=0.14,
                        alpha=left_alpha,
                    )
                draw_gripper(
                    scene_axis,
                    [right_x, 0.0, 0.50],
                    "right",
                    right_color,
                    0.72 + 0.28 * action_p,
                )
                scene_axis.quiver(
                    right_x + 0.08,
                    0.0,
                    0.50,
                    0.82,
                    0.0,
                    0.0,
                    color=right_color,
                    linewidth=2.6,
                    arrow_length_ratio=0.14,
                    alpha=0.80,
                )

            orbit = np.sin(np.pi * min(time_value / 7.0, 1.0))
            scene_axis.set_xlim(-3.30, 3.30)
            scene_axis.set_ylim(-3.25, 1.35)
            scene_axis.set_zlim(-0.22, 2.35)
            scene_axis.set_box_aspect((6.6, 4.6, 2.7))
            scene_axis.view_init(elev=23.5 + 1.6 * orbit, azim=-66.0 + 5.0 * orbit)
            scene_axis.set_axis_off()

            figure.text(
                0.035,
                0.955,
                "TASK-AWARE ACTIVE VERIFICATION UNDER OCCLUSION",
                color=white,
                fontsize=19,
                fontweight="bold",
                va="top",
            )
            figure.text(
                0.035,
                0.918,
                "A closed loop from partial geometry to safe manipulation",
                color=muted,
                fontsize=10.5,
                va="top",
            )
            figure.text(
                0.035,
                0.075,
                stage_title,
                color=green,
                fontsize=14,
                fontweight="bold",
                va="bottom",
            )
            figure.text(
                0.035,
                0.045,
                stage_detail,
                color=white,
                fontsize=10.5,
                va="bottom",
            )

            progress_axis = figure.add_axes([0.035, 0.022, 0.93, 0.006])
            progress_axis.barh([0.5], [time_value / duration_seconds], color=green, height=1.0)
            progress_axis.barh(
                [0.5],
                [1.0 - time_value / duration_seconds],
                left=[time_value / duration_seconds],
                color="#23364E",
                height=1.0,
            )
            progress_axis.set_xlim(0, 1)
            progress_axis.axis("off")

            if view_p > 0.60:
                hud_alpha = interval_progress(view_p, 0.60, 0.92)
                belief = PRIOR * (1.0 - belief_p) + posterior_target * belief_p
                belief_axis = figure.add_axes([0.755, 0.72, 0.205, 0.17], facecolor=navy)
                bars = belief_axis.barh(
                    [2, 1, 0],
                    belief,
                    color=[cyan, orange, muted],
                    alpha=0.95 * hud_alpha,
                    height=0.48,
                )
                belief_axis.set_xlim(0, 1.0)
                belief_axis.set_ylim(-0.7, 2.75)
                belief_axis.set_yticks([2, 1, 0], ["H1", "H2", "H3"])
                belief_axis.set_xticks([])
                belief_axis.tick_params(axis="y", colors=white, labelsize=9)
                belief_axis.set_title(
                    "BELIEF UPDATE",
                    color=white,
                    fontsize=10,
                    fontweight="bold",
                    loc="left",
                    pad=5,
                )
                for spine in belief_axis.spines.values():
                    spine.set_visible(False)
                for bar, value in zip(bars, belief):
                    belief_axis.text(
                        min(value + 0.035, 0.94),
                        bar.get_y() + bar.get_height() / 2,
                        f"{value:.2f}",
                        color=white,
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        alpha=hud_alpha,
                    )
                belief_axis.text(
                    0.0,
                    -0.55,
                    "H1 and H3 agree on GRASP-RIGHT" if belief_p > 0.82 else "selected view arriving...",
                    color=green if belief_p > 0.82 else muted,
                    fontsize=8.5,
                    alpha=hud_alpha,
                )

            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba())
            rgb_frame = np.ascontiguousarray(rgba[:, :, :3])
            writer.append_data(rgb_frame)

            if frame_index % 2 == 0:
                gif_image = Image.fromarray(rgb_frame).resize(
                    (1280, 720), Image.Resampling.LANCZOS
                )
                gif_frames.append(gif_image)
            if frame_index == int(6.65 * fps):
                Image.fromarray(rgb_frame).save(keyframe_path)
            plt.close(figure)
    finally:
        writer.close()

    quantized_frames = [
        frame.quantize(colors=192, method=Image.Quantize.MEDIANCUT)
        for frame in gif_frames
    ]
    quantized_frames[0].save(
        gif_path,
        save_all=True,
        append_images=quantized_frames[1:],
        duration=int(round(2000 / fps)),
        loop=0,
        disposal=2,
        optimize=False,
    )
    return mp4_path, gif_path, keyframe_path


def make_concise_research_animation(parameters, output_dir):
    """Render a short three-act animation with one visual claim per act."""
    family_index = 2
    params = parameters[family_index]
    fps = 24
    duration_seconds = 6.0
    frame_count = int(fps * duration_seconds)

    mp4_path = output_dir / "research_demo_active_verification_1080p.mp4"
    gif_path = output_dir / "research_demo_active_verification.gif"
    keyframe_path = output_dir / "research_demo_animation_keyframe.png"

    navy = "#07111F"
    white = "#F5F8FC"
    muted = "#91A4BC"
    cyan = "#54BCEB"
    orange = "#FF8A32"
    green = "#49D17D"
    red = "#FF626C"
    purple = "#A78BFA"
    shelf = "#52647B"

    gif_frames = []
    writer = imageio.get_writer(
        mp4_path,
        fps=fps,
        codec="libx264",
        quality=9,
        macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    try:
        for frame_index in range(frame_count):
            time_value = frame_index / fps
            ambiguity_p = interval_progress(time_value, 0.25, 1.15)
            camera_p = interval_progress(time_value, 2.00, 3.25)
            eliminate_p = interval_progress(time_value, 3.10, 3.90)
            action_p = interval_progress(time_value, 4.05, 5.10)

            if time_value < 2.0:
                act_number = "01"
                stage_title = "WHICH SIDE IS SAFE?"
                stage_detail = "The front view cannot reveal which side contains the hidden hazard."
                accent = orange
            elif time_value < 4.0:
                act_number = "02"
                stage_title = "TAKE ONE TARGETED VIEW"
                stage_detail = "The side observation contradicts one task-relevant hypothesis."
                accent = purple if camera_p < 0.65 else green
            else:
                act_number = "03"
                stage_title = "ACT WHEN THE TASK IS RESOLVED"
                stage_detail = "The remaining shapes agree on the same safe side grasp."
                accent = green

            figure = plt.figure(figsize=(16, 9), dpi=100, facecolor=navy)
            scene_axis = figure.add_axes([0.02, 0.12, 0.96, 0.80], projection="3d")
            scene_axis.set_facecolor(navy)
            width, depth, height = params["body_size"]

            # A quiet shelf context; it fades once the side view is reached so
            # that the evidence and final grasp remain visually unambiguous.
            scene_alpha = 0.34 - 0.18 * camera_p
            draw_box_3d(
                scene_axis,
                [4.9, 2.55, 0.10],
                [-2.45, -1.38, -0.16],
                shelf,
                scene_alpha + 0.08,
            )
            draw_box_3d(
                scene_axis,
                [4.9, 0.10, 2.25],
                [-2.45, 1.07, -0.10],
                shelf,
                scene_alpha * 0.52,
            )
            for x_position in (-2.45, 2.35):
                draw_box_3d(
                    scene_axis,
                    [0.10, 2.55, 2.25],
                    [x_position, -1.38, -0.10],
                    shelf,
                    scene_alpha * 0.55,
                )

            for part in build_shape(2, family_index, params):
                part_color = cyan if part["role"] == "body" else DISTRACTOR_COLOR
                add_mesh_to_axis(scene_axis, part["mesh"], part_color, 1.0)

            # Both orange parts are alternative hypotheses, not simultaneous
            # physical geometry. The contradicted H2 hypothesis fades away.
            for class_index in (0, 1):
                hypothesis_alpha = 0.58 * ambiguity_p
                if class_index == 1:
                    hypothesis_alpha *= 1.0 - eliminate_p
                for part in build_shape(class_index, family_index, params):
                    if part["role"] == "grasp hazard" and hypothesis_alpha > 0.01:
                        add_mesh_to_axis(
                            scene_axis,
                            part["mesh"],
                            orange,
                            hypothesis_alpha,
                            edgecolor="#FFC28E",
                        )

            occluder_alpha = 0.86 * (1.0 - 0.78 * camera_p)
            for part in add_front_occluders([], params):
                add_mesh_to_axis(
                    scene_axis,
                    part["mesh"],
                    "#9AA5B5",
                    occluder_alpha,
                )

            # During the first act, symmetric red approach arrows make the
            # decision ambiguity explicit without introducing two full grippers.
            approach_alpha = ambiguity_p * (1.0 - action_p)
            if approach_alpha > 0.02:
                for origin_x, direction in ((2.65, -1.0), (-2.65, 1.0)):
                    scene_axis.quiver(
                        origin_x,
                        -0.05,
                        0.50,
                        1.05 * direction,
                        0.0,
                        0.0,
                        color=red,
                        linewidth=3.0,
                        arrow_length_ratio=0.16,
                        alpha=0.72 * approach_alpha,
                    )
                scene_axis.text(
                    2.20,
                    -0.20,
                    0.82,
                    "?",
                    color=orange,
                    fontsize=23,
                    fontweight="bold",
                    alpha=approach_alpha,
                )
                scene_axis.text(
                    -2.20,
                    -0.20,
                    0.82,
                    "?",
                    color=orange,
                    fontsize=23,
                    fontweight="bold",
                    alpha=approach_alpha,
                )

            # One camera moves continuously from the ambiguous front view to
            # the informative right-side view.
            camera_angle = -np.pi / 2.0 + camera_p * np.pi / 2.0
            camera_radius = 3.20
            camera_eye = [
                camera_radius * np.cos(camera_angle),
                camera_radius * np.sin(camera_angle),
                1.58 + 0.06 * np.sin(np.pi * camera_p),
            ]
            camera_color = blend_hex(purple, green, camera_p)
            draw_camera_frustum(
                scene_axis,
                camera_eye,
                [0.0, 0.0, 0.48],
                camera_color,
                "targeted view" if camera_p > 0.55 else "current view",
                0.72,
                0.96,
            )

            # Put a visible cross directly on the hypothesis rejected by the
            # side-depth evidence. It then fades with that orange geometry.
            if eliminate_p > 0.03:
                rejected_x = -width / 2.0 - params["extension"] / 2.0
                rejected_z = params["feature_z"] + params["feature_height"] * 0.65
                cross_alpha = np.sin(np.pi * min(eliminate_p, 1.0))
                scene_axis.text(
                    rejected_x,
                    -0.42,
                    rejected_z,
                    "X",
                    color=red,
                    fontsize=28,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    alpha=max(0.32, cross_alpha),
                )

            # Only one gripper appears in the final act. Moving it slightly in
            # front of the occluder makes the safe action immediately legible.
            if action_p > 0.01:
                gripper_x = -2.75 + 1.36 * action_p
                draw_gripper(
                    scene_axis,
                    [gripper_x, -0.55, 0.50],
                    "right",
                    green,
                    0.98,
                )
                scene_axis.quiver(
                    gripper_x + 0.05,
                    -0.55,
                    0.50,
                    0.72,
                    0.0,
                    0.0,
                    color=green,
                    linewidth=3.2,
                    arrow_length_ratio=0.16,
                    alpha=0.95,
                )
                if action_p > 0.72:
                    scene_axis.text(
                        -1.78,
                        -0.82,
                        1.06,
                        "SAFE GRASP",
                        color=green,
                        fontsize=11,
                        fontweight="bold",
                        alpha=interval_progress(action_p, 0.72, 1.0),
                    )

            orbit = 2.0 * np.sin(np.pi * min(time_value / 5.6, 1.0))
            scene_axis.set_xlim(-3.25, 3.25)
            scene_axis.set_ylim(-3.15, 1.30)
            scene_axis.set_zlim(-0.22, 2.25)
            scene_axis.set_box_aspect((6.5, 4.45, 2.55), zoom=1.25)
            scene_axis.view_init(elev=24.0, azim=-65.0 + orbit)
            scene_axis.set_axis_off()

            figure.text(
                0.045,
                0.94,
                "TASK-AWARE ACTIVE VERIFICATION",
                color=white,
                fontsize=18,
                fontweight="bold",
                va="top",
            )
            figure.text(
                0.045,
                0.905,
                "Observe only when hidden geometry can change the safe action",
                color=muted,
                fontsize=10.5,
                va="top",
            )

            figure.text(
                0.045,
                0.078,
                act_number,
                color=accent,
                fontsize=15,
                fontweight="bold",
                va="bottom",
            )
            figure.text(
                0.085,
                0.078,
                stage_title,
                color=white,
                fontsize=14,
                fontweight="bold",
                va="bottom",
            )
            figure.text(
                0.085,
                0.047,
                stage_detail,
                color=muted,
                fontsize=10.5,
                va="bottom",
            )

            # Three unobtrusive segments show the short structure without a
            # quantitative progress dashboard.
            segment_axis = figure.add_axes([0.765, 0.055, 0.19, 0.012])
            active_segment = min(int(time_value // 2.0), 2)
            for segment_index in range(3):
                segment_axis.barh(
                    [0.5],
                    [0.29],
                    left=[segment_index * 0.35],
                    height=0.72,
                    color=accent if segment_index == active_segment else "#263A53",
                )
            segment_axis.set_xlim(0, 1.0)
            segment_axis.axis("off")

            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba())
            rgb_frame = np.ascontiguousarray(rgba[:, :, :3])
            writer.append_data(rgb_frame)

            if frame_index % 2 == 0:
                gif_frames.append(
                    Image.fromarray(rgb_frame).resize(
                        (1280, 720), Image.Resampling.LANCZOS
                    )
                )
            if frame_index == int(5.0 * fps):
                Image.fromarray(rgb_frame).save(keyframe_path)
            plt.close(figure)
    finally:
        writer.close()

    quantized_frames = [
        frame.quantize(colors=192, method=Image.Quantize.MEDIANCUT)
        for frame in gif_frames
    ]
    quantized_frames[0].save(
        gif_path,
        save_all=True,
        append_images=quantized_frames[1:],
        duration=int(round(2000 / fps)),
        loop=0,
        disposal=2,
        optimize=False,
    )
    return mp4_path, gif_path, keyframe_path


def main(show_figure=True, make_gif=True):
    rng = np.random.default_rng(RANDOM_SEED)
    parameters = sample_family_parameters(rng)
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    gallery_path = make_shape_gallery(parameters, output_dir)
    parameter_path = write_parameter_table(parameters, output_dir)
    demo_figure, demo_path, decision_path, summary_rows = make_decision_demo(
        parameters, output_dir
    )
    plt.close(demo_figure)
    hero_figure, hero_path, geometry_trace_path, hero_result = (
        make_research_hero_figure(parameters, output_dir)
    )
    animation_path = None
    storyboard_path = None
    mp4_path = None
    keyframe_path = None
    if make_gif:
        mp4_path, animation_path, keyframe_path = make_concise_research_animation(
            parameters, output_dir
        )

    print("\nTASK-ADAPTIVE SELECTIVE VERIFICATION DEMO")
    print("-----------------------------------------")
    print(f"Prior over H1/H2/H3: {PRIOR.tolist()}")
    print(f"Cost per extra observation: {SENSING_COST:.2f}")
    for row in summary_rows:
        print(
            f"{row['scenario']}: task={row['task']}, "
            f"best-now={row['best_action_now']}, EVPI={row['evpi']:.2f}, "
            f"best-view-net={row['best_view_net_value']:+.2f}"
        )
    print(f"Saved main demo: {demo_path}")
    print(f"Saved research hero figure: {hero_path}")
    print(f"Geometry-derived utility matrix:\n{hero_result['utilities']}")
    print(f"Selected view: {hero_result['selected_view']}")
    print(f"Posterior: {hero_result['posterior'].round(3).tolist()}")
    print(f"Final action: {hero_result['final_action']}")
    print(f"Saved geometry trace: {geometry_trace_path}")
    print(f"Saved decision table: {decision_path}")
    print(f"Saved auxiliary shape gallery: {gallery_path}")
    print(f"Saved shape parameters: {parameter_path}")
    if animation_path is not None:
        print(f"Saved 1080p research video: {mp4_path}")
        print(f"Saved animated research demo: {animation_path}")
        print(f"Saved animation keyframe: {keyframe_path}")

    if show_figure:
        plt.show()
    else:
        plt.close(hero_figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening a window.",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Skip generation of the animated research demo.",
    )
    arguments = parser.parse_args()
    main(show_figure=not arguments.no_show, make_gif=not arguments.no_gif)
