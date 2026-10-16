from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240


def create_box(size, position):
    """Create a box mesh using a lower-left-front corner position."""
    box = o3d.geometry.TriangleMesh.create_box(
        width=size[0],
        height=size[1],
        depth=size[2],
    )
    box.translate(position)
    box.compute_vertex_normals()
    return box


def create_shape_hypothesis(hidden_part):
    """Return the complete object for one hidden-shape hypothesis."""
    meshes = [
        create_box(
            size=[1.4, 0.8, 0.8],
            position=[-0.7, -0.4, 0.0],
        )
    ]

    if hidden_part == "left":
        meshes.append(
            create_box(
                size=[0.55, 0.35, 0.35],
                position=[-1.25, -0.175, 0.225],
            )
        )
    elif hidden_part == "right":
        meshes.append(
            create_box(
                size=[0.55, 0.35, 0.35],
                position=[0.70, -0.175, 0.225],
            )
        )
    elif hidden_part != "none":
        raise ValueError(f"Unknown hidden-part type: {hidden_part}")

    return meshes


def create_occluders():
    """Create two front panels that hide the task-relevant side geometry."""
    left_panel = create_box(
        size=[1.0, 0.10, 1.2],
        position=[-1.5, -0.65, -0.1],
    )
    right_panel = create_box(
        size=[1.0, 0.10, 1.2],
        position=[0.5, -0.65, -0.1],
    )
    return [left_panel, right_panel]


def render_depth(hidden_part):
    """Render the robot's front-view depth observation."""
    raycasting_scene = o3d.t.geometry.RaycastingScene()

    for mesh in create_shape_hypothesis(hidden_part) + create_occluders():
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        raycasting_scene.add_triangles(tensor_mesh)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=50.0,
        center=[0.0, 0.0, 0.4],
        eye=[0.0, -4.0, 0.65],
        up=[0.0, 0.0, 1.0],
        width_px=IMAGE_WIDTH,
        height_px=IMAGE_HEIGHT,
    )

    depth = raycasting_scene.cast_rays(rays)["t_hit"].numpy()
    depth[~np.isfinite(depth)] = np.nan
    return depth


def finite_max_difference(depth_a, depth_b):
    """Compare two depth maps while treating shared background as equal."""
    valid_a = np.isfinite(depth_a)
    valid_b = np.isfinite(depth_b)

    if not np.array_equal(valid_a, valid_b):
        return float("inf")
    if not np.any(valid_a):
        return 0.0

    return float(np.max(np.abs(depth_a[valid_a] - depth_b[valid_b])))


def main(show_figure=True):
    hypothesis_names = ["H1: hidden-left", "H2: hidden-right", "H3: no protrusion"]
    hidden_parts = ["left", "right", "none"]
    depth_maps = [render_depth(part) for part in hidden_parts]

    finite_values = np.concatenate(
        [depth[np.isfinite(depth)] for depth in depth_maps]
    )
    vmin = float(finite_values.min())
    vmax = float(finite_values.max())

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    last_image = None

    for axis, name, depth in zip(axes, hypothesis_names, depth_maps):
        masked_depth = np.ma.masked_invalid(depth)
        last_image = axis.imshow(masked_depth, cmap="viridis_r", vmin=vmin, vmax=vmax)
        axis.set_title(name)
        axis.set_xlabel("camera x (pixels)")
        axis.set_ylabel("camera y (pixels)")
        axis.set_facecolor("white")

    figure.colorbar(last_image, ax=axes, shrink=0.82, label="ray depth (scene units)")
    figure.suptitle(
        "Same Partial Observation from Three Different Complete Shapes",
        fontsize=14,
        fontweight="bold",
    )

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "partial_observations.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")

    difference_12 = finite_max_difference(depth_maps[0], depth_maps[1])
    difference_13 = finite_max_difference(depth_maps[0], depth_maps[2])
    difference_23 = finite_max_difference(depth_maps[1], depth_maps[2])

    print("\nDepth-observation comparison")
    print("--------------------------------")
    print(f"max |H1 - H2|: {difference_12:.8f}")
    print(f"max |H1 - H3|: {difference_13:.8f}")
    print(f"max |H2 - H3|: {difference_23:.8f}")
    print(f"Saved figure: {output_path}")

    if max(difference_12, difference_13, difference_23) < 1e-6:
        print("Result: the three complete shapes produce the same partial observation.")
    else:
        print("Result: the observations are not identical; the occlusion design needs revision.")

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
