import open3d as o3d


def create_box(size, position, color):
    """创建一个指定尺寸、位置和颜色的长方体。"""
    box = o3d.geometry.TriangleMesh.create_box(
        width=size[0],
        height=size[1],
        depth=size[2],
    )
    box.translate(position)
    box.paint_uniform_color(color)
    box.compute_vertex_normals()
    return box


def create_hypothesis(center_x, hidden_part):
    """
    创建一个候选形状。

    hidden_part:
        "left"  -> 左侧有隐藏凸起
        "right" -> 右侧有隐藏凸起
        "none"  -> 没有隐藏凸起
    """
    body_color = [0.25, 0.60, 0.95]
    hidden_color = [1.00, 0.45, 0.10]

    geometries = []

    # 所有候选形状都有完全相同的主体
    body = create_box(
        size=[1.4, 0.8, 0.8],
        position=[center_x - 0.7, -0.4, 0.0],
        color=body_color,
    )
    geometries.append(body)

    # 不同候选形状拥有不同的隐藏部分
    if hidden_part == "left":
        protrusion = create_box(
            size=[0.55, 0.35, 0.35],
            position=[center_x - 1.25, -0.175, 0.225],
            color=hidden_color,
        )
        geometries.append(protrusion)

    elif hidden_part == "right":
        protrusion = create_box(
            size=[0.55, 0.35, 0.35],
            position=[center_x + 0.70, -0.175, 0.225],
            color=hidden_color,
        )
        geometries.append(protrusion)

    return geometries


scene = []

# H1：左侧有凸起
scene.extend(create_hypothesis(-3.0, "left"))

# H2：右侧有凸起
scene.extend(create_hypothesis(0.0, "right"))

# H3：没有凸起
scene.extend(create_hypothesis(3.0, "none"))

# 坐标轴，帮助理解3D方向
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.8,
    origin=[0.0, -1.3, 0.0],
)
scene.append(coordinate_frame)

print("H1：左侧有隐藏凸起")
print("H2：右侧有隐藏凸起")
print("H3：没有隐藏凸起")
print("蓝色：可见主体")
print("橙色：隐藏部分")

o3d.visualization.draw_geometries(
    scene,
    window_name="Three Hidden-Shape Hypotheses",
    width=1200,
    height=700,
)