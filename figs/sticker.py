import math
from collections import namedtuple
import numpy as np
from scipy.spatial.distance import cdist
import regot
import svgwrite

# ==========================================
# Configuration
# ==========================================
Config = namedtuple("Config",
    [
        "output_filename",
        "canvas_size",
        "hex_radius",
        "padding",
        "col_bg",
        "col_border",
        "col_source",
        "col_target",
        "col_line_start",
        "col_line_end",
        "col_text",
        "n_source_points",
        "n_target_points"
    ]
)
config = Config(
    # Filename
    output_filename="sticker-regot.svg",
    # Dimensions
    canvas_size=800,
    hex_radius=380,
    # Inner padding for the point cloud
    padding=40,
    # Colors
    col_bg="#021129",
    col_border="#EFEFEF",  # "#09306D"
    col_source="#0088FF",
    col_target="#FF8800",
    col_line_start="#0088FF",
    col_line_end="#FF8800",
    col_text="#FFFFFF",
    # Numbers of data points
    n_source_points=120,
    n_target_points=80
)

# ==========================================
# Helper functions
# ==========================================

# Simulate point clouds
def generate_data(n, m):
    np.random.seed(123)
    # Source distribution
    Xs = np.random.normal(scale=0.5, size=(n, 2)) + np.array([-1.5, 0.2])
    # Target distribution
    Xt = np.random.normal(scale=0.4, size=(m, 2)) + np.array([2.0, -0.2])
    return Xs, Xt

# Map data points to canvas coordinates
def data_to_canvas_coords(points, shift, scale):
    # Reverse y axis, since in SVG y axis points downwards
    points[:, 1] = -points[:, 1]
    # Scale and shift
    canvas_points = points * scale + shift
    return canvas_points

# ==========================================
# Compute transport plan
# ==========================================

def compute_transport_plan(Xs, Xt):
    # Cost matrix
    M = cdist(Xs, Xt, metric="sqeuclidean")
    # Uniform marginals
    a = np.ones(len(Xs)) / len(Xs)
    b = np.ones(len(Xt)) / len(Xt)
    # Compute plan
    plan = regot.sinkhorn_splr(M, a, b, reg=0.1).plan
    return plan

# ==========================================
# SVG graphics
# ==========================================

def create_svg_sticker():
    # Initialize canvas
    dwg = svgwrite.Drawing(config.output_filename, size=(config.canvas_size, config.canvas_size), profile="full")
    center = np.array([config.canvas_size / 2, config.canvas_size / 2])

    # Path of the sticker hexagon
    hex_points = []
    for i in range(6):
        angle_deg = 30 + 60 * i
        angle_rad = math.radians(angle_deg)
        x = center[0] + config.hex_radius * math.cos(angle_rad)
        y = center[1] + config.hex_radius * math.sin(angle_rad)
        hex_points.append((x, y))

    # Clip content outside the hexagon
    clip_hex = dwg.clipPath(id="hex_clip")
    clip_hex.add(dwg.polygon(points=hex_points))
    dwg.defs.add(clip_hex)

    # Background
    bg_group = dwg.g(clip_path="url(#hex_clip)")
    bg_group.add(dwg.polygon(points=hex_points, fill=config.col_bg))

    # Add grid lines to the background
    grid_size = config.canvas_size // 16
    grid_pattern = dwg.pattern(size=(grid_size, grid_size), patternUnits="userSpaceOnUse")
    grid_pattern.add(dwg.path(d=f"M {grid_size} 0 L 0 0 0 {grid_size}", stroke="#FFFFFF", stroke_width=3, opacity=0.1, fill="none"))
    dwg.defs.add(grid_pattern)
    bg_group.add(dwg.rect(insert=(0,0), size=("100%", "100%"), fill=grid_pattern.get_paint_server()))
    dwg.add(bg_group)

    # Prepare data
    Xs_raw, Xt_raw = generate_data(config.n_source_points, config.n_target_points)
    # Compute shifting and scaling factors
    shift = np.array([center[0], center[1] - config.hex_radius * 0.1])
    data_range = np.max([np.max(np.abs(Xs_raw)), np.max(np.abs(Xt_raw))])
    scale_factor = (0.5 * math.sqrt(3) * config.hex_radius - config.padding) / data_range
    # Convert coordinates
    Xs_canvas = data_to_canvas_coords(Xs_raw, shift, scale_factor)
    Xt_canvas = data_to_canvas_coords(Xt_raw, shift, scale_factor)

    # Compute transport plan
    T_matrix = compute_transport_plan(Xs_raw, Xt_raw)
    # Normalize matrix for opacity mapping
    # Maximum value is one
    T_norm = T_matrix / np.max(T_matrix)

    # Apply clip to all contents
    content_group = dwg.g(clip_path="url(#hex_clip)")

    # Define line color gradient
    line_grad = dwg.linearGradient(id="line_grad", 
                                   start=(0, 0), end=(config.canvas_size, 0), 
                                   gradientUnits="userSpaceOnUse")
    line_grad.add_stop_color(offset="0%", color=config.col_line_start, opacity=1)
    line_grad.add_stop_color(offset="100%", color=config.col_line_end, opacity=1)
    dwg.defs.add(line_grad)

    # Draw connecting segments
    # Only keep weights above the threshold to avoid making SVG too large
    weight_thresh = 0.01
    seg_count = 0
    for i in range(config.n_source_points):
        for j in range(config.n_target_points):
            weight = T_norm[i, j]
            if weight > weight_thresh:
                start_pt = (Xs_canvas[i, 0], Xs_canvas[i, 1])
                end_pt = (Xt_canvas[j, 0], Xt_canvas[j, 1])
                opacity = np.power(weight, 0.7) * 0.8
                width = 0.5 + weight * 1.5

                # Draw segment
                line = dwg.line(start=start_pt, end=end_pt,
                                stroke="url(#line_grad)",
                                stroke_width=width,
                                stroke_opacity=opacity,
                                stroke_linecap="round")
                content_group.add(line)
                seg_count += 1
    print(f"Finished drawing {seg_count} segments")

    # Draw source point cloud
    for i in range(config.n_source_points):
        circle = dwg.circle(center=(Xs_canvas[i, 0], Xs_canvas[i, 1]),
                            r=4, fill=config.col_source, stroke="none", opacity=0.7)
        # Glow effect
        glow = dwg.circle(center=(Xs_canvas[i, 0], Xs_canvas[i, 1]),
                          r=6, fill=config.col_source, opacity=0.3)
        content_group.add(glow)
        content_group.add(circle)

    # Draw target point cloud
    for i in range(config.n_target_points):
        circle = dwg.circle(center=(Xt_canvas[i, 0], Xt_canvas[i, 1]),
                            r=4, fill=config.col_target, stroke="none", opacity=0.7)
        glow = dwg.circle(center=(Xt_canvas[i, 0], Xt_canvas[i, 1]),
                          r=6, fill=config.col_target, opacity=0.3)
        content_group.add(glow)
        content_group.add(circle)

    dwg.add(content_group)

    # Draw border lines
    dwg.add(dwg.polygon(points=hex_points, fill="none", stroke=config.col_border, stroke_width=15))

    # Add text
    text_group = dwg.g(style="font-family:Ink Free,serif; text-anchor:middle")
    text_label = dwg.text("RegOT",
                          insert=(center[0], center[1] + config.hex_radius * 0.6),
                          fill=config.col_text,
                          font_size=96,
                          letter_spacing=2)
    text_group.add(text_label)
    dwg.add(text_group)

    # Save file
    dwg.save()
    print(f"Successfully created sticker: {config.output_filename}")

if __name__ == "__main__":
    create_svg_sticker()
