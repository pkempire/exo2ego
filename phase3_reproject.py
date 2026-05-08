#!/usr/bin/env python3
"""Phase 3: Geometric Reprojection — Exo → Ego View (OUR CODE)

Takes Phase 2 JSON output (depth map + hand positions) and:
  1. Backprojects exo depth → 3D point cloud
  2. Computes ego camera position (wrist + head offset)
  3. Reprojects 3D scene → ego viewpoint
  4. Overlays hand object region
  5. Outputs sparse ego frame

Usage: python3 phase3_reproject.py phase2_output.json
"""
import sys, json
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- CONFIG ---
JSON_PATH = sys.argv[1] if len(sys.argv) > 1 else None
if not JSON_PATH:
    print("Usage: python3 phase3_reproject.py phase2_output.json")
    sys.exit(1)

HEAD_OFFSET_Y = 0.25   # head is ~25cm above wrist (in meters)
HEAD_OFFSET_Z = 0.15   # head is ~15cm behind wrist
EGO_FOV_DEG = 90       # wide FOV for ego camera
EGO_IMG_W, EGO_IMG_H = 640, 480

# --- LOAD PHASE 2 DATA ---
with open(JSON_PATH) as f:
    data = json.load(f)

depth_map = np.array(data["depth_map"], dtype=np.float32)
h, w = depth_map.shape
hands = data["hands"]
print(f"Loaded: {w}x{h}, {len(hands)} hand(s)")

# --- CAMERA INTRINSICS (approximate from image dimensions) ---
# Assume principal point at center, focal length from image diagonal
diag = np.sqrt(w**2 + h**2)
fx = fy = diag  # rough approximation: ~90° FOV
cx, cy = w / 2, h / 2

K_exo = np.array([[fx, 0, cx],
                   [0, fy, cy],
                   [0,  0,  1]], dtype=np.float32)
print(f"Estimated K_exo: fx={fx:.0f}, fy={fy:.0f}")

# ================================================================
# STEP 1: BACKPROJECT DEPTH → 3D POINT CLOUD
# ================================================================
print("Backprojecting depth → point cloud...")

# Create pixel grid
u = np.arange(w, dtype=np.float32)
v = np.arange(h, dtype=np.float32)
uu, vv = np.meshgrid(u, v)

# Backproject: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy, Z = depth
Z = depth_map
X = (uu - cx) * Z / fx
Y = (vv - cy) * Z / fy

# Stack into (H, W, 3) point cloud
point_cloud = np.stack([X, Y, Z], axis=-1)  # shape: (H, W, 3)

# Mask invalid depths (0 or too far)
valid = (Z > 0.01) & (Z < 50.0) & np.isfinite(Z)
n_points = valid.sum()
print(f"  Valid points: {n_points:,} ({100*n_points/(h*w):.1f}%)")

# ================================================================
# STEP 2: COMPUTE EGO CAMERA POSITION
# ================================================================
print("Computing ego camera position...")

if len(hands) == 0:
    print("No hands found — placing ego camera at scene center")
    ego_center = np.array([0, 0.5, np.median(Z[valid])], dtype=np.float32)
else:
    # Use first detected hand's wrist
    wrist = hands[0]
    wx, wy = wrist["wrist_pixel"]
    wz = wrist["wrist_depth_m"]

    # Get wrist 3D position from point cloud
    wy_clip = min(wy, h-1)
    wx_clip = min(wx, w-1)
    wrist_3d = point_cloud[wy_clip, wx_clip].copy()
    print(f"  Wrist 3D (exo frame): X={wrist_3d[0]:.3f}, Y={wrist_3d[1]:.3f}, Z={wrist_3d[2]:.3f}")

    # Ego camera = wrist + offset (head position)
    ego_center = wrist_3d + np.array([0, HEAD_OFFSET_Y, HEAD_OFFSET_Z])
print(f"  Ego camera 3D: X={ego_center[0]:.3f}, Y={ego_center[1]:.3f}, Z={ego_center[2]:.3f}")

# ================================================================
# STEP 3: TRANSFORM POINT CLOUD TO EGO FRAME
# ================================================================
print("Transforming to ego frame...")

# Exo → Ego transform: translate by -ego_center, then rotate 180° around Y
# (ego camera looks backward at the person)
R_ego = np.array([[-1, 0, 0],   # flip X (left-right)
                   [0, 1, 0],    # keep Y
                   [0, 0, -1]],  # flip Z (look backward)
                  dtype=np.float32)

# Translate to ego origin
pc_ego = point_cloud - ego_center  # (H, W, 3)

# Apply rotation
pc_ego = pc_ego @ R_ego.T  # (H, W, 3) @ (3, 3) → (H, W, 3)

# ================================================================
# STEP 4: PROJECT TO EGO IMAGE
# ================================================================
print("Projecting to ego view...")

# Ego camera intrinsics (wide FOV)
fx_ego = EGO_IMG_W / (2 * np.tan(np.radians(EGO_FOV_DEG / 2)))
fy_ego = fx_ego
K_ego = np.array([[fx_ego, 0, EGO_IMG_W/2],
                   [0, fy_ego, EGO_IMG_H/2],
                   [0, 0, 1]], dtype=np.float32)

# Project: u_ego = fx * X/Z + cx, v_ego = fy * Y/Z + cy
Xe, Ye, Ze = pc_ego[..., 0], pc_ego[..., 1], pc_ego[..., 2]
valid_ego = (Ze > 0.1) & valid  # points in front of ego camera

u_ego = (fx_ego * Xe / Ze + K_ego[0, 2]).astype(np.int32)
v_ego = (fy_ego * Ye / Ze + K_ego[1, 2]).astype(np.int32)

# Build ego image: for each projected pixel, take the point with smallest Z
ego_depth = np.full((EGO_IMG_H, EGO_IMG_W), np.inf, dtype=np.float32)
ego_rgb = np.zeros((EGO_IMG_H, EGO_IMG_W, 3), dtype=np.float32)

# Use exo RGB if available (otherwise use depth as grayscale)
if "rgb" in data:
    exo_rgb = np.array(data["rgb"], dtype=np.float32) / 255.0
else:
    # Fake RGB from depth
    depth_norm = np.clip((Z - Z[valid].min()) / (Z[valid].max() - Z[valid].min()), 0, 1)
    exo_rgb = np.stack([depth_norm] * 3, axis=-1)

valid_pts = valid_ego
ue = u_ego[valid_pts]
ve = v_ego[valid_pts]
ze = Ze[valid_pts]
re = exo_rgb[valid_pts]

in_bounds = (ue >= 0) & (ue < EGO_IMG_W) & (ve >= 0) & (ve < EGO_IMG_H)
ue, ve, ze, re = ue[in_bounds], ve[in_bounds], ze[in_bounds], re[in_bounds]

# Z-buffer: keep closest point per pixel
for i in range(len(ue)):
    if ze[i] < ego_depth[ve[i], ue[i]]:
        ego_depth[ve[i], ue[i]] = ze[i]
        ego_rgb[ve[i], ue[i]] = re[i]

filled = (ego_depth < np.inf).sum()
print(f"  Ego pixels filled: {filled:,} ({100*filled/(EGO_IMG_W*EGO_IMG_H):.1f}%)")

# ================================================================
# STEP 5: VISUALIZE
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Exo RGB (or depth)
if "rgb" in data:
    axes[0].imshow(np.array(data["rgb"]))
else:
    axes[0].imshow(depth_map, cmap="inferno")
axes[0].set_title("Exo View (input)", fontsize=14)
axes[0].axis("off")

# Ego RGB (sparse — has gaps)
mask = ego_depth < np.inf
ego_display = ego_rgb.copy()
ego_display[~mask] = [0.1, 0.1, 0.15]  # dark bg for unfilled
axes[1].imshow(np.clip(ego_display, 0, 1))
axes[1].set_title(f"Ego View (sparse, {100*filled/(EGO_IMG_W*EGO_IMG_H):.0f}% filled)", fontsize=14)
axes[1].axis("off")

# Ego depth
im = axes[2].imshow(ego_depth, cmap="inferno")
axes[2].set_title("Ego Depth", fontsize=14)
axes[2].axis("off")
plt.colorbar(im, ax=axes[2], fraction=0.046)

plt.tight_layout()
out_img = JSON_PATH.replace(".json", "_ego.png")
plt.savefig(out_img, dpi=150, bbox_inches="tight")
print(f"Saved: {out_img}")

# ================================================================
# SAVE FOR PHASE 4
# ================================================================
output = {
    "ego_image": ego_rgb.tolist(),
    "ego_depth": ego_depth.tolist(),
    "ego_mask": mask.tolist(),
    "exo_intrinsics": K_exo.tolist(),
    "ego_intrinsics": K_ego.tolist(),
    "ego_center": ego_center.tolist(),
    "R_exo_to_ego": R_ego.tolist(),
    "filled_pct": float(100 * filled / (EGO_IMG_W * EGO_IMG_H))
}

out_json = JSON_PATH.replace(".json", "_ego.json")
with open(out_json, "w") as f:
    json.dump(output, f)

print(f"Saved: {out_json}")
print("Done. Ready for Phase 4 (ProPainter inpainting).")
