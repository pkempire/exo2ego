#!/usr/bin/env python3
"""3D visualization: point cloud + ego camera position"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

JSON_PATH = sys.argv[1] if len(sys.argv) > 1 else None
if not JSON_PATH:
    print("Usage: python3 visualize_3d.py phase2_output.json")
    sys.exit(1)

with open(JSON_PATH) as f:
    data = json.load(f)

rgb = np.array(data["rgb"], dtype=np.float32) / 255.0
depth = np.array(data["depth_map"], dtype=np.float32)
h, w = depth.shape

# Basic intrinsics
hfov_rad = np.radians(55)
fx = (w/2) / np.tan(hfov_rad/2)
K = np.array([[fx, 0, w/2], [0, fx, h/2], [0, 0, 1]], dtype=np.float32)

# Backproject
uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
Z = depth
X = (uu - K[0,2]) * Z / K[0,0]
Y = (vv - K[1,2]) * Z / K[1,1]
valid = (Z > 0.01) & (Z < 20) & np.isfinite(Z)

# Subsample for visualization
n_sample = 3000
valid_idx = np.where(valid.flat)[0]
if len(valid_idx) > n_sample:
    idx = np.random.choice(valid_idx, n_sample, replace=False)
else:
    idx = valid_idx

px = X.flat[idx]
py = Y.flat[idx]
pz = Z.flat[idx]
colors = rgb.reshape(-1, 3)[idx]

# Hand wrist positions
wrist_positions = []
for hand in data.get("hands", []):
    wx, wy = hand["wrist_pixel"]
    if 0 <= wy < h and 0 <= wx < w:
        wrist_positions.append((X[wy, wx], Y[wy, wx], Z[wy, wx]))

# Exo camera at origin
exo_cam = np.array([0, 0, 0])

# Ego camera = wrist + offset (head position)
if wrist_positions:
    head_pos = np.array(wrist_positions[0]) + np.array([0, 0.3, 0.15])
else:
    head_pos = np.array([0, 0.5, np.median(pz)])

# FIGURE
fig = plt.figure(figsize=(16, 10))

# View 1: Top-down
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.scatter(px, pz, py, c=colors, s=1, alpha=0.3)
ax1.scatter(*exo_cam, c='red', s=200, marker='^', label='Exo Camera')
ax1.scatter(*head_pos, c='cyan', s=200, marker='o', label='Ego Camera (head)')
for wp in wrist_positions:
    ax1.scatter(*wp, c='lime', s=100, marker='x', label='Wrist')
# Draw line from exo to ego
ax1.plot([exo_cam[0], head_pos[0]], [exo_cam[2], head_pos[2]], [exo_cam[1], head_pos[1]], 
         'c--', linewidth=1, alpha=0.5)
ax1.set_xlabel('X'); ax1.set_ylabel('Z (depth)'); ax1.set_zlabel('Y (height)')
ax1.set_title('Top-Down View (XZ plane)')
ax1.legend()

# View 2: Side view
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.scatter(pz, py, px, c=colors, s=1, alpha=0.3)
ax2.scatter(*[exo_cam[2], exo_cam[1], exo_cam[0]], c='red', s=200, marker='^')
ax2.scatter(*[head_pos[2], head_pos[1], head_pos[0]], c='cyan', s=200, marker='o')
for wp in wrist_positions:
    ax2.scatter(wp[2], wp[1], wp[0], c='lime', s=100, marker='x')
ax2.set_xlabel('Z (depth)'); ax2.set_ylabel('Y (height)'); ax2.set_zlabel('X')
ax2.set_title('Side View')
ax2.view_init(elev=10, azim=-60)

# View 3: Front view (what exo camera sees — just the 2D image)
ax3 = fig.add_subplot(2, 2, 3)
ax3.imshow(rgb)
ax3.set_title('Exo Camera View (what the image shows)')
for hand in data.get("hands", []):
    ax3.plot(hand["wrist_pixel"][0], hand["wrist_pixel"][1], 'go', markersize=10, label='Wrist')
if data.get("hands"):
    ax3.legend()

# View 4: Explanation
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')
explanation = f"""
WHY REPROJECTION FAILS (5% fill rate):

Exo camera position:   (0, 0, 0) — the phone/camera
Ego camera position:   ({head_pos[0]:.2f}, {head_pos[1]:.2f}, {head_pos[2]:.2f}) — your eyes
Distance between them: {np.linalg.norm(head_pos - exo_cam):.2f}m

THE PROBLEM:
The exo camera photographed YOU (your face, body).
The ego camera (your eyes) sees what's BEHIND the camera.
These are DIFFERENT parts of the scene.

3D point cloud extent:
  X: [{px.min():.1f}, {px.max():.1f}]
  Y: [{py.min():.1f}, {py.max():.1f}]  
  Z: [{pz.min():.1f}, {pz.max():.1f}]

The point cloud only contains points the EXO camera saw.
From the EGO position, most of these points are BEHIND you.

TO FIX:
Record exo video from BEHIND or SIDE while you do a task.
Then exo camera sees the same scene your eyes see.
Example: tripod behind you filming you at a desk.
"""
ax4.text(0.05, 0.95, explanation, transform=ax4.transAxes, fontsize=10,
         fontfamily='monospace', verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='#0f172a', edgecolor='#334155', alpha=0.9))

plt.tight_layout()
out = JSON_PATH.replace('.json', '_3d_viz.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
print(explanation)
