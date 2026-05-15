#!/usr/bin/env python3
"""Wearer pose extraction from an exocentric frame.

The point of this script is to answer a single question per frame:

  Is the camera wearer visible in the exo image? If so, where are their
  eyes, where is their torso, and what direction are they roughly facing?

If yes, we can use that information as a strong POV anchor in the
generation prompt. If no, the prompt has to fall back to scene-graph hints,
and our experience is that the generated POV drifts.

Strategy:
  1. MediaPipe FaceDetection for face bbox + 6 landmarks (2 eyes, nose,
     mouth, 2 ears).
  2. MediaPipe Pose for shoulders, torso center, and a body-side direction.
  3. Optional: face-orientation estimate from the 6 face landmarks
     (rough yaw/roll only; pitch is unreliable without a face model).

Output JSON (per frame):
  {
    "frame": "...",
    "wearer_visible": true,
    "n_faces": 1,
    "face_bbox_norm": [x, y, w, h],
    "eye_left_norm":  [x, y],
    "eye_right_norm": [x, y],
    "eye_center_norm":[x, y],
    "interocular_norm": float,
    "face_roll_deg": float,
    "face_yaw_proxy": float,   # +1 right, -1 left, from eye-vs-nose offset
    "torso_center_norm": [x, y],
    "shoulder_vector_norm": [dx, dy],
    "body_side": "left|right|center",
    "implied_camera_height_norm": float
  }

Usage:
  python scripts/wearer_pose_extract.py \
      --frames experiments/.../exo_frames/*.jpg \
      --out experiments/.../wearer_pose

  python scripts/wearer_pose_extract.py \
      --frames experiments/.../exo_frames/*.jpg \
      --out experiments/.../wearer_pose --visualize
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def _try_load_mediapipe():
    try:
        import mediapipe as mp  # type: ignore
        return mp
    except Exception as e:
        print(f"mediapipe import failed: {e}")
        return None


def detect_face(mp, image_bgr: np.ndarray) -> Optional[dict]:
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.4
    ) as fd:
        result = fd.process(img_rgb)
    if not result.detections:
        return None
    det = max(
        result.detections,
        key=lambda d: d.score[0] if d.score else 0,
    )
    bb = det.location_data.relative_bounding_box
    kp = det.location_data.relative_keypoints
    # mediapipe gives 6 keypoints in order:
    #   RIGHT_EYE, LEFT_EYE, NOSE_TIP, MOUTH_CENTER, RIGHT_EAR_TRAGION, LEFT_EAR_TRAGION
    right_eye = (kp[0].x, kp[0].y)
    left_eye = (kp[1].x, kp[1].y)
    nose = (kp[2].x, kp[2].y)
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    interocular = math.hypot(
        left_eye[0] - right_eye[0], left_eye[1] - right_eye[1]
    )
    roll_rad = math.atan2(
        right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
    )
    roll_deg = math.degrees(roll_rad)
    # crude yaw proxy: nose-x relative to eye midpoint, normalized by interocular distance
    yaw_proxy = 0.0
    if interocular > 1e-4:
        yaw_proxy = (nose[0] - eye_center[0]) / interocular
    return {
        "face_bbox_norm": [bb.xmin, bb.ymin, bb.width, bb.height],
        "eye_left_norm": list(left_eye),
        "eye_right_norm": list(right_eye),
        "eye_center_norm": list(eye_center),
        "nose_norm": list(nose),
        "interocular_norm": interocular,
        "face_roll_deg": roll_deg,
        "face_yaw_proxy": yaw_proxy,
        "score": float(det.score[0]) if det.score else 0.0,
    }


def detect_pose(mp, image_bgr: np.ndarray) -> Optional[dict]:
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2,
    ) as pose:
        result = pose.process(img_rgb)
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks.landmark
    # MediaPipe Pose head landmarks: 0 nose, 1-3 left eye (inner/center/outer),
    # 4-6 right eye, 7-8 ears, 9-10 mouth.
    nose = lm[0]
    le_inner, le, le_outer = lm[1], lm[2], lm[3]
    re_inner, re, re_outer = lm[4], lm[5], lm[6]
    l_ear, r_ear = lm[7], lm[8]
    # Body
    ls, rs = lm[11], lm[12]
    lh, rh = lm[23], lm[24]
    sh_center = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
    hip_center = ((lh.x + rh.x) / 2, (lh.y + rh.y) / 2)
    torso_center = (
        (sh_center[0] + hip_center[0]) / 2,
        (sh_center[1] + hip_center[1]) / 2,
    )
    shoulder_vec = (rs.x - ls.x, rs.y - ls.y)

    # Head-from-pose: eyes if both visible, else nose+ears, else nose only.
    head_eye_center = None
    interocular = None
    face_roll_deg = None
    face_yaw_proxy = None
    eye_vis_threshold = 0.5
    if min(le.visibility, re.visibility) > eye_vis_threshold:
        head_eye_center = ((le.x + re.x) / 2, (le.y + re.y) / 2)
        interocular = math.hypot(le.x - re.x, le.y - re.y)
        face_roll_deg = math.degrees(math.atan2(re.y - le.y, re.x - le.x))
        if interocular > 1e-4:
            face_yaw_proxy = (nose.x - head_eye_center[0]) / interocular
    elif nose.visibility > eye_vis_threshold and (
        l_ear.visibility > eye_vis_threshold or r_ear.visibility > eye_vis_threshold
    ):
        # back/side view: estimate "eye" near the nose, slightly above
        head_eye_center = (nose.x, max(0.0, nose.y - 0.02))
        # if only one ear is visible, the head is turned that way
        if l_ear.visibility > r_ear.visibility:
            face_yaw_proxy = -0.5  # facing left
        else:
            face_yaw_proxy = 0.5  # facing right
    elif nose.visibility > eye_vis_threshold:
        head_eye_center = (nose.x, max(0.0, nose.y - 0.02))

    return {
        "nose_norm": [nose.x, nose.y, nose.visibility],
        "left_eye_norm": [le.x, le.y, le.visibility],
        "right_eye_norm": [re.x, re.y, re.visibility],
        "left_ear_norm": [l_ear.x, l_ear.y, l_ear.visibility],
        "right_ear_norm": [r_ear.x, r_ear.y, r_ear.visibility],
        "shoulder_left_norm": [ls.x, ls.y, ls.visibility],
        "shoulder_right_norm": [rs.x, rs.y, rs.visibility],
        "shoulder_center_norm": list(sh_center),
        "hip_center_norm": list(hip_center),
        "torso_center_norm": list(torso_center),
        "shoulder_vector_norm": list(shoulder_vec),
        "shoulder_visibility": float(min(ls.visibility, rs.visibility)),
        # Derived head fields (None if not recoverable)
        "head_eye_center_norm": list(head_eye_center) if head_eye_center else None,
        "interocular_norm": interocular,
        "face_roll_deg": face_roll_deg,
        "face_yaw_proxy": face_yaw_proxy,
        "head_orientation": (
            "back" if face_yaw_proxy is None and nose.visibility < 0.3 else
            "left_profile" if face_yaw_proxy is not None and face_yaw_proxy < -0.3 else
            "right_profile" if face_yaw_proxy is not None and face_yaw_proxy > 0.3 else
            "frontal" if face_yaw_proxy is not None else "unknown"
        ),
    }


def derive_pov_anchor(face: Optional[dict], pose: Optional[dict]) -> dict:
    """Combine face + pose into a single high-level POV anchor."""
    anchor: dict = {"wearer_visible": False}
    if face is None and pose is None:
        return anchor
    anchor["wearer_visible"] = True
    if face is not None:
        anchor["face"] = face
        ec = face["eye_center_norm"]
        anchor["implied_eye_norm"] = ec
        # crude depth proxy: bigger interocular = closer = lower implied height
        anchor["implied_depth_proxy"] = (
            "near" if face["interocular_norm"] > 0.07 else
            "mid" if face["interocular_norm"] > 0.035 else "far"
        )
    if pose is not None:
        anchor["pose"] = pose
        tc = pose["torso_center_norm"]
        # Prefer head-from-pose, then face from FaceDetection, then shoulder-derived
        if pose.get("head_eye_center_norm"):
            anchor["implied_eye_norm"] = pose["head_eye_center_norm"]
            anchor["head_orientation"] = pose["head_orientation"]
            anchor["face_roll_deg"] = pose["face_roll_deg"]
            anchor["face_yaw_proxy"] = pose["face_yaw_proxy"]
        elif "implied_eye_norm" not in anchor:
            sc = pose["shoulder_center_norm"]
            anchor["implied_eye_norm"] = [sc[0], max(0.0, sc[1] - 0.10)]
        # body side
        cx = tc[0]
        anchor["body_side"] = (
            "left" if cx < 0.4 else "right" if cx > 0.6 else "center"
        )
    return anchor


def visualize(image_bgr: np.ndarray, anchor: dict) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    vis = image_bgr.copy()
    if not anchor.get("wearer_visible"):
        cv2.putText(
            vis,
            "no wearer detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return vis
    face = anchor.get("face")
    if face:
        bb = face["face_bbox_norm"]
        x0, y0 = int(bb[0] * w), int(bb[1] * h)
        x1, y1 = int((bb[0] + bb[2]) * w), int((bb[1] + bb[3]) * h)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
        ec = face["eye_center_norm"]
        cv2.circle(vis, (int(ec[0] * w), int(ec[1] * h)), 7, (0, 255, 0), -1)
        cv2.putText(
            vis,
            f"eyes  roll={face['face_roll_deg']:+.0f}  yaw_proxy={face['face_yaw_proxy']:+.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    pose = anchor.get("pose")
    if pose:
        sc = pose["shoulder_center_norm"]
        cv2.circle(vis, (int(sc[0] * w), int(sc[1] * h)), 8, (255, 200, 0), -1)
        tc = pose["torso_center_norm"]
        cv2.circle(vis, (int(tc[0] * w), int(tc[1] * h)), 6, (255, 100, 0), -1)
        cv2.putText(
            vis,
            f"torso side={anchor.get('body_side','?')}",
            (10, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2,
        )
    ec = anchor.get("implied_eye_norm")
    if ec:
        cv2.drawMarker(
            vis,
            (int(ec[0] * w), int(ec[1] * h)),
            (255, 0, 255),
            cv2.MARKER_CROSS,
            16,
            2,
        )
        cv2.putText(
            vis,
            "implied ego cam",
            (int(ec[0] * w) + 8, int(ec[1] * h) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            1,
        )
    return vis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--require-face", action="store_true",
                   help="Skip frames without a face detection.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        (out / "viz").mkdir(exist_ok=True)
    mp = _try_load_mediapipe()
    if mp is None:
        print("mediapipe not installed. exit.")
        return
    rows = []
    for path in args.frames:
        img = cv2.imread(path)
        if img is None:
            continue
        face = detect_face(mp, img)
        pose = detect_pose(mp, img)
        anchor = derive_pov_anchor(face, pose)
        anchor["frame"] = path
        rows.append(anchor)
        if args.visualize:
            vis = visualize(img, anchor)
            cv2.imwrite(str(out / "viz" / (Path(path).stem + "_anchor.jpg")), vis)
    summary = {
        "n_frames": len(rows),
        "n_with_wearer": sum(1 for r in rows if r["wearer_visible"]),
        "n_with_face": sum(1 for r in rows if r.get("face")),
        "n_with_pose": sum(1 for r in rows if r.get("pose")),
    }
    if args.require_face:
        rows = [r for r in rows if r.get("face") is not None]
    (out / "wearer_pose.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
