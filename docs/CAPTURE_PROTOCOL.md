# Own-Data Capture Protocol

Goal: collect ~20–50 paired exo/ego frames where **the exo camera deliberately sees the wearer's head, torso, and hands.** This is the single biggest gap in the current evaluation (Mendeley exo is overhead; the wearer is invisible). The same EgoJudge harness then runs unchanged.

## Hardware (use what you have)

- **Exo camera:** any phone or DSLR on a tripod, ~2 m back, ~30°–60° off your seating axis, height at your chest level so it sees both your head/torso and the table surface.
- **Ego camera:** second phone on a head/cap/glasses mount, OR taped to the brim of a baseball cap pointed slightly down, OR clipped to a chest harness. *The camera direction matters more than the mount: it should point at the table where your hands work.*
- **Calibration target:** print a 6×9 checkerboard PDF (any A4 page from `https://docs.opencv.org/4.x/_downloads/...` style). Tape it flat near the edge of the table where it's visible in both views.
- **Lighting:** daylight from a window or a single soft overhead light. Lock exposure / white balance in your phone's camera app if available.

## Sync

At the start of every clip:
1. Start exo recording, then start ego recording.
2. Wait 2 seconds.
3. Clap once, hard, with both hands in the exo frame (audible in both audio tracks).
4. Begin the task.

The audio peak gives a sub-frame-accurate time offset, recovered later by `scripts/ingest_paired_capture.py`.

## Tasks (5 × ~15–20s)

Each task should start with hands out of frame, perform the action, then return hands out of frame.

| # | Task | Required items | Why it's robotics-relevant |
|---|---|---|---|
| 1 | **Stack cups** | 3 plastic cups | clean pick/place, vertical assembly |
| 2 | **Pick screwdriver** | screwdriver + scrap board | tool grasp, alignment, contact |
| 3 | **Pour pebbles** | 2 cups + small objects | granular manipulation, contact transition |
| 4 | **Open & close box** | small cardboard box | articulated object, two-handed |
| 5 | **Insert cable** | USB or aux cable + powered-off device | fine motor, insertion |

Each clip ~15–20 seconds. Total recording time ~5 minutes.

## Folder layout (after ingest)

```
data/own_capture/
  <task_name>/
    raw/
      exo.mp4
      ego.mp4
    pairs.json                      # same schema as Mendeley pairs.json
    frames/
      <task>_<frame_id>_exo.jpg
      <task>_<frame_id>_ego_gt.jpg
    annotations/
      <task>_<frame_id>.json        # verb, noun, contact, action_phase
    calibration/
      checkerboard_exo.npz          # K, dist
      checkerboard_ego.npz
```

## After recording

1. Drop `exo.mp4` and `ego.mp4` into `data/own_capture/<task>/raw/`.
2. Run `python -m exo2ego ingest --task <task_name>` (we'll add this in repo cleanup).
3. The ingest script: detects clap, aligns timelines, extracts paired frames at 2 fps, optionally estimates intrinsics if a checkerboard clip exists.
4. From there the existing EgoJudge harness runs unchanged.

## Quality checklist before stopping a take

- [ ] Are both your hands and the active object visible in the exo frame?
- [ ] Is your head/upper torso visible in the exo frame? (This is the whole point of the recapture.)
- [ ] Is the ego camera pointed at the work area, not up at the ceiling?
- [ ] Did you clap visibly within the first 3 seconds?
- [ ] Is the checkerboard visible in at least one second of footage somewhere in the clip?
- [ ] Is exposure stable (no auto-exposure pumping)?

If any of these fail on a take, just re-record. Bad takes are cheaper to redo than to salvage.
