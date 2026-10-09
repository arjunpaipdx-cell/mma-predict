"""Film study: track two fighters in a video and turn their poses into habits.

Pipeline
  1. Pose model   YOLO11n-pose exported to ONNX (models/yolo11n-pose.onnx). Runs through
                  ONNX Runtime, using NVIDIA TensorRT / CUDA when a GPU is available
                  and the CPU otherwise. Each person: box + 17 COCO keypoints.
  2. Who's who    The user points at the two fighters in one frame. After that a
                  tracker follows them frame to frame (box overlap + position + a
                  colour histogram of their shorts/torso), ignoring the referee.
  3. Camera       Broadcast cameras pan and zoom. Background feature points (outside
                  the people) give the camera's motion between frames, which is
                  removed before measuring who moved forward.
  4. Habits       Body-relative measurements (in torso lengths, so zoom doesn't
                  matter): stance, range, pressure, guard height, punch and kick
                  attempts, level changes, time on the ground.

Everything here is an estimate from 2D video. It's good at "how often" and "how
much" questions over a few minutes of footage, and bad at single moments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolo11n-pose.onnx"

# COCO keypoint indices
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 5), (0, 6)]
KP_MIN = 0.35  # keypoint confidence needed to use a joint


# ------------------------------------------------------------------ pose model
@dataclass
class Person:
    box: np.ndarray        # x1, y1, x2, y2
    score: float
    kpts: np.ndarray       # (17, 3): x, y, confidence


class PoseModel:
    def __init__(self, path: Path = MODEL_PATH, size: int = 640):
        import onnxruntime as ort

        wanted = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        providers = [p for p in wanted if p in available] or ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        self.provider = self.session.get_providers()[0]
        self.input = self.session.get_inputs()[0].name
        self.size = size

    @property
    def device(self) -> str:
        return {"TensorrtExecutionProvider": "NVIDIA GPU (TensorRT)",
                "CUDAExecutionProvider": "NVIDIA GPU (CUDA)"}.get(self.provider, "CPU")

    def _letterbox(self, img):
        h, w = img.shape[:2]
        r = self.size / max(h, w)
        nh, nw = int(round(h * r)), int(round(w * r))
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, r, left, top

    def __call__(self, frame_bgr: np.ndarray, conf: float = 0.3, iou: float = 0.5) -> list[Person]:
        img, r, dx, dy = self._letterbox(frame_bgr)
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input: blob})[0][0].T  # (8400, 56)
        out = out[out[:, 4] > conf]
        if len(out) == 0:
            return []
        cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)
        keep = cv2.dnn.NMSBoxes(boxes.tolist(), out[:, 4].tolist(), conf, iou)
        keep = np.array(keep).reshape(-1)
        people = []
        for i in keep:
            x, y, bw, bh = boxes[i]
            box = (np.array([x, y, x + bw, y + bh]) - [dx, dy, dx, dy]) / r
            k = out[i, 5:].reshape(17, 3).copy()
            k[:, 0] = (k[:, 0] - dx) / r
            k[:, 1] = (k[:, 1] - dy) / r
            people.append(Person(box=box, score=float(out[i, 4]), kpts=k))
        people.sort(key=lambda p: -(p.box[2] - p.box[0]) * (p.box[3] - p.box[1]))
        return people


# ------------------------------------------------------------------ appearance + tracking
def appearance(frame, box) -> np.ndarray | None:
    """Hue/saturation histogram of the middle of the body (shorts and torso)."""
    x1, y1, x2, y2 = box.astype(int)
    h = y2 - y1
    x1, x2 = max(x1, 0), min(x2, frame.shape[1])
    ya, yb = max(y1 + int(h * 0.3), 0), min(y1 + int(h * 0.7), frame.shape[0])
    if x2 - x1 < 4 or yb - ya < 4:
        return None
    hsv = cv2.cvtColor(frame[ya:yb, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class TwoFighterTracker:
    """Keeps identities for two chosen people; anyone else (the referee) is ignored."""

    def __init__(self, frame, red: Person, blue: Person):
        self.last = {"red": red, "blue": blue}
        self.hist = {"red": appearance(frame, red.box), "blue": appearance(frame, blue.box)}
        self.missing = {"red": 0, "blue": 0}

    def _cost(self, frame, name, p):
        last = self.last[name]
        diag = math.hypot(last.box[2] - last.box[0], last.box[3] - last.box[1]) + 1e-6
        c_last = (last.box[:2] + last.box[2:]) / 2
        c_new = (p.box[:2] + p.box[2:]) / 2
        motion = min(np.linalg.norm(c_new - c_last) / diag, 2.0) * (0.5 if self.missing[name] else 1.0)
        app = 1.0
        h_new = appearance(frame, p.box)
        if h_new is not None and self.hist[name] is not None:
            app = cv2.compareHist(self.hist[name].astype(np.float32), h_new.astype(np.float32),
                                  cv2.HISTCMP_BHATTACHARYYA)
        return 0.4 * (1 - iou(last.box, p.box)) + 0.3 * motion + 0.6 * app, h_new

    def update(self, frame, people: list[Person]) -> dict[str, Person | None]:
        out = {"red": None, "blue": None}
        if not people:
            for n in out:
                self.missing[n] += 1
            return out
        names = ["red", "blue"]
        costs = np.full((2, len(people)), 9.0)
        hists = {}
        for i, n in enumerate(names):
            for j, p in enumerate(people):
                costs[i, j], hists[(n, j)] = self._cost(frame, n, p)
        # best joint assignment of 2 tracks to detections (small problem: brute force)
        best, pair = 9e9, None
        for j1 in range(len(people)):
            for j2 in range(len(people)):
                if j1 == j2 and len(people) > 1:
                    continue
                c = costs[0, j1] + costs[1, j2]
                if c < best:
                    best, pair = c, (j1, j2)
        for i, n in enumerate(names):
            j = pair[i] if pair else None
            if j is None or costs[i, j] > 1.25 or (len(people) == 1 and i == 1):
                self.missing[n] += 1
                continue
            p = people[j]
            out[n] = p
            self.last[n] = p
            self.missing[n] = 0
            h = hists.get((n, j))
            if h is not None and self.hist[n] is not None:
                self.hist[n] = 0.9 * self.hist[n] + 0.1 * h
        return out


# ------------------------------------------------------------------ camera motion
class CameraMotion:
    """Estimate pan/zoom between frames from background points, so it can be removed."""

    def __init__(self):
        self.prev = None

    def step(self, frame, boxes) -> np.ndarray | None:
        """Returns a 2x3 affine transform mapping previous-frame points to this frame."""
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, np.uint8)
        for b in boxes:
            x1, y1, x2, y2 = (np.asarray(b) * 0.5).astype(int)
            mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 0
        M = None
        if self.prev is not None:
            prev_gray, prev_mask = self.prev
            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=8, mask=prev_mask)
            if pts is not None and len(pts) >= 12:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                good = st.reshape(-1) == 1
                if good.sum() >= 10:
                    M, _ = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC,
                                                       ransacReprojThreshold=2.0)
                    if M is not None:
                        M = M.copy()
                        M[:, 2] *= 2.0  # back to full resolution
        self.prev = (gray, mask)
        return M


# ------------------------------------------------------------------ body geometry
def _pt(k, i):
    return k[i, :2] if k[i, 2] >= KP_MIN else None


def _mid(k, i, j):
    a, b = _pt(k, i), _pt(k, j)
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return (a + b) / 2


def body(k) -> dict | None:
    """Key body measurements for one pose, or None if too little of the body is visible."""
    sh, hip = _mid(k, L_SH, R_SH), _mid(k, L_HIP, R_HIP)
    if sh is None or hip is None:
        return None
    torso = float(np.linalg.norm(sh - hip))
    if torso < 5:
        return None
    v = sh - hip
    lean = math.degrees(math.atan2(abs(v[0]), -v[1]))  # 0 = upright, 90 = horizontal
    ankles = [a for a in (_pt(k, L_AN), _pt(k, R_AN)) if a is not None]
    low = max(a[1] for a in ankles) if ankles else None  # the foot that's on the floor
    return {"center": hip, "shoulders": sh, "torso": torso, "lean": lean,
            "hip_to_ankle": None if low is None else float(low - hip[1]),
            "head_y": None if _pt(k, NOSE) is None else float(_pt(k, NOSE)[1])}


# ------------------------------------------------------------------ analysis
@dataclass
class FrameRecord:
    t: float
    red: Person | None
    blue: Person | None
    cam: np.ndarray | None = None


@dataclass
class Analysis:
    frames: list[FrameRecord] = field(default_factory=list)
    fps_used: float = 0.0
    device: str = "CPU"


def analyze(video_path: str, red_box, blue_box, start: float, end: float, sample_fps: float,
            model: PoseModel, progress=None, keep_frames: int = 120) -> tuple[Analysis, list]:
    """Track both fighters from `start` to `end` seconds.

    `red_box` / `blue_box` are the boxes the user picked on the frame at `start`.
    Returns the analysis and a list of (t, annotated JPEG bytes) for the frame scrubber.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))
    first = int(start * fps)
    last = int(end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    total = max(1, (last - first) // step)
    res = Analysis(fps_used=fps / step, device=model.device)
    tracker = cam = None
    thumbs = []
    thumb_every = max(1, total // keep_frames)
    idx, n = first, 0
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - first) % step:
            idx += 1
            continue
        people = model(frame)
        t = idx / fps
        if tracker is None:
            red = _closest(people, red_box)
            blue = _closest(people, blue_box, exclude=red)
            if red is None or blue is None:
                idx += 1
                continue
            tracker, cam = TwoFighterTracker(frame, red, blue), CameraMotion()
            assigned = {"red": red, "blue": blue}
        else:
            assigned = tracker.update(frame, people)
        M = cam.step(frame, [p.box for p in people])
        res.frames.append(FrameRecord(t=t, red=assigned["red"], blue=assigned["blue"], cam=M))
        if n % thumb_every == 0 and len(thumbs) < keep_frames:
            thumbs.append((t, to_jpeg(draw(frame, assigned, scale=min(1.0, 720 / frame.shape[1])))))
        n += 1
        if progress:
            progress(min(n / total, 1.0), n, total)
        idx += 1
    cap.release()
    return res, thumbs


def _closest(people, box, exclude=None):
    if box is None:
        return None
    box = np.asarray(box, dtype=float)
    cands = [p for p in people if p is not exclude]
    if not cands:
        return None
    best = max(cands, key=lambda p: iou(p.box, box))
    return best if iou(best.box, box) > 0.2 else None


# ------------------------------------------------------------------ habits
RANGES = (("Clinch", 0, 1.4), ("Pocket", 1.4, 3.0), ("Outside", 3.0, 99))


def habits(res: Analysis) -> dict:
    """Turn tracked poses into per-fighter habit numbers and a timeline."""
    dt = 1.0 / res.fps_used if res.fps_used else 0.1
    F = res.frames
    rows = []
    prev = {"red": None, "blue": None}
    prev_t = None
    counters = {n: {"punch": [], "kick": [], "level": []} for n in ("red", "blue")}
    last_event = {(n, e): -9 for n in ("red", "blue") for e in ("punch", "kick", "level")}
    head_base = {"red": [], "blue": []}
    prev_wrist = {"red": None, "blue": None}

    for fr in F:
        b = {n: (body(getattr(fr, n).kpts) if getattr(fr, n) is not None else None) for n in ("red", "blue")}
        row = {"t": fr.t, "both": b["red"] is not None and b["blue"] is not None}
        for n in ("red", "blue"):
            row[f"{n}_seen"] = b[n] is not None
        if row["both"]:
            ca, cb = b["red"]["center"], b["blue"]["center"]
            scale = (b["red"]["torso"] + b["blue"]["torso"]) / 2
            row["distance"] = float(np.linalg.norm(ca - cb) / scale)
        for n, o in (("red", "blue"), ("blue", "red")):
            bn, bo, p = b[n], b[o], getattr(fr, n)
            if bn is None:
                prev[n] = None
                prev_wrist[n] = None
                continue
            k = p.kpts
            s = bn["torso"]
            grounded = bn["lean"] > 55 or (bn["hip_to_ankle"] is not None and bn["hip_to_ankle"] < 0.55 * s)
            row[f"{n}_ground"] = bool(grounded)
            # guard: both wrists at or above shoulder height (image y grows downward)
            wl, wr = _pt(k, L_WR), _pt(k, R_WR)
            if wl is not None and wr is not None and not grounded:
                row[f"{n}_guard_high"] = bool(max(wl[1], wr[1]) < bn["shoulders"][1] + 0.25 * s)
            if bo is None:
                prev[n] = None
                continue
            facing = 1.0 if bo["center"][0] > bn["center"][0] else -1.0
            # stance: which ankle is closer to the opponent
            al, ar = _pt(k, L_AN), _pt(k, R_AN)
            if al is not None and ar is not None and not grounded and abs(al[0] - ar[0]) > 0.15 * s:
                row[f"{n}_stance"] = "Orthodox" if facing * (al[0] - ar[0]) > 0 else "Southpaw"
            # pressure: movement toward the opponent after removing camera motion
            if prev[n] is not None and prev_t is not None and fr.t - prev_t < 3 * dt:
                p0 = prev[n]
                if fr.cam is not None:
                    p0 = fr.cam[:, :2] @ p0 + fr.cam[:, 2]
                v = (bn["center"] - p0) / s / (fr.t - prev_t)  # torso lengths per second
                to_opp = bo["center"] - bn["center"]
                to_opp = to_opp / (np.linalg.norm(to_opp) + 1e-6)
                row[f"{n}_fwd"] = float(np.dot(v, to_opp))
                row[f"{n}_speed"] = float(np.linalg.norm(v))
            prev[n] = bn["center"].copy()
            if grounded:
                prev_wrist[n] = None
                continue
            # punches: a wrist shoots out toward the opponent and the arm straightens
            wrists = {}
            for side, (w_i, s_i) in {"L": (L_WR, L_SH), "R": (R_WR, R_SH)}.items():
                w, shp = _pt(k, w_i), _pt(k, s_i)
                if w is not None and shp is not None:
                    wrists[side] = (w, float(np.linalg.norm(w - shp) / s), float(facing * (w[0] - shp[0]) / s))
            pw = prev_wrist[n]
            if pw is not None and fr.t - last_event[(n, "punch")] > 0.35:
                for side, (w, ext, reach) in wrists.items():
                    if side in pw:
                        moved = facing * (w[0] - pw[side][0]) / s
                        if ext > 1.05 and reach > 0.8 and moved > 0.35:
                            lead = row.get(f"{n}_stance")
                            hand = ("lead" if (lead == "Orthodox") == (side == "L") else "rear") if lead else "unknown"
                            counters[n]["punch"].append((fr.t, hand))
                            last_event[(n, "punch")] = fr.t
                            break
            prev_wrist[n] = {sd: w for sd, (w, _, _) in wrists.items()}
            # kicks: an ankle rises to hip height while the other foot stays down
            hip_y = bn["center"][1]
            for a_i in (L_AN, R_AN):
                a = _pt(k, a_i)
                if a is not None and a[1] < hip_y + 0.35 * s and fr.t - last_event[(n, "kick")] > 0.6:
                    counters[n]["kick"].append((fr.t, "kick"))
                    last_event[(n, "kick")] = fr.t
            # level changes: their height in the frame drops fast relative to the opponent's
            # (comparing with the opponent cancels out camera zoom and walking toward the camera)
            po = getattr(fr, o)
            hn, ho = p.box[3] - p.box[1], po.box[3] - po.box[1]
            legs = bn["hip_to_ankle"] / s if bn["hip_to_ankle"] is not None else None
            if ho > 0 and legs is not None and iou(p.box, po.box) < 0.15:  # overlapping boxes distort heights
                hb = head_base[n]
                hb.append((fr.t, hn / ho, legs))
                head_base[n] = [x for x in hb if fr.t - x[0] <= 1.0]
                peak_h = max(x[1] for x in head_base[n])
                peak_legs = max(x[2] for x in head_base[n])
                # a real level change: they get shorter AND their hips sink toward their feet
                if (hn / ho < 0.75 * peak_h and legs < 0.75 * peak_legs
                        and fr.t - last_event[(n, "level")] > 1.5):
                    counters[n]["level"].append((fr.t, "level"))
                    last_event[(n, "level")] = fr.t
        prev_t = fr.t
        rows.append(row)

    import pandas as pd

    tl = pd.DataFrame(rows)
    if tl.empty:
        return {"timeline": tl, "red": {}, "blue": {}, "summary": {}}
    minutes = max((tl["t"].max() - tl["t"].min()) / 60, 1e-6)
    both = tl[tl["both"]] if "both" in tl else tl.iloc[0:0]
    summary = {"seconds": float(tl["t"].max() - tl["t"].min()), "frames": int(len(tl)),
               "both_visible": float(tl["both"].mean()) if len(tl) else 0.0}
    if "distance" in both and len(both):
        d = both["distance"]
        standing = ~(both.get("red_ground", False).fillna(False).astype(bool) |
                     both.get("blue_ground", False).fillna(False).astype(bool))
        summary["ground_share"] = float(1 - standing.mean())
        ds = d[standing]
        for name, lo, hi in RANGES:
            summary[f"range_{name.lower()}"] = float(((ds >= lo) & (ds < hi)).mean()) if len(ds) else float("nan")
        summary["median_distance"] = float(ds.median()) if len(ds) else float("nan")

    out = {"timeline": tl, "summary": summary}
    for n in ("red", "blue"):
        col = lambda c: tl[c].dropna() if c in tl else pd.Series(dtype=float)  # noqa: E731
        st_ = col(f"{n}_stance")
        fwd = col(f"{n}_fwd")
        punches = counters[n]["punch"]
        lead = sum(1 for _, h in punches if h == "lead")
        rear = sum(1 for _, h in punches if h == "rear")
        out[n] = {
            "seen": float(tl[f"{n}_seen"].mean()),
            "stance": (st_.value_counts(normalize=True).to_dict() if len(st_) else {}),
            "advancing": float((fwd > 0.4).mean()) if len(fwd) else float("nan"),
            "retreating": float((fwd < -0.4).mean()) if len(fwd) else float("nan"),
            "activity": float(col(f"{n}_speed").median()) if len(col(f"{n}_speed")) else float("nan"),
            "guard_high": float(col(f"{n}_guard_high").astype(float).mean()) if len(col(f"{n}_guard_high")) else float("nan"),
            "ground": float(col(f"{n}_ground").astype(float).mean()) if len(col(f"{n}_ground")) else float("nan"),
            "punches_pm": len(punches) / minutes,
            "lead_share": lead / (lead + rear) if lead + rear else float("nan"),
            "kicks_pm": len(counters[n]["kick"]) / minutes,
            "level_changes_pm": len(counters[n]["level"]) / minutes,
            "events": [(t, "punch") for t, _ in punches] + counters[n]["kick"] + counters[n]["level"],
        }
    return out


# ------------------------------------------------------------------ drawing
RED_BGR, BLUE_BGR, GREY_BGR = (46, 16, 200), (145, 79, 29), (170, 170, 170)


def draw(frame, assigned: dict, scale: float = 1.0, others: list[Person] | None = None, labels: bool = False):
    """Skeletons in corner colours; returns an RGB image."""
    img = frame.copy()
    for p in others or []:
        x1, y1, x2, y2 = p.box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), GREY_BGR, 2)
    for name, color in (("red", RED_BGR), ("blue", BLUE_BGR)):
        p = assigned.get(name)
        if p is None:
            continue
        k = p.kpts
        for i, j in SKELETON:
            if k[i, 2] >= KP_MIN and k[j, 2] >= KP_MIN:
                cv2.line(img, tuple(k[i, :2].astype(int)), tuple(k[j, :2].astype(int)), color, 3, cv2.LINE_AA)
        for i in range(17):
            if k[i, 2] >= KP_MIN:
                cv2.circle(img, tuple(k[i, :2].astype(int)), 3, (255, 255, 255), -1, cv2.LINE_AA)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_jpeg(rgb, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def clean(obj):
    """Make habit output JSON-safe (NumPy numbers -> float, NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(float(obj)) else round(float(obj), 4)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def numbered(frame, people: list[Person], max_width: int = 960):
    """Frame with every detected person boxed and numbered, for the who's-who step."""
    img = frame.copy()
    for i, p in enumerate(people, 1):
        x1, y1, x2, y2 = p.box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), 1)
        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(img, (x1, max(y1 - th - 14, 0)), (x1 + tw + 14, max(y1, th + 14)), (20, 20, 20), -1)
        cv2.putText(img, label, (x1 + 7, max(y1 - 7, th + 7)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3,
                    cv2.LINE_AA)
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_frame(video_path: str, t: float):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    info = {"ok": cap.isOpened() and fps > 0, "fps": fps, "frames": int(n),
            "seconds": (n / fps) if fps else 0, "width": int(cap.get(3)), "height": int(cap.get(4))}
    cap.release()
    return info
