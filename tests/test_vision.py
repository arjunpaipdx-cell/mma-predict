"""Synthetic checks for the habit logic: poses are built by hand so we know the right answer.

Run:  python -m pytest tests/  (or: python tests/test_vision.py)
"""
import numpy as np

from mmapred import vision as V


def pose(cx, ground_y=400, s=100, facing=1, stance="Orthodox", punch=None, kick=False, crouch=False, prone=False):
    """A standing fighter centred at cx; s = torso length in pixels."""
    k = np.zeros((17, 3))
    k[:, 2] = 0.9
    hip_y = ground_y - (0.6 if crouch else 1.1) * s   # crouching: knees bend, hips sink
    sh_y = hip_y - s
    if prone:  # lying flat: shoulders level with hips
        sh_y = hip_y
        hip_y = ground_y - 0.2 * s
        sh_y = hip_y - 0.05 * s
    k[V.L_HIP] = [cx - 0.15 * s, hip_y, .9]
    k[V.R_HIP] = [cx + 0.15 * s, hip_y, .9]
    sh_x = cx + (facing * 0.9 * s if prone else 0)
    k[V.L_SH] = [sh_x - 0.25 * s, sh_y, .9]
    k[V.R_SH] = [sh_x + 0.25 * s, sh_y, .9]
    k[V.NOSE] = [sh_x, sh_y - (0.05 if prone else 0.4) * s * (0.5 if crouch else 1), .9]
    lead_x, rear_x = cx + facing * 0.35 * s, cx - facing * 0.35 * s
    left_lead = stance == "Orthodox"
    k[V.L_AN] = [lead_x if left_lead else rear_x, ground_y, .9]
    k[V.R_AN] = [rear_x if left_lead else lead_x, ground_y, .9]
    k[V.L_KN] = [k[V.L_AN][0], (hip_y + ground_y) / 2, .9]
    k[V.R_KN] = [k[V.R_AN][0], (hip_y + ground_y) / 2, .9]
    # guard: hands at chin height
    for wr, el, shi in ((V.L_WR, V.L_EL, V.L_SH), (V.R_WR, V.R_EL, V.R_SH)):
        k[wr] = [k[shi][0] + facing * 0.2 * s, sh_y - 0.2 * s, .9]
        k[el] = [k[shi][0] + facing * 0.1 * s, sh_y + 0.4 * s, .9]
    if punch:
        wr, shi = (V.L_WR, V.L_SH) if punch == "L" else (V.R_WR, V.R_SH)
        k[wr] = [k[shi][0] + facing * 1.4 * s, sh_y, .9]
    if kick:
        k[V.R_AN] = [cx + facing * 1.0 * s, hip_y - 0.1 * s, .9]
    for i in (1, 2, 3, 4):  # eyes and ears sit beside the nose
        k[i] = [k[V.NOSE][0] + (i - 2.5) * 0.05 * s, k[V.NOSE][1], .9]
    box = np.array([k[:, 0].min() - 10, k[:, 1].min() - 10, k[:, 0].max() + 10, k[:, 1].max() + 10])
    if crouch:
        box[1] = box[3] - (box[3] - box[1]) * 0.55
    return V.Person(box=box, score=0.9, kpts=k)


def run(frames, fps=10):
    res = V.Analysis(fps_used=fps)
    for i, (r, b) in enumerate(frames):
        res.frames.append(V.FrameRecord(t=i / fps, red=r, blue=b, cam=None))
    return V.habits(res)


def test_stance_range_guard():
    frames = [(pose(300, facing=1), pose(550, facing=-1, stance="Southpaw")) for _ in range(30)]
    h = run(frames)
    assert max(h["red"]["stance"], key=h["red"]["stance"].get) == "Orthodox"
    assert max(h["blue"]["stance"], key=h["blue"]["stance"].get) == "Southpaw"
    assert h["summary"]["range_pocket"] > 0.9          # 2.5 torso lengths apart
    assert h["red"]["guard_high"] > 0.9
    assert h["red"]["punches_pm"] == 0 and h["red"]["kicks_pm"] == 0


def test_pressure_and_punches():
    frames = []
    for i in range(40):
        red_x = 200 + i * 8                              # red walks forward 0.8 torso lengths/s
        punch = "L" if i % 10 == 5 else None              # a jab every second
        frames.append((pose(red_x, punch=punch), pose(red_x + 280, facing=-1)))
    h = run(frames)
    assert h["red"]["advancing"] > 0.8
    assert 3 <= h["red"]["punches_pm"] / 60 * 4 <= 4.5   # ~4 jabs in 4 s
    assert h["red"]["lead_share"] == 1.0                # orthodox, left hand = lead


def test_kicks_and_ground():
    frames = [(pose(300, kick=(i % 15 == 7)), pose(560, facing=-1)) for i in range(30)]
    assert run(frames)["red"]["kicks_pm"] > 0
    frames = [(pose(300, prone=True), pose(420, facing=-1, prone=True)) for _ in range(20)]
    h = run(frames)
    assert h["red"]["ground"] > 0.9 and h["summary"]["ground_share"] > 0.9


def test_level_change():
    frames = [(pose(300, crouch=(12 <= i <= 14)), pose(560, facing=-1)) for i in range(30)]
    assert run(frames)["red"]["level_changes_pm"] > 0




def test_camera_pan_is_measured():
    import cv2
    rng = np.random.default_rng(0)
    base = (rng.random((480, 640)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 3)
    base = cv2.cvtColor(cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
    shifted = np.roll(base, 12, axis=1)  # camera pans: the scene moves 12 px right
    cam = V.CameraMotion()
    cam.step(base, [])
    M = cam.step(shifted, [])
    assert M is not None and abs(M[0, 2] - 12) < 1.5 and abs(M[1, 2]) < 1.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
