"""Research page: track two fighters in uploaded footage and measure their habits."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from mmapred import ui
from mmapred import vision as V
from views.common import film_library, get_predictor

MAX_SECONDS = 300
EST_SEC_PER_FRAME = 0.09  # measured on a 2-core CPU; a GPU is far faster


@st.cache_resource(show_spinner="Loading the pose model…")
def pose_model() -> V.PoseModel:
    return V.PoseModel()


def mmss(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 60}:{t % 60:02d}"


pr = get_predictor()
all_names = sorted(pr.index.names(min_fights=1))
saved = st.session_state.get("matchup") or {}
if not saved:  # same default matchup as the Predict page: the two top-rated active lightweights
    t = pr.index.table
    lw = [n for n in pr.index.names(min_fights=2, active_since="2024-01-01") if t.loc[n, "weightclass"] == "Lightweight"]
    lw = sorted(lw, key=lambda n: -t.loc[n, "elo"])
    saved = {"a": lw[0], "b": lw[1]} if len(lw) > 1 else {}

st.markdown('<h2 class="oo-page">Film study</h2>'
            '<p class="oo-lede">Upload a fight clip, point out the two fighters, and the app tracks them frame by '
            'frame to measure how they actually fight: stance, range, pressure, hands, strikes, level changes.</p>',
            unsafe_allow_html=True)

# ------------------------------------------------------------------ 1. clip + fighters
st.markdown("### 1. Clip and fighters")
c_up, c_f = st.columns([1.25, 1], gap="large")
with c_up:
    up = st.file_uploader("Fight video", type=["mp4", "mov", "m4v", "webm", "mkv", "avi"],
                          help="MP4 (H.264) works best. Up to 500 MB.")
    st.markdown('<p class="oo-cap">Use footage you have the right to use. Videos stay in this session and are '
                'deleted when you close the page.</p>', unsafe_allow_html=True)
with c_f:
    red_name = st.selectbox("Red corner (tracked in red)", all_names,
                            index=all_names.index(saved["a"]) if saved.get("a") in all_names else 0)
    blue_opts = [n for n in all_names if n != red_name]
    blue_name = st.selectbox("Blue corner (tracked in blue)", blue_opts,
                             index=blue_opts.index(saved["b"]) if saved.get("b") in blue_opts else 0)

if up is None:
    lib = film_library()
    if lib:
        st.markdown(f'<p class="oo-note">You have {len(lib)} saved stud{"y" if len(lib) == 1 else "ies"} below. '
                    f'Upload another clip to add more.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="oo-note">Start by uploading a clip. A minute or two of standing exchanges gives the '
                    'most reliable numbers.</p>', unsafe_allow_html=True)

video_path = None
info = None
if up is not None:
    key = f"video_{up.file_id}"
    if key not in st.session_state:
        suffix = Path(up.name).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(up.getbuffer())
        tmp.close()
        st.session_state[key] = tmp.name
    video_path = st.session_state[key]
    info = V.video_info(video_path)
    if not info["ok"] or info["seconds"] <= 0:
        st.error("This video couldn't be read. Convert it to MP4 (H.264) and upload it again.")
        video_path = None

# ------------------------------------------------------------------ 2. who's who
if video_path:
    dur = info["seconds"]
    st.markdown("### 2. Who's who")
    st.markdown('<p class="oo-note">Pick a moment where both fighters are clearly visible, then match them to the '
                'numbered boxes. Tracking starts from this moment.</p>', unsafe_allow_html=True)
    c_t, c_end, c_fps = st.columns([1.4, 1.4, 1])
    start = c_t.slider("Start at", 0.0, max(dur - 1, 0.1), 0.0, step=0.5, format="%.1fs")
    end = c_end.slider("Track until", min(start + 1, dur), dur, min(dur, start + 120), step=0.5, format="%.1fs")
    sample_fps = c_fps.radio("Detail", [5, 10, 15], index=1, horizontal=True,
                             format_func=lambda f: f"{f} fps",
                             help="Frames analysed per second of video. More frames catch more punches but take longer.")
    if end - start > MAX_SECONDS:
        st.warning(f"That's {mmss(end - start)} of footage. Tracking is limited to {MAX_SECONDS // 60} minutes at a "
                   f"time, so it will stop at {mmss(start + MAX_SECONDS)}.")
        end = start + MAX_SECONDS

    frame = V.read_frame(video_path, start)
    if frame is None:
        st.error("Couldn't read a frame at that time. Try a slightly earlier start.")
        st.stop()
    model = pose_model()
    people = model(frame)
    img_col, pick_col = st.columns([2.2, 1], gap="large")
    img_col.image(V.numbered(frame, people), width="stretch")
    with pick_col:
        if len(people) < 2:
            st.warning("Fewer than two people found here. Move **Start at** to a frame where both fighters are "
                       "fully in view.")
            red_i = blue_i = None
        else:
            labels = [f"Person {i}" for i in range(1, len(people) + 1)]
            red_i = st.selectbox(f"{red_name} is", range(len(people)), format_func=lambda i: labels[i], index=0)
            blue_choices = [i for i in range(len(people)) if i != red_i]
            blue_i = st.selectbox(f"{blue_name} is", blue_choices, format_func=lambda i: labels[i], index=0)
            n_frames = int((end - start) * sample_fps)
            est = n_frames * EST_SEC_PER_FRAME * (0.15 if model.device != "CPU" else 1)
            st.markdown(f'<p class="oo-cap">{mmss(end - start)} of video · {n_frames:,} frames · about '
                        f'{max(1, round(est / 60))} min on {model.device}</p>', unsafe_allow_html=True)
            go = st.button("Track fighters", type="primary", width="stretch")
            if go:
                bar = st.progress(0.0, text="Starting…")
                t0 = time.perf_counter()

                def report(f, n, total):
                    if n % 5 == 0 or f >= 1:
                        bar.progress(f, text=f"Tracking frame {n:,} of {total:,}")

                res, thumbs = V.analyze(video_path, people[red_i].box, people[blue_i].box, start, end,
                                        sample_fps, model, progress=report)
                bar.empty()
                if len(res.frames) < 5:
                    st.error("Lost both fighters almost immediately. Try a start time with a clearer, wider shot.")
                else:
                    h = V.habits(res)
                    study = {
                        "version": 1,
                        "fighters": {"red": red_name, "blue": blue_name},
                        "clip": up.name, "range": [round(start, 1), round(end, 1)], "fps": res.fps_used,
                        "device": res.device, "seconds_taken": round(time.perf_counter() - t0, 1),
                        "summary": V.clean(h["summary"]),
                        "profiles": {n: V.clean({k: v for k, v in h[n].items()}) for n in ("red", "blue")},
                    }
                    st.session_state["film_current"] = {"study": study, "timeline": h["timeline"], "thumbs": thumbs}
                    film_library().insert(0, study)
                    st.rerun()

# ------------------------------------------------------------------ 3. results
cur = st.session_state.get("film_current")
if cur:
    study, tl, thumbs = cur["study"], cur["timeline"], cur["thumbs"]
    fr, fb = study["fighters"]["red"], study["fighters"]["blue"]
    sm, P = study["summary"], study["profiles"]
    st.markdown("### 3. What the footage shows")
    st.markdown(f'<p class="oo-note">{ui.e(fr)} vs {ui.e(fb)} · {ui.e(study["clip"])} · '
                f'{mmss(study["range"][0])}–{mmss(study["range"][1])} · both fighters tracked in '
                f'{(sm.get("both_visible") or 0):.0%} of frames · analysed on {ui.e(study["device"])} in '
                f'{mmss(study["seconds_taken"])}</p>', unsafe_allow_html=True)
    if (sm.get("both_visible") or 0) < 0.5:
        st.warning("Both fighters were visible in under half the frames (close-ups, replays or cutaways). "
                   "Treat these numbers loosely, or pick a section with a wide shot.")

    def val(p, k, fmt):
        v = p.get(k)
        return "—" if v is None else fmt.format(v)

    def main_stance(p):
        st_ = p.get("stance") or {}
        if not st_:
            return "—", None
        k = max(st_, key=st_.get)
        return f"{k} {st_[k]:.0%}", k

    def edge(a, b, higher=True, eps=0.0):
        if a is None or b is None or abs(a - b) <= eps:
            return 0
        return 1 if (a > b) == higher else -1

    sr, sb = main_stance(P["red"])[0], main_stance(P["blue"])[0]
    rows = [
        ("Stance", sr, sb, 0),
        ("Moving forward", val(P["red"], "advancing", "{:.0%}"), val(P["blue"], "advancing", "{:.0%}"),
         edge(P["red"].get("advancing"), P["blue"].get("advancing"), eps=.03)),
        ("Moving backward", val(P["red"], "retreating", "{:.0%}"), val(P["blue"], "retreating", "{:.0%}"), 0),
        ("Footwork (torso lengths / s)", val(P["red"], "activity", "{:.2f}"), val(P["blue"], "activity", "{:.2f}"),
         edge(P["red"].get("activity"), P["blue"].get("activity"), eps=.05)),
        ("Hands up (guard high)", val(P["red"], "guard_high", "{:.0%}"), val(P["blue"], "guard_high", "{:.0%}"),
         edge(P["red"].get("guard_high"), P["blue"].get("guard_high"), eps=.03)),
        ("Punch attempts / min", val(P["red"], "punches_pm", "{:.1f}"), val(P["blue"], "punches_pm", "{:.1f}"),
         edge(P["red"].get("punches_pm"), P["blue"].get("punches_pm"), eps=.5)),
        ("Share thrown with lead hand", val(P["red"], "lead_share", "{:.0%}"), val(P["blue"], "lead_share", "{:.0%}"), 0),
        ("Kick attempts / min", val(P["red"], "kicks_pm", "{:.1f}"), val(P["blue"], "kicks_pm", "{:.1f}"),
         edge(P["red"].get("kicks_pm"), P["blue"].get("kicks_pm"), eps=.3)),
        ("Level changes / min", val(P["red"], "level_changes_pm", "{:.1f}"),
         val(P["blue"], "level_changes_pm", "{:.1f}"),
         edge(P["red"].get("level_changes_pm"), P["blue"].get("level_changes_pm"), eps=.3)),
        ("Time on the ground", val(P["red"], "ground", "{:.0%}"), val(P["blue"], "ground", "{:.0%}"), 0),
    ]
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown(ui.tape(fr, fb, rows), unsafe_allow_html=True)
        st.markdown('<p class="oo-cap">Estimated from 2D video, so read these as tendencies over the clip, not '
                    'exact counts. Distances are in torso lengths, so camera zoom doesn\'t matter. Forward and '
                    'backward movement has the camera\'s panning removed.</p>', unsafe_allow_html=True)
    with right:
        st.markdown("**Where the fight happened**")
        rng = [("Outside kicking range", sm.get("range_outside")), ("Punching range", sm.get("range_pocket")),
               ("Clinch", sm.get("range_clinch"))]
        ground = sm.get("ground_share") or 0
        rows_r = [[k, (f"{(v or 0) * (1 - ground):.0%}{ui.bar((v or 0) * (1 - ground), ui.INK)}", "")] for k, v in rng]
        rows_r.append(["On the ground", (f"{ground:.0%}{ui.bar(ground, ui.INK)}", "")])
        st.markdown(ui.table(["", "Share of time"], rows_r), unsafe_allow_html=True)

        if "distance" in tl:
            d = tl[["t", "distance"]].dropna()
            ev = []
            for corner, name in (("red", fr), ("blue", fb)):
                for t_, kind in P[corner].get("events", []) or []:
                    ev.append({"t": t_, "fighter": name, "kind": {"punch": "Punch", "kick": "Kick",
                                                                  "level": "Level change"}.get(kind, kind)})
            base = alt.Chart(d).mark_line(color=ui.INK, strokeWidth=1.5).encode(
                x=alt.X("t:Q", title="Seconds into the video"),
                y=alt.Y("distance:Q", title="Distance apart (torso lengths)", scale=alt.Scale(zero=True)),
                tooltip=[alt.Tooltip("t:Q", format=".1f", title="Time"),
                         alt.Tooltip("distance:Q", format=".1f", title="Distance")])
            layers = [base]
            if ev:
                layers.append(alt.Chart(pd.DataFrame(ev)).mark_tick(thickness=2, size=14).encode(
                    x="t:Q", y=alt.value(6),
                    color=alt.Color("fighter:N", scale=alt.Scale(domain=[fr, fb], range=[ui.RED, ui.BLUE]), title=None),
                    tooltip=["fighter", "kind", alt.Tooltip("t:Q", format=".1f", title="Time")]))
            st.altair_chart(ui.style_chart(alt.layer(*layers).properties(height=200)), width="stretch")
            st.markdown('<p class="oo-cap">Ticks along the top are detected strikes and level changes.</p>',
                        unsafe_allow_html=True)

    if thumbs:
        st.markdown("**Check the tracking**")
        i = st.slider("Frame", 0, len(thumbs) - 1, 0, format="%d", label_visibility="collapsed")
        t_i, jpg = thumbs[i]
        img_c, _ = st.columns([2.2, 1])
        img_c.image(jpg, caption=f"{mmss(t_i)} · red = {fr}, blue = {fb}", width="stretch")

    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("Download study (JSON)", json.dumps(study, indent=1),
                       file_name=f"film_{fr.split()[-1]}_{fb.split()[-1]}.json".lower(), mime="application/json",
                       width="stretch")
    d2.download_button("Download frame data (CSV)", tl.to_csv(index=False),
                       file_name=f"film_{fr.split()[-1]}_{fb.split()[-1]}_frames.csv".lower(), mime="text/csv",
                       width="stretch")
    st.page_link("views/predict.py", label=f"See {fr} vs {fb} on the Predict page →")

# ------------------------------------------------------------------ library
st.markdown("### Film library")
lib = film_library()
if lib:
    rows = [[f'<span style="color:{ui.RED}">{ui.e(s_["fighters"]["red"])}</span> vs '
             f'<span style="color:{ui.BLUE}">{ui.e(s_["fighters"]["blue"])}</span>',
             (ui.e(s_["clip"]), "muted"), (mmss((s_["summary"].get("seconds") or 0)), "")] for s_ in lib]
    st.markdown(ui.table(["Fighters", "Clip", "Tracked"], rows, left=2), unsafe_allow_html=True)
    st.markdown('<p class="oo-cap">Studies appear on the Predict page whenever one of these fighters is picked. '
                'They don\'t change the odds: a few clips aren\'t enough to retrain the model on, and the habits '
                'haven\'t been tested against results yet. Download them to build up a dataset that can be.</p>',
                unsafe_allow_html=True)
else:
    st.markdown('<p class="oo-note">No studies yet. Finished studies are kept here until you close the page.</p>',
                unsafe_allow_html=True)

with st.expander("Load studies you downloaded earlier"):
    files = st.file_uploader("Study files (JSON)", type=["json"], accept_multiple_files=True,
                             label_visibility="collapsed")
    if files:
        added, bad = 0, []
        have = {(s_["clip"], tuple(s_["range"]), s_["fighters"]["red"]) for s_ in lib}
        for f in files:
            try:
                s_ = json.loads(f.getvalue())
                keyt = (s_["clip"], tuple(s_["range"]), s_["fighters"]["red"])
                assert "profiles" in s_ and "summary" in s_
            except Exception:
                bad.append(f.name)
                continue
            if keyt not in have:
                lib.append(s_)
                have.add(keyt)
                added += 1
        if added:
            st.success(f"Added {added} stud{'y' if added == 1 else 'ies'} to the library.")
        if bad:
            st.error(f"Couldn't read {', '.join(bad)}. Only files downloaded from this page work here.")
