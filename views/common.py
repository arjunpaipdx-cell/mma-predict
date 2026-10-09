"""Shared state for every page: the predictor, saved metrics and the film-study library."""
from __future__ import annotations

import json

import streamlit as st

from mmapred.predict import Predictor
from mmapred.train import ART


@st.cache_resource(show_spinner="Loading every UFC result since 1994…")
def get_predictor() -> Predictor:
    return Predictor()


@st.cache_data
def load_metrics() -> dict:
    return json.loads((ART / "metrics.json").read_text())


def film_library() -> list[dict]:
    """Film-study results saved in this browser session (newest first)."""
    return st.session_state.setdefault("film", [])


def film_for(name: str) -> list[dict]:
    """Every saved study where `name` was one of the two tracked fighters, as that fighter's profile."""
    out = []
    for study in film_library():
        for corner in ("red", "blue"):
            if study["fighters"][corner] == name:
                out.append({**study["profiles"][corner], "opponent": study["fighters"]["blue" if corner == "red" else "red"],
                            "clip": study["clip"], "seconds": study["summary"].get("seconds", 0),
                            "summary": study["summary"]})
    return out
