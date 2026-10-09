"""Octagon Odds: Streamlit entry point.   streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from mmapred import ui
from mmapred.train import ART

st.set_page_config(page_title="Octagon Odds", page_icon="🥊", layout="wide", initial_sidebar_state="collapsed")
st.markdown(ui.CSS, unsafe_allow_html=True)

if not (ART / "model.json").exists():
    st.error("The trained model is missing. Run `python -m mmapred.train` from the project folder, then reload.")
    st.stop()

from views.common import get_predictor  # noqa: E402  (after the model check)

pages = [
    st.Page("views/predict.py", title="Predict", default=True),
    st.Page("views/research.py", title="Research"),
    st.Page("views/about.py", title="About"),
]
# Our own navigation row (same on desktop and phone) instead of Streamlit's built-in menu,
# which collapses into a hidden sidebar on small screens.
nav = st.navigation(pages, position="hidden")
pr = get_predictor()
st.markdown(ui.masthead(pr.data_through.strftime("%b %-d, %Y")), unsafe_allow_html=True)
with st.container(horizontal=True, key="oo_nav"):
    for page in pages:
        st.page_link(page, label=page.title)
nav.run()
