"""Streamlit automatically exposes this page in the sidebar navigation."""

import streamlit as st

from business.review_ui import render_review_page


st.set_page_config(page_title="ServicePilot AI")
with st.sidebar:
    st.markdown("### ServicePilot AI")

render_review_page(st)
