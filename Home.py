import streamlit as st
import os

# Set page configuration
st.set_page_config(
    page_title="A/B Testing Analysis",
    page_icon="🧪",
    layout="wide"
)

st.sidebar.success("Select a page above.")

st.title("🧪 A/B Testing Statistical Analysis")
st.write("Upload your experiment data and run statistical tests between groups.")
st.write("Select a page from the sidebar that fits your stats use-case!")
st.write("Binary variables are metrics like conversion rates, click rates, retention, etc. which only have two possible options. Continuous variables are metrics like amount spent, average frame rates, total matches played, etc.")
