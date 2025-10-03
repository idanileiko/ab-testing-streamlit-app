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
st.markdown("""
    ***Binary variables*** are metrics that only have two options. Examples are:
         - conversion (did users spend or no?)
         - click rates (did users click a link?)
         - retention (did users retain on d7 or no?)

    ***Continuous variables*** are metrics that can be any value along a continuous scale. Examples are:
        - total amount spent
        - average frame rate
        - total matches played
""")
