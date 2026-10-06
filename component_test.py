import streamlit as st
import streamlit.components.v1 as components

st.title("Component Test")

components.html(
    """
    <div style="
        width:100%;
        height:400px;
        background:#e8f4ff;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:30px;
        font-family:Arial;
    ">
        STREAMLIT COMPONENT IS WORKING
    </div>
    """,
    height=400
)