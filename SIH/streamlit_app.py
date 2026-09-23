import streamlit as st

st.set_page_config(
    page_title="IBVAP",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ IBVAP")
st.subheader("AI-Based Intelligent Video Analytics Platform")

st.success("IBVAP Streamlit application is running!")

st.markdown("""
### AI Analytics Layer

- 👤 Human Detection & Tracking
- 🚗 Vehicle Detection
- 🔢 ANPR + OCR
- 🚧 Digital Fence / Intrusion Detection
- ⚠️ Suspicious Activity Detection
- 📹 CCTV / RTSP Video Analytics
""")

st.info("Streamlit deployment test — AI pipeline will be connected next.")