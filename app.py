import streamlit as st
import tempfile, os
from model import load_model, predict_file

st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️")


@st.cache_resource
def get_model():
    return load_model("deploy_model.pt", device="cpu")


model, cfg = get_model()

st.title("Deepfake Audio Detector")
st.write(
    "Upload an audio clip"
)

uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3", "flac", "ogg"])

if uploaded is not None:
    st.audio(uploaded)

    suffix = os.path.splitext(uploaded.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Analyzing audio..."):
        verdict, p_fake = predict_file(tmp_path, model, cfg)
    os.remove(tmp_path)

    if "Deepfake" in verdict:
        st.error(f"{verdict}")
    else:
        st.success(f"{verdict}")

    st.metric("Probability of being AI-generated", f"{p_fake * 100:.1f}%")
    st.progress(min(max(p_fake, 0.0), 1.0))
    st.caption(
        f"Decision threshold: {cfg['threshold']:.3f}  "
        f"(above → Deepfake, below → Genuine). "
        "Model: custom CNN on log-mel spectrograms, trained on FoR + ASVspoof."
    )