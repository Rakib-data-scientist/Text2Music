import os
import base64
import torch
import torchaudio
import streamlit as st
from audiocraft.models import MusicGen

# Configure the Streamlit page
st.set_page_config(page_icon="musical_note", page_title="GenAI-Music")


@st.cache_resource
def load_model():
    """Load the pretrained music generation model."""
    return MusicGen.get_pretrained('facebook/musicgen-small')


def generate_music_tensors(description: str, duration: int):
    """Generate music tensors based on the given description and duration."""
    model = load_model()
    model.set_generation_params(use_sampling=True, top_k=250, duration=duration)
    output = model.generate(descriptions=[description], progress=True, return_tokens=True)
    return output[0]


def save_audio(samples: torch.Tensor):
    """Save generated audio samples to a file."""
    sample_rate = 32000
    save_path = "audio_output/"
    
    samples = samples.detach().cpu().unsqueeze(0) if samples.dim() == 2 else samples.detach().cpu()
    torchaudio.save(os.path.join(save_path, "audio_0.wav"), samples[0], sample_rate)


def get_binary_file_downloader_html(bin_file: str, file_label='File'):
    """Create a downloadable link for the generated audio file."""
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href


def main():
    """Main function to run the Streamlit app."""
    st.title("Text2Music Generator🎵")

    description = st.text_area("Music Description")
    duration = st.slider("Time Duration", 0, 20, 10)

    if description and duration:
        st.json({'Your Description': description, 'Selected Time Duration (in Seconds)': duration})

        st.subheader("Generated Music")
        music_tensors = generate_music_tensors(description, duration)
        save_audio(music_tensors)
        
        audio_filepath = 'audio_output/audio_0.wav'
        with open(audio_filepath, 'rb') as audio_file:
            st.audio(audio_file.read())
        
        st.markdown(get_binary_file_downloader_html(audio_filepath, 'Audio'), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
