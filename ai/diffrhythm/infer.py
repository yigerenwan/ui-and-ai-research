# Copyright (c) 2025 ASLP-LAB
#               2025 Huakang Chen  (huakang@mail.nwpu.edu.cn)
#               2025 Guobin Ma     (guobin.ma@gmail.com)
#
# Licensed under the Stability AI License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE.md
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import time

import torch
import torchaudio
from einops import rearrange
import streamlit as st

print("Current working directory:", os.getcwd())

from utils import (
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
)


def inference(
    cfm_model,
    vae_model,
    cond,
    text,
    duration,
    style_prompt,
    negative_style_prompt,
    start_time,
    chunked=False,
):
    with torch.inference_mode():
        generated, _ = cfm_model.sample(
            cond=cond,
            text=text,
            duration=duration,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            steps=32,
            cfg_strength=4.0,
            start_time=start_time,
        )

        generated = generated.to(torch.float32)
        latent = generated.transpose(1, 2)  # [b d t]

        output = decode_audio(latent, vae_model, chunked=chunked)

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")
        # Peak normalize, clip, convert to int16, and save to file
        output = (
            output.to(torch.float32)
            .div(torch.max(torch.abs(output)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )

        return output


def parse_lrc(lrc_text):
    """Parse LRC format lyrics into a list of (timestamp, lyric) tuples"""
    lyrics = []
    for line in lrc_text.split('\n'):
        if line.strip() and '[' in line:
            time_str = line[line.find('[')+1:line.find(']')]
            text = line[line.find(']')+1:].strip()
            
            # Convert timestamp to seconds
            try:
                min_sec = time_str.split(':')
                seconds = float(min_sec[0]) * 60 + float(min_sec[1])
                lyrics.append((seconds, text))
            except:
                continue
    return sorted(lyrics)


def main():
    st.title("DiffRhythm Music Generator")
    
    # Create single column layout with audio at top and lyrics at bottom
    st.subheader("Generated Music")
    audio_placeholder = st.empty()  # Placeholder for audio widget
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    # Text area for lyrics input
    lrc = st.sidebar.text_area(
        "Enter Lyrics (LRC format)",
        """[00:00.00]Example lyrics line 1
[00:05.00]Example lyrics line 2
[00:10.00]Example lyrics line 3""",
        help="Enter lyrics in LRC format with timestamps [mm:ss.xx]",
        height=200
    )
    
    # Style controls
    style_option = st.sidebar.radio(
        "Style Input Method",
        ["Text Prompt", "Audio Reference"]
    )
    
    example_prompts = {
        "Classical Piano": "classical genres, hopeful mood, piano, emotional, orchestral",
        "Rock Band": "rock music, electric guitar, drums, energetic, powerful vocals",
        "Electronic Dance": "electronic dance music, upbeat, synthesizer, modern beats",
        "Jazz Ensemble": "jazz music, smooth saxophone, improvisation, swing rhythm",
        "Pop Music": "pop music, catchy melody, modern production, radio-friendly"
    }
    
    if style_option == "Text Prompt":
        # Add example prompts dropdown
        selected_example = st.sidebar.selectbox(
            "Example Style Prompts (Select to copy)",
            ["Custom"] + list(example_prompts.keys())
        )
        
        default_prompt = example_prompts[selected_example] if selected_example != "Custom" else "classical genres, hopeful mood, piano."
        
        ref_prompt = st.sidebar.text_area(
            "Style Prompt",
            default_prompt,
            help="Enter a description of the musical style or select an example above"
        )
        ref_audio_path = None
    else:
        ref_audio = st.sidebar.file_uploader("Upload reference audio", type=['wav', 'mp3'])
        ref_prompt = None
        if ref_audio:
            ref_audio_path = "temp_reference.wav"
            with open(ref_audio_path, "wb") as f:
                f.write(ref_audio.getbuffer())
        else:
            ref_audio_path = None
    
    # Adjustable audio length
    audio_length = st.sidebar.number_input(
        "Audio Length (seconds)",
        min_value=10,
        max_value=180,
        value=95,
        step=5,
        help="Length of generated song (10-180 seconds)"
    )
    
    chunked = st.sidebar.checkbox("Use Chunked Decoding", value=True)
    
    if st.sidebar.button("Generate Music"):
        if not (ref_prompt or ref_audio_path):
            st.error("Please provide either a style prompt or reference audio")
            return
            
        with st.spinner("Preparing models..."):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                st.warning("GPU not available. Processing might be slow.")
                
            # Calculate max_frames based on audio length
            max_frames = int((audio_length / 95) * 2048)  # Scale frames proportionally
            cfm, tokenizer, muq, vae = prepare_model(device)
            
        with st.spinner("Processing..."):
            # Process lyrics
            lrc_prompt, start_time = get_lrc_token(lrc, tokenizer, device)
            
            # Get style prompt
            if ref_audio_path:
                style_prompt = get_style_prompt(muq, ref_audio_path)
            else:
                style_prompt = get_style_prompt(muq, prompt=ref_prompt)
                
            negative_style_prompt = get_negative_style_prompt(device)
            latent_prompt = get_reference_latent(device, max_frames)
            
            # Generate music
            generated_song = inference(
                cfm_model=cfm,
                vae_model=vae,
                cond=latent_prompt,
                text=lrc_prompt,
                duration=max_frames,
                style_prompt=style_prompt,
                negative_style_prompt=negative_style_prompt,
                start_time=start_time,
                chunked=chunked,
            )
            
            # Display result - audio first
            output_path = "output.wav"
            torchaudio.save(output_path, generated_song, sample_rate=44100)
            
            # Parse lyrics
            parsed_lyrics = parse_lrc(lrc)
            
            # Create audio player at top
            audio_file = open(output_path, 'rb')
            audio_bytes = audio_file.read()
            audio_placeholder.audio(audio_bytes, format='audio/wav')
            
            # Display lyrics section at bottom
            st.subheader("Lyrics")
            lyrics_text = "\n".join([f"{int(t//60):02d}:{t%60:05.2f} {l}" for t, l in parsed_lyrics])
            st.text(lyrics_text)
            
            # Add JavaScript for syncing lyrics with audio playback
            st.markdown("""
                <script>
                const audio = document.querySelector('audio');
                const currentLyric = document.getElementById('current-lyric');
                
                audio.addEventListener('timeupdate', function() {
                    const time = audio.currentTime;
                    // Update current lyric based on time
                    // This is handled by the Streamlit app
                });
                </script>
                """, unsafe_allow_html=True)
            
            # Cleanup temporary files
            if ref_audio_path == "temp_reference.wav" and os.path.exists(ref_audio_path):
                os.remove(ref_audio_path)
            if os.path.exists(output_path):
                os.remove(output_path)


if __name__ == "__main__":
    main()