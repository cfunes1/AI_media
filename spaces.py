import gradio as gr
import time
import numpy as np
from carlos_tools_audio import OpenAI_transcribe, local_whisper_transcribe, local_faster_whisper_transcribe
from carlos_tools_misc import clear_GPU_cache
import tempfile
import soundfile as sf

# Import your local whisper and faster-whisper models
from transformers import pipeline
import faster_whisper

# Dummy GPT remote transcription function (replace with your actual API call)
def gpt_transcribe(path):
    response = OpenAI_transcribe(
        path,
        model="whisper-1",
        response_format="text"
    )
    text=response["text"]
    duration=response["inference_time"]
    return text, duration

# Local Whisper
def whisper_transcribe(path):
    clear_GPU_cache()
    response = local_whisper_transcribe(
        path,
        model_size="large-v3",
    )
    text= response["text"]
    duration= response["inference_time"]
    return text, duration

# Faster Whisper
def faster_whisper_transcribe(path):
    clear_GPU_cache()
    response = local_faster_whisper_transcribe(
        path,
        model_size="distil-large-v3",
    )
    text = response["text"]
    duration = response["inference_time"]
    return text, duration

def compare_transcriptions(path):
    gpt_text, gpt_time = gpt_transcribe(path)
    faster_text, faster_time = faster_whisper_transcribe(path)
    whisper_text, whisper_time = whisper_transcribe(path)
    table = [
        # ["Model", "Transcription", "Duration (s)"],
        ["GPT (remote)", gpt_text, round(gpt_time, 2)],
        ["Whisper (local)", whisper_text, round(whisper_time, 2)],
        ["Faster Whisper (local)", faster_text, round(faster_time, 2)],
    ]
    return table

with gr.Blocks() as demo:   # may tray themes e.g. theme=gr.themes.Soft()
    with gr.Sidebar(position="left", width=200, visible=True):
        gr.Markdown("# Carlos' tests")
        gr.Markdown("## Audio")
        gr.Markdown("### [Speech-To-Text](https://huggingface.co/openai/whisper-large-v3)")

    gr.Markdown("""
            # Speech-To-Text tasks
            """)
    with gr.Tab("Transcribe"):
            with gr.Row():
                gr.Markdown("""
                            # TRANSCRIPTION
                            ## compare transcriptions from different models
                            ### Models used: 
                            """)
            with gr.Row():
                gr.Markdown("""
                            ### [Whisper](https://huggingface.co/openai/whisper-large-v3)

                            Whisper is  essentially a language model grounded in audio — an audio-conditional GPT. It was trained by OpenAI on a large and diverse dataset of multilingual audio, enabling it to perform automatic speech recognition (ASR) and translation tasks across many languages.

                            Whisper is trained in a similar fashion to the original GPT, using self-supervised learning with a next-token prediction objective. However, while GPT is trained solely on text, Whisper is trained on paired audio and text, where the model learns to generate transcriptions (or translations) token by token from audio inputs. During training, the encoder processes the audio into latent representations, and the decoder learns to predict the next text token given the previous tokens and the audio context. Unlike GPT, which relies purely on textual continuity, Whisper must also learn alignment between speech and language, making it a multimodal model trained end-to-end on large-scale audio-text datasets.
                            """)
                gr.Markdown("""
                            ### [Faster Whisper](https://huggingface.co/Systran/faster-whisper-large-v3)

                            Faster-Whisper is a high-performance, inference-optimized implementation of OpenAI's Whisper model — effectively a language model grounded in audio, engineered for fast, resource-efficient deployment.

                            While it retains the same underlying architecture and training paradigm as Whisper — a multimodal encoder-decoder transformer trained via self-supervised next-token prediction on paired audio-text data — Faster-Whisper focuses entirely on inference-time efficiency. It uses CTranslate2, a highly optimized inference engine for transformer models, to significantly accelerate transcription and translation while reducing memory usage.

                            Like Whisper, Faster-Whisper takes in raw audio, encodes it into latent representations via the audio encoder, and then decodes text token by token, conditioned on both the encoded audio and previously generated tokens. However, all training is inherited directly from the original Whisper checkpoints — Faster-Whisper is not retrained, but instead recompiled and optimized for speed and portability (e.g., on CPU, GPU, or ARM devices).

                            As a result, Faster-Whisper makes Whisper’s powerful multilingual speech recognition capabilities more accessible in production environments, edge devices, and real-time applications where latency and efficiency are critical.
                            """)

            audio_input = gr.Audio(sources="upload", type= "filepath", label="Upload Audio")
            output_table = gr.Dataframe(
                headers=["Model", "Translation", "Duration (s)"],
                datatype=["str", "str", "number"],  # Ensure "Translation" is string
                row_count=5,  # Adjust as needed for visible rows
                interactive=False
)
            transcribe_btn = gr.Button("Transcribe with All Models")
            transcribe_btn.click(compare_transcriptions, inputs=audio_input, outputs=output_table)
    with gr.Tab("Translate"):
            gr.Markdown("""
                        # Speech-To-Text tasks: TRANSLATION
                        ## compare translations from different models
                        ### Models used: 
                        ### [Whisper](https://huggingface.co/openai/whisper-large-v3)

                        Whisper is  essentially a language model grounded in audio — an audio-conditional GPT. It was trained by OpenAI on a large and diverse dataset of multilingual audio, enabling it to perform automatic speech recognition (ASR) and translation tasks across many languages.

                        Whisper is trained in a similar fashion to the original GPT, using self-supervised learning with a next-token prediction objective. However, while GPT is trained solely on text, Whisper is trained on paired audio and text, where the model learns to generate transcriptions (or translations) token by token from audio inputs. During training, the encoder processes the audio into latent representations, and the decoder learns to predict the next text token given the previous tokens and the audio context. Unlike GPT, which relies purely on textual continuity, Whisper must also learn alignment between speech and language, making it a multimodal model trained end-to-end on large-scale audio-text datasets.

                        ### [Faster Whisper](https://huggingface.co/Systran/faster-whisper-large-v3)

                        Faster-Whisper is a high-performance, inference-optimized implementation of OpenAI's Whisper model — effectively a language model grounded in audio, engineered for fast, resource-efficient deployment.

                        While it retains the same underlying architecture and training paradigm as Whisper — a multimodal encoder-decoder transformer trained via self-supervised next-token prediction on paired audio-text data — Faster-Whisper focuses entirely on inference-time efficiency. It uses CTranslate2, a highly optimized inference engine for transformer models, to significantly accelerate transcription and translation while reducing memory usage.

                        Like Whisper, Faster-Whisper takes in raw audio, encodes it into latent representations via the audio encoder, and then decodes text token by token, conditioned on both the encoded audio and previously generated tokens. However, all training is inherited directly from the original Whisper checkpoints — Faster-Whisper is not retrained, but instead recompiled and optimized for speed and portability (e.g., on CPU, GPU, or ARM devices).

                        As a result, Faster-Whisper makes Whisper’s powerful multilingual speech recognition capabilities more accessible in production environments, edge devices, and real-time applications where latency and efficiency are critical.
                        """)
            audio_input = gr.Audio(sources="upload", type= "filepath", label="Upload Audio")
            # output_table = gr.Dataframe(headers=["Model", "Translation", "Duration (s)"], interactive=False)
            output_table = gr.Dataframe(
                headers=["Model", "Translation", "Duration (s)"],
                datatype=["str", "str", "number"],  # Ensure "Translation" is string
                row_count=5,  # Adjust as needed for visible rows
                interactive=False
)
            transcribe_btn = gr.Button("Translate with All Models")
            transcribe_btn.click(compare_transcriptions, inputs=audio_input, outputs=output_table)
    with gr.Tab("Detect language"):
            gr.Markdown("""
                        # Speech-To-Text tasks: LANGUAGE DETECTION
                        ## compare language detection from different models
                        ### Models used: 
                        ### [Whisper](https://huggingface.co/openai/whisper-large-v3)

                        Whisper is  essentially a language model grounded in audio — an audio-conditional GPT. It was trained by OpenAI on a large and diverse dataset of multilingual audio, enabling it to perform automatic speech recognition (ASR) and translation tasks across many languages.

                        Whisper is trained in a similar fashion to the original GPT, using self-supervised learning with a next-token prediction objective. However, while GPT is trained solely on text, Whisper is trained on paired audio and text, where the model learns to generate transcriptions (or translations) token by token from audio inputs. During training, the encoder processes the audio into latent representations, and the decoder learns to predict the next text token given the previous tokens and the audio context. Unlike GPT, which relies purely on textual continuity, Whisper must also learn alignment between speech and language, making it a multimodal model trained end-to-end on large-scale audio-text datasets.

                        ### [Faster Whisper](https://huggingface.co/Systran/faster-whisper-large-v3)

                        Faster-Whisper is a high-performance, inference-optimized implementation of OpenAI's Whisper model — effectively a language model grounded in audio, engineered for fast, resource-efficient deployment.

                        While it retains the same underlying architecture and training paradigm as Whisper — a multimodal encoder-decoder transformer trained via self-supervised next-token prediction on paired audio-text data — Faster-Whisper focuses entirely on inference-time efficiency. It uses CTranslate2, a highly optimized inference engine for transformer models, to significantly accelerate transcription and translation while reducing memory usage.

                        Like Whisper, Faster-Whisper takes in raw audio, encodes it into latent representations via the audio encoder, and then decodes text token by token, conditioned on both the encoded audio and previously generated tokens. However, all training is inherited directly from the original Whisper checkpoints — Faster-Whisper is not retrained, but instead recompiled and optimized for speed and portability (e.g., on CPU, GPU, or ARM devices).

                        As a result, Faster-Whisper makes Whisper’s powerful multilingual speech recognition capabilities more accessible in production environments, edge devices, and real-time applications where latency and efficiency are critical.
                        """)
            audio_input = gr.Audio(sources="upload", type= "filepath", label="Upload Audio")
            output_table = gr.Dataframe(headers=["Model", "Language", "Duration (s)"], interactive=False)
            transcribe_btn = gr.Button("Detect Language with All Models")

demo.launch()