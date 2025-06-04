from carlos_tools_audio import local_whisper, local_faster_whisper

def test_local_whisper():
    # Test the local_whisper function with a sample audio file
    audio_file = '001-About-me.mp3'  # Replace with your actual audio file path

    # Call the local_whisper function
    transcription = local_faster_whisper(
        directory='media',
        file_name=audio_file,
    )
    # Print the transcription
    print("Transcription:", transcription['text'])
if __name__ == "__main__":
    test_local_whisper()