import project
import pytest


def test_get_video_stream():
    assert project.get_video_stream("https://www.youtube.com/shorts/fN-_j1yOXCE",audio_only=True).title == "A one-handed 5-way polyrhythmic game I frequent."

    assert project.get_video_stream("https://www.youtube.com/watch?v=TV7Dj8lch4k",audio_only=True).title == "Vídeo motivacional animado  | By Operary"

    with pytest.raises(ValueError):
        assert project.get_video_stream("https://www.youtube.com/invalidName")    

def test_speech_to_text():
    assert project.speech_to_text("./media/A one-handed 5-way polyrhythmic game I frequent.mp4").lower().strip().startswith("when i was a teenager") == True

    assert project.speech_to_text("./media/Vídeo motivacional animado   By Operary.mp4").lower().strip().startswith("sale el sol") == True

def test_speech_to_English_text():
    assert project.speech_to_English_text("./media/Vídeo motivacional animado   By Operary.mp4").lower().strip().startswith("the sun rises") == True