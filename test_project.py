import project
import pytest
import os


def test_get_video():
    #invalid file format
    with pytest.raises(FileNotFoundError):
        assert project.get_video("asdfasdf")

    #not existing file
    with pytest.raises(FileNotFoundError):
        assert project.get_video("https://www.youtube.com/invalidNamehgfd")

    #existing file correctly found
    ytObject = project.get_video("https://www.youtube.com/watch?v=TV7Dj8lch4k")
    assert ytObject.title == "Vídeo motivacional animado  | By Operary"

def test_save_smallest_audio_stream():
    #file correctly saved
    ytObject = project.get_video("https://www.youtube.com/watch?v=fN-_j1yOXCE")
    assert project.save_smallest_audio_stream(ytObject,"./media","fN-_j1yOXCE.mp3").endswith("fN-_j1yOXCE.mp3")

def test_cut_file():
    ytObject = project.get_video("https://www.youtube.com/watch?v=WbzNRTTrX0g")
    download_full_path = project.save_smallest_audio_stream(ytObject,"./media","WbzNRTTrX0g.mp3")
    assert project.cut_file(download_full_path, 10 * 60) == True
    assert project.cut_file(download_full_path, 10 * 60) == False


def test_speech_to_text():
    
    ytObject = project.get_video("https://www.youtube.com/watch?v=fN-_j1yOXCE")
    download_full_path = project.save_smallest_audio_stream(ytObject,"./media","fN-_j1yOXCE.mp3")
    transcription = project.speech_to_text(download_full_path)
    assert transcription.text.lower().strip().startswith("when i was a teenager") 
    assert transcription.language == "english"

    ytObject = project.get_video("https://www.youtube.com/watch?v=TV7Dj8lch4k")
    download_full_path = project.save_smallest_audio_stream(ytObject,"./media","TV7Dj8lch4k.mp3")

    transcription = project.speech_to_text(download_full_path)
    assert transcription.text.lower().strip().startswith("sale el sol")
    assert transcription.language == "spanish"

    translation = project.speech_to_English_text(download_full_path)
    assert translation.text.lower().strip().startswith("the sun rises")
    assert translation.language == "english"

def save_text_to_file():
    with pytest.raises(ValueError):
        assert project.save_text_to_file("","")

def test_summarize_text():
    with pytest.raises(ValueError):
        assert project.summarize_text("")

def test_cut_text():
    assert project.cut_text("This is a test of the cut_text function",4096) == "This is a test of the cut_text function"
    assert project.cut_text("This is a test of the cut_text function",4) == "This"

def test_text_to_speech():
    with pytest.raises(ValueError):
        assert project.text_to_speech("","")

def test_generate_image():
    with pytest.raises(ValueError):
        assert project.generate_image("abc","")
    with pytest.raises(ValueError):
        assert project.generate_image("","url")
    with pytest.raises(ValueError):
        assert project.generate_image("","b64_json")
    with pytest.raises(ValueError):
        assert project.generate_image("abc","abc")

def test_save_image_from_url():
    with pytest.raises(ValueError):
        assert project.save_image_from_URL("","")

def test_save_image_from_b64data():
    with pytest.raises(ValueError):
        assert project.save_image_from_b64data("","")
