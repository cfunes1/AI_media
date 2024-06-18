from project import yt_transcript, summarize_text

url = input("Enter the url of the video: ")
if url == "":
    url= "https://www.youtube.com/watch?v=WbzNRTTrX0g"

transcript = yt_transcript(url)

text = ""
for sentence in transcript:
    #print(sentence)
    text += f"{sentence["text"]}\n"

with open("transcript.txt","w") as f:
    f.write(text)
    #summary = summarize_text(text, run_local = False)
    #print(summary)
    #f.write(summary)