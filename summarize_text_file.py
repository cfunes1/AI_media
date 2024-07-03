from project import summarize_text

f = input("Text file to summarize: ")
with open(f, "r") as file:
    text = file.read()
    print(text)
    print("Summarizing text...")
    summary = summarize_text(text, run_local=True)
    print(summary)
    with open("summary.txt", "w") as file:
        file.write(summary)
