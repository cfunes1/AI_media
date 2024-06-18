import ollama
response = ollama.chat(model="llama3",messages=[
    {"role": "user", 
     "content": "say hello world in spanish"
     },
])
print(response["message"]["content"])