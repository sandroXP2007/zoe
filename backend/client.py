import requests
import re
import sys
import time

def format_markdown(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\033[1m\1\033[0m', text)
    text = re.sub(r'\*(.*?)\*', r'\033[3m\1\033[0m', text)
    text = re.sub(r'`(.*?)`', r'\033[4m\1\033[0m', text)
    text = re.sub(r'```(\w*)\n(.*?)```', r'\033[94m\1\n\2\033[0m', text, flags=re.DOTALL)
    text = re.sub(r'^(#{1,6})\s+(.*)$', r'\033[1m\2\033[0m', text, flags=re.MULTILINE)
    text = re.sub(r'^(\s*)- (.*)$', r'\1• \2', text, flags=re.MULTILINE)
    return text

def stream_response(text, delay=0.02):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

url = "http://localhost:8000/chat"
messages = []

while True:
    user_input = input("> ")
    if user_input.lower() in ["exit", "quit"]:
        break
   
    messages.append({"role": "user", "content": user_input})
    response = requests.post(url, json={"messages": messages, "add_generation_prompt": True}, stream=True)
    assistant_response = ""
    for line in response.iter_lines():
        if line:
            text = line.decode()
            assistant_response += text
            formatted_text = format_markdown(assistant_response)
            sys.stdout.write("\r\033[K") 
            sys.stdout.write(formatted_text)
            sys.stdout.flush()
    print()  
    messages.append({"role": "assistant", "content": assistant_response})
