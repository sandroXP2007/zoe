import os
import toml
from pathlib import Path
import llama_cpp
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import jinja2
import asyncio
import time
from threading import Thread, Event
from typing import Optional, Dict, Any


current_generation_task = None
generation_cancel_event = Event()
model_speed_tracker = {"start_time": None, "tokens_count": 0}
thinking_mode_enabled = True
current_config = {}

config_path = Path("config.toml")
config = None
if config_path.exists():
    try:
        with open(config_path, 'r') as f:
            config = toml.load(f)
    except Exception as e:
        print(f"Error reading config.toml: {e}")
        config = None

model_path = None
if config and 'model' in config and 'path' in config['model']:
    model_path = os.path.expanduser(config['model']['path'])
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        exit(1)
else:
    cache_dir = Path.home() / ".cache" / "llama.cpp"
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
    models = list(cache_dir.glob("*.gguf"))
    if not models:
        print(f"No models found in {cache_dir}")
        exit(1)
    if len(models) > 1:
        print("Multiple models found. Choose one:")
        for i, model in enumerate(models):
            print(f"{i}: {model.name}")
        choice = int(input("Enter choice: "))
        model_path = str(models[choice])
    else:
        model_path = str(models[0])

model_config = config['model'] if config and 'model' in config else {}
n_ctx = model_config.get('n_ctx', 2048)
temperature = model_config.get('temperature', 0.7)
top_k = model_config.get('top_k', 40)
top_p = model_config.get('top_p', 0.95)

system_prompt = "You are Zoe, a helpful AI assistant. Your task is to assist users with their questions and tasks."
if config and 'system' in config and 'prompt' in config['system']:
    system_prompt = config['system']['prompt']

llm = llama_cpp.Llama(
    model_path=model_path,
    n_ctx=n_ctx,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    verbose=True,
    n_threads=10
)

plugins_dir = Path("plugins")
plugins_dir.mkdir(exist_ok=True)
for plugin in plugins_dir.glob("*.py"):
    print(f"Loaded plugin: {plugin.name}")

env = jinja2.Environment()
env.globals['len'] = len  

template_str = """{%- if tools %} 
{{- '<|im_start|>system\n' }}
{%- if messages and messages[0].role == 'system' %} 
{{- messages[0].content + '\n\n' }} 
{%- endif %} 
{{- '# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>' }}
{%- for tool in tools %} 
{{- '\n' }} 
{{- tool | tojson }} 
{%- endfor %} 
{{- '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tools_call></tools_call> XML tags:\n<tools_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tools_call><|im_end|>\n' }}
{%- else %} 
{%- if messages and messages[0].role == 'system' %} 
{{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }} 
{%- endif %} 
{%- endif %}

{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
{%- set index = (messages|length - 1) - loop.index0 %}
{%- if ns.multi_step_tool and message.role == "user" and not(message.content.startswith('<|im_start|>tool') and message.content.endswith('<|im_end|>')) %}
{%- set ns.multi_step_tool = false %}
{%- set ns.last_query_index = index %}
{%- endif %}
{%- endfor %}

{%- for message in messages %}
{%- if message.role == "user" or (message.role == "system" and not loop.first) %}
{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
{%- elif message.role == "assistant" %}
{%- set content = message.content %}
{%- set reasoning_content = '' %}
{%- if 'think>' in content and '</think>' in content %}
{%- set start_tag = content.find('think>') + len('think>') %}
{%- set end_tag = content.find('</think>') %}
{%- set reasoning_content = content[start_tag:end_tag] %}
{%- set content = content[end_tag + len('</think>'):] %}
{%- endif %}
{%- if loop.index0 > ns.last_query_index %}
{%- if loop.last or (not loop.last and reasoning_content) %}
{{- '<|im_start|>assistant\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
{%- else %}
{{- '<|im_start|>assistant\n' + content }}
{%- endif %}
{%- else %}
{{- '<|im_start|>assistant\n' + content }}
{%- endif %}
{%- if message.tool_calls %}
{%- for tool_call in message.tool_calls %}
{%- if (loop.first and content) or (not loop.first) %}
{{- '\n' }}
{%- endif %}
{%- if tool_call.function %}
{%- set tool_call = tool_call.function %}
{%- endif %}
{{- '<tools_call>\n{"name": "' }}
{{- tool_call.name }}
{{- '", "arguments": ' }}
{%- if tool_call.arguments is string %}
{{- tool_call.arguments }}
{%- else %}
{{- tool_call.arguments | tojson }}
{%- endif %}
{{- '}\n</tools_call>' }}
{%- endfor %}
{%- endif %}
{{- '<|im_end|>\n' }}
{%- elif message.role == "tool" %}
{%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
{{- '<|im_start|>user' }}
{%- endif %}
{{- '\n<|im_start|>tool\n' }}
{{- message.content }}
{{- '\n<|im_end|>' }}
{%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
{{- '\n' }}
{%- endif %}
{%- endif %}
{%- endfor %}

{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- if enable_thinking is defined and enable_thinking is false %}
{{- '<think>\n\n</think>\n\n' }}
{%- endif %}
{%- endif %}"""

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: Request):
    global current_generation_task, generation_cancel_event, model_speed_tracker
    

    if current_generation_task and not current_generation_task.done():
        generation_cancel_event.set()
        await asyncio.sleep(0.1)  
    
    generation_cancel_event.clear()
    current_generation_task = asyncio.current_task()
    
    data = await request.json()
    messages = data.get("messages", [])
    tools = data.get("tools", [])
    add_generation_prompt = data.get("add_generation_prompt", True)
    enable_thinking = data.get("enable_thinking", thinking_mode_enabled)
    
    if not messages or messages[0].get('role') != 'system':
        messages = [{'role': 'system', 'content': system_prompt}] + messages
    
    template = env.from_string(template_str)
    prompt = template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking
    )
    
    print("Prompt gerado:\n", prompt)
    

    model_speed_tracker["start_time"] = time.time()
    model_speed_tracker["tokens_count"] = 0
    
    output = llm.create_completion(
        prompt,
        max_tokens=8126,
        stream=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    
    def generate():
        global current_generation_task, generation_cancel_event, model_speed_tracker
        
        for chunk in output:
            if generation_cancel_event.is_set():
                break
            
            text = chunk["choices"][0]["text"]
            model_speed_tracker["tokens_count"] += len(text.split())  # Aproximação
            yield text
    
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/cancel")
async def cancel_generation():
    global generation_cancel_event
    generation_cancel_event.set()
    return {"message": "Generation cancelled"}

@app.get("/speed")
async def get_speed():
    if model_speed_tracker["start_time"] is None:
        return {"tokens_per_second": 0, "tokens_count": 0}
    
    elapsed_time = time.time() - model_speed_tracker["start_time"]
    tokens_per_second = model_speed_tracker["tokens_count"] / elapsed_time if elapsed_time > 0 else 0
    
    return {
        "tokens_per_second": round(tokens_per_second, 2),
        "tokens_count": model_speed_tracker["tokens_count"],
        "elapsed_time": round(elapsed_time, 2)
    }

@app.get("/thinking")
async def get_thinking_mode():
    return {"thinking_enabled": thinking_mode_enabled}

@app.post("/thinking")
async def set_thinking_mode(request: Request):
    global thinking_mode_enabled
    data = await request.json()
    thinking_mode_enabled = data.get("enabled", True)
    return {"thinking_enabled": thinking_mode_enabled}

@app.get("/config")
async def get_config():
    global current_config
    current_config = load_config()
    return current_config

@app.post("/config")
async def update_config(request: Request):
    global current_config
    new_config = await request.json()
    current_config.update(new_config)
    save_config(current_config)
    
    return {"message": "Configuration updated", "config": current_config}

def load_config():
    global config_path
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return toml.load(f)
        except Exception as e:
            print(f"Error reading config.toml: {e}")
            return {}
    return {}

def save_config(config_data):
    global config_path
    try:
        with open(config_path, 'w') as f:
            toml.dump(config_data, f)
        print("Configuration saved to config.toml")
    except Exception as e:
        print(f"Error saving config.toml: {e}")

@app.get("/model/info")
async def get_model_info():
    return {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "system_prompt": system_prompt
    }

@app.post("/model/reload")
async def reload_model():
    global llm, n_ctx, temperature, top_k, top_p, system_prompt
    
    config = load_config()
    model_config = config.get('model', {})
    
    n_ctx = model_config.get('n_ctx', 2048)
    temperature = model_config.get('temperature', 0.7)
    top_k = model_config.get('top_k', 40)
    top_p = model_config.get('top_p', 0.95)
    
    if config and 'system' in config and 'prompt' in config['system']:
        system_prompt = config['system']['prompt']
    
    llm = llama_cpp.Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        verbose=True,
        n_threads=4
    )
    
    return {"message": "Model reloaded with new configuration"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
