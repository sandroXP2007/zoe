import os
import toml
from pathlib import Path
import llama_cpp
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import jinja2
import asyncio
import time
import importlib.util
import inspect
import json
import re
from threading import Event
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

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
        print(f"[ERROR] Error reading config.toml: {e}")
        config = None

model_path = None
if config and 'model' in config and 'path' in config['model']:
    model_path = os.path.expanduser(config['model']['path'])
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}")
        exit(1)
else:
    cache_dir = Path.home() / ".cache" / "llama.cpp"
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
    models = list(cache_dir.glob("*.gguf"))
    if not models:
        print(f"[ERROR] No models found in {cache_dir}")
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

chat_template_name = None
if config and 'model' in config and 'chat_template' in config['model']:
    chat_template_name = config['model']['chat_template']
else:
    model_filename = Path(model_path).stem.lower()
    if 'qwen3' in model_filename:
        chat_template_name = "qwen3"
    elif 'llama' in model_filename:
        chat_template_name = "llama"
    elif 'mistral' in model_filename:
        chat_template_name = "mistral"
    elif 'gemma' in model_filename:
        chat_template_name = "gemma"
    elif 'deepseek' in model_filename:
        chat_template_name = "deepseek"
    elif 'phi' in model_filename:
        chat_template_name = "phi"
    elif 'yi' in model_filename:
        chat_template_name = "yi"
    else:
        chat_template_name = "default"

template_str = None
chat_templates_dir = Path("chat_templates")
chat_templates_dir.mkdir(exist_ok=True)

template_file = chat_templates_dir / chat_template_name
if template_file.exists():
    try:
        with open(template_file, 'r') as f:
            template_str = f.read()
        print(f"[INFO] Loaded chat template: {chat_template_name}")
    except Exception as e:
        print(f"[ERROR] Error loading chat template {chat_template_name}: {e}")

if not template_str:
    print(f"[INFO] Using default template for {chat_template_name}")
    if chat_template_name == "qwen3":
        template_str = """{%- if tools %} {{- '<<|im_start|>>system\n' }} {%- if messages and messages[0].role == 'system' %} {{- messages[0].content + '\n\n' }} {%- endif %} {{- '# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>' }} {%- for tool in tools %} {{- '\n' }} {{- tool | tojson }} {%- endfor %} {{- '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tools></tools> XML tags:\n<tools>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tools><|im_end|>\n' }} {%- else %} {%- if messages and messages[0].role == 'system' %} {{- '<<|im_start|>>system\n' + messages[0].content + '<|im_end|>\n' }} {%- endif %} {%- endif %} {%- for message in messages %} {%- if message.role == "user" or (message.role == "system" and not loop.first) %} {{- '<<|im_start|>>' + message.role + '\n' + message.content + '<|im_end|>\n' }} {%- elif message.role == "assistant" %} {{- '<<|im_start|>>assistant\n' + message.content + '<|im_end|>\n' }} {%- elif message.role == "tool" %} {{- '<<|im_start|>>tool\n' + message.content + '<|im_end|>\n' }} {%- endif %} {%- endfor %} {%- if add_generation_prompt %} {{- '<<|im_start|>>assistant\n' }} {%- if enable_thinking is defined and enable_thinking is false %} {{- '<think>\n\n</think>\n\n' }} {%- endif %} {%- endif %}"""
    else:
        template_str = """{%- if tools %} {{- '<<|system|>>\n' }} {%- if messages and messages[0].role == 'system' %} {{- messages[0].content + '\n\n' }} {%- endif %} {{- '# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>' }} {%- for tool in tools %} {{- '\n' }} {{- tool | tojson }} {%- endfor %} {{- '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tools></tools> XML tags:\n<tools>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tools>\n' }} {%- else %} {%- if messages and messages[0].role == 'system' %} {{- '<<|system|>>\n' + messages[0].content + '\n' }} {%- endif %} {%- endif %} {%- for message in messages %} {%- if message.role == "user" or (message.role == "system" and not loop.first) %} {{- '<<|user|>>\n' + message.content + '<|end|>\n' }} {%- elif message.role == "assistant" %} {{- '<<|assistant|>>\n' + message.content + '<|end|>\n' }} {%- elif message.role == "tool" %} {{- '<<|tool|>>\n' + message.content + '<|end|>\n' }} {%- endif %} {%- endfor %} {%- if add_generation_prompt %} {{- '<<|assistant|>>\n' }} {%- if enable_thinking is defined and enable_thinking is false %} {{- '<think>\n\n</think>\n\n' }} {%- endif %} {%- endif %}"""

llm = llama_cpp.Llama(
    model_path=model_path,
    n_ctx=n_ctx,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    verbose=True,
    n_threads=8
)

class PluginManager:
    def __init__(self):
        self.plugins = {}
        self.tools = {}
        self._load_plugins()
    
    def _load_plugins(self):
        plugins_dir = Path("plugins")
        plugins_dir.mkdir(exist_ok=True)
        
        for plugin_file in plugins_dir.glob("*.py"):
            try:
                spec = importlib.util.spec_from_file_location(
                    plugin_file.stem, plugin_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, 'PLUGIN_NAME'):
                    plugin_name = getattr(module, 'PLUGIN_NAME')
                    self.plugins[plugin_name] = module
                    print(f"[PLUGIN] Loaded plugin: {plugin_name}")
                    self._register_plugin_tools(module, plugin_name)
                    if hasattr(module, 'initialize'):
                        module.initialize()
                        
            except Exception as e:
                print(f"[PLUGIN ERROR] Error loading plugin {plugin_file.name}: {e}")
    
    def _register_plugin_tools(self, module, plugin_name: str):
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith('_') and name not in ['initialize', 'cleanup']:
                tool_spec = self._create_tool_spec(func, name, plugin_name)
                full_tool_name = f"{plugin_name}_{name}"
                self.tools[full_tool_name] = {
                    "function": func,
                    "spec": tool_spec
                }
                print(f"[PLUGIN] Registered tool: {full_tool_name}")
    
    def _create_tool_spec(self, func, func_name: str, plugin_name: str) -> Dict[str, Any]:
        sig = inspect.signature(func)
        parameters = []
        
        for param_name, param in sig.parameters.items():
            param_spec = {
                "name": param_name,
                "type": "string"
            }
            
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_spec["type"] = "integer"
                elif param.annotation == float:
                    param_spec["type"] = "number"
                elif param.annotation == bool:
                    param_spec["type"] = "boolean"
                elif param.annotation == list:
                    param_spec["type"] = "array"
                elif param.annotation == dict:
                    param_spec["type"] = "object"
            
            if param.default != inspect.Parameter.empty:
                param_spec["default"] = param.default
            
            parameters.append(param_spec)
        
        return {
            "name": f"{plugin_name}_{func_name}",
            "description": getattr(func, '__doc__', f"Function {func_name} from {plugin_name} plugin") or f"Function {func_name} from {plugin_name} plugin",
            "parameters": {
                "type": "object",
                "properties": {p["name"]: {"type": p["type"]} for p in parameters},
                "required": [p["name"] for p in parameters if "default" not in p]
            }
        }
    
    def get_available_tools(self) -> list:
        tools_list = [tool["spec"] for tool in self.tools.values()]
        print(f"[PLUGIN] Available tools: {[t['name'] for t in tools_list]}")
        return tools_list
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[PLUGIN] Executing tool: {tool_name} with args: {arguments}")
        
        if tool_name in self.tools:
            try:
                tool_func = self.tools[tool_name]["function"]
                result = tool_func(**arguments)
                print(f"[PLUGIN] Tool {tool_name} executed successfully")
                return {
                    "success": True,
                    "tool_name": tool_name,
                    "result": result
                }
            except Exception as e:
                print(f"[PLUGIN ERROR] Error executing tool {tool_name}: {e}")
                return {
                    "success": False,
                    "tool_name": tool_name,
                    "error": str(e)
                }
        else:
            print(f"[PLUGIN ERROR] Tool not found: {tool_name}")
            return {
                "success": False,
                "tool_name": tool_name,
                "error": "Tool not found"
            }

plugin_manager = PluginManager()

plugins_dir = Path("plugins")
plugins_dir.mkdir(exist_ok=True)
for plugin in plugins_dir.glob("*.py"):
    print(f"[PLUGIN] Loaded plugin: {plugin.name}")

env = jinja2.Environment()
env.globals['len'] = len  

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    """Extrai chamadas de ferramentas do texto gerado"""
    print(f"[TOOLS] Extracting tool calls from text")
    tool_calls = []
    
    start_tag = "<tools>"
    end_tag = "</tools>"
    start_pos = 0

    while True:
        start_idx = text.find(start_tag, start_pos)
        if start_idx == -1:
            break
        start_idx += len(start_tag)
        end_idx = text.find(end_tag, start_idx)
        if end_idx == -1:
            break
        
        json_str = text[start_idx:end_idx].strip()
        try:
            data = json.loads(json_str)
            tool_calls.append(data)
            print(f"[TOOLS] Parsed tool call: {data.get('name')}")
        except json.JSONDecodeError:
            print(f"[TOOLS ERROR] Invalid JSON in tool call: {json_str}")
        
        start_pos = end_idx + len(end_tag)
    
    print(f"[TOOLS] Found {len(tool_calls)} tool calls")
    return tool_calls

def execute_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Executa chamadas de ferramentas e retorna resultados"""
    print(f"[TOOLS] Executing {len(tool_calls)} tool calls")
    results = []
    
    for tool_call in tool_calls:
        try:
            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})
            
            print(f"[TOOLS] Executing tool: {tool_name}")
            result = plugin_manager.execute_tool(tool_name, arguments)
            results.append(result)
            print(f"[TOOLS] Tool {tool_name} executed successfully")
        except Exception as e:
            print(f"[TOOLS ERROR] Error executing tool call: {e}")
            error_result = {
                "success": False,
                "tool_name": tool_call.get("name", "unknown"),
                "error": str(e)
            }
            results.append(error_result)
    
    return results

def add_tool_results_to_messages(messages: List[Dict[str, Any]], tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Adiciona resultados de ferramentas às mensagens"""
    print(f"[TOOLS] Adding {len(tool_results)} tool results to messages")
    updated_messages = messages.copy()
    
    for result in tool_results:
        tool_message = {
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": f"call_{int(time.time()*1000)}"
        }
        updated_messages.append(tool_message)
        print(f"[TOOLS] Added tool result message: {result.get('tool_name')}")
    
    return updated_messages

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    tools: Optional[List[Dict]] = None
    add_generation_prompt: bool = True
    enable_thinking: bool = True

@app.post("/chat")
async def chat(request: Request):
    global current_generation_task, generation_cancel_event, model_speed_tracker
    
    print("[CHAT] === CHAT REQUEST STARTED ===")
    
    if current_generation_task and not current_generation_task.done():
        generation_cancel_event.set()
        await asyncio.sleep(0.1)
    
    generation_cancel_event.clear()
    current_generation_task = asyncio.current_task()
    
    data = await request.json()
    print(f"[CHAT] Received  {data}")
    
    validated_data = ChatRequest(**data)
    messages = [m.dict() for m in validated_data.messages]
    tools = validated_data.tools
    add_generation_prompt = validated_data.add_generation_prompt
    enable_thinking = validated_data.enable_thinking
    
    if not messages or messages[0].get('role') != 'system':
        messages = [{'role': 'system', 'content': system_prompt}] + messages
        print("[CHAT] Added system prompt to messages")
    
    if not tools:
        tools = plugin_manager.get_available_tools()
        print(f"[CHAT] Using all available tools: {len(tools)}")
    
    template = env.from_string(template_str)
    prompt = template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking
    )
    
    print("[CHAT] Generated prompt:")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    
    model_speed_tracker["start_time"] = time.time()
    model_speed_tracker["tokens_count"] = 0
    
    prompt_tokens = len(llm.tokenize(prompt.encode()))
    max_tokens = min(8126, n_ctx - prompt_tokens - 1)
    if max_tokens <= 0:
        raise HTTPException(status_code=400, detail="Prompt too long for context window.")
    
    output = llm.create_completion(
        prompt,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    
    def generate():
        global current_generation_task, generation_cancel_event, model_speed_tracker
        nonlocal messages
        
        try:
            full_response = ""
            for chunk in output:
                if generation_cancel_event.is_set():
                    print("[CHAT] Generation cancelled")
                    break
                
                text = chunk["choices"][0]["text"]
                full_response += text
                model_speed_tracker["tokens_count"] += len(text.split())
                print(f"[CHAT] Yielding text: {text[:50]}...")
                yield text
            
            print("[CHAT] Checking for tool calls in generated response...")
            tool_calls = extract_tool_calls_from_text(full_response)
            
            if tool_calls:
                print(f"[CHAT] Found {len(tool_calls)} tool calls, executing...")
                tool_results = execute_tool_calls(tool_calls)
                
                messages_with_results = add_tool_results_to_messages(messages + [{"role": "assistant", "content": full_response}], tool_results)
                
                print("[CHAT] Generating response with tool results...")
                new_template = env.from_string(template_str)
                new_prompt = new_template.render(
                    messages=messages_with_results,
                    tools=[],
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                
                print("[CHAT] New prompt with tool results:")
                print(new_prompt[:500] + "..." if len(new_prompt) > 500 else new_prompt)
                
                new_prompt_tokens = len(llm.tokenize(new_prompt.encode()))
                new_max_tokens = min(4096, n_ctx - new_prompt_tokens - 1)
                if new_max_tokens <= 0:
                    print("[CHAT] Second prompt too long, skipping.")
                    return
                
                new_output = llm.create_completion(
                    new_prompt,
                    max_tokens=new_max_tokens,
                    stream=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                
                for chunk in new_output:
                    if generation_cancel_event.is_set():
                        print("[CHAT] Final generation cancelled")
                        break
                    
                    text = chunk["choices"][0]["text"]
                    model_speed_tracker["tokens_count"] += len(text.split())
                    print(f"[CHAT] Yielding final text: {text[:50]}...")
                    yield text
            
        except Exception as e:
            print(f"[CHAT ERROR] Error in generation: {e}")
            yield "❌ Error during generation."
    
    print("[CHAT] === CHAT REQUEST COMPLETED ===")
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/cancel")
async def cancel_generation():
    global generation_cancel_event
    generation_cancel_event.set()
    print("[CANCEL] Generation cancelled")
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
    print(f"[THINKING] Thinking mode set to: {thinking_mode_enabled}")
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
    print("[CONFIG] Configuration updated")
    return {"message": "Configuration updated", "config": current_config}

def load_config():
    global config_path
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return toml.load(f)
        except Exception as e:
            print(f"[CONFIG ERROR] Error reading config.toml: {e}")
            return {}
    return {}

def save_config(config_data):
    global config_path
    try:
        with open(config_path, 'w') as f:
            toml.dump(config_data, f)
        print("[CONFIG] Configuration saved to config.toml")
    except Exception as e:
        print(f"[CONFIG ERROR] Error saving config.toml: {e}")

@app.get("/model/info")
async def get_model_info():
    return {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "system_prompt": system_prompt,
        "chat_template": chat_template_name
    }

@app.post("/model/reload")
async def reload_model():
    global llm, n_ctx, temperature, top_k, top_p, system_prompt, template_str, chat_template_name, plugin_manager
    
    config = load_config()
    model_config = config.get('model', {})
    
    n_ctx = model_config.get('n_ctx', 2048)
    temperature = model_config.get('temperature', 0.7)
    top_k = model_config.get('top_k', 40)
    top_p = model_config.get('top_p', 0.95)
    
    if config and 'system' in config and 'prompt' in config['system']:
        system_prompt = config['system']['prompt']
    
    chat_template_name = None
    if 'chat_template' in model_config:
        chat_template_name = model_config['chat_template']
    else:
        model_filename = Path(model_path).stem.lower()
        if 'qwen3' in model_filename:
            chat_template_name = "qwen3"
        elif 'llama' in model_filename:
            chat_template_name = "llama"
        elif 'mistral' in model_filename:
            chat_template_name = "mistral"
        elif 'gemma' in model_filename:
            chat_template_name = "gemma"
        elif 'deepseek' in model_filename:
            chat_template_name = "deepseek"
        elif 'phi' in model_filename:
            chat_template_name = "phi"
        elif 'yi' in model_filename:
            chat_template_name = "yi"
        else:
            chat_template_name = "default"
    
    template_file = Path("chat_templates") / chat_template_name
    if template_file.exists():
        try:
            with open(template_file, 'r') as f:
                template_str = f.read()
            print(f"[MODEL] Loaded chat template: {chat_template_name}")
        except Exception as e:
            print(f"[MODEL ERROR] Error loading chat template {chat_template_name}: {e}")
    
    llm = llama_cpp.Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        verbose=True,
        n_threads=8
    )
    
    plugin_manager = PluginManager()
    
    print("[MODEL] Model reloaded with new configuration")
    return {"message": "Model reloaded with new configuration"}

@app.get("/tools/list")
async def list_tools():
    tools = plugin_manager.get_available_tools()
    return {
        "tools": tools,
        "count": len(tools)
    }

@app.post("/tools/execute")
async def execute_tool_endpoint(request: Request):
    data = await request.json()
    tool_name = data.get("name")
    arguments = data.get("arguments", {})
    
    if not tool_name:
        raise HTTPException(status_code=400, detail="Tool name is required")
    
    result = plugin_manager.execute_tool(tool_name, arguments)
    return result

if __name__ == "__main__":
    print("[SERVER] Starting ZoeAI server...")
    print(f"[SERVER] Model path: {model_path}")
    print(f"[SERVER] Chat template: {chat_template_name}")
    print(f"[SERVER] Available plugins: {len(plugin_manager.plugins)}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
