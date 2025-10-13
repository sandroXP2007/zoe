# ZoeAI

## AI-Powered Chat Application



### Table of Contents
- [Description](#description)
- [Future Plans](#future-plans)
- [Dependencies](#dependencies)
- [Installation Instructions](#installation-instructions)
- [Usage Modes](#usage-modes)
  - [Web Dashboard](#web-dashboard)
  - [CLI for Testing](#cli-for-testing)
- [API Endpoints](#api-endpoints)
- [FAQ](#faq)

---

## Description

ZoeAI is a **modern and user-friendly local AI implementation** designed to provide a seamless interface for interacting with local AI models. 
Divided between the **API** and the **web dashboard** as frontend, it currently has the following capabilities:


- **Thinking mode** for step-by-step reasoning
- **Markdown support** for rich content formatting
- **Conversation history** support
- **Customizable settings**
- **Context menu** for conversation management

The interface prioritizes **user privacy** by keeping all data local and **ethical AI practices** through transparent thinking processes.

---

## Future Plans

- [ ]  **Plugin Support** for integrations and better feature support

---

## Dependencies

### Required
- **CPU** with ***AVX2*** instruction support
- 8GB **RAM** (I recommend 16GB or more for larger models)
- Nvidia **GPU** with a minimum of ***8GB of VRAM*** to run larger models *(optional)*

- **Web browser**
- **Python**

> The application is both **client-side** and **server-side**.

---

## Installation Instructions

### 1. Set Up Backend
After cloning this repository, we must create a **virtual environment** *(optional, but recommended)* and **install the dependencies**

```bash
git clone https://github.com/sandroXP2007/zoe

cd zoe/backend
mkdir models plugins
./download_model

python -m venv venv (recommended)
source venv/bin/activate

pip install -r requirements.txt
```
Once the environment is configured, you can **download the LLM models** in the models/ folder or use the **pre-configured model** (Qwen3-4B quantized *q4_k_s*)

After the installation process is finished, we can **start the API server**

```bash
python server.py
```


### 2. Run the Web Interface 
Option A: **Direct browser access**  (for local-only)
- Open `frontend/index.html` in your browser

Option B: **HTTP server** (to share on the local network or on the web)
- Enter the ``frontend/`` folder and start the server
```bash
sudo python -m http.server 80
```

Then, just open the browser on [localhost](http://localhost).


### 3. Configure Settings
1. Open the web interface
2. Click **Settings**
3. Set your model path and parameters
4. Click **Save**

---

## Usage Modes

### Web Dashboard
1. **Start a new conversation** with the "New Chat" button
2. **Type messages** in the input field
3. **Toggle thinking mode** with the brain icon
4. **Manage conversations** via right-click menu (rename/delete)
5. **Monitor performance** with the token speed indicator

### CLI for Testing
We also have a CLI client for testing purposes, and you can use it by going to the ``backend/`` folder and starting the client.

```bash
source venv/bin/activate (if you have set up a virtual environment)

python client.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Process chat messages with streaming |
| `/speed` | GET | Get current token generation speed |
| `/thinking` | POST | Toggle thinking mode |
| `/thinking` | GET | Check thinking mode status |
| `/config` | GET | Retrieve current configuration |
| `/config` | POST | Update model configuration |
| `/model/reload` | POST | Reload the AI model |
| `/cancel` | POST | Stop current generation |

### Sample Configuration
```json
{
  "model": {
    "path": "models/qwen3-4b-q4_k_s.gguf",
    "n_ctx": 2048,
    "temperature": 0.7
  },
  "system": {
    "prompt": "You are Zoe, a helpful AI assistant."
  }
}
```

---

## FAQ

### How do I change the AI model?
1. Open **Settings** (cog icon)
2. Enter the new model path (e.g., `models/mistral-7b.gguf`)
3. Click **Save**
4. Click **Reload** to apply changes

### Why is my response taking too long?
- Check **token speed** indicator
- Ensure your local server has adequate resources
- Verify model compatibility with your hardware
- Reduce `n_ctx` value in settings for faster responses

### How can I export conversations?
Currently, conversations are stored **locally in browser storage**. 

If you want to export your conversations, you can do so by following these steps:
1. Go to **Settings** → **Export Data**
2. Copy the JSON data
3. Save to a file


---

Thanks to other promising projects such as [Qwen](https://qwen.ai/home) (by our project's default model, Qwen3-4B), [llama.cpp](https://github.com/ggml-org/llama.cpp) and [llama-cpp-python](https://github.com/abetlen/llama-cpp-python).
