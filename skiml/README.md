# Gemini ReAct Agent

A ReAct (Reasoning and Acting) agent implementation using Google's Gemini API and Wikipedia for question answering.

## Structure

The codebase is organized into the following modules:

- **`config.py`**: Configuration and API setup, including retry policies and API key management
- **`prompts.py`**: ReAct prompt templates and few-shot examples
- **`tools.py`**: Wikipedia search and lookup tool implementations
- **`react_agent.py`**: Main ReAct agent class that orchestrates reasoning and acting
- **`main.py`**: Command-line entry point for running the agent

## Installation

```bash
# Install required dependencies
pip install google-genai wikipedia
```

## Configuration

Set your Google API key as an environment variable:

```bash
export GOOGLE_API_KEY='your-api-key-here'
```

Alternatively, if using Kaggle, configure it in Kaggle secrets.

## Usage

### Command Line

Run with default question:
```bash
python main.py
```

Ask a custom question:
```bash
python main.py "What is the capital of France?"
```

With custom parameters:
```bash
python main.py "Your question here" \
    --model gemini-2.0-flash-exp \
    --temperature 0.2 \
    --max-calls 8
```