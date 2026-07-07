# Reference project for agent development workshop 2026

## Setup

1. install the LangGraph cli

```bash
# Python >= 3.11 is required.
pip install --upgrade "langgraph-cli[inmem]"
```

2. get openai api key at: https://platform.openai.com/settings/organization/api-keys
3. set up environment variables in `.env`

```bash
OPENAI_API_KEY=<your_openai_api_key>
```
4. lunch the langgraph server: `langgraph dev`


