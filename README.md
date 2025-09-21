## Project Overview

This repository contains multiple FastAPI applications built around sentiment analysis and conversational AI.  
Each file serves a different purpose as described below:

### 1. `main.py`
- Original API of the trained sentiment analysis models.  
- Provides endpoints to analyze text sentiment using custom-trained models.  

### 2. `main_open_ai.py`
- API that integrates **GPT-4o-mini** with system prompts.  
- Designed for conversational AI tasks where system instructions guide responses.  

### 3. `main_with_validation_layer.py`
- Hybrid API combining:
  - The original trained sentiment analysis models.  
  - An additional **LLM validation layer** that converts traditional *therapist/psychologist/psychiatrist* terms into equivalent **Mind Science** terminology.  

---

## Environment

- **Python Version**: `3.10.11`  

RUN this to run the trained model api
```bash
  uvicorn main:app --reload
```

RUN this to run the openai system prompted api
```bash
  uvicorn main_open_ai:app --reload
```
