"""Integración con Groq: routing de intención (LLM) y transcripción (Whisper)."""

import json

from fastapi import HTTPException, status
from groq import Groq
from pydantic import ValidationError

from src.app.core.config import get_settings
from src.app.schemas.voice import InstructionPayload
from src.app.services import task_store

SYSTEM_PROMPT = """You are a strict intent router for a to-do list REST API.
Your ONLY job is to convert a user's voice command (in any language) into ONE JSON object.
Never answer with free text, explanations, markdown or code fences. Output JSON only.

Output format (exact keys, nothing else):
{"endpoint": "<path>", "method": "<HTTP method>", "params": {<arguments>}}

Available endpoints:
1. List all tasks            -> {"endpoint": "/tasks", "method": "GET", "params": {}}
2. Create a task             -> {"endpoint": "/tasks", "method": "POST", "params": {"title": "<title>"}}
3. Replace a whole task      -> {"endpoint": "/tasks/<id>", "method": "PUT", "params": {"title": "<title>", "done": <true|false>}}
4. Partially update a task   -> {"endpoint": "/tasks/<id>", "method": "PATCH", "params": {"title": "<new title>"} and/or {"done": <true|false>}}
5. Delete a task             -> {"endpoint": "/tasks/<id>", "method": "DELETE", "params": {}}

Rules:
- <id> must be the integer id of an existing task. Use the CURRENT TASKS list provided by the user
  message to find the task the user is talking about (match by title, meaning or position).
- Marking a task as done/completed/finished -> PATCH with {"done": true}.
  Marking it as pending/not done -> PATCH with {"done": false}.
- Renaming / changing the text of a task -> PATCH with {"title": "..."}.
- Use PUT only when the user explicitly wants to replace both the title and the status.
- Titles: short, capitalize the first letter, keep the user's language, no trailing punctuation,
  and remove filler like "add ... to my list".
- If the user refers to a task that does not exist, or the request is unclear or unrelated
  to tasks, return the list action: {"endpoint": "/tasks", "method": "GET", "params": {}}.

Examples (assume CURRENT TASKS = [{"id": 1, "title": "Comprar leche", "done": false}, {"id": 2, "title": "Call mom", "done": false}]):
"añade comprar pan a mi lista" -> {"endpoint": "/tasks", "method": "POST", "params": {"title": "Comprar pan"}}
"qué tareas tengo" -> {"endpoint": "/tasks", "method": "GET", "params": {}}
"ya compré la leche" -> {"endpoint": "/tasks/1", "method": "PATCH", "params": {"done": true}}
"cambia comprar leche por comprar leche de almendra" -> {"endpoint": "/tasks/1", "method": "PATCH", "params": {"title": "Comprar leche de almendra"}}
"delete the call mom task" -> {"endpoint": "/tasks/2", "method": "DELETE", "params": {}}
"reemplaza la tarea 2 por pagar la luz, sin completar" -> {"endpoint": "/tasks/2", "method": "PUT", "params": {"title": "Pagar la luz", "done": false}}
"""


def _get_client() -> Groq:
    settings = get_settings()
    return Groq(api_key=settings.groq_api_key, timeout=settings.request_timeout_seconds)


def get_instruction(transcription: str) -> InstructionPayload:
    """Pide al LLM el JSON de routing. No ejecuta ninguna acción."""
    settings = get_settings()
    user_message = (
        f"CURRENT TASKS: {json.dumps(task_store.list_tasks(), ensure_ascii=False)}\n"
        f"USER COMMAND: {transcription}"
    )

    try:
        completion = _get_client().chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            response_format={"type": "json_object"},  # fuerza JSON válido
        )
    except Exception as exc:  # errores de red / API key / rate limit
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Groq request failed: {exc}",
        ) from exc

    content = completion.choices[0].message.content or ""
    try:
        instruction = InstructionPayload.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM returned an invalid routing JSON: {content}",
        ) from exc

    instruction.method = instruction.method.upper()
    return instruction


def transcribe_audio(filename: str, audio: bytes, language: str | None = None) -> str:
    """Convierte audio a texto con Whisper en Groq."""
    settings = get_settings()
    try:
        result = _get_client().audio.transcriptions.create(
            file=(filename, audio),
            model=settings.groq_transcription_model,
            language=language,
            temperature=0,
            response_format="json",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Groq transcription failed: {exc}",
        ) from exc

    text = (result.text or "").strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not understand any speech in the audio.",
        )
    return text
