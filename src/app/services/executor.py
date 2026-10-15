"""Ejecuta una instrucción de routing contra el almacén de tareas en memoria."""

import re
from typing import Any

from fastapi import HTTPException, status
from pydantic import ValidationError

from src.app.schemas.voice import InstructionPayload, TaskCreate, TaskReplace, TaskUpdate
from src.app.services import task_store

_TASK_ID_PATH = re.compile(r"^/tasks/(\d+)/?$")


def execute_instruction(instruction: InstructionPayload) -> Any:
    endpoint = instruction.endpoint.strip()
    method = instruction.method.upper()
    params = instruction.params

    try:
        if endpoint.rstrip("/") == "/tasks":
            if method == "GET":
                return task_store.list_tasks()
            if method == "POST":
                data = TaskCreate.model_validate(params)
                return task_store.create_task(data.title, data.done)

        match = _TASK_ID_PATH.match(endpoint)
        if match:
            task_id = int(match.group(1))
            if method == "PUT":
                data = TaskReplace.model_validate(params)
                return task_store.replace_task(task_id, data.title, data.done)
            if method == "PATCH":
                data = TaskUpdate.model_validate(params)
                return task_store.update_task(task_id, data.title, data.done)
            if method == "DELETE":
                return task_store.delete_task(task_id)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid params from LLM: {exc.errors()}",
        ) from exc

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unsupported instruction: {method} {endpoint}",
    )
