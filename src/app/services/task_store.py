"""Almacenamiento en memoria de tareas (sin base de datos)."""

from fastapi import HTTPException, status

# Almacén de datos a nivel de módulo. Se reinicia al reiniciar el servidor.
tasks: list[dict] = []
_next_id = 1


def _get_next_id() -> int:
    global _next_id
    task_id = _next_id
    _next_id += 1  # Los IDs nunca se reutilizan, aunque se borren tareas.
    return task_id


def _find_index(task_id: int) -> int:
    for index, task in enumerate(tasks):
        if task["id"] == task_id:
            return index
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Task with id {task_id} not found",
    )


def list_tasks() -> list[dict]:
    return tasks


def create_task(title: str, done: bool = False) -> dict:
    task = {"id": _get_next_id(), "title": title, "done": done}
    tasks.append(task)
    return task


def replace_task(task_id: int, title: str, done: bool) -> dict:
    index = _find_index(task_id)
    tasks[index] = {"id": task_id, "title": title, "done": done}
    return tasks[index]


def update_task(task_id: int, title: str | None = None, done: bool | None = None) -> dict:
    task = tasks[_find_index(task_id)]
    if title is not None:
        task["title"] = title
    if done is not None:
        task["done"] = done
    return task


def delete_task(task_id: int) -> dict:
    task = tasks.pop(_find_index(task_id))
    return {"message": f"Task {task_id} deleted", "task": task}
