from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from src.app.schemas.voice import InstructionRequest, TranscribeFlowResponse
from src.app.services.executor import execute_instruction
from src.app.services.groq_service import get_instruction, transcribe_audio
from src.app.utils.language import normalize_transcription_language

router = APIRouter(tags=["transcribe"])


@router.get("/")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/transcribe", response_model=TranscribeFlowResponse)
async def transcribe_and_run_flow(request: Request) -> TranscribeFlowResponse:
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("application/json"):
        # Fallback manual: el frontend envía {"transcription": "..."}
        try:
            body = InstructionRequest.model_validate(await request.json())
        except (ValueError, ValidationError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Body must be JSON with a non-empty 'transcription' field.",
            ) from exc
        transcription = body.transcription.strip()
    else:
        form = await request.form()
        file = form.get("file")
        if not isinstance(file, UploadFile):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Send an audio file in the 'file' form field.",
            )
        audio = await file.read()
        if not audio:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The audio file is empty.",
            )
        language = normalize_transcription_language(form.get("language"))
        transcription = await run_in_threadpool(
            transcribe_audio, file.filename or "command.webm", audio, language
        )

    # Misma lógica que POST /instruction, y luego se ejecuta la acción.
    instruction = await run_in_threadpool(get_instruction, transcription)
    result = execute_instruction(instruction)

    return TranscribeFlowResponse(
        transcription=transcription,
        instruction=instruction,
        result=result,
    )
