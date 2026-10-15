from fastapi import APIRouter

from src.app.schemas.voice import InstructionPayload, InstructionRequest
from src.app.services.groq_service import get_instruction

router = APIRouter(tags=["instruction"])


@router.post("/instruction", response_model=InstructionPayload)
def route_instruction(payload: InstructionRequest) -> InstructionPayload:
    # Solo devuelve el JSON de routing; NO ejecuta la acción.
    return get_instruction(payload.transcription)
