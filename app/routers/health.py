"""Rota de monitoramento de saúde do serviço."""

import time
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Request, status

from app.core.config import settings
from app.dependencies import obter_servico_modelo
from app.schemas.health import HealthResposta
from app.services.model_service import DigitModelService

router = APIRouter(tags=["Monitoramento"])


@router.get(
    "/health",
    response_model=HealthResposta,
    status_code=status.HTTP_200_OK,
    summary="Verifica se o serviço está ativo e se o modelo está em memória",
)
async def health(
    request: Request,
    servico: Annotated[DigitModelService, Depends(obter_servico_modelo)],
) -> HealthResposta:
    """Mostra se a API está no ar e se o modelo está carregado."""
    return HealthResposta(
        status="ok" if servico.carregado else "degradado",
        servico_ativo=True,
        modelo_carregado=servico.carregado,
        versao_modelo=servico.versao,
        versao_api=settings.app_versao,
        uptime_segundos=round(time.time() - request.app.state.iniciado_em, 2),
        timestamp=datetime.now(timezone.utc),
    )
