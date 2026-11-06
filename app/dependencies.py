"""Dependências injetadas nas rotas (FastAPI ``Depends``)."""

from fastapi import Request

from app.core.exceptions import ModeloIndisponivelError
from app.services.model_service import DigitModelService


def obter_servico_modelo(request: Request) -> DigitModelService:
    """Entrega às rotas a instância do serviço criada no lifespan (o modelo já está em RAM)."""
    return request.app.state.servico_modelo


def obter_servico_modelo_carregado(request: Request) -> DigitModelService:
    """Como ``obter_servico_modelo``, mas exige que o modelo esteja carregado (senão HTTP 503)."""
    servico = obter_servico_modelo(request)
    if not servico.carregado:
        raise ModeloIndisponivelError("O modelo de Machine Learning não está carregado em memória.")
    return servico
