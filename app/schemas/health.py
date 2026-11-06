"""Contrato de resposta do endpoint de monitoramento."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HealthResposta(BaseModel):
    status: Literal["ok", "degradado"] = Field(
        ..., description="'ok' quando o serviço está ativo e o modelo está em memória; 'degradado' caso contrário."
    )
    servico_ativo: bool = Field(..., description="Indica que o servidor HTTP está respondendo.")
    modelo_carregado: bool = Field(..., description="Indica se o modelo de ML reside na memória RAM.")
    versao_modelo: str | None = Field(None, description="Versão do modelo carregado, quando houver.")
    versao_api: str = Field(..., description="Versão da API.")
    uptime_segundos: float = Field(..., ge=0, description="Tempo desde a inicialização do serviço.")
    timestamp: datetime = Field(..., description="Momento da checagem (UTC).")
