"""Formato dos dados de entrada e saída das rotas de predição.

Dados fora do formato são barrados pelo Pydantic com erro 422.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from app.ml.preprocessing import N_PIXELS
from app.schemas.exemplos import EXEMPLO_PIXELS_DIGITO_7

Pixel = Annotated[int, Field(strict=True, ge=0, le=255, description="Intensidade do pixel (0 = preto, 255 = branco).")]


class PixelsEntrada(BaseModel):
    """Imagem 28 × 28 em escala de cinza, vetorizada linha a linha (padrão MNIST: fundo preto, traço branco)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{"pixels": EXEMPLO_PIXELS_DIGITO_7}]})

    pixels: list[Pixel] = Field(
        ...,
        min_length=N_PIXELS,
        max_length=N_PIXELS,
        description=f"Exatamente {N_PIXELS} inteiros de 0 a 255 (28 linhas × 28 colunas).",
    )


class ProbabilidadeDigito(BaseModel):
    digito: int = Field(..., ge=0, le=9, description="Dígito candidato.")
    probabilidade: float = Field(..., ge=0, le=1, description="Probabilidade estimada pelo modelo.")


class PredicaoSaida(BaseModel):
    """Resultado da classificação de um dígito."""

    digito_previsto: int = Field(..., ge=0, le=9, description="Dígito com maior probabilidade.")
    confianca: float = Field(..., ge=0, le=1, description="Probabilidade do dígito previsto.")
    previsao_confiavel: bool = Field(
        ..., description="False quando a confiança fica abaixo do limiar: a previsão deve ser revisada por uma pessoa."
    )
    limiar_confianca: float = Field(..., ge=0, le=1, description="Confiança mínima para considerar a previsão confiável.")
    top_3: list[ProbabilidadeDigito] = Field(..., min_length=3, max_length=3,
                                             description="Os três dígitos mais prováveis, em ordem decrescente.")
    probabilidades: list[float] = Field(..., min_length=10, max_length=10,
                                        description="Probabilidade de cada dígito, do 0 ao 9.")
    versao_modelo: str = Field(..., min_length=1, max_length=20, description="Versão do artefato que gerou a previsão.")

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "digito_previsto": 7, "confianca": 0.9998, "previsao_confiavel": True, "limiar_confianca": 0.9,
        "top_3": [{"digito": 7, "probabilidade": 0.9998}, {"digito": 2, "probabilidade": 0.0001},
                  {"digito": 3, "probabilidade": 0.0001}],
        "probabilidades": [0.0, 0.0, 0.0001, 0.0001, 0.0, 0.0, 0.0, 0.9998, 0.0, 0.0],
        "versao_modelo": "1.0.0"}]})


class MetricasModelo(BaseModel):
    acuracia: float = Field(..., ge=0, le=1)
    precisao_macro: float = Field(..., ge=0, le=1)
    recall_macro: float = Field(..., ge=0, le=1)
    f1_macro: float = Field(..., ge=0, le=1)
    f1_ponderado: float = Field(..., ge=0, le=1)
    roc_auc_ovr: float = Field(..., ge=0, le=1)


class ModeloInfo(BaseModel):
    """Metadados do modelo em produção."""

    nome: str
    versao: str
    algoritmo: str
    arquitetura: str
    limiar_confianca: float = Field(..., ge=0, le=1)
    treinado_em: str
    scikit_learn: str
    metricas_teste: MetricasModelo
