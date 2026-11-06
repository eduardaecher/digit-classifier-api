"""Rotas de negócio: inferência e informações do modelo (versão 1)."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.core.config import settings
from app.core.exceptions import ArquivoMuitoGrandeError
from app.dependencies import obter_servico_modelo_carregado
from app.schemas.prediction import ModeloInfo, PixelsEntrada, PredicaoSaida
from app.services.model_service import DigitModelService

router = APIRouter(prefix="/v1", tags=["Classificação de dígitos"])

Servico = Annotated[DigitModelService, Depends(obter_servico_modelo_carregado)]

RESPOSTAS_ERRO = {
    400: {"description": "JSON malformado, imagem vazia, cores invertidas ou arquivo que não é imagem."},
    422: {"description": "Campo ausente/extra, tamanho diferente de 784, pixel fora de 0–255 ou tipo inválido."},
    500: {"description": "Falha inesperada ao gerar a previsão."},
    503: {"description": "Modelo não carregado em memória."},
}


@router.post(
    "/predict",
    response_model=PredicaoSaida,
    status_code=status.HTTP_200_OK,
    summary="Classifica um dígito a partir de 784 pixels (JSON)",
    responses=RESPOSTAS_ERRO,
)
def prever_digito(entrada: PixelsEntrada, servico: Servico) -> PredicaoSaida:
    """Recebe os 784 pixels da imagem e devolve o dígito previsto."""
    return servico.prever_pixels(entrada.pixels)


@router.post(
    "/predict/imagem",
    response_model=PredicaoSaida,
    status_code=status.HTTP_200_OK,
    summary="Classifica um dígito a partir de um arquivo de imagem",
    responses={**RESPOSTAS_ERRO, 413: {"description": "Arquivo maior que 2 MB."}},
)
def prever_digito_imagem(
    servico: Servico,
    arquivo: Annotated[UploadFile, File(description="Imagem PNG, JPG ou BMP com um único dígito.")],
) -> PredicaoSaida:
    """Recebe um arquivo de imagem, converte para o formato do MNIST e devolve o dígito previsto."""
    conteudo = arquivo.file.read(settings.tamanho_maximo_upload + 1)
    if len(conteudo) > settings.tamanho_maximo_upload:
        raise ArquivoMuitoGrandeError(
            f"O arquivo excede o limite de {settings.tamanho_maximo_upload // (1024 * 1024)} MB."
        )
    return servico.prever_imagem(conteudo)


@router.get(
    "/model/info",
    response_model=ModeloInfo,
    status_code=status.HTTP_200_OK,
    summary="Mostra a versão, a arquitetura e as métricas do modelo em produção",
    responses={503: RESPOSTAS_ERRO[503]},
)
async def info_modelo(servico: Servico) -> ModeloInfo:
    m = servico.metadados
    return ModeloInfo(
        nome=m["nome"], versao=m["versao"], algoritmo=m["algoritmo"], arquitetura=m["arquitetura"],
        limiar_confianca=m["limiar_confianca"], treinado_em=m["treinado_em"], scikit_learn=m["scikit_learn"],
        metricas_teste=m["metricas_teste"],
    )
