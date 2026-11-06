"""Erros da aplicação e o status HTTP de cada um.

Todas as respostas de erro usam o mesmo formato: erro, mensagem e detalhes.
"""

import logging

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("digit_api")


class RegraNegocioError(Exception):
    """Os dados passaram na validação de formato, mas não representam um dígito utilizável."""

    def __init__(self, mensagens: list[str]):
        self.mensagens = mensagens
        super().__init__("; ".join(mensagens))


class ArquivoInvalidoError(Exception):
    """O arquivo enviado está vazio ou não é uma imagem que o Pillow consiga abrir."""


class ArquivoMuitoGrandeError(Exception):
    """O arquivo enviado excede o tamanho máximo permitido."""


class ModeloIndisponivelError(Exception):
    """O artefato do modelo não está carregado em memória."""


class InferenciaError(Exception):
    """O modelo falhou ao gerar a previsão."""


def _resumir_valor(valor):
    # Evita devolver listas de 784 números inteiras na mensagem de erro
    if isinstance(valor, (list, tuple)) and len(valor) > 10:
        return f"<lista com {len(valor)} itens>"
    if isinstance(valor, dict):
        return "<objeto JSON>"
    return valor


def _erro(status_code: int, erro: str, mensagem: str, detalhes=None) -> JSONResponse:
    corpo = {"erro": erro, "mensagem": mensagem}
    if detalhes is not None:
        corpo["detalhes"] = detalhes
    return JSONResponse(status_code=status_code, content=jsonable_encoder(corpo))


def registrar_handlers(app: FastAPI) -> None:
    """Registra no app os tratadores de exceção da API."""

    @app.exception_handler(RequestValidationError)
    async def validacao_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        erros = exc.errors()
        if any(e.get("type") == "json_invalid" for e in erros):
            return _erro(status.HTTP_400_BAD_REQUEST, "json_malformado",
                         "O corpo da requisição não é um JSON válido.")
        detalhes = [
            {"campo": ".".join(str(p) for p in e.get("loc", []) if p != "body"),
             "mensagem": e.get("msg"), "valor_recebido": _resumir_valor(e.get("input")), "tipo": e.get("type")}
            for e in erros[:20]
        ]
        return _erro(status.HTTP_422_UNPROCESSABLE_CONTENT, "dados_invalidos",
                     "Um ou mais campos não atendem ao contrato de entrada.", detalhes)

    @app.exception_handler(RegraNegocioError)
    async def regra_negocio_handler(request: Request, exc: RegraNegocioError) -> JSONResponse:
        return _erro(status.HTTP_400_BAD_REQUEST, "imagem_inadequada",
                     "Os pixels enviados não representam um dígito no padrão esperado.", exc.mensagens)

    @app.exception_handler(ArquivoInvalidoError)
    async def arquivo_invalido_handler(request: Request, exc: ArquivoInvalidoError) -> JSONResponse:
        return _erro(status.HTTP_400_BAD_REQUEST, "arquivo_invalido", str(exc))

    @app.exception_handler(ArquivoMuitoGrandeError)
    async def arquivo_grande_handler(request: Request, exc: ArquivoMuitoGrandeError) -> JSONResponse:
        return _erro(status.HTTP_413_CONTENT_TOO_LARGE, "arquivo_muito_grande", str(exc))

    @app.exception_handler(ModeloIndisponivelError)
    async def modelo_indisponivel_handler(request: Request, exc: ModeloIndisponivelError) -> JSONResponse:
        return _erro(status.HTTP_503_SERVICE_UNAVAILABLE, "modelo_indisponivel", str(exc))

    @app.exception_handler(InferenciaError)
    async def inferencia_handler(request: Request, exc: InferenciaError) -> JSONResponse:
        logger.exception("Falha na inferência")
        return _erro(status.HTTP_500_INTERNAL_SERVER_ERROR, "erro_inferencia",
                     "Não foi possível gerar a previsão. Tente novamente mais tarde.")

    @app.exception_handler(Exception)
    async def erro_inesperado_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Erro inesperado")
        return _erro(status.HTTP_500_INTERNAL_SERVER_ERROR, "erro_interno",
                     "Ocorreu um erro interno inesperado.")
