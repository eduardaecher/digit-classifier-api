"""Ponto de entrada da API.

Para rodar, a partir da raiz do projeto: uvicorn app.main:app --reload
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.core.exceptions import registrar_handlers
from app.routers import health, prediction
from app.services.model_service import DigitModelService

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("digit_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo uma única vez quando a API inicia.

    Se o arquivo não carregar, a API sobe mesmo assim: /health mostra o problema
    e as rotas de predição respondem 503.
    """
    app.state.iniciado_em = time.time()
    servico = DigitModelService(settings.caminho_modelo)
    try:
        servico.carregar()
    except Exception:
        logger.exception("Não foi possível carregar o modelo em %s", settings.caminho_modelo)
    app.state.servico_modelo = servico

    yield  # --- a aplicação atende requisições aqui ---

    servico.descarregar()
    logger.info("Modelo liberado da memória. Encerrando a aplicação.")


app = FastAPI(
    title=settings.app_nome,
    version=settings.app_versao,
    description=settings.app_descricao,
    lifespan=lifespan,
)

registrar_handlers(app)
app.include_router(health.router)
app.include_router(prediction.router)


@app.get("/", include_in_schema=False)
async def raiz() -> RedirectResponse:
    return RedirectResponse(url="/docs")
