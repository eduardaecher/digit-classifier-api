"""Carrega o modelo e faz as previsões. Não depende de nada do HTTP."""

import io
import logging
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.core.exceptions import (ArquivoInvalidoError, InferenciaError, ModeloIndisponivelError,
                                 RegraNegocioError)
from app.ml.preprocessing import LIMIAR_TRACO, N_PIXELS, ImagemSemDigitoError, preparar_imagem
from app.schemas.prediction import PredicaoSaida, ProbabilidadeDigito

logger = logging.getLogger("digit_api")

# Nenhum dígito do MNIST tem mais de 40% dos pixels claros; acima de 50% a imagem
# quase certamente está com as cores invertidas (fundo branco e traço preto).
FRACAO_MAXIMA_PIXELS_CLAROS = 0.5


def validar_regras_negocio(pixels: np.ndarray) -> None:
    """Confere se os pixels representam um dígito no padrão MNIST. Levanta ``RegraNegocioError`` (HTTP 400)."""
    if pixels.max() <= LIMIAR_TRACO:
        raise RegraNegocioError([
            f"A imagem está vazia: nenhum pixel ultrapassa a intensidade {LIMIAR_TRACO}, então não há traço a classificar."
        ])
    fracao_clara = float((pixels > 127).mean())
    if fracao_clara > FRACAO_MAXIMA_PIXELS_CLAROS:
        raise RegraNegocioError([
            f"{fracao_clara:.0%} dos pixels são claros. O modelo espera fundo preto (0) e traço branco (255): "
            "inverta as cores (255 - valor) ou envie o arquivo pelo endpoint /v1/predict/imagem, que inverte automaticamente."
        ])


class DigitModelService:
    """Mantém o pipeline em memória e executa as previsões."""

    def __init__(self, caminho_modelo: Path):
        self.caminho_modelo = Path(caminho_modelo)
        self.pipeline = None
        self.limiar_confianca: float | None = None
        self.metadados: dict = {}

    @property
    def carregado(self) -> bool:
        return self.pipeline is not None

    @property
    def versao(self) -> str | None:
        return self.metadados.get("versao")

    def carregar(self) -> None:
        """Lê o artefato do disco. Chamado UMA única vez, no lifespan da aplicação."""
        inicio = time.perf_counter()
        artefato = joblib.load(self.caminho_modelo)
        self.pipeline = artefato["pipeline"]
        self.limiar_confianca = float(artefato["limiar_confianca"])
        self.metadados = artefato["metadados"]
        logger.info("Modelo %s v%s carregado de %s em %.2fs", self.metadados.get("nome"), self.versao,
                    self.caminho_modelo.name, time.perf_counter() - inicio)

    def descarregar(self) -> None:
        self.pipeline = None
        self.limiar_confianca = None
        self.metadados = {}

    def _garantir_carregado(self) -> None:
        if not self.carregado:
            raise ModeloIndisponivelError("O modelo de Machine Learning não está carregado em memória.")

    def _inferir(self, pixels: np.ndarray) -> PredicaoSaida:
        try:
            probabilidades = self.pipeline.predict_proba(pixels.reshape(1, N_PIXELS).astype(np.float64))[0]
        except Exception as exc:  # qualquer falha do modelo vira 500 controlado, sem derrubar a API
            raise InferenciaError(str(exc)) from exc

        ordem = np.argsort(probabilidades)[::-1]
        digito = int(ordem[0])
        confianca = float(probabilidades[digito])
        return PredicaoSaida(
            digito_previsto=digito,
            confianca=round(confianca, 4),
            previsao_confiavel=confianca >= self.limiar_confianca,
            limiar_confianca=self.limiar_confianca,
            top_3=[ProbabilidadeDigito(digito=int(d), probabilidade=round(float(probabilidades[d]), 4)) for d in ordem[:3]],
            probabilidades=[round(float(p), 4) for p in probabilidades],
            versao_modelo=self.versao,
        )

    def prever_pixels(self, pixels: list[int]) -> PredicaoSaida:
        """Classifica um vetor de 784 pixels já no padrão MNIST."""
        self._garantir_carregado()
        vetor = np.asarray(pixels, dtype=np.uint8)
        validar_regras_negocio(vetor)
        return self._inferir(vetor)

    def prever_imagem(self, conteudo: bytes) -> PredicaoSaida:
        """Classifica um arquivo de imagem (PNG, JPG, BMP...) de qualquer tamanho."""
        self._garantir_carregado()
        if not conteudo:
            raise ArquivoInvalidoError("O arquivo enviado está vazio.")
        try:
            with Image.open(io.BytesIO(conteudo)) as imagem:
                imagem.load()
                vetor = preparar_imagem(imagem)
        except (UnidentifiedImageError, OSError) as exc:
            raise ArquivoInvalidoError("O arquivo enviado não é uma imagem válida (use PNG, JPG ou BMP).") from exc
        except ImagemSemDigitoError as exc:
            raise RegraNegocioError([str(exc)]) from exc
        validar_regras_negocio(vetor)
        return self._inferir(vetor)
