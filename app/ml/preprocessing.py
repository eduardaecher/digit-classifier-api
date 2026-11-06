"""Converte uma imagem de dígito para o formato do MNIST (28x28, fundo preto, traço branco).

Baseado no pré-processamento do Desafio C do projeto anterior. É usado pelo
notebook e pela API.
"""

import numpy as np
from PIL import Image
from scipy import ndimage

LADO = 28                 # imagens MNIST têm 28 × 28 pixels
N_PIXELS = LADO * LADO    # 784 atributos após a vetorização
LADO_DIGITO = 20          # no MNIST o dígito ocupa uma caixa de 20 × 20 com margem
LIMIAR_TRACO = 30         # intensidade mínima para considerar um pixel parte do traço
LIMIAR_FUNDO_CLARO = 127  # borda com média acima disso indica papel branco / traço escuro


class ImagemSemDigitoError(ValueError):
    """A imagem não contém nenhum traço identificável."""


def _para_escala_de_cinza(imagem: Image.Image) -> np.ndarray:
    # Transparência é composta sobre fundo branco antes da conversão
    if imagem.mode in ("RGBA", "LA", "PA") or (imagem.mode == "P" and "transparency" in imagem.info):
        rgba = imagem.convert("RGBA")
        fundo = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        imagem = Image.alpha_composite(fundo, rgba) if rgba.getchannel("A").getextrema()[0] < 255 else rgba
    return np.array(imagem.convert("L"), dtype=np.float64)


def fundo_e_claro(cinza: np.ndarray) -> bool:
    """True se a borda da imagem é clara (ex.: foto de papel branco com caneta)."""
    borda = np.concatenate([cinza[0], cinza[-1], cinza[:, 0], cinza[:, -1]])
    return float(borda.mean()) > LIMIAR_FUNDO_CLARO


def preparar_imagem(imagem: Image.Image, inverter_cores: bool | None = None) -> np.ndarray:
    """Devolve a imagem como um vetor de 784 inteiros (0 a 255).

    Passos: escala de cinza, inversão de cores se o fundo for claro, recorte do dígito,
    redimensionamento para 20x20, colagem numa tela 28x28 e centralização pelo centro de massa.
    """
    cinza = _para_escala_de_cinza(imagem)

    if inverter_cores is None:
        inverter_cores = fundo_e_claro(cinza)
    if inverter_cores:
        cinza = 255.0 - cinza

    mascara = cinza > LIMIAR_TRACO
    if not mascara.any():
        raise ImagemSemDigitoError("A imagem não contém nenhum traço identificável.")

    linhas = np.where(mascara.any(axis=1))[0]
    colunas = np.where(mascara.any(axis=0))[0]
    recorte = cinza[linhas[0]:linhas[-1] + 1, colunas[0]:colunas[-1] + 1]

    digito = Image.fromarray(recorte.astype(np.uint8))
    digito.thumbnail((LADO_DIGITO, LADO_DIGITO), Image.LANCZOS)

    tela = Image.new("L", (LADO, LADO), color=0)
    largura, altura = digito.size
    tela.paste(digito, ((LADO - largura) // 2, (LADO - altura) // 2))
    tela = np.array(tela, dtype=np.float64)

    centro_y, centro_x = ndimage.center_of_mass(tela)
    tela = ndimage.shift(tela, (LADO / 2 - centro_y, LADO / 2 - centro_x), order=1, mode="constant", cval=0)

    return np.clip(np.rint(tela), 0, 255).astype(np.uint8).reshape(N_PIXELS)


def preparar_arquivo(caminho) -> np.ndarray:
    """Atalho para abrir um arquivo de imagem do disco e prepará-lo."""
    with Image.open(caminho) as imagem:
        imagem.load()
        return preparar_imagem(imagem)
