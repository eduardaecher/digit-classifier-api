"""Testes automatizados da API (execute com: pytest)."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageOps

from app.main import app
from app.schemas.exemplos import EXEMPLO_PIXELS_DIGITO_7

PASTA_IMAGENS = Path(__file__).resolve().parents[1] / "data" / "imagens_proprias"


@pytest.fixture
def client():
    # O "with" dispara o lifespan: o modelo é carregado na inicialização, como em produção
    with TestClient(app) as c:
        yield c


def pixels(**alteracoes):
    lista = list(EXEMPLO_PIXELS_DIGITO_7)
    for indice, valor in alteracoes.items():
        lista[int(indice.removeprefix("p"))] = valor
    return {"pixels": lista}


def enviar_imagem(client, conteudo: bytes, nome="digito.png", tipo="image/png"):
    return client.post("/v1/predict/imagem", files={"arquivo": (nome, conteudo, tipo)})


# ----------------------------------------------------------------- 200 OK
def test_health_responde_200_com_modelo_em_memoria(client):
    resposta = client.get("/health")
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["status"] == "ok"
    assert corpo["servico_ativo"] is True
    assert corpo["modelo_carregado"] is True
    assert corpo["versao_modelo"] == "1.0.0"


def test_predict_pixels_do_digito_7(client):
    resposta = client.post("/v1/predict", json={"pixels": EXEMPLO_PIXELS_DIGITO_7})
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert set(corpo) == {"digito_previsto", "confianca", "previsao_confiavel", "limiar_confianca",
                          "top_3", "probabilidades", "versao_modelo"}
    assert corpo["digito_previsto"] == 7
    assert corpo["previsao_confiavel"] is True
    assert len(corpo["probabilidades"]) == 10
    assert abs(sum(corpo["probabilidades"]) - 1) < 0.01
    assert corpo["top_3"][0]["digito"] == 7


def test_predict_imagem_acerta_ao_menos_9_das_10_imagens_proprias(client):
    acertos = 0
    for digito in range(10):
        conteudo = (PASTA_IMAGENS / f"digito_{digito}.png").read_bytes()
        resposta = enviar_imagem(client, conteudo)
        assert resposta.status_code == 200
        acertos += resposta.json()["digito_previsto"] == digito
    assert acertos >= 9


def test_predict_imagem_inverte_cores_automaticamente(client):
    # Simula uma foto de papel branco com traço escuro, em JPEG
    imagem = ImageOps.invert(Image.open(PASTA_IMAGENS / "digito_4.png").convert("RGB"))
    buffer = io.BytesIO()
    imagem.save(buffer, format="JPEG")
    resposta = enviar_imagem(client, buffer.getvalue(), "papel.jpg", "image/jpeg")
    assert resposta.status_code == 200
    assert resposta.json()["digito_previsto"] == 4


def test_model_info(client):
    resposta = client.get("/v1/model/info")
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["algoritmo"] == "MLPClassifier"
    assert corpo["metricas_teste"]["acuracia"] > 0.97


def test_modelo_nao_e_recarregado_a_cada_requisicao(client, monkeypatch):
    import joblib

    def falhar(*args, **kwargs):
        raise AssertionError("O modelo não deve ser lido do disco durante uma requisição")

    monkeypatch.setattr(joblib, "load", falhar)
    for _ in range(3):
        assert client.post("/v1/predict", json={"pixels": EXEMPLO_PIXELS_DIGITO_7}).status_code == 200


# ----------------------------------------------------------------- 422
@pytest.mark.parametrize("payload, descricao", [
    ({"pixels": EXEMPLO_PIXELS_DIGITO_7[:783]}, "783 pixels"),
    ({"pixels": EXEMPLO_PIXELS_DIGITO_7 + [0]}, "785 pixels"),
    ({"pixels": []}, "lista vazia"),
    (pixels(p300=256), "pixel acima de 255"),
    (pixels(p300=-1), "pixel negativo"),
    (pixels(p300="abc"), "texto no lugar de número"),
    (pixels(p300="12"), "número enviado como texto (modo estrito)"),
    (pixels(p300=12.5), "decimal no lugar de inteiro"),
    (pixels(p300=None), "pixel nulo"),
    ({}, "campo pixels ausente"),
    ({"pixels": "0" * 784}, "texto no lugar da lista"),
    ({**pixels(), "rotulo": 7}, "campo extra"),
])
def test_predict_dados_invalidos_retorna_422(client, payload, descricao):
    resposta = client.post("/v1/predict", json=payload)
    assert resposta.status_code == 422, descricao
    assert resposta.json()["erro"] == "dados_invalidos"


def test_predict_imagem_sem_arquivo_retorna_422(client):
    assert client.post("/v1/predict/imagem").status_code == 422


# ----------------------------------------------------------------- 400
def test_predict_json_malformado_retorna_400(client):
    resposta = client.post("/v1/predict", content=b'{"pixels": [0, 0,', headers={"Content-Type": "application/json"})
    assert resposta.status_code == 400
    assert resposta.json()["erro"] == "json_malformado"


def test_predict_imagem_em_branco_retorna_400(client):
    resposta = client.post("/v1/predict", json={"pixels": [0] * 784})
    assert resposta.status_code == 400
    assert resposta.json()["erro"] == "imagem_inadequada"


def test_predict_cores_invertidas_retorna_400(client):
    resposta = client.post("/v1/predict", json={"pixels": [255 - p for p in EXEMPLO_PIXELS_DIGITO_7]})
    assert resposta.status_code == 400
    assert resposta.json()["erro"] == "imagem_inadequada"


def test_predict_imagem_arquivo_que_nao_e_imagem_retorna_400(client):
    resposta = enviar_imagem(client, b"isto nao e uma imagem", "dados.txt", "text/plain")
    assert resposta.status_code == 400
    assert resposta.json()["erro"] == "arquivo_invalido"


def test_predict_imagem_arquivo_vazio_retorna_400(client):
    assert enviar_imagem(client, b"").status_code == 400


def test_predict_imagem_sem_digito_retorna_400(client):
    buffer = io.BytesIO()
    Image.new("L", (100, 100), color=0).save(buffer, format="PNG")
    resposta = enviar_imagem(client, buffer.getvalue())
    assert resposta.status_code == 400
    assert resposta.json()["erro"] == "imagem_inadequada"


# ----------------------------------------------------------------- 413 / 503 / 500
def test_predict_imagem_acima_de_2mb_retorna_413(client):
    resposta = enviar_imagem(client, b"0" * (2 * 1024 * 1024 + 1))
    assert resposta.status_code == 413


def test_api_sem_modelo_fica_degradada_sem_cair(monkeypatch, tmp_path):
    from app.core.config import Settings

    monkeypatch.setattr("app.main.settings", Settings(caminho_modelo=tmp_path / "inexistente.joblib"))
    with TestClient(app) as c:
        saude = c.get("/health")
        assert saude.status_code == 200
        assert saude.json()["modelo_carregado"] is False
        assert saude.json()["status"] == "degradado"
        assert c.post("/v1/predict", json={"pixels": EXEMPLO_PIXELS_DIGITO_7}).status_code == 503


def test_falha_na_inferencia_retorna_500(client, monkeypatch):
    class PipelineQuebrado:
        def predict_proba(self, X):
            raise RuntimeError("falha simulada")

    monkeypatch.setattr(app.state.servico_modelo, "pipeline", PipelineQuebrado())
    resposta = client.post("/v1/predict", json={"pixels": EXEMPLO_PIXELS_DIGITO_7})
    assert resposta.status_code == 500
    assert resposta.json()["erro"] == "erro_inferencia"
