"""Configurações da API. Os caminhos são relativos à raiz do projeto."""

import os
from dataclasses import dataclass, field
from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    app_nome: str = "Digit Classifier API"
    app_versao: str = "1.0.0"
    app_descricao: str = "Classifica dígitos escritos à mão (0 a 9) a partir dos pixels ou de um arquivo de imagem."
    caminho_modelo: Path = field(
        default_factory=lambda: Path(os.getenv("DIGIT_MODEL_PATH", RAIZ_PROJETO / "models" / "digit_mlp_pipeline.joblib"))
    )
    tamanho_maximo_upload: int = 2 * 1024 * 1024  # 2 MB


settings = Settings()
