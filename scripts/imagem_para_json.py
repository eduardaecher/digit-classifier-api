"""Converte uma imagem de dígito no JSON aceito por POST /v1/predict.

Uso (a partir da raiz do projeto):

    python scripts/imagem_para_json.py data/imagens_proprias/digito_3.png > payload.json

O arquivo gerado pode ser colado no Swagger ou enviado com:

    curl -X POST http://127.0.0.1:8000/v1/predict -H "Content-Type: application/json" -d @payload.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.preprocessing import preparar_arquivo  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    pixels = preparar_arquivo(sys.argv[1]).tolist()
    print(json.dumps({"pixels": pixels}))


if __name__ == "__main__":
    main()
