# Klike Data Science Challenge

> 🚧 Em construção: EDA, modelagem do `klike_score` e um recommendations engine para criativos em vídeo.

## Como rodar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Estrutura

```
data/        dataset original
docs/        log de decisões
notebooks/   01 EDA · 02 modelagem · 03 demo do engine
src/klike/   código reutilizável (limpeza, features, modelo, recomendações)
reports/     figuras
tests/       testes da engine
```
