from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "reports" / "figures"
FIGURES_EDA = FIGURES / "eda"
FIGURES_MODEL = FIGURES / "modelagem"
MODELS = ROOT / "models"

for _dir in (FIGURES_EDA, FIGURES_MODEL):
    _dir.mkdir(parents=True, exist_ok=True)
