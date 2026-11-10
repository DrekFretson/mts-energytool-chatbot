"""Собирает dist/submission.zip: только корневые .py из core/ и requirements.txt."""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"
TARGET = ROOT / "dist" / "submission.zip"


def pins(path):
    return {line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")}


forbidden = pins(CORE / "requirements.txt") - pins(ROOT / "allowed-requirements.txt")
if forbidden:
    raise SystemExit(f"Пакеты вне разрешённого списка платформы: {sorted(forbidden)}")

TARGET.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(TARGET, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in [*sorted(CORE.glob("*.py")), CORE / "requirements.txt"]:
        archive.write(path, path.name)
print(TARGET, f"{TARGET.stat().st_size} байт")
