"""Собирает dist/<имя>.zip: только корневые .py из core/ и requirements.txt.

python build_zip.py [имя] [КЛЮЧ=значение ...]: значения подменяют строки "КЛЮЧ = ..." в config.py внутри архива,
например python build_zip.py V6-think 'REASONING_EFFORT="xhigh"'.
"""

import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"
NAME, *OVERRIDES = sys.argv[1:] or ["submission"]
TARGET = ROOT / "dist" / f"{NAME}.zip"


def pins(path):
    return {line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")}


forbidden = pins(CORE / "requirements.txt") - pins(ROOT / "allowed-requirements.txt")
if forbidden:
    raise SystemExit(f"Пакеты вне разрешённого списка платформы: {sorted(forbidden)}")

TARGET.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(TARGET, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in [*sorted(CORE.glob("*.py")), CORE / "requirements.txt"]:
        text = path.read_text(encoding="utf-8")
        if path.name == "config.py":
            for item in OVERRIDES:
                key, value = item.split("=", 1)
                text, replaced = re.subn(rf"(?m)^{key} = .*$", lambda _: f"{key} = {value}", text, count=1)
                if not replaced:
                    raise SystemExit(f"В config.py нет настройки {key}")
        archive.writestr(path.name, text)
print(TARGET, f"{TARGET.stat().st_size} байт")
