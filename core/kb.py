"""Разбор базы знаний: zip с Markdown -> список фрагментов с крошками."""

import re
import zipfile
from pathlib import PurePosixPath

import config

COLUMNS = re.compile(r"^\s*-+(?: +-+)+\s*$")
RULE = re.compile(r"^\s*-{5,}\s*$")
HEADING = re.compile(r"^(#{1,3}) +(.+?)\s*$")


def _cells(block, spans):
    return [" ".join(" ".join(line[a:b].split()) for line in block).strip() for a, b in spans]


def _format_row(header, cells):
    if header:
        return "- " + "; ".join(f"{h}: {c}" if h else c for h, c in zip(header, cells) if c)
    return "- " + " — ".join(c for c in cells if c)


def normalize_tables(text):
    """Pandoc-таблицы -> строки "- Колонка: значение; ...", чтобы оговорка не отрывалась от числа."""
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        if not COLUMNS.match(lines[i]):
            out.append(lines[i])
            i += 1
            continue

        starts = [m.start() for m in re.finditer(r"-+", lines[i])]
        spans = list(zip(starts, starts[1:] + [10_000]))
        head = []
        while out and out[-1].strip() and not RULE.match(out[-1]):
            head.insert(0, out.pop())
        framed = bool(out) and bool(RULE.match(out[-1]))
        if framed:
            out.pop()
        header = _cells(head, spans) if head else None
        # Без заголовка и рамки таблица всё равно закрывается такой же строкой дефисов.
        multiline = framed or not head

        rows, block = [], []
        i += 1
        while i < len(lines):
            line = lines[i]
            if RULE.match(line) or COLUMNS.match(line):
                i += 1
                break
            if not line.strip():
                if not multiline:
                    break
                if block:
                    rows.append(block)
                    block = []
            elif not line.startswith(" "):
                break
            elif multiline:
                block.append(line)
            else:
                rows.append([line])
            i += 1
        if block:
            rows.append(block)
        out.extend(_format_row(header, _cells(row, spans)) for row in rows)
        out.append("")
    return "\n".join(out)


def _split_long(text, limit):
    parts, current = [], ""
    for piece in re.split(r"\n\s*\n|\n(?=- )", text):
        if current and len(current) + len(piece) + 1 > limit:
            parts.append(current)
            current = ""
        current = f"{current}\n{piece}" if current else piece
    if current.strip():
        parts.append(current)
    return parts


def split_sections(name, text):
    """Режет по H1-H3. Крошка H1 > H2 > H3 различает одинаковые разделы "Чего утверждать нельзя"."""
    path, body, sections = {}, [], []

    def flush():
        content = "\n".join(body).strip()
        body.clear()
        if content:
            crumb = " > ".join(path[level] for level in sorted(path)) or PurePosixPath(name).stem
            sections.append((crumb, content))

    for line in normalize_tables(text).split("\n"):
        match = HEADING.match(line)
        if match:
            flush()
            level = len(match[1])
            path = {k: v for k, v in path.items() if k < level}
            path[level] = match[2]
        elif not re.fullmatch(r"-{5,}", line.strip()):
            body.append(line)
    flush()
    return sections


def load_chunks(zip_path):
    chunks = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in sorted(archive.namelist()):
            if name.startswith("__MACOSX/") or PurePosixPath(name).suffix.lower() not in (".md", ".txt"):
                continue
            text = archive.read(name).decode("utf-8-sig", errors="replace")
            for section, (crumb, content) in enumerate(split_sections(name, text), start=len(chunks)):
                # section: номер раздела, общий для его кусков; растёт по порядку документа.
                for piece in _split_long(content, config.MAX_CHUNK_CHARS):
                    chunks.append({"id": len(chunks), "section": section, "file": name, "crumb": crumb, "text": piece})
    if not chunks:
        raise ValueError("В базе знаний нет файлов .md или .txt")
    return chunks
