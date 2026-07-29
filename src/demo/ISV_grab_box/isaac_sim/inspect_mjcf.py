#!/usr/bin/env python3
"""Validate the S63 lab MJCF and all local files it references."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_scene = (
        script_dir.parent
        / "biped_s63"
        / "xml"
        / "scenes"
        / "lab_sence_static_with_box.xml"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf", nargs="?", type=Path, default=default_scene)
    return parser.parse_args()


def collect_documents(entry: Path) -> tuple[list[tuple[Path, ET.Element]], list[Path]]:
    documents: list[tuple[Path, ET.Element]] = []
    missing: list[Path] = []
    visited: set[Path] = set()

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in visited:
            return
        visited.add(path)
        if not path.is_file():
            missing.append(path)
            return

        root = ET.parse(path).getroot()
        documents.append((path, root))
        for include in root.iter("include"):
            include_file = include.get("file")
            if include_file:
                visit(path.parent / include_file)

    visit(entry)
    return documents, missing


def referenced_assets(path: Path, root: ET.Element) -> list[Path]:
    compiler = root.find("compiler")
    mesh_dir = Path(compiler.get("meshdir", ".")) if compiler is not None else Path(".")
    texture_dir = (
        Path(compiler.get("texturedir", "."))
        if compiler is not None
        else Path(".")
    )
    result: list[Path] = []
    for element in root.iter():
        file_name = element.get("file")
        if not file_name or element.tag == "include":
            continue
        if element.tag == "mesh":
            base = path.parent / mesh_dir
        elif element.tag == "texture":
            base = path.parent / texture_dir
        else:
            base = path.parent
        result.append((base / file_name).resolve())
    return result


def main() -> int:
    entry = parse_args().mjcf.resolve()
    documents, missing = collect_documents(entry)
    if not documents:
        print(f"[ERROR] MJCF does not exist: {entry}", file=sys.stderr)
        return 2

    counts: Counter[str] = Counter()
    assets: list[Path] = []
    body_names: set[str] = set()
    for path, root in documents:
        for element in root.iter():
            counts[element.tag] += 1
            if element.tag == "body" and element.get("name"):
                body_names.add(element.get("name", ""))
        assets.extend(referenced_assets(path, root))

    missing.extend(asset for asset in assets if not asset.is_file())
    duplicate_assets = len(assets) - len(set(assets))
    target_ok = "lab_target_box" in body_names

    print(f"MJCF entry       : {entry}")
    print(f"XML documents    : {len(documents)}")
    print(f"Bodies / geoms   : {counts['body']} / {counts['geom']}")
    print(f"Joints / free    : {counts['joint']} / {counts['freejoint']}")
    print(f"Cameras / lights : {counts['camera']} / {counts['light']}")
    print(f"Textures/materials: {counts['texture']} / {counts['material']}")
    print(
        f"Referenced assets: {len(assets)} "
        f"({duplicate_assets} repeated references)"
    )
    print(f"Target box found : {'yes' if target_ok else 'NO'}")

    if missing:
        print("[ERROR] Missing referenced files:", file=sys.stderr)
        for path in sorted(set(missing)):
            print(f"  - {path}", file=sys.stderr)
        return 1
    if not target_ok:
        print("[ERROR] lab_target_box was not found.", file=sys.stderr)
        return 1

    print("Validation       : OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

