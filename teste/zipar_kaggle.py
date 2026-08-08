#!/usr/bin/env python3
"""
zipar_kaggle.py — Empacota o projeto + datasets para upload no Kaggle.

Uso:
    python zipar_kaggle.py
    python zipar_kaggle.py --output projeto.zip
    python zipar_kaggle.py --include-gpt2       # ~500 MB extra

Saída: autotimes_kaggle.zip (ou nome customizado) na pasta atual.
Assume que o dataset está em ../dataset/ (raiz do AutoTimes).
"""

import argparse
import zipfile
from pathlib import Path

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".venv", "env", "venv", ".mypy_cache", ".pytest_cache",
    "checkpoints", "logs", "graficos", "test_results",
    ".ipynb_checkpoints", ".vscode", ".idea",
}

EXCLUDE_FILES = {
    "benchmark_results.csv",
    ".DS_Store",
}

EXCLUDE_SUFFIXES = {
    ".zip", ".rar", ".7z",
    ".pth", ".pt", ".ckpt", ".bin",
    ".pyc", ".pyo",
    ".npy", ".npz",
    ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".pdf", ".pptx", ".docx", ".xlsx",
}

INCLUDE_SUFFIXES = {
    ".py", ".sh", ".ps1", ".md", ".txt", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".json", ".gitignore",
}


def _add_file(zipf, path, archive_name, stats):
    if path.name in EXCLUDE_FILES:
        stats["skipped_files"] += 1
        return
    if path.suffix.lower() in EXCLUDE_SUFFIXES and path.suffix.lower() not in INCLUDE_SUFFIXES:
        stats["skipped_files"] += 1
        return
    zipf.write(path, archive_name)
    stats["file_count"] += 1
    stats["total_size"] += path.stat().st_size
    print(f"  + {archive_name}  ({path.stat().st_size / 1024:.1f} KB)")


def _walk_and_add(zipf, root_dir, prefix_in_zip, stats, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = EXCLUDE_DIRS
    root = Path(root_dir)
    if not root.exists():
        return
    for entry in sorted(root.rglob("*")):
        relative = entry.relative_to(root)
        parts = relative.parts

        skip = False
        for part in parts:
            if part in exclude_dirs:
                skip = True
                break
        if skip:
            continue

        if entry.is_dir():
            continue

        archive_name = str(Path(prefix_in_zip) / relative).replace("\\", "/")
        _add_file(zipf, entry, archive_name, stats)


def zipar_projeto(output: str = "autotimes_kaggle.zip", include_gpt2: bool = False):
    root = Path.cwd()
    dataset_root = root.parent / "dataset"
    stats = {"file_count": 0, "total_size": 0, "skipped_files": 0, "skipped_dirs": 0}

    print(f"Empacotando codigo:  {root}")
    print(f"Empacotando dataset: {dataset_root}")
    print(f"  ignorando diretorios: {', '.join(sorted(EXCLUDE_DIRS))}")
    print()

    exclude_dirs = EXCLUDE_DIRS | {"gpt2"} if not include_gpt2 else EXCLUDE_DIRS

    import shutil
    src_models = root.parent / "models"
    dst_models = root / "models"
    dst_models.mkdir(exist_ok=True)
    for f in src_models.glob("AutoTimes_*.py"):
        shutil.copy2(f, dst_models / f.name)

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Codigo (teste/)
        _walk_and_add(zipf, root, ".", stats, exclude_dirs)

        # Dataset (../dataset/ -> dataset/ dentro do zip)
        if dataset_root.exists():
            print(f"\n  --- dataset/ ---")
            _walk_and_add(zipf, dataset_root, "dataset", stats)

    size_mb = stats["total_size"] / (1024 * 1024)
    print(f"\nConcluido: {output}")
    print(f"  Arquivos: {stats['file_count']}")
    print(f"  Tamanho:  {size_mb:.2f} MB")
    if stats["skipped_files"]:
        print(f"  Ignorados: {stats['skipped_files']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Empacota projeto + datasets para Kaggle")
    parser.add_argument("--output", default="autotimes_kaggle.zip", help="Nome do arquivo .zip de saida")
    parser.add_argument("--include-gpt2", action="store_true", help="Incluir pasta gpt2/ (pesado ~500 MB)")
    args = parser.parse_args()
    zipar_projeto(output=args.output, include_gpt2=args.include_gpt2)
