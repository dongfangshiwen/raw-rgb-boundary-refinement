"""Collect the first-party source and configuration needed for a study export."""
import ast
from importlib.util import resolve_name

from boundary_refinement.paths import SOURCE_ROOT


def source_files(entry_points):
    pending = list(entry_points)
    files = set()
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen or name.split(".", 1)[0] != "boundary_refinement":
            continue
        seen.add(name)
        stem = SOURCE_ROOT.joinpath(*name.split("."))
        is_package = stem.is_dir()
        path = stem / "__init__.py" if is_package else stem.with_suffix(".py")
        if not path.is_file():
            continue  # A symbol imported from a module can also appear in this list.
        files.add(path)
        package = name if is_package else name.rpartition(".")[0]
        if "." in name:
            pending.append(name.rpartition(".")[0])
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = resolve_name("." * node.level + (node.module or ""), package)
                pending.append(base)
                pending.extend(f"{base}.{alias.name}" for alias in node.names)

    data_files = {
        "boundary_refinement.data.source_integrity": "boundary_refinement/data/released_source_digests.json",
        "boundary_refinement.inference.score_grid": "boundary_refinement/models/parameters/decoder_coefficients.json",
    }
    files.update(SOURCE_ROOT / filename for module, filename in data_files.items() if module in seen)
    return sorted(files)
