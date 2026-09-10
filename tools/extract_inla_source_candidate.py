#!/usr/bin/env python3
"""Reconstruct ``SimfolioEngine._bdes_cagr_candidate`` without importing the source app."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def _extract(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, object] = {}

    def literal(node: ast.AST) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Tuple):
            return tuple(literal(v) for v in node.elts)
        if isinstance(node, ast.List):
            return [literal(v) for v in node.elts]
        if isinstance(node, ast.Dict):
            return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return literal(node.left) + literal(node.right)
        if isinstance(node, ast.Name):
            return values[node.id]
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            return values[node.attr]
        raise ValueError(f"unsupported source expression: {ast.dump(node)}")

    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SimfolioEngine")
    for node in tree.body + cls.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = literal(node.value)
        except (KeyError, ValueError, TypeError):
            pass

    class Dummy:
        pass

    dummy = Dummy()
    for key, value in values.items():
        setattr(dummy, key, value)

    def evaluate(node: ast.AST) -> object:
        if isinstance(node, ast.Name) and node.id == "self":
            return dummy
        if isinstance(node, ast.Name):
            return getattr(dummy, node.id)
        if isinstance(node, ast.Attribute):
            return getattr(evaluate(node.value), node.attr)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Tuple):
            return tuple(evaluate(v) for v in node.elts)
        if isinstance(node, ast.List):
            return [evaluate(v) for v in node.elts]
        if isinstance(node, ast.Dict):
            return {evaluate(k): evaluate(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return evaluate(node.left) + evaluate(node.right)
        raise ValueError(f"unsupported source expression: {ast.dump(node)}")

    def body(name: str, base: dict[str, object] | None = None) -> dict[str, object]:
        function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        result = dict(base or {})
        for statement in function.body:
            if isinstance(statement, ast.Return):
                if isinstance(statement.value, ast.Name) and statement.value.id == "candidate":
                    return result
                return dict(evaluate(statement.value))
            if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
                continue
            call = statement.value
            if not isinstance(call.func, ast.Attribute):
                continue
            if call.func.attr == "pop":
                result.pop(evaluate(call.args[0]), None)
            elif call.func.attr == "update":
                result.update(evaluate(call.args[0]))
        return result

    return body(
        "_bdes_cagr_candidate",
        body("_bdes_cagr_fastmap_candidate", body("_bdes_cagr_mcmc_candidate")),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_engine", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = _extract(args.source_engine)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(candidate, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(candidate)} keys to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
