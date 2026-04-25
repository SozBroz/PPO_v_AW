"""Phase 8 Lane K — AST helpers for slack inventory (read-only analysis)."""
from __future__ import annotations

import ast
import pathlib


def main() -> None:
    p = pathlib.Path(__file__).resolve().parent / "oracle_zip_replay.py"
    tree = ast.parse(p.read_text(encoding="utf-8"))
    raises: list[tuple[int, str]] = []

    class V(ast.NodeVisitor):
        def visit_Raise(self, n: ast.Raise) -> None:
            exc = n.exc
            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                if exc.func.id == "UnsupportedOracleAction" and exc.args:
                    a0 = exc.args[0]
                    if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                        raises.append((n.lineno, a0.value))
                    elif isinstance(a0, ast.JoinedStr):
                        raises.append((n.lineno, "<fstring>"))
                    else:
                        raises.append((n.lineno, ast.unparse(a0)[:100]))
            self.generic_visit(n)

    V().visit(tree)
    print("unsupported_oracle_action_raise_count", len(raises))
    for ln, msg in raises:
        if msg != "<fstring>" and len(msg) < 72:
            print(ln, repr(msg))


if __name__ == "__main__":
    main()
