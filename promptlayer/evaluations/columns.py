import ast
import inspect
import json
import textwrap
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from promptlayer.evaluations.utils import is_reserved_eval_column_title
from promptlayer.evaluations.validation import validation_error
from promptlayer.types.table import ColumnType, ColumnTypeValue, EvalScorerColumn

# Scorer args mapped onto eval row column titles.
# CODE_EXECUTION sandboxes expose row columns via the `data` object.
_SCORER_PARAM_COLUMN_ALIASES = {
    "input": "input",
    "output": "Output",
    "expected": "expected",
    "expected_trace": "expected_trace",
    "trace": "Trace",
}


def _function_body_source(fn: Callable[..., Any], fn_name: str) -> str:
    """Return the dedented body of ``fn`` (no outer ``def``), docstring stripped."""
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as exc:
        raise validation_error(
            f"Could not read source for scorer '{fn_name}'. "
            "Define a named function in a .py file, or pass "
            "code_execution_column(...).",
        ) from exc

    if not source.strip().startswith(("def ", "@")):
        raise validation_error(
            f"Scorer '{fn_name}' must be a named function definition.",
        )

    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        raise validation_error(
            f"Could not parse source for scorer '{fn_name}'.",
        ) from exc

    func_node = next(
        (node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if func_node is None:
        raise validation_error(
            f"Scorer '{fn_name}' must be a named function definition.",
        )

    body_nodes = list(func_node.body)
    if (
        body_nodes
        and isinstance(body_nodes[0], ast.Expr)
        and isinstance(body_nodes[0].value, ast.Constant)
        and isinstance(body_nodes[0].value.value, str)
    ):
        body_nodes = body_nodes[1:]
    if not body_nodes:
        raise validation_error(
            f"Scorer '{fn_name}' has an empty body.",
        )

    # Keep top-level ``return`` (CODE_EXECUTION wraps the body in ``def execute(data)``
    # and requires a real return). Nested helper returns are left alone.
    # Normalize ``return {"score": x}`` → ``return x``.
    rewritten: List[ast.stmt] = []
    for stmt in body_nodes:
        if isinstance(stmt, ast.Return):
            value: ast.expr = ast.Constant(value=None) if stmt.value is None else stmt.value
            # _result = <expr>
            # if isinstance(_result, dict) and "score" in _result:
            #     _result = _result["score"]
            # return _result
            assign = ast.Assign(
                targets=[ast.Name(id="_result", ctx=ast.Store())],
                value=value,
            )
            normalize_if = ast.If(
                test=ast.BoolOp(
                    op=ast.And(),
                    values=[
                        ast.Call(
                            func=ast.Name(id="isinstance", ctx=ast.Load()),
                            args=[
                                ast.Name(id="_result", ctx=ast.Load()),
                                ast.Name(id="dict", ctx=ast.Load()),
                            ],
                            keywords=[],
                        ),
                        ast.Compare(
                            left=ast.Constant(value="score"),
                            ops=[ast.In()],
                            comparators=[ast.Name(id="_result", ctx=ast.Load())],
                        ),
                    ],
                ),
                body=[
                    ast.Assign(
                        targets=[ast.Name(id="_result", ctx=ast.Store())],
                        value=ast.Subscript(
                            value=ast.Name(id="_result", ctx=ast.Load()),
                            slice=ast.Constant(value="score"),
                            ctx=ast.Load(),
                        ),
                    )
                ],
                orelse=[],
            )
            ret = ast.Return(value=ast.Name(id="_result", ctx=ast.Load()))
            for node in (assign, normalize_if, ret):
                rewritten.append(ast.copy_location(node, stmt))
        else:
            rewritten.append(_rewrite_trace_column_lookups(stmt))

    module = ast.Module(body=rewritten, type_ignores=[])
    ast.fix_missing_locations(module)
    return ast.unparse(module)


def _rewrite_trace_column_lookups(node: ast.AST) -> ast.AST:
    class _Rewriter(ast.NodeTransformer):
        def visit_Call(self, call: ast.Call) -> ast.AST:
            self.generic_visit(call)
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "get"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "data"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value in _SCORER_PARAM_COLUMN_ALIASES
            ):
                call.args[0] = ast.Constant(value=_SCORER_PARAM_COLUMN_ALIASES[call.args[0].value])
            return call

        def visit_Subscript(self, sub: ast.Subscript) -> ast.AST:
            self.generic_visit(sub)
            if (
                isinstance(sub.value, ast.Name)
                and sub.value.id == "data"
                and isinstance(sub.slice, ast.Constant)
                and sub.slice.value in _SCORER_PARAM_COLUMN_ALIASES
            ):
                sub.slice = ast.Constant(value=_SCORER_PARAM_COLUMN_ALIASES[sub.slice.value])
            return sub

    return _Rewriter().visit(node)


def _bind_params_from_data(fn: Callable[..., Any]) -> List[str]:
    """Emit ``param = data.get("Column")`` lines for each scorer parameter."""
    params = [
        (name, param)
        for name, param in inspect.signature(fn).parameters.items()
        if param.kind
        not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    ]
    if len(params) == 1 and params[0][0] == "data":
        return []

    lines: List[str] = []
    for param_name, _param in params:
        if param_name == "data":
            # Avoid shadowing the sandbox ``data`` object.
            continue
        column_title = _SCORER_PARAM_COLUMN_ALIASES.get(param_name, param_name)
        lines.append(f"{param_name} = data.get({json.dumps(column_title)})")
    return lines


def column(
    title: str,
    type: Union[ColumnType, ColumnTypeValue],
    config: Optional[Dict[str, Any]] = None,
) -> EvalScorerColumn:
    if not isinstance(title, str) or not title.strip():
        raise validation_error(
            "Column title must be a non-empty string.",
        )
    if is_reserved_eval_column_title(title):
        raise validation_error(
            f"Eval column title {title!r} is reserved for built-in eval columns.",
        )
    resolved_type: ColumnTypeValue = type.value if isinstance(type, ColumnType) else type
    if str(resolved_type).upper() == "TEXT":
        raise validation_error(
            "Eval columns cannot be TEXT; use dataset fields or built-in input/expected/output columns.",
        )
    payload: EvalScorerColumn = {
        "title": title,
        "type": resolved_type,
    }
    if config is not None:
        payload["config"] = config
    return payload


def code_execution_column(
    title: str,
    *,
    code: str,
    language: Literal["PYTHON", "JAVASCRIPT"] = "PYTHON",
) -> EvalScorerColumn:
    if not isinstance(code, str) or not code.strip():
        raise validation_error(
            "code_execution_column requires non-empty code.",
        )
    return column(
        title,
        ColumnType.CODE_EXECUTION,
        {"code": code, "language": language},
    )


def scorer_from_function(
    fn: Callable[..., Any],
    *,
    title: Optional[str] = None,
) -> EvalScorerColumn:
    """Turn a Python scorer function into a server-side CODE_EXECUTION column."""
    if not callable(fn):
        raise validation_error(
            "scorer_from_function requires a callable.",
        )
    if inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn):
        raise validation_error(
            "Async scorer functions are not supported; use a sync function or code_execution_column(...).",
        )

    fn_name = getattr(fn, "__name__", "") or "scorer"
    if fn_name == "<lambda>":
        raise validation_error(
            "Lambda scorers are not supported; use a named function or code_execution_column(...).",
        )

    body = _function_body_source(fn, fn_name)
    bind_lines = _bind_params_from_data(fn)
    code = "\n".join([*bind_lines, body.rstrip()]) + "\n"
    column_title = title if title is not None else fn_name.replace("_", " ").strip() or fn_name
    return code_execution_column(
        column_title,
        code=code,
        language="PYTHON",
    )
