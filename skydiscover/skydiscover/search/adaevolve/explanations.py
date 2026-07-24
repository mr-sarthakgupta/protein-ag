"""Post-iteration scientific explanations for symbolic AdaEvolve programs."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import sympy as sp

from skydiscover.search.base_database import Program

logger = logging.getLogger(__name__)

_SAFE_EXPRESSION = re.compile(r"^[A-Za-z0-9_+\-*/()., \t]+$")
_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_FUNCTION_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_FUNCTIONS: dict[str, Any] = {
    "Abs": sp.Abs,
    "acos": sp.acos,
    "acosh": sp.acosh,
    "asin": sp.asin,
    "asinh": sp.asinh,
    "atan": sp.atan,
    "atanh": sp.atanh,
    "cos": sp.cos,
    "cosh": sp.cosh,
    "exp": sp.exp,
    "erf": sp.erf,
    "erfc": sp.erfc,
    "log": sp.log,
    "sin": sp.sin,
    "sinh": sp.sinh,
    "sqrt": sp.sqrt,
    "tan": sp.tan,
    "tanh": sp.tanh,
}


def select_iteration_programs(
    *,
    current_iteration_programs: Sequence[Program],
    previous_best_id: str | None,
    previous_best_score: float | None,
    current_best: Program | None,
    previous_pareto_ids: set[str],
    current_pareto: Sequence[Program],
    score_key,
) -> list[tuple[Program, list[str]]]:
    """Select the batch winner and genuinely new best/Pareto programs by ID."""
    selected: dict[str, tuple[Program, list[str]]] = {}

    def add(program: Program | None, reason: str) -> None:
        if program is None:
            return
        if program.id not in selected:
            selected[program.id] = (program, [])
        reasons = selected[program.id][1]
        if reason not in reasons:
            reasons.append(reason)

    if current_iteration_programs:
        add(max(current_iteration_programs, key=score_key), "iteration_winner")

    current_ids = {program.id for program in current_iteration_programs}
    if (
        current_best is not None
        and current_best.id in current_ids
        and current_best.id != previous_best_id
        and (previous_best_score is None or score_key(current_best) > previous_best_score)
    ):
        add(current_best, "new_global_best")

    for program in current_pareto:
        if program.id in current_ids and program.id not in previous_pareto_ids:
            add(program, "new_pareto_representative")

    return list(selected.values())


def _parse_expression(text: str) -> sp.Expr:
    """Parse evaluator-produced SymPy text with a deliberately tiny namespace."""
    text = text.strip()
    if not text or not _SAFE_EXPRESSION.fullmatch(text) or "__" in text:
        raise ValueError(f"unsafe or empty symbolic expression: {text!r}")
    called = set(_FUNCTION_CALL.findall(text))
    unsupported = called - set(_FUNCTIONS)
    if unsupported:
        raise ValueError(f"unsupported symbolic functions: {sorted(unsupported)}")
    local_dict = dict(_FUNCTIONS)
    local_dict.update({"E": sp.E, "pi": sp.pi})
    for name in _IDENTIFIER.findall(text):
        if name not in local_dict:
            local_dict[name] = sp.Symbol(name)
    return sp.sympify(text, locals=local_dict, evaluate=False)


def _equation_parts(template: str) -> tuple[str, str, str]:
    if "=" in template:
        lhs, rhs = template.split("=", 1)
        lhs = lhs.strip()
        kind = "ode" if lhs.startswith("d(") and "/dt" in lhs else "algebraic"
        return kind, lhs, rhs.strip()
    return "ode", "d(x4)/dt", template.strip()


def _expression_tree(expr: sp.Expr, equation_id: str) -> tuple[list[dict[str, Any]], str]:
    nodes: list[dict[str, Any]] = []

    def visit(node: sp.Basic, parent_id: str, path: tuple[int, ...]) -> str:
        suffix = "root" if not path else ".".join(str(index) for index in path)
        node_id = f"{equation_id}.node.{suffix}"
        child_ids = [
            f"{equation_id}.node.{'.'.join(str(value) for value in path + (index,))}"
            for index in range(len(node.args))
        ]
        nodes.append(
            {
                "id": node_id,
                "operator": node.func.__name__,
                "type": type(node).__name__,
                "expression": str(node),
                "parent_id": parent_id,
                "children_ids": child_ids,
            }
        )
        for index, child in enumerate(node.args):
            visit(child, node_id, path + (index,))
        return node_id

    root_id = visit(expr, equation_id, ())
    return nodes, root_id


def build_equation_manifest(
    program: Program,
    *,
    observed_variables: Mapping[str, str] | None = None,
    selection_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a deterministic complete preorder manifest from evaluator metrics."""
    metrics = program.metrics or {}
    templates = metrics.get("equation_templates") or []
    if isinstance(templates, str):
        templates = [templates]
    if not templates:
        fallback = metrics.get("equation_template") or metrics.get("equation")
        if fallback:
            templates = [fallback]
    if not templates:
        raise ValueError("program metrics do not contain equation-system metadata")

    equations: list[dict[str, Any]] = []
    all_nodes: list[dict[str, Any]] = []
    for index, template_value in enumerate(templates):
        template = str(template_value)
        kind, target, expression_text = _equation_parts(template)
        equation_id = f"equation.{index:03d}"
        nodes, root_id = _expression_tree(_parse_expression(expression_text), equation_id)
        equations.append(
            {
                "id": equation_id,
                "order": index,
                "kind": kind,
                "operator": "=",
                "type": "equation",
                "target": target,
                "expression": expression_text,
                "root_node_id": root_id,
                "parent_id": None,
                "children_ids": [root_id],
            }
        )
        all_nodes.extend(nodes)

    resolved_text = str(metrics.get("resolved_ode_template") or "")
    resolved = None
    if resolved_text:
        resolved = {
            "kind": "resolved_ode",
            "target": "d(x4)/dt",
            "expression": resolved_text,
        }

    coverage_ids = [equation["id"] for equation in equations]
    coverage_ids.extend(node["id"] for node in all_nodes)
    fitted_constants = {
        str(dataset): result.get("constants", {})
        for dataset, result in (metrics.get("per_dataset") or {}).items()
        if isinstance(result, Mapping)
    }
    return {
        "schema_version": 1,
        "program_id": program.id,
        "iteration": program.iteration_found,
        "selection_reasons": list(selection_reasons),
        "system_fingerprint": metrics.get("system_fingerprint"),
        "ordered_equations": equations,
        "resolved_ode": resolved,
        "expression_nodes_preorder": all_nodes,
        "coverage_ids": coverage_ids,
        "observed_variables": dict(observed_variables or {}),
        "fitted_constants_by_dataset": fitted_constants,
        "metrics": _json_safe(metrics),
        "candidate_source": program.solution,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def parse_and_validate_explanation(
    response_text: str, required_ids: Iterable[str]
) -> tuple[dict[str, Any], list[str]]:
    """Parse strict JSON and return manifest IDs not covered by explanation records."""
    response = json.loads(response_text)
    if not isinstance(response, dict) or not isinstance(response.get("explanations"), list):
        raise ValueError("response must be a JSON object with an explanations array")
    if not isinstance(response.get("summary"), str) or not isinstance(
        response.get("limitations"), list
    ):
        raise ValueError("response must contain string summary and limitations array")
    if any(not isinstance(item, str) for item in response["limitations"]):
        raise ValueError("every limitation must be a string")
    required = set(required_ids)
    covered: set[str] = set()
    for entry in response["explanations"]:
        if not isinstance(entry, dict):
            raise ValueError("each explanation must be a JSON object")
        manifest_id = entry.get("manifest_id")
        required_fields = (
            "mathematical_role",
            "scientific_interpretation",
            "concentration_relationship",
            "evidence_level",
        )
        if not isinstance(manifest_id, str) or any(
            not isinstance(entry.get(field), str) or not entry[field].strip()
            for field in required_fields
        ):
            raise ValueError("explanation entry is missing required string fields")
        if entry["evidence_level"] not in {
            "supported",
            "plausible_hypothesis",
            "purely_mathematical",
        }:
            raise ValueError(f"invalid evidence_level for {manifest_id}")
        if manifest_id not in required:
            raise ValueError(f"unknown manifest_id: {manifest_id}")
        covered.add(manifest_id)
    missing = sorted(required - covered)
    return response, missing


def _prompt(manifest: Mapping[str, Any], missing_ids: Sequence[str] | None = None) -> str:
    repair = ""
    if missing_ids is not None:
        repair = (
            "\nYour previous response was incomplete. Return the entire corrected JSON, "
            f"with these missing manifest IDs included: {json.dumps(list(missing_ids))}\n"
        )
    return f"""Explain this fitted symbolic equation system conservatively.
{repair}
Return ONLY one valid JSON object, with no markdown fences or commentary:
{{
  "summary": "string",
  "limitations": ["string"],
  "explanations": [{{
    "manifest_id": "exact ID from coverage_ids",
    "mathematical_role": "string",
    "scientific_interpretation": "string",
    "concentration_relationship": "explicit relation to inhibitor, monomer, seed, state/concentration, or why none",
    "evidence_level": "supported|plausible_hypothesis|purely_mathematical"
  }}]
}}
Include every ID in coverage_ids at least once, including every equation and every
expression node/term. Do not assert causality beyond the supplied data. Clearly
separate direct mathematical facts from scientific hypotheses.

MANIFEST:
{json.dumps(manifest, sort_keys=True, separators=(",", ":"))}
"""


class IterationExplanationWriter:
    """Generate, validate, and atomically persist optional explanations."""

    def __init__(self, llm_pool, config, output_dir: str | os.PathLike[str]):
        self.llm_pool = llm_pool
        self.config = config
        self.root = Path(output_dir) / "iteration_explanations"
        self.index_path = self.root / "index.jsonl"

    async def explain(
        self, program: Program, iteration: int, selection_reasons: Sequence[str]
    ) -> dict[str, Any]:
        iteration_dir = self.root / f"iteration_{iteration}"
        json_path = iteration_dir / f"{program.id}.json"
        markdown_path = iteration_dir / f"{program.id}.md"
        if json_path.exists():
            try:
                existing = json.loads(json_path.read_text())
                return self._index_record(existing, json_path, markdown_path, reused=True)
            except (OSError, json.JSONDecodeError):
                logger.warning("Ignoring corrupt explanation entry %s", json_path)

        manifest: dict[str, Any] | None = None
        response: dict[str, Any] | None = None
        raw_responses: list[str] = []
        status = "error"
        error: str | None = None
        missing_ids: list[str] = []
        try:
            manifest = build_equation_manifest(
                program,
                observed_variables=self.config.observed_variables,
                selection_reasons=selection_reasons,
            )
            for attempt in range(2):
                result = await self.llm_pool.generate(
                    self.config.system_message,
                    [
                        {
                            "role": "user",
                            "content": _prompt(manifest, missing_ids if attempt else None),
                        }
                    ],
                    max_tokens=self.config.max_tokens,
                    retries=self.config.retries,
                    timeout=self.config.timeout,
                )
                raw = result.text or ""
                raw_responses.append(raw)
                try:
                    response, missing_ids = parse_and_validate_explanation(
                        raw, manifest["coverage_ids"]
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    error = str(exc)
                    missing_ids = list(manifest["coverage_ids"])
                    response = None
                if response is not None and not missing_ids:
                    status = "complete"
                    error = None
                    break
            else:
                status = "incomplete" if response is not None else "error"
                if missing_ids:
                    error = f"missing manifest IDs: {', '.join(missing_ids)}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("Program explanation failed for %s: %s", program.id, error)

        document = {
            "schema_version": 1,
            "status": status,
            "program_id": program.id,
            "iteration": iteration,
            "selection_reasons": list(selection_reasons),
            "manifest": manifest,
            "explanation": response,
            "missing_ids": missing_ids,
            "error": error,
            "attempts": len(raw_responses),
            "raw_responses": raw_responses,
        }
        iteration_dir.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(json_path, document)
        self._write_text_atomic(markdown_path, self._markdown(document))
        record = self._index_record(document, json_path, markdown_path)
        self._append_index_once(record)
        return record

    def _index_record(self, document, json_path, markdown_path, reused=False):
        return {
            "iteration": document.get("iteration"),
            "program_id": document.get("program_id"),
            "status": document.get("status", "error"),
            "json_path": str(json_path),
            "markdown_path": str(markdown_path),
            "reused": reused,
        }

    @staticmethod
    def _write_json_atomic(path: Path, value: Any) -> None:
        IterationExplanationWriter._write_text_atomic(
            path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )

    @staticmethod
    def _write_text_atomic(path: Path, text: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        os.replace(temporary, path)

    def _append_index_once(self, record: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        key = (record["iteration"], record["program_id"])
        if self.index_path.exists():
            for line in self.index_path.read_text().splitlines():
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (existing.get("iteration"), existing.get("program_id")) == key:
                    return
        with self.index_path.open("a") as index_file:
            index_file.write(json.dumps(record, sort_keys=True) + "\n")
            index_file.flush()
            os.fsync(index_file.fileno())

    @staticmethod
    def _markdown(document: Mapping[str, Any]) -> str:
        lines = [
            f"# Program {document['program_id']}",
            "",
            f"Status: `{document['status']}`",
            f"Iteration: {document['iteration']}",
            f"Selection: {', '.join(document['selection_reasons'])}",
            "",
        ]
        explanation = document.get("explanation")
        if isinstance(explanation, Mapping):
            lines.extend([str(explanation.get("summary", "")), "", "## Terms", ""])
            for entry in explanation.get("explanations", []):
                lines.extend(
                    [
                        f"### {entry.get('manifest_id', '')}",
                        f"- Mathematical role: {entry.get('mathematical_role', '')}",
                        f"- Scientific interpretation: {entry.get('scientific_interpretation', '')}",
                        f"- Concentration relationship: {entry.get('concentration_relationship', '')}",
                        f"- Evidence: `{entry.get('evidence_level', '')}`",
                        "",
                    ]
                )
        else:
            lines.extend(["## Error", "", str(document.get("error") or "Unknown error"), ""])
        return "\n".join(lines)
