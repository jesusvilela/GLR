# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Score OlympiadBench OE_TO_maths_en_COMP custom decodes (exact_match + math_verify).

This mirrors the intent of lm-evaluation-harness's Minerva math task metrics:
  - exact_match: extract a final answer string, normalize it, and check symbolic
    equivalence via SymPy.
  - math_verify: use `math_verify.verify` between gold and model output.

It is designed for the custom decoding JSONL format used in this repo:
  {"question": <prompt>, "answer": <gold answer>, "predicted": <model output>}

Example:
  python src/metric/olympiadbench_oe_to_maths_en_comp_score.py \
    --pred-file /path/to/olympiadbench_oe_to_maths_en_comp_train_latent10_gen8192_temp0.0.json

Dependencies:
  - exact_match: `pip install sympy antlr4-python3-runtime==4.11`
  - math_verify: `pip install sympy antlr4-python3-runtime==4.11 math-verify`
"""

import argparse
import json
import re
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm


@dataclass(frozen=True)
class ExampleResult:
    gold_raw: str
    pred_raw: str
    gold_norm: str
    gold_solution_used: str | None
    pred_final_raw: str
    pred_norm: str
    exact_match: int
    math_verify: int


def _extract_doc_solution_from_sample(rec: dict[str, Any]) -> str | None:
    doc = rec.get("doc")
    if not isinstance(doc, dict):
        return None
    sol = doc.get("solution")
    if sol in (None, ""):
        return None
    return str(sol)


def _iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    """Yield dict records from a JSONL file or a JSON list."""
    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            return
        first = first.strip()
        try:
            first_obj = json.loads(first)
        except json.JSONDecodeError:
            first_obj = None

    if isinstance(first_obj, dict):
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Failed to parse JSON on line {line_no} of {path}: {e}"
                    ) from e
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Expected JSON object on line {line_no} of {path}, got {type(obj)}"
                    )
                yield obj
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for idx, obj in enumerate(data):
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected list of objects in {path}, but element {idx} is {type(obj)}"
                )
            yield obj
        return

    raise ValueError(
        f"Unsupported JSON structure in {path}: expected JSONL objects or a JSON list"
    )


def _detect_is_jsonl(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                return False
            return isinstance(obj, dict)
    return False


def _get_doc_identifier(rec: dict[str, Any]) -> int | str | None:
    """Return an identifier for joining predictions to gold."""
    if "doc_id" in rec:
        return rec.get("doc_id")
    if "id" in rec:
        return rec.get("id")
    return None


def _extract_first_resp_from_resps(resps: Any) -> str | None:
    """Extract the first response string from lm-eval `resps`."""
    if not isinstance(resps, list) or not resps:
        return None
    first = resps[0]
    if isinstance(first, list):
        if not first:
            return None
        first = first[0]
    if first is None:
        return None
    return str(first)


def _load_gold_answers(
    gold_file: Path,
    *,
    id_key: str = "id",
) -> tuple[list[dict[str, Any]], dict[int | str, dict[str, Any]]]:
    gold_list: list[dict[str, Any]] = []
    gold_by_id: dict[int | str, dict[str, Any]] = {}
    for rec in _iter_json_records(gold_file):
        gold_list.append(rec)
        if id_key in rec:
            rid = rec.get(id_key)
            if rid is not None:
                gold_by_id[rid] = rec
    return gold_list, gold_by_id


_PROMPT_QUESTION_RE = re.compile(r"\bQuestion:\s*(.*?)\nAnswer:\s*$", re.DOTALL)


def _extract_problem_from_prompt(prompt: str) -> str | None:
    if prompt is None:
        return None
    m = _PROMPT_QUESTION_RE.search(prompt)
    if not m:
        return None
    prob = m.group(1)
    prob = prob.replace("<|im_start|>", "").replace("<|im_end|>", "")
    return prob.strip() if prob.strip() else None


def _load_gold_solutions(
    gold_file: Path,
    *,
    problem_key: str,
    solution_key: str,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for rec in _iter_json_records(gold_file):
        if problem_key not in rec or solution_key not in rec:
            continue
        prob = str(rec[problem_key]).strip()
        sol = str(rec[solution_key])
        if prob and (prob not in mapping):
            mapping[prob] = sol
    if not mapping:
        raise ValueError(
            f"Loaded 0 gold (problem, solution) pairs from {gold_file}. "
            f"Expected keys '{problem_key}' and '{solution_key}'."
        )
    return mapping


def _last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def _remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] != left:
            return s
        return s[len(left) :]
    left = "\\boxed{"
    if not s.startswith(left) or not s.endswith("}"):
        return s
    return s[len(left) : -1]


def _response_region(text: str) -> str:
    if text is None:
        return ""

    markers = (
        "<|im_start|>assistant",
        "<|assistant|>",
        "\nAssistant:",
        "\n### Assistant:",
    )

    best_idx = -1
    best_marker = None
    for marker in markers:
        idx = text.rfind(marker)
        if idx > best_idx:
            best_idx = idx
            best_marker = marker

    if best_idx == -1 or best_marker is None:
        return text
    return text[best_idx + len(best_marker) :]


def _extract_balanced_braces(text: str, left_brace_idx: int) -> tuple[str, int] | None:
    if left_brace_idx < 0 or left_brace_idx >= len(text) or text[left_brace_idx] != "{":
        return None
    depth = 0
    content_start: int | None = None
    for i in range(left_brace_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
            if depth == 1:
                content_start = i + 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and content_start is not None:
                return text[content_start:i], i + 1
            if depth < 0:
                return None
    return None


def _extract_last_boxed_content(text: str) -> str | None:
    if not text:
        return None

    idx_boxed = text.rfind("\\boxed")
    idx_fbox = text.rfind("\\fbox")
    idx = max(idx_boxed, idx_fbox)
    if idx == -1:
        return None

    cmd = "\\boxed" if idx == idx_boxed else "\\fbox"
    j = idx + len(cmd)
    while j < len(text) and text[j].isspace():
        j += 1

    if j < len(text) and text[j] == "{":
        parsed = _extract_balanced_braces(text, j)
        if parsed is None:
            return None
        content, _ = parsed
        content = content.strip()
        return content if content else None

    m = re.match(r"([^\s$<]+)", text[j:])
    if not m:
        return None
    content = m.group(1).strip()
    return content if content else None


_FINAL_ANSWER_RE = re.compile(
    r"Final Answer: The final answer is(.*?). I hope it is correct\.", re.DOTALL
)


def extract_minerva_final_answer(predicted: str) -> str:
    if predicted is None:
        return "[invalidanswer]"

    region = _response_region(predicted)

    m = _FINAL_ANSWER_RE.search(region + " I hope it is correct.")
    if m:
        ans = m.group(1).strip()
        return ans if ans else "[invalidanswer]"

    boxed = _extract_last_boxed_content(region)
    if boxed is not None:
        return boxed

    math_spans: list[str] = []
    math_spans.extend(re.findall(r"\$(.*?)\$", region, flags=re.DOTALL))
    math_spans.extend(re.findall(r"\\\((.*?)\\\)", region, flags=re.DOTALL))
    math_spans.extend(re.findall(r"\\\[(.*?)\\\]", region, flags=re.DOTALL))
    if math_spans:
        last = math_spans[-1].strip()
        return last if last else "[invalidanswer]"

    return "[invalidanswer]"


# Normalization copied character-for-character from lm-eval-harness Minerva utils.
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\\mathrm{th}",
    r"^\\circ",
    r"^{\\circ}",
    r"\\;",
    r",\\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    if final_answer is None:
        return "[invalidanswer]"

    # Strip a single outer $...$ pair before splitting on "=" so that
    # "$f(n)=n$" → "f(n)=n" → "n" instead of "n$".
    s = final_answer.strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2 and s.count("$") == 2:
        final_answer = s[1:-1]

    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        # Pure-alphabetic unit tokens (e.g. "ft", "cm", "square") must not be
        # removed when they appear inside a LaTeX command such as \left, \lfloor,
        # or \square.  Use a word-boundary-aware regex for those.
        if re.fullmatch(r"[a-zA-Z]+", expr):
            final_answer = re.sub(
                r"(?<![\\\w])" + re.escape(expr) + r"(?![\w])", "", final_answer
            )
        else:
            final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textit\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"\\mathbf\{(.*?)\}", r"\\1", final_answer)
    final_answer = re.sub(r"\\mbox\{(.*?)\}", r"\\1", final_answer)

    if "\\begin" in final_answer:
        final_answer = final_answer.split("\\begin")[0]

    # In some cases, the model outputs an equation of the form `\frac{a}{b} = \frac{c}{d}`.
    if final_answer.count("=") == 1:
        final_answer = final_answer.split("=")[-1]

    final_answer = final_answer.strip(".\n")

    return final_answer if final_answer else "[invalidanswer]"


class TimeoutError(Exception):
    pass


def _timeout_handler(_signum: int, _frame: Any) -> None:
    raise TimeoutError


def _is_equiv_latex(pred_norm: str, gold_norm: str) -> bool:
    """Return True if two normalized answers are symbolically equivalent."""
    if pred_norm in (None, "", "[invalidanswer]") or gold_norm in (
        None,
        "",
        "[invalidanswer]",
    ):
        return False

    try:
        import sympy
        from sympy.parsing.latex import parse_latex

        # Avoid hangs on pathological parses.
        old = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(5)
        try:
            try:
                parsed_pred = parse_latex(pred_norm)
                parsed_gold = parse_latex(gold_norm)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                return False

            try:
                diff = parsed_pred - parsed_gold
            except TypeError:
                return False

            try:
                return sympy.simplify(diff) == 0
            except ValueError:
                return False
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    except TimeoutError:
        return False
    except Exception:
        return False


_MEMBERSHIP_RE = re.compile(r"^\s*(?:[a-zA-Z]|\\[a-zA-Z]+)\s*\\in\s*(.+?)\s*$")


_MULTI_DOLLAR_RE = re.compile(r"^\$[^$]+\$(?:,\s*\$[^$]+\$)+$")


def _gold_for_math_verify(gold_answer_raw: str) -> str:
    s = gold_answer_raw.strip() if gold_answer_raw is not None else ""
    if not s:
        return "[invalidanswer]"
    if "\\boxed" in s or "\\fbox" in s:
        return s
    # Comma-separated multi-value: "$90^{\circ}$,$60^{\circ}$"
    # Extract each inner expression and join them so the boxed form is valid.
    if _MULTI_DOLLAR_RE.match(s):
        inner_parts = re.findall(r"\$([^$]+)\$", s)
        return f"\\boxed{{{', '.join(inner_parts)}}}"
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1]
    return f"\\boxed{{{s}}}"


def _gold_candidates_for_math_verify(gold_answer_raw: str) -> list[str]:
    s = gold_answer_raw.strip() if gold_answer_raw is not None else ""
    if not s:
        return ["[invalidanswer]"]

    candidates = [_gold_for_math_verify(s)]
    m = _MEMBERSHIP_RE.match(s)
    if m:
        rhs = m.group(1).strip()
        if rhs:
            candidates.append(_gold_for_math_verify(rhs))

    out: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


_FLEX_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def _extract_flexible_numeric(region: str) -> str | None:
    matches = _FLEX_RE.findall(region)
    if not matches:
        return None
    last = matches[-1]
    chosen = next((m for m in last if m), "") if isinstance(last, tuple) else str(last)
    chosen = chosen.strip()
    return chosen if chosen else None


def _pred_answer_candidates(pred_raw: str) -> list[str]:
    region = _response_region(pred_raw)
    cands: list[str] = []

    primary = extract_minerva_final_answer(pred_raw)
    if primary and primary != "[invalidanswer]":
        cands.append(primary)

    boxed = _extract_last_boxed_content(region)
    if boxed:
        cands.append(boxed)

    flex = _extract_flexible_numeric(region)
    if flex:
        cands.append(flex)

    out: list[str] = []
    seen: set[str] = set()
    for c in cands:
        c2 = str(c).strip()
        if not c2:
            continue
        if c2 not in seen:
            out.append(c2)
            seen.add(c2)
    return out


def score_file(
    pred_file: Path,
    *,
    pred_key: str = "predicted",
    gold_key: str = "answer",
    question_key: str = "question",
    compute_math_verify: bool = True,
    gold_solution_file: Path | None = None,
    gold_problem_key: str = "problem",
    gold_solution_key: str = "solution",
    pred_format: str = "simple",
) -> tuple[list[ExampleResult], dict[str, float]]:
    results: list[ExampleResult] = []

    # Require sympy for exact_match to avoid silent all-zeros.
    try:
        import sympy  # noqa: F401
        from sympy.parsing.latex import parse_latex  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing dependency for exact_match. Install: `pip install sympy antlr4-python3-runtime==4.11`"
        ) from e

    if compute_math_verify:
        try:
            from math_verify import parse, verify  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependency for math_verify. Install: `pip install sympy antlr4-python3-runtime==4.11 math-verify`"
            ) from e
    else:
        parse = verify = None  # type: ignore

    gold_list: list[dict[str, Any]] | None = None
    gold_by_id: dict[int | str, dict[str, Any]] | None = None
    gold_solutions: dict[str, str] | None = None

    if gold_solution_file is not None:
        gold_list, gold_by_id = _load_gold_answers(gold_solution_file)
        # Also build a problem->solution mapping when possible. This is needed
        # for custom decode logs that have no doc_id/id to join on.
        tmp_map: dict[str, str] = {}
        for rec in gold_list:
            if gold_problem_key in rec and gold_solution_key in rec:
                prob = str(rec.get(gold_problem_key, "")).strip()
                sol = str(rec.get(gold_solution_key, ""))
                if prob and prob not in tmp_map:
                    tmp_map[prob] = sol
        gold_solutions = tmp_map if tmp_map else None

    exact_sum = 0
    mv_sum = 0

    def _score_one(
        *,
        pred_raw: str,
        gold_raw: str,
        gold_solution_used: str | None,
        pred_doc_solution: str | None,
    ) -> ExampleResult:
        nonlocal exact_sum, mv_sum

        candidates = _pred_answer_candidates(pred_raw)
        pred_final_raw = candidates[0] if candidates else "[invalidanswer]"

        # exact_match: try sympy equivalence, fall back to normalized string equality.
        gold_norm = normalize_final_answer(gold_raw)
        em = 0
        for cand in candidates if candidates else [pred_final_raw]:
            cand_norm = normalize_final_answer(cand)
            if _is_equiv_latex(cand_norm, gold_norm):
                em = 1
                break
            if cand_norm != "[invalidanswer]" and cand_norm == gold_norm:
                em = 1
                break
        exact_sum += em

        mv = 0
        if compute_math_verify:
            assert parse is not None and verify is not None

            gold_candidates: list[str] = []
            if gold_solution_used is not None:
                gold_candidates.append(gold_solution_used)
            elif pred_doc_solution is not None:
                gold_candidates.append(pred_doc_solution)

            # Always include boxed-answer fallback; helps when solutions are huge/noisy.
            gold_candidates.extend(_gold_candidates_for_math_verify(gold_raw))

            # Try parsing the full model output first, then extracted candidates.
            target_candidates: list[str] = [pred_raw]
            target_candidates.extend(candidates)

            for gold_for_mv in gold_candidates:
                try:
                    g_parsed = parse(gold_for_mv)
                except Exception:
                    continue
                for target_str in target_candidates:
                    try:
                        if verify(gold=g_parsed, target=parse(target_str)):
                            mv = 1
                            break
                    except Exception:
                        continue
                if mv:
                    break
        mv_sum += mv

        return ExampleResult(
            gold_raw=gold_raw,
            pred_raw=pred_raw,
            gold_norm=gold_norm,
            gold_solution_used=gold_solution_used,
            pred_final_raw=pred_final_raw,
            pred_norm=normalize_final_answer(pred_final_raw),
            exact_match=em,
            math_verify=mv,
        )

    # Gold-driven evaluation (join samples -> gold) when predictions have ids.
    if gold_list is not None and gold_by_id is not None:
        pred_by_join_id: dict[int | str, tuple[str, str | None]] = {}
        for rec in _iter_json_records(pred_file):
            join_id = _get_doc_identifier(rec)
            if join_id is None:
                continue

            if pred_format == "lm_eval_samples":
                resp = _extract_first_resp_from_resps(rec.get("resps"))
                if resp is None:
                    continue
                pred_by_join_id[join_id] = (
                    resp,
                    _extract_doc_solution_from_sample(rec),
                )
            else:
                if pred_key not in rec:
                    continue
                pred_by_join_id[join_id] = (str(rec[pred_key]), None)

        # If we couldn't extract any join ids, fall back to per-record scoring below.
        if pred_by_join_id:
            for doc_id, gold_rec in tqdm(enumerate(gold_list), total=len(gold_list)):
                pred_raw: str | None = None
                pred_doc_solution: str | None = None
                gold_id = gold_rec.get("id")
                if gold_id is not None and gold_id in pred_by_join_id:
                    pred_raw, pred_doc_solution = pred_by_join_id[gold_id]
                elif doc_id in pred_by_join_id:
                    pred_raw, pred_doc_solution = pred_by_join_id[doc_id]
                if pred_raw is None:
                    continue

                gold_raw = str(
                    gold_rec.get("answer")
                    or gold_rec.get("a")
                    or gold_rec.get("target")
                    or ""
                )
                gold_solution_used = None
                sol = gold_rec.get("solution")
                if sol not in (None, ""):
                    gold_solution_used = str(sol)

                results.append(
                    _score_one(
                        pred_raw=pred_raw,
                        gold_raw=gold_raw,
                        gold_solution_used=gold_solution_used,
                        pred_doc_solution=pred_doc_solution,
                    )
                )

            n = len(results)
            metrics = {
                "exact_match": (exact_sum / n) if n else 0.0,
                "math_verify": (mv_sum / n) if n else 0.0,
                "n": float(n),
            }
            return results, metrics

    # Default behavior: score directly from pred file records.
    for rec in tqdm(_iter_json_records(pred_file)):
        if pred_format == "lm_eval_samples":
            resp = _extract_first_resp_from_resps(rec.get("resps"))
            if resp is None:
                continue
            pred_raw = resp
            pred_doc_solution = _extract_doc_solution_from_sample(rec)
        else:
            if pred_key not in rec:
                raise KeyError(
                    f"Record missing prediction key '{pred_key}', got keys={list(rec.keys())}"
                )
            pred_raw = str(rec[pred_key])
            pred_doc_solution = None

        if gold_key not in rec:
            raise KeyError(
                f"Record missing gold key '{gold_key}', got keys={list(rec.keys())}"
            )
        gold_raw = str(rec[gold_key])

        gold_solution_used: str | None = None
        if gold_solutions is not None:
            if question_key not in rec:
                raise KeyError(
                    f"gold_solution_file was provided, but records have no '{question_key}' field for problem matching."
                )
            prob = _extract_problem_from_prompt(str(rec[question_key]))
            if prob is None:
                raise ValueError("Failed to extract problem text from question prompt.")
            if prob not in gold_solutions:
                raise KeyError(
                    "Problem not found in gold_solution_file mapping. "
                    "Check that the gold file uses the same problem text."
                )
            gold_solution_used = gold_solutions[prob]
            boxed = _last_boxed_only_string(gold_solution_used)
            if boxed:
                gold_raw = normalize_final_answer(_remove_boxed(boxed))

        results.append(
            _score_one(
                pred_raw=pred_raw,
                gold_raw=gold_raw,
                gold_solution_used=gold_solution_used,
                pred_doc_solution=pred_doc_solution,
            )
        )

    n = len(results)
    metrics = {
        "exact_match": (exact_sum / n) if n else 0.0,
        "math_verify": (mv_sum / n) if n else 0.0,
        "n": float(n),
    }
    return results, metrics


def _save_annotated_file(
    pred_file: Path,
    save_file: Path,
    *,
    pred_key: str,
    gold_key: str,
    question_key: str,
    compute_math_verify: bool,
    gold_solution_file: Path | None,
    gold_problem_key: str,
    gold_solution_key: str,
    pred_format: str,
) -> None:
    # Validate deps early (mirror score_file).
    try:
        import sympy  # noqa: F401
        from sympy.parsing.latex import parse_latex  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing dependency for exact_match. Install: `pip install sympy antlr4-python3-runtime==4.11`"
        ) from e

    if compute_math_verify:
        try:
            from math_verify import parse, verify  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependency for math_verify. Install: `pip install sympy antlr4-python3-runtime==4.11 math-verify`"
            ) from e
    else:
        parse = verify = None  # type: ignore

    gold_list: list[dict[str, Any]] | None = None
    gold_by_id: dict[int | str, dict[str, Any]] | None = None
    gold_solutions: dict[str, str] | None = None

    if gold_solution_file is not None:
        gold_list, gold_by_id = _load_gold_answers(gold_solution_file)
        tmp_map: dict[str, str] = {}
        for rec in gold_list:
            if gold_problem_key in rec and gold_solution_key in rec:
                prob = str(rec.get(gold_problem_key, "")).strip()
                sol = str(rec.get(gold_solution_key, ""))
                if prob and prob not in tmp_map:
                    tmp_map[prob] = sol
        gold_solutions = tmp_map if tmp_map else None

    is_jsonl = _detect_is_jsonl(pred_file)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    def _annotate_record(rec: dict[str, Any]) -> dict[str, Any]:
        pred_text: str | None
        if pred_format == "lm_eval_samples":
            pred_text = _extract_first_resp_from_resps(rec.get("resps"))
            pred_doc_solution = _extract_doc_solution_from_sample(rec)
        else:
            pred_text = str(rec.get(pred_key)) if pred_key in rec else None
            pred_doc_solution = None

        gold_raw: str | None = str(rec.get(gold_key)) if gold_key in rec else None
        gold_solution_used: str | None = None

        # Prefer joining to a gold record when possible.
        if gold_list is not None and gold_by_id is not None:
            join_id = _get_doc_identifier(rec)
            gold_rec = None
            if join_id is not None and join_id in gold_by_id:
                gold_rec = gold_by_id[join_id]
            elif isinstance(join_id, int) and 0 <= join_id < len(gold_list):
                gold_rec = gold_list[join_id]

            if gold_rec is not None:
                gold_raw = str(
                    gold_rec.get("answer")
                    or gold_rec.get("a")
                    or gold_rec.get("target")
                    or gold_raw
                    or ""
                )
                sol = gold_rec.get("solution")
                if sol not in (None, ""):
                    gold_solution_used = str(sol)

        # If we have a problem->solution mapping, use it to supply math_verify
        # and (optionally) extract a boxed gold answer.
        if gold_solutions is not None and question_key in rec:
            prob = _extract_problem_from_prompt(str(rec[question_key]))
            if prob is not None and prob in gold_solutions:
                gold_solution_used = gold_solutions[prob]
                boxed = _last_boxed_only_string(gold_solution_used)
                if boxed:
                    gold_raw = normalize_final_answer(_remove_boxed(boxed))

        rec2 = dict(rec)
        if pred_text is None or gold_raw is None:
            rec2["exact_match_correct"] = None
            rec2["math_verify_correct"] = None
            return rec2

        candidates = _pred_answer_candidates(pred_text)
        pred_final_raw = candidates[0] if candidates else "[invalidanswer]"
        gold_norm = normalize_final_answer(gold_raw)

        em = 0
        for cand in candidates if candidates else [pred_final_raw]:
            cand_norm = normalize_final_answer(cand)
            if _is_equiv_latex(cand_norm, gold_norm) or (
                cand_norm != "[invalidanswer]" and cand_norm == gold_norm
            ):
                em = 1
                break

        mv = 0
        if compute_math_verify:
            assert parse is not None and verify is not None
            gold_candidates: list[str] = []
            if gold_solution_used is not None:
                gold_candidates.append(gold_solution_used)
            elif pred_doc_solution is not None:
                gold_candidates.append(pred_doc_solution)
            gold_candidates.extend(_gold_candidates_for_math_verify(gold_raw))

            target_candidates: list[str] = [pred_text]
            target_candidates.extend(candidates)

            for gold_for_mv in gold_candidates:
                try:
                    g_parsed = parse(gold_for_mv)
                except Exception:
                    continue
                for target_str in target_candidates:
                    try:
                        if verify(gold=g_parsed, target=parse(target_str)):
                            mv = 1
                            break
                    except Exception:
                        continue
                if mv:
                    break

        rec2["pred_final_raw"] = pred_final_raw
        rec2["pred_norm"] = normalize_final_answer(pred_final_raw)
        rec2["gold_norm"] = gold_norm
        rec2["exact_match_correct"] = int(em)
        rec2["math_verify_correct"] = int(mv)
        return rec2

    if is_jsonl:
        with (
            pred_file.open("r", encoding="utf-8") as fin,
            save_file.open("w", encoding="utf-8") as fout,
        ):
            for line_no, line in enumerate(fin, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Failed to parse JSON on line {line_no} of {pred_file}: {e}"
                    ) from e
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Expected JSON object on line {line_no} of {pred_file}, got {type(obj)}"
                    )
                fout.write(json.dumps(_annotate_record(obj), ensure_ascii=False) + "\n")
        return

    data = json.loads(pred_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"Unsupported JSON structure in {pred_file}: expected JSONL objects or a JSON list"
        )
    out: list[dict[str, Any]] = []
    for idx, obj in enumerate(data):
        if not isinstance(obj, dict):
            raise ValueError(
                f"Expected list of objects in {pred_file}, but element {idx} is {type(obj)}"
            )
        out.append(_annotate_record(obj))
    save_file.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-file", type=Path, required=True)
    parser.add_argument("--pred-key", type=str, default="predicted")
    parser.add_argument("--gold-key", type=str, default="answer")
    parser.add_argument("--question-key", type=str, default="question")
    parser.add_argument(
        "--gold-solution-file",
        type=Path,
        default=None,
        help=(
            "Optional local JSONL/JSON containing gold solutions. If it has keys 'problem' and 'solution', "
            "math_verify uses the full solution by matching on extracted problem text from the prompt. "
            "If it has ids, predictions can also be joined by id/doc_id."
        ),
    )
    parser.add_argument("--gold-problem-key", type=str, default="problem")
    parser.add_argument("--gold-solution-key", type=str, default="solution")
    parser.add_argument(
        "--math-verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute math_verify using `math_verify` (requires extra deps).",
    )
    parser.add_argument(
        "--pred-format",
        type=str,
        default="simple",
        choices=("simple", "lm_eval_samples"),
        help=(
            "Prediction file format. 'simple' expects a per-line dict with a prediction field (default: predicted). "
            "'lm_eval_samples' expects lm-eval-harness log_samples JSONL with nested `resps`."
        ),
    )
    parser.add_argument(
        "--print-mismatches",
        type=int,
        default=0,
        help="Print the first N exact_match mismatches.",
    )
    parser.add_argument(
        "--save-file",
        type=Path,
        default=None,
        help="Optional path to save an annotated copy of the prediction file with per-record 'correct' fields.",
    )
    args = parser.parse_args()

    results, metrics = score_file(
        args.pred_file,
        pred_key=args.pred_key,
        gold_key=args.gold_key,
        question_key=args.question_key,
        compute_math_verify=args.math_verify,
        gold_solution_file=args.gold_solution_file,
        gold_problem_key=args.gold_problem_key,
        gold_solution_key=args.gold_solution_key,
        pred_format=args.pred_format,
    )

    exact_correct = sum(r.exact_match for r in results)
    mv_correct = sum(r.math_verify for r in results)

    print(f"file: {args.pred_file}")
    print(f"total: {int(metrics['n'])}")
    print(f"exact_match_correct: {exact_correct}")
    print(f"math_verify_correct: {mv_correct}")
    print(f"exact_match: {metrics['exact_match']:.6f}")
    print(f"math_verify: {metrics['math_verify']:.6f}")

    if args.print_mismatches:
        shown = 0
        for i, r in enumerate(results):
            if r.exact_match:
                continue
            print("-" * 80)
            print(f"idx: {i}")
            print(f"math_verify: {r.math_verify}")
            print(f"gold_raw: {r.gold_raw!r}")
            print(f"pred_final_raw: {r.pred_final_raw!r}")
            print(f"gold_norm: {r.gold_norm!r}")
            print(f"pred_norm: {r.pred_norm!r}")
            shown += 1
            if shown >= args.print_mismatches:
                break

    if args.save_file is not None:
        _save_annotated_file(
            args.pred_file,
            args.save_file,
            pred_key=args.pred_key,
            gold_key=args.gold_key,
            question_key=args.question_key,
            compute_math_verify=args.math_verify,
            gold_solution_file=args.gold_solution_file,
            gold_problem_key=args.gold_problem_key,
            gold_solution_key=args.gold_solution_key,
            pred_format=args.pred_format,
        )


if __name__ == "__main__":
    main()
