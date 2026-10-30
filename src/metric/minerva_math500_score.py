# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Score Minerva MATH-500 custom decodes (exact_match + math_verify).

This mirrors the intent of lm-evaluation-harness's Minerva Math task metrics:
  - exact_match: normalize a final answer string and check symbolic equivalence.
  - math_verify: use `math_verify.verify` between gold and model output.

It is designed for the custom decoding JSONL format used in this repo:
  {"question": <prompt>, "answer": <gold answer>, "predicted": <model output>}

Example:
  python src/metric/minerva_math500_score.py \
    --pred-file /path/to/minerva_math500_zeroshot_50_latent10_gen8192_temp0.0.json
"""

import argparse
import json
import re
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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
    """Return an identifier for joining predictions to gold.

    Preference:
      - `doc_id` (lm-eval samples JSONL)
      - `id`
    """
    if "doc_id" in rec:
        return rec.get("doc_id")
    if "id" in rec:
        return rec.get("id")
    return None


def _extract_first_resp_from_resps(resps: Any) -> str | None:
    """Extract the first response string from lm-eval `resps`.

    lm-eval samples commonly store `resps` as a nested list: [["..."]]
    but sometimes it can be a flat list: ["..."]
    """
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
    target_keys: tuple[str, ...] = ("target", "a", "answer"),
    solution_key: str = "solution",
) -> tuple[list[dict[str, Any]], dict[int | str, dict[str, Any]]]:
    """Load gold records.

    Returns:
      - gold_list: list of records in file order
      - gold_by_id: mapping from `id` (if present) to record

    Note: We always can join by implicit index (doc_id) using gold_list[i].
    """
    gold_list: list[dict[str, Any]] = []
    gold_by_id: dict[int | str, dict[str, Any]] = {}

    for rec in _iter_json_records(gold_file):
        gold_list.append(rec)
        if id_key in rec:
            rid = rec.get(id_key)
            if rid is not None:
                gold_by_id[rid] = rec

    # sanity: must have at least one target key somewhere
    if gold_list:
        has_any = any(any(k in r for k in target_keys) for r in gold_list)
        if not has_any:
            raise ValueError(
                f"Gold file {gold_file} did not contain any of answer keys {target_keys}."
            )
    return gold_list, gold_by_id


def _get_gold_answer_and_solution(
    gold_rec: dict[str, Any],
    *,
    target_keys: tuple[str, ...] = ("target", "a", "answer"),
    solution_key: str = "solution",
) -> tuple[str, str | None]:
    gold_answer = None
    for k in target_keys:
        if k in gold_rec and gold_rec.get(k) not in (None, ""):
            gold_answer = str(gold_rec.get(k))
            break
    if gold_answer is None:
        gold_answer = "[invalidanswer]"

    gold_solution = gold_rec.get(solution_key)
    if gold_solution in (None, ""):
        gold_solution = None
    else:
        gold_solution = str(gold_solution)
    return gold_answer, gold_solution


def _response_region(text: str) -> str:
    """Best-effort slice of the assistant's response (avoid prompt numbers)."""
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
    """Extract content of the last \boxed{...} (or \fbox{...}) if present."""
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

    # Handle "\\boxed 15" style.
    m = re.match(r"([^\s$<]+)", text[j:])
    if not m:
        return None
    content = m.group(1).strip()
    return content if content else None


def _extract_problem_from_prompt(prompt: str) -> str | None:
    if not prompt:
        return None
    # Common for minerva_math prompts in this repo:
    #   "Problem:\n<problem>\n\nSolution:"
    # plus possible chat template wrappers.
    region = prompt
    # Strip chat wrapper prefix.
    for marker in ("Problem:\n", "\nProblem:\n"):
        idx = region.find(marker)
        if idx != -1:
            region = region[idx + len(marker) :]
            break
    end = region.find("\n\nSolution:")
    if end != -1:
        region = region[:end]
    out = region.strip()
    return out or None


def _load_gold_solutions(
    gold_file: Path,
    *,
    problem_key: str = "problem",
    solution_key: str = "solution",
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


_FINAL_ANSWER_RE = re.compile(
    r"Final Answer: The final answer is(.*?). I hope it is correct\.", re.DOTALL
)


def extract_minerva_final_answer(predicted: str) -> str:
    """Extract a final answer string from the model output.

    Priority:
      1) Minerva-style phrase: "Final Answer: The final answer is ..."
      2) Last \boxed{...}
      3) Last LaTeX math span ($...$, \(...\), \[...\])
    """
    if predicted is None:
        return "[invalidanswer]"

    region = _response_region(predicted)

    # 1) Canonical Minerva pattern (lm-eval-harness expects this)
    # Make sure the sentinel exists to allow a match across end-of-string.
    m = _FINAL_ANSWER_RE.search(region + " I hope it is correct.")
    if m:
        ans = m.group(1).strip()
        return ans if ans else "[invalidanswer]"

    # 2) Last boxed answer
    boxed = _extract_last_boxed_content(region)
    if boxed is not None:
        return boxed

    # 3) Last math span
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
    (r"\ ", ""),
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
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer (Lewkowycz et al., 2022 Appendix D)."""
    if final_answer is None:
        return "[invalidanswer]"

    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


class _timeout:
    def __init__(self, seconds: int = 5, error_message: str = "Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def _is_equiv_latex(pred_norm: str, gold_norm: str) -> bool:
    """Return True if two normalized LaTeX strings are symbolically equivalent."""
    try:
        # Lazy imports so the file can be present even if deps aren't.
        import sympy
        from sympy.parsing.latex import parse_latex

        with _timeout(seconds=5):
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
    except TimeoutError:
        return False
    except Exception:
        return False


def _gold_for_math_verify(gold_answer_raw: str) -> str:
    s = gold_answer_raw.strip() if gold_answer_raw is not None else ""
    if not s:
        return "[invalidanswer]"
    # Help `math_verify.parse` by presenting a boxed final answer.
    if "\\boxed" in s or "\\fbox" in s:
        return s
    # Strip outer $...$ if present.
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1]
    return f"\\boxed{{{s}}}"


_MEMBERSHIP_RE = re.compile(r"^\s*(?:[a-zA-Z]|\\[a-zA-Z]+)\s*\\in\s*(.+?)\s*$")


def _gold_candidates_for_math_verify(gold_answer_raw: str) -> list[str]:
    """Generate one or more gold strings to try with math_verify.

    When the gold is a membership statement like `x\in[-2,7]`, some models output
    just the interval `[-2,7]` (especially when the prompt asks for interval
    notation). In that case we also try verifying against the RHS set.
    """
    s = gold_answer_raw.strip() if gold_answer_raw is not None else ""
    if not s:
        return ["[invalidanswer]"]

    candidates = [_gold_for_math_verify(s)]
    m = _MEMBERSHIP_RE.match(s)
    if m:
        rhs = m.group(1).strip()
        if rhs:
            candidates.append(_gold_for_math_verify(rhs))
    # de-dup while preserving order
    out: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
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

    # Gold source handling:
    # - If `gold_solution_file` is provided, we evaluate only over that gold file,
    #   joining predictions by `doc_id` (index) or by `id`.
    # - Otherwise we evaluate using the per-record gold field in the pred file.
    gold_list: list[dict[str, Any]] | None = None
    gold_by_id: dict[int | str, dict[str, Any]] | None = None
    gold_solutions: dict[str, str] | None = None
    if gold_solution_file is not None:
        # First try loading as generic gold answers file (id/target/a).
        try:
            gold_list, gold_by_id = _load_gold_answers(gold_solution_file)
        except Exception:
            # Fallback: interpret as (problem -> solution) mapping file.
            gold_solutions = _load_gold_solutions(
                gold_solution_file,
                problem_key=gold_problem_key,
                solution_key=gold_solution_key,
            )

    if compute_math_verify:
        try:
            from math_verify import parse, verify
        except Exception as e:
            raise RuntimeError(
                "Missing dependency for math_verify. Install: "
                "`pip install sympy antlr4-python3-runtime==4.11 math-verify`"
            ) from e

    exact_sum = 0
    mv_sum = 0

    # Build prediction map if we are evaluating from a gold file.
    pred_by_join_id: dict[int | str, tuple[str, str | None]] | None = None
    if gold_solution_file is not None:
        pred_by_join_id = {}
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

        # Iterate gold in file order and score only where a prediction exists.
        assert gold_list is not None
        for doc_id, gold_rec in enumerate(gold_list):
            # Preferred join: gold `id` (dataset id) against prediction `doc_id`.
            # Fallback join: positional index within the provided gold file.
            pred_raw: str | None = None
            pred_doc_solution: str | None = None
            gold_id = gold_rec.get("id")
            if gold_id is not None and gold_id in pred_by_join_id:
                pred_raw, pred_doc_solution = pred_by_join_id[gold_id]
            elif doc_id in pred_by_join_id:
                pred_raw, pred_doc_solution = pred_by_join_id[doc_id]
            if pred_raw is None:
                continue

            gold_answer, gold_solution_used = _get_gold_answer_and_solution(gold_rec)
            gold_raw = gold_answer

            pred_final_raw = extract_minerva_final_answer(pred_raw)
            pred_norm = normalize_final_answer(pred_final_raw)
            gold_norm = normalize_final_answer(gold_raw)

            em = 1 if _is_equiv_latex(pred_norm, gold_norm) else 0
            exact_sum += em

            mv = 0
            if compute_math_verify:
                # Prefer: explicit gold solution from gold file.
                # Fallback (lm-eval samples): per-example `doc.solution` from the samples file.
                # Final fallback: boxed gold answer (+ membership RHS fallback).
                gold_candidates: list[str]
                if gold_solution_used is not None:
                    gold_candidates = [gold_solution_used]
                elif pred_doc_solution is not None:
                    gold_candidates = [pred_doc_solution]
                else:
                    gold_candidates = _gold_candidates_for_math_verify(gold_raw)

                for gold_for_mv in gold_candidates:
                    try:
                        if verify(gold=parse(gold_for_mv), target=parse(pred_raw)):
                            mv = 1
                            break
                    except Exception:
                        continue
            mv_sum += mv

            results.append(
                ExampleResult(
                    gold_raw=gold_raw,
                    pred_raw=pred_raw,
                    gold_norm=gold_norm,
                    gold_solution_used=gold_solution_used,
                    pred_final_raw=pred_final_raw,
                    pred_norm=pred_norm,
                    exact_match=em,
                    math_verify=mv,
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
    for rec in _iter_json_records(pred_file):
        if pred_format == "lm_eval_samples":
            resp = _extract_first_resp_from_resps(rec.get("resps"))
            if resp is None:
                continue
            pred_raw = resp
        else:
            if pred_key not in rec:
                raise KeyError(
                    f"Record missing prediction key '{pred_key}', got keys={list(rec.keys())}"
                )
            pred_raw = str(rec[pred_key])

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

        pred_final_raw = extract_minerva_final_answer(pred_raw)
        pred_norm = normalize_final_answer(pred_final_raw)
        gold_norm = normalize_final_answer(gold_raw)

        em = 1 if _is_equiv_latex(pred_norm, gold_norm) else 0
        exact_sum += em

        mv = 0
        if compute_math_verify:
            gold_candidates: list[str]
            if gold_solution_used is not None:
                gold_candidates = [gold_solution_used]
            else:
                gold_candidates = _gold_candidates_for_math_verify(gold_raw)

            for gold_for_mv in gold_candidates:
                try:
                    if verify(gold=parse(gold_for_mv), target=parse(pred_raw)):
                        mv = 1
                        break
                except Exception:
                    continue
        mv_sum += mv

        results.append(
            ExampleResult(
                gold_raw=gold_raw,
                pred_raw=pred_raw if question_key not in rec else pred_raw,
                gold_norm=gold_norm,
                gold_solution_used=gold_solution_used,
                pred_final_raw=pred_final_raw,
                pred_norm=pred_norm,
                exact_match=em,
                math_verify=mv,
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
    try:
        import sympy
        from sympy.parsing.latex import parse_latex
    except Exception:
        pass
    if compute_math_verify:
        try:
            from math_verify import parse, verify
        except Exception:
            pass

    gold_list = None
    gold_by_id = None
    gold_solutions = None
    if gold_solution_file is not None:
        try:
            gold_list, gold_by_id = _load_gold_answers(gold_solution_file)
        except Exception:
            gold_solutions = _load_gold_solutions(
                gold_solution_file,
                problem_key=gold_problem_key,
                solution_key=gold_solution_key,
            )

    pred_records = list(_iter_json_records(pred_file))

    for doc_id, rec in enumerate(pred_records):
        join_id = _get_doc_identifier(rec)

        pred_raw = None
        if pred_format == "lm_eval_samples":
            resp = _extract_first_resp_from_resps(rec.get("resps"))
            if resp is not None:
                pred_raw = resp
        else:
            if pred_key in rec:
                pred_raw = str(rec[pred_key])

        gold_raw = None
        gold_solution_used = None

        if gold_list is not None:
            gold_rec = None
            if gold_by_id is not None and join_id is not None and join_id in gold_by_id:
                gold_rec = gold_by_id[join_id]
            elif (
                join_id is not None
                and isinstance(join_id, int)
                and 0 <= join_id < len(gold_list)
            ):
                gold_rec = gold_list[join_id]
            elif 0 <= doc_id < len(gold_list):
                gold_rec = gold_list[doc_id]

            if gold_rec is not None:
                g_ans, g_sol = _get_gold_answer_and_solution(gold_rec)
                gold_raw = g_ans
                gold_solution_used = g_sol
        else:
            if gold_key in rec:
                gold_raw = str(rec[gold_key])
                if gold_solutions is not None and question_key in rec:
                    prob = _extract_problem_from_prompt(str(rec[question_key]))
                    if prob is not None and prob in gold_solutions:
                        gold_solution_used = gold_solutions[prob]
                        boxed = _last_boxed_only_string(gold_solution_used)
                        if boxed:
                            gold_raw = normalize_final_answer(_remove_boxed(boxed))

        rec["exact_match_correct"] = None
        rec["math_verify_correct"] = None

        if pred_raw is not None and gold_raw is not None:
            pred_final_raw = extract_minerva_final_answer(pred_raw)
            pred_norm = normalize_final_answer(pred_final_raw)
            gold_norm = normalize_final_answer(gold_raw)

            em = 1 if _is_equiv_latex(pred_norm, gold_norm) else 0
            rec["exact_match_correct"] = em

            mv_val = 0
            if compute_math_verify:
                pred_doc_solution = _extract_doc_solution_from_sample(rec)
                g_cands = []
                if gold_solution_used is not None:
                    g_cands = [gold_solution_used]
                elif pred_doc_solution is not None:
                    g_cands = [pred_doc_solution]
                else:
                    g_cands = _gold_candidates_for_math_verify(gold_raw)

                for c in g_cands:
                    try:
                        if verify(gold=parse(c), target=parse(pred_raw)):
                            mv_val = 1
                            break
                    except Exception:
                        pass
                rec["math_verify_correct"] = mv_val

    is_jsonl = _detect_is_jsonl(pred_file)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    if is_jsonl:
        with save_file.open("w", encoding="utf-8") as f:
            for rec in pred_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        save_file.write_text(
            json.dumps(pred_records, ensure_ascii=False), encoding="utf-8"
        )


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
            "Optional local JSONL/JSON containing gold solutions, with keys 'problem' and 'solution'. "
            "If provided, math_verify uses the full solution (lm-eval-harness-style)."
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
