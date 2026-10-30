# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Score AMC23 custom decodes (numeric exact_match + math_verify).

Designed for the decoding JSONL format used in this repo:
  {"question": <prompt>, "answer": <gold answer>, "predicted": <model output>}

Also supports scoring lm-eval-harness `samples_*.jsonl` logs via `--pred-format
lm_eval_samples` (reads predictions from `resps`) and optional joining against a
gold file by id.

Example:
  python src/metric/amc23_score.py \
    --pred-file /path/to/amc23_latent10_gen8192_temp0.0.json

Dependencies:
  - exact_match: no extra deps
  - math_verify: `pip install sympy antlr4-python3-runtime==4.11 math-verify`
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

# Mirrors `gsm8k_zeroshot.yaml` exact_match normalizations.
DEFAULT_REGEXES_TO_IGNORE: tuple[str, ...] = (
    ",",
    r"\$",
    r"(?s).*#### ",
    r"\.$",
)


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


_STRICT_HASH_RE = re.compile(r"####\s*(\-?[0-9\.\,]+)")
_ALT_HASH_RE = re.compile(r"###\s*(\-?[0-9\.\,]+)")

# Mirrors lm-eval-harness flexible-extract filter:
# regex_pattern: "(-?[$0-9.,]{2,})|(-?[0-9]+)", group_select: -1
_FLEX_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")

_NUMERIC_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def _parse_numeric(text: str) -> int | float | None:
    s = text.strip()
    if not s:
        return None

    if s.endswith(".") and s.count(".") == 1:
        s2 = s[:-1]
        if s2 and _NUMERIC_RE.match(s2):
            try:
                return int(s2) if "." not in s2 else float(s2)
            except ValueError:
                return None

    if not _NUMERIC_RE.match(s):
        return None
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return None


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
    if "doc_id" in rec:
        return rec.get("doc_id")
    if "id" in rec:
        return rec.get("id")
    return None


def _extract_first_resp_from_resps(resps: Any) -> str | None:
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


def _load_gold_records(
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


def _get_gold_answer_and_solution(gold_rec: dict[str, Any]) -> tuple[str, str | None]:
    gold_answer = ""
    for k in ("answer", "a", "target"):
        if k in gold_rec and gold_rec.get(k) not in (None, ""):
            gold_answer = str(gold_rec.get(k))
            break

    gold_solution_used: str | None = None
    sol = gold_rec.get("solution")
    if sol not in (None, ""):
        sol_s = str(sol)
        # AMC23 solutions in this repo are URLs; avoid feeding them to math_verify.
        if not sol_s.strip().lower().startswith("http"):
            gold_solution_used = sol_s

    return gold_answer, gold_solution_used


def _normalize_exact_match(
    text: str,
    *,
    ignore_case: bool = True,
    regexes_to_ignore: tuple[str, ...] = DEFAULT_REGEXES_TO_IGNORE,
) -> str:
    out = text
    for rgx in regexes_to_ignore:
        out = re.sub(rgx, "", out)
    if ignore_case:
        out = out.lower()
    return out


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


def _extract_boxed(text: str) -> str | None:
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


def _extract_by_hash_marker(text: str) -> str | None:
    matches = _STRICT_HASH_RE.findall(text)
    if matches:
        return matches[-1].strip()
    matches = _ALT_HASH_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return None


def _extract_flexible(text: str) -> str | None:
    matches = _FLEX_RE.findall(text)
    if not matches:
        return None
    last = matches[-1]
    if isinstance(last, tuple):
        chosen = next((m for m in last if m), "")
    else:
        chosen = last
    chosen = chosen.strip()
    return chosen if chosen else None


def _extract_last_math_span(text: str) -> str | None:
    if not text:
        return None
    spans: list[str] = []
    spans.extend(re.findall(r"\$(.*?)\$", text, flags=re.DOTALL))
    spans.extend(re.findall(r"\\\((.*?)\\\)", text, flags=re.DOTALL))
    spans.extend(re.findall(r"\\\[(.*?)\\\]", text, flags=re.DOTALL))
    if not spans:
        return None
    last = spans[-1].strip()
    return last if last else None


def extract_amc23_final_answer(text: str, *, mode: str = "flexible-extract") -> str:
    """GSM8K-style flexible numeric extraction with extra AMC-friendly fallbacks."""
    if text is None:
        return "[invalid]"

    region = _response_region(text)

    boxed = _extract_boxed(region)
    if boxed is not None:
        # Prefer the raw boxed content so that symbolic math_verify can evaluate
        # fractions / expressions correctly.  Only fall back to numeric extraction
        # if the boxed string is clearly non-parseable (e.g. sentence text).
        boxed_num = _extract_flexible(boxed)
        if boxed_num is not None and boxed_num == boxed.strip():
            # Boxed content IS a plain number — return it directly.
            return boxed_num
        # Boxed content is an expression (e.g. \frac{3}{4}); return it raw so
        # _pred_candidates can pass it to math_verify.  _extract_flexible is
        # still added as a separate candidate there.
        return boxed

    marked = _extract_by_hash_marker(region)
    if marked is not None:
        return marked

    lower_region = region.lower()

    final_idx = lower_region.rfind("final answer")
    if final_idx != -1:
        after = region[final_idx + len("final answer") :]
        flex = _extract_flexible(after)
        if flex is not None:
            return flex

    answer_markers = []
    answer_markers.extend(re.finditer(r"\banswer\b\s*[:\-]", lower_region))
    answer_markers.extend(re.finditer(r"\banswer\b\s+is\s+", lower_region))
    if answer_markers:
        last = max(answer_markers, key=lambda m: m.start())
        after = region[last.end() :]
        flex = _extract_flexible(after)
        if flex is not None:
            return flex

    if mode == "strict-match":
        return "[invalid]"

    flex = _extract_flexible(region)
    return flex if flex is not None else "[invalid]"


def _gold_for_math_verify(gold_answer_raw: str) -> str:
    s = gold_answer_raw.strip() if gold_answer_raw is not None else ""
    if not s:
        return "[invalidanswer]"
    if "\\boxed" in s or "\\fbox" in s:
        return s
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1]
    return f"\\boxed{{{s}}}"


def _pred_candidates(pred_raw: str, *, extraction_mode: str) -> list[str]:
    region = _response_region(pred_raw)
    cands: list[str] = []

    primary = extract_amc23_final_answer(pred_raw, mode=extraction_mode)
    if primary and primary not in ("[invalid]", "[invalidanswer]"):
        cands.append(primary)

    boxed = _extract_boxed(region)
    if boxed:
        cands.append(boxed)

    marked = _extract_by_hash_marker(region)
    if marked:
        cands.append(marked)

    last_math = _extract_last_math_span(region)
    if last_math:
        cands.append(last_math)

    flex = _extract_flexible(region)
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


def _exact_match_from_candidates(
    pred_raw: str,
    gold_raw: str,
    *,
    extraction_mode: str,
) -> tuple[int, str, str, str]:
    """Return (exact_match, pred_final_raw, pred_norm, gold_norm)."""
    gold_norm = _normalize_exact_match(gold_raw)
    gold_num = _parse_numeric(gold_norm)

    candidates = _pred_candidates(pred_raw, extraction_mode=extraction_mode)
    if not candidates:
        return 0, "[invalid]", "[invalid]", gold_norm

    pred_final_raw = candidates[0]
    pred_norm = _normalize_exact_match(pred_final_raw)

    for cand in candidates:
        cand_norm = _normalize_exact_match(cand)
        cand_num = _parse_numeric(cand_norm)
        if (gold_num is not None) and (cand_num is not None):
            # If either side is float, compare in float space with tolerance.
            if isinstance(gold_num, float) or isinstance(cand_num, float):
                if abs(float(cand_num) - float(gold_num)) < 1e-9:
                    return 1, pred_final_raw, pred_norm, gold_norm
            else:
                if int(cand_num) == int(gold_num):
                    return 1, pred_final_raw, pred_norm, gold_norm
        else:
            if cand_norm == gold_norm and cand not in ("[invalid]", "[invalidanswer]"):
                return 1, pred_final_raw, pred_norm, gold_norm

    return 0, pred_final_raw, pred_norm, gold_norm


def _math_verify_one(
    pred_raw: str,
    gold_raw: str,
    *,
    gold_solution_used: str | None,
    extraction_mode: str,
    parse: Any,
    verify: Any,
) -> int:
    gold_candidates = (
        [gold_solution_used]
        if gold_solution_used
        else [_gold_for_math_verify(gold_raw)]
    )
    target_candidates = [pred_raw]
    target_candidates.extend(
        _pred_candidates(pred_raw, extraction_mode=extraction_mode)
    )

    for gold_for_mv in gold_candidates:
        try:
            g_parsed = parse(gold_for_mv)
        except Exception:
            continue
        for target_str in target_candidates:
            try:
                if verify(gold=g_parsed, target=parse(target_str)):
                    return 1
            except Exception:
                continue
    return 0


def score_file(
    pred_file: Path,
    *,
    pred_key: str = "predicted",
    gold_key: str = "answer",
    question_key: str = "question",
    extraction_mode: str = "flexible-extract",
    compute_math_verify: bool = True,
    gold_solution_file: Path | None = None,
    pred_format: str = "simple",
) -> tuple[list[ExampleResult], dict[str, float]]:
    results: list[ExampleResult] = []

    if compute_math_verify:
        try:
            from math_verify import parse, verify  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependency for math_verify. Install: `pip install sympy antlr4-python3-runtime==4.11 math-verify`"
            ) from e
    else:
        parse = verify = None  # type: ignore

    def _score_one(
        *, pred_raw: str, gold_raw: str, gold_solution_used: str | None
    ) -> ExampleResult:
        em, pred_final_raw, pred_norm, gold_norm = _exact_match_from_candidates(
            pred_raw, gold_raw, extraction_mode=extraction_mode
        )
        mv = (
            _math_verify_one(
                pred_raw,
                gold_raw,
                gold_solution_used=gold_solution_used,
                extraction_mode=extraction_mode,
                parse=parse,
                verify=verify,
            )
            if compute_math_verify
            else 0
        )
        return ExampleResult(
            gold_raw=gold_raw,
            pred_raw=pred_raw,
            gold_norm=gold_norm,
            gold_solution_used=gold_solution_used,
            pred_final_raw=pred_final_raw,
            pred_norm=pred_norm,
            exact_match=em,
            math_verify=mv,
        )

    exact_sum = 0
    mv_sum = 0

    if gold_solution_file is not None:
        gold_list, _gold_by_id = _load_gold_records(gold_solution_file)

        pred_by_join_id: dict[int | str, str] = {}
        for rec in _iter_json_records(pred_file):
            join_id = _get_doc_identifier(rec)
            if join_id is None:
                continue

            if pred_format == "lm_eval_samples":
                resp = _extract_first_resp_from_resps(rec.get("resps"))
                if resp is None:
                    continue
                pred_by_join_id[join_id] = resp
            else:
                if pred_key not in rec:
                    continue
                pred_by_join_id[join_id] = str(rec[pred_key])

        # If predictions have no join identifiers, fall back to per-record scoring
        # (using the embedded gold `answer` field in the prediction file).
        if not pred_by_join_id:
            gold_list = None

        if gold_list is not None:
            for doc_id, gold_rec in tqdm(enumerate(gold_list), total=len(gold_list)):
                pred_raw = None
                gold_id = gold_rec.get("id")
                if gold_id is not None and gold_id in pred_by_join_id:
                    pred_raw = pred_by_join_id[gold_id]
                elif doc_id in pred_by_join_id:
                    pred_raw = pred_by_join_id[doc_id]
                if pred_raw is None:
                    continue

                gold_answer, gold_solution_used = _get_gold_answer_and_solution(
                    gold_rec
                )
                r = _score_one(
                    pred_raw=str(pred_raw),
                    gold_raw=str(gold_answer),
                    gold_solution_used=gold_solution_used,
                )
                exact_sum += r.exact_match
                mv_sum += r.math_verify
                results.append(r)

            n = len(results)
            metrics = {
                "exact_match": (exact_sum / n) if n else 0.0,
                "math_verify": (mv_sum / n) if n else 0.0,
                "n": float(n),
            }
            return results, metrics

    for rec in tqdm(_iter_json_records(pred_file)):
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

        r = _score_one(pred_raw=pred_raw, gold_raw=gold_raw, gold_solution_used=None)
        exact_sum += r.exact_match
        mv_sum += r.math_verify
        results.append(r)

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
    extraction_mode: str,
    compute_math_verify: bool,
    gold_solution_file: Path | None,
    pred_format: str,
) -> None:
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
    if gold_solution_file is not None:
        gold_list, gold_by_id = _load_gold_records(gold_solution_file)

    is_jsonl = _detect_is_jsonl(pred_file)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    def _annotate_record(rec: dict[str, Any]) -> dict[str, Any]:
        pred_text: str | None
        if pred_format == "lm_eval_samples":
            pred_text = _extract_first_resp_from_resps(rec.get("resps"))
        else:
            pred_text = str(rec.get(pred_key)) if pred_key in rec else None

        gold_raw: str | None = None
        gold_solution_used: str | None = None
        if gold_solution_file is None:
            if gold_key in rec:
                gold_raw = str(rec.get(gold_key))
        else:
            join_id = _get_doc_identifier(rec)
            gold_rec = None
            if gold_by_id is not None and join_id is not None and join_id in gold_by_id:
                gold_rec = gold_by_id[join_id]
            elif (
                gold_list is not None
                and isinstance(join_id, int)
                and 0 <= join_id < len(gold_list)
            ):
                gold_rec = gold_list[join_id]
            if gold_rec is not None:
                gold_raw, gold_solution_used = _get_gold_answer_and_solution(gold_rec)
            elif gold_key in rec:
                # If we can't join to gold (e.g., custom decodes have no ids),
                # fall back to the embedded gold answer in the prediction file.
                gold_raw = str(rec.get(gold_key))

        rec2 = dict(rec)
        if pred_text is None or gold_raw is None:
            rec2["exact_match_correct"] = None
            rec2["math_verify_correct"] = None
            return rec2

        em, pred_final_raw, pred_norm, gold_norm = _exact_match_from_candidates(
            pred_text, gold_raw, extraction_mode=extraction_mode
        )
        mv = (
            _math_verify_one(
                pred_text,
                gold_raw,
                gold_solution_used=gold_solution_used,
                extraction_mode=extraction_mode,
                parse=parse,
                verify=verify,
            )
            if compute_math_verify
            else 0
        )

        rec2["pred_final_raw"] = pred_final_raw
        rec2["pred_norm"] = pred_norm
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
            "Optional gold JSONL/JSON. If provided, evaluation runs over this file and "
            "predictions are joined by id (gold `id` to pred `doc_id`, with a positional fallback)."
        ),
    )
    parser.add_argument(
        "--math-verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute math_verify using `math_verify` (requires extra deps).",
    )
    parser.add_argument(
        "--extraction-mode",
        type=str,
        default="flexible-extract",
        choices=("flexible-extract", "strict-match"),
        help="Answer extraction mode for exact_match.",
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
        extraction_mode=args.extraction_mode,
        compute_math_verify=args.math_verify,
        gold_solution_file=args.gold_solution_file,
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
            extraction_mode=args.extraction_mode,
            compute_math_verify=args.math_verify,
            gold_solution_file=args.gold_solution_file,
            pred_format=args.pred_format,
        )


if __name__ == "__main__":
    main()
