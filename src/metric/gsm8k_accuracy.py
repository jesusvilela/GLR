# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Compute GSM8K exact-match accuracy from custom decoding outputs.

This script is intended to match lm-evaluation-harness behavior for GSM8K
(`gsm8k_zeroshot.yaml`) in the common case:
  - Extraction: "flexible-extract" (last numeric-looking match), but with
    support for final-answer markers your runs may emit: `####`, `###`, and
    `\\boxed{...}`.
  - Scoring: exact match after applying the same regex normalizations as the
    harness config (ignore commas, '$', leading text up to '#### ', and a
    trailing period), plus optional case folding.

Input format:
  - JSONL (one JSON object per line) or a single JSON list.
  - Each record must contain prediction and gold fields (defaults:
    `predicted` and `answer`).

Additionally, this script can score lm-eval-harness `samples_*.jsonl` logs by
reading predictions from `resps` and joining to an external gold file by id.

Example:
  python src/metric/gsm8k_accuracy.py \
    --pred-file /path/to/gsm8k_zeroshot_100_latent100_gen8192_temp0.0.json
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from tqdm import tqdm

# Mirrors `gsm8k_zeroshot.yaml` metric config.
DEFAULT_REGEXES_TO_IGNORE: tuple[str, ...] = (
    ",",
    r"\$",
    r"(?s).*#### ",
    r"\.$",
)


_STRICT_HASH_RE = re.compile(r"####\s*(\-?[0-9\.\,]+)")
_ALT_HASH_RE = re.compile(r"###\s*(\-?[0-9\.\,]+)")

# Mirrors lm-eval-harness flexible-extract filter:
# regex_pattern: "(-?[$0-9.,]{2,})|(-?[0-9]+)", group_select: -1
_FLEX_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


@dataclass(frozen=True)
class ExampleResult:
    gold_extracted: str
    pred_extracted: str
    gold_norm: str
    pred_norm: str
    correct: bool
    invalid_pred: bool


_NUMERIC_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def _parse_numeric(text: str) -> int | float | None:
    """Parse a numeric string after normalization.

    - Returns int when no decimal point is present.
    - Returns float when a decimal point is present.
    - Returns None if it cannot be parsed as a simple number.
    """
    s = text.strip()
    if not s:
        return None

    # Be tolerant to a dangling decimal point (e.g., "14.")
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
    """Yield dict records from a JSONL file or JSON list."""
    # Try JSONL first (common for decoding dumps).
    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            return
        # If the file is JSONL, the first line should parse as an object.
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

    # Otherwise treat as a single JSON value (commonly a list).
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


def _get_gold_solution_text(gold_rec: dict[str, Any]) -> str:
    """Choose the gold solution/answer text from a gold record.

    For GSM8K, both `target` and `a` commonly contain the full solution with
    the final answer marked as `#### <answer>`.
    """
    for k in ("target", "a", "answer"):
        if k in gold_rec and gold_rec.get(k) not in (None, ""):
            return str(gold_rec.get(k))
    # Fall back to empty string; downstream extraction will mark invalid.
    return ""


def _normalize_exact_match(
    text: str,
    *,
    ignore_case: bool = True,
    regexes_to_ignore: tuple[str, ...] = DEFAULT_REGEXES_TO_IGNORE,
) -> str:
    """Apply lm-eval-harness `exact_match` normalizations (GSM8K config).

    Note: `exact_match_hf_evaluate` does not strip whitespace; we keep that
    behavior to stay aligned with the harness.
    """

    out = text
    for rgx in regexes_to_ignore:
        out = re.sub(rgx, "", out)
    if ignore_case:
        out = out.lower()
    return out


def _response_region(text: str) -> str:
    """Best-effort slice of the assistant's response (avoid prompt numbers)."""
    if text is None:
        return ""

    # Common chat templates.
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
    """Return (content, next_index) for a {...} block starting at left_brace_idx."""
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
    """Extract the content of the last \boxed{...} (or \fbox{...}) if present."""
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

    # \boxed{...}
    if j < len(text) and text[j] == "{":
        parsed = _extract_balanced_braces(text, j)
        if parsed is None:
            return None
        content, _ = parsed
        content = content.strip()
        return content if content else None

    # \boxed 15 (space form)
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
    # group_select = -1 (take last match), then pick first non-empty group
    last = matches[-1]
    if isinstance(last, tuple):
        chosen = next((m for m in last if m), "")
    else:
        chosen = last
    chosen = chosen.strip()
    return chosen if chosen else None


def extract_gsm8k_final_answer(text: str, *, mode: str = "flexible-extract") -> str:
    """Extract a final answer string from a model output or gold solution.

    mode:
      - "strict-match": requires a `####`/`###` or `\\boxed` marker (else invalid)
      - "flexible-extract": prefer markers, else last numeric-like match
    """
    if text is None:
        return "[invalid]"

    region = _response_region(text)

    boxed = _extract_boxed(region)
    if boxed is not None:
        # To stay consistent with lm-eval-harness flexible-extract behavior,
        # prefer pulling a numeric-like substring from the boxed content.
        boxed_num = _extract_flexible(boxed)
        return boxed_num if boxed_num is not None else boxed

    marked = _extract_by_hash_marker(region)
    if marked is not None:
        return marked

    # Common patterns in free-form generations.
    # We slice from the last marker occurrence to avoid earlier numbers.
    lower_region = region.lower()

    final_idx = lower_region.rfind("final answer")
    if final_idx != -1:
        after = region[final_idx + len("final answer") :]
        flex = _extract_flexible(after)
        if flex is not None:
            return flex

    # "Answer: 15" or "Answer is 15"
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


def score_file(
    pred_file: Path,
    *,
    pred_key: str = "predicted",
    gold_key: str = "answer",
    extraction_mode: str = "flexible-extract",
    ignore_case: bool = True,
    regexes_to_ignore: tuple[str, ...] = DEFAULT_REGEXES_TO_IGNORE,
    gold_solution_file: Path | None = None,
    pred_format: str = "simple",
) -> tuple[list[ExampleResult], float]:
    results: list[ExampleResult] = []
    correct = 0

    def _score_one(*, pred_text: str, gold_text: str) -> ExampleResult:
        nonlocal correct

        pred_extracted = extract_gsm8k_final_answer(pred_text, mode=extraction_mode)
        # Gold in lm-eval-harness GSM8K is the full solution string with "#### <answer>".
        # The harness then relies on `regexes_to_ignore` (not an extraction filter)
        # to drop everything before the final "#### ".
        gold_extracted = extract_gsm8k_final_answer(gold_text, mode="flexible-extract")

        pred_norm = _normalize_exact_match(
            pred_extracted, ignore_case=ignore_case, regexes_to_ignore=regexes_to_ignore
        )
        gold_norm = _normalize_exact_match(
            gold_text, ignore_case=ignore_case, regexes_to_ignore=regexes_to_ignore
        )

        gold_num = _parse_numeric(gold_norm)
        pred_num = _parse_numeric(pred_norm)

        invalid_pred = pred_extracted == "[invalid]" or pred_num is None

        if (gold_num is not None) and (pred_num is not None):
            # Gold is always an integer for GSM8K, but allow float preds.
            if isinstance(pred_num, float):
                is_correct = abs(pred_num - float(gold_num)) < 1e-9
            else:
                is_correct = pred_num == int(gold_num)
        else:
            # Fallback to string equality if parsing fails (should be rare).
            is_correct = (not invalid_pred) and (pred_norm == gold_norm)

        if is_correct:
            correct += 1

        return ExampleResult(
            gold_extracted=gold_extracted,
            pred_extracted=pred_extracted,
            gold_norm=gold_norm,
            pred_norm=pred_norm,
            correct=is_correct,
            invalid_pred=invalid_pred,
        )

    # Gold-driven evaluation (join samples -> gold).
    if gold_solution_file is not None:
        gold_list, _gold_by_id = _load_gold_records(gold_solution_file)

        pred_by_id: dict[int | str, str] = {}
        for rec in _iter_json_records(pred_file):
            join_id = _get_doc_identifier(rec)
            if join_id is None:
                continue

            if pred_format == "lm_eval_samples":
                resp = _extract_first_resp_from_resps(rec.get("resps"))
                if resp is None:
                    continue
                pred_by_id[join_id] = resp
            else:
                if pred_key not in rec:
                    continue
                pred_by_id[join_id] = str(rec[pred_key])

        # Evaluate only over the provided gold file (skip missing predictions).
        for idx, gold_rec in tqdm(enumerate(gold_list)):
            gold_id = gold_rec.get("id")
            pred_text = None
            if gold_id is not None and gold_id in pred_by_id:
                pred_text = pred_by_id[gold_id]
            elif idx in pred_by_id:
                # Fallback when gold has no ids and pred ids are positional.
                pred_text = pred_by_id[idx]
            if pred_text is None:
                continue

            gold_text = _get_gold_solution_text(gold_rec)
            results.append(
                _score_one(pred_text=str(pred_text), gold_text=str(gold_text))
            )

        acc = correct / len(results) if results else 0.0
        return results, acc

    # Default behavior: score directly from pred file records.
    for rec in tqdm(_iter_json_records(pred_file)):
        if pred_format == "lm_eval_samples":
            pred_text = _extract_first_resp_from_resps(rec.get("resps"))
            if pred_text is None:
                continue
        else:
            if pred_key not in rec:
                raise KeyError(
                    f"Missing prediction key '{pred_key}' in record keys={list(rec.keys())}"
                )
            pred_text = str(rec[pred_key])

        if gold_key not in rec:
            raise KeyError(
                f"Missing gold key '{gold_key}' in record keys={list(rec.keys())}"
            )
        gold_text = str(rec[gold_key])

        results.append(_score_one(pred_text=str(pred_text), gold_text=str(gold_text)))

    acc = correct / len(results) if results else 0.0
    return results, acc


def _save_annotated_file(
    pred_file: Path,
    save_file: Path,
    *,
    pred_key: str,
    gold_key: str,
    extraction_mode: str,
    ignore_case: bool,
    regexes_to_ignore: tuple[str, ...],
    gold_solution_file: Path | None,
    pred_format: str,
) -> None:
    gold_list: list[dict[str, Any]] | None = None
    gold_by_id: dict[int | str, dict[str, Any]] | None = None
    if gold_solution_file is not None:
        gold_list, gold_by_id = _load_gold_records(gold_solution_file)

    def _score_correct(pred_text: str, gold_text: str) -> bool:
        pred_extracted = extract_gsm8k_final_answer(pred_text, mode=extraction_mode)
        pred_norm = _normalize_exact_match(
            pred_extracted, ignore_case=ignore_case, regexes_to_ignore=regexes_to_ignore
        )
        gold_norm = _normalize_exact_match(
            gold_text, ignore_case=ignore_case, regexes_to_ignore=regexes_to_ignore
        )

        gold_num = _parse_numeric(gold_norm)
        pred_num = _parse_numeric(pred_norm)
        invalid_pred = pred_extracted == "[invalid]" or pred_num is None
        if invalid_pred:
            return False

        if (gold_num is not None) and (pred_num is not None):
            if isinstance(pred_num, float):
                return abs(pred_num - float(gold_num)) < 1e-9
            return pred_num == int(gold_num)

        return pred_norm == gold_norm

    is_jsonl = _detect_is_jsonl(pred_file)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    def _annotate_record(rec: dict[str, Any]) -> dict[str, Any]:
        pred_text: str | None
        if pred_format == "lm_eval_samples":
            pred_text = _extract_first_resp_from_resps(rec.get("resps"))
        else:
            pred_text = str(rec.get(pred_key)) if pred_key in rec else None

        correct_val: bool | None = None
        if pred_text is not None:
            if gold_solution_file is None:
                if gold_key in rec:
                    correct_val = _score_correct(pred_text, str(rec.get(gold_key)))
            else:
                join_id = _get_doc_identifier(rec)
                gold_rec = None
                if (
                    gold_by_id is not None
                    and join_id is not None
                    and join_id in gold_by_id
                ):
                    gold_rec = gold_by_id[join_id]
                elif (
                    gold_list is not None
                    and isinstance(join_id, int)
                    and 0 <= join_id < len(gold_list)
                ):
                    gold_rec = gold_list[join_id]
                if gold_rec is not None:
                    gold_text = _get_gold_solution_text(gold_rec)
                    correct_val = _score_correct(pred_text, str(gold_text))

        rec2 = dict(rec)
        rec2["correct"] = correct_val
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
    parser.add_argument(
        "--gold-solution-file",
        type=Path,
        default=None,
        help=(
            "Optional gold test-set JSONL/JSON. If provided, evaluation runs over this file and "
            "predictions are joined by id (gold `id` to pred `doc_id`, with a positional fallback)."
        ),
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
        "--extraction-mode",
        type=str,
        default="flexible-extract",
        choices=("flexible-extract", "strict-match"),
    )
    parser.add_argument(
        "--ignore-case",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match lm-eval-harness gsm8k_zeroshot.yaml (default: true).",
    )
    parser.add_argument(
        "--print-mismatches",
        type=int,
        default=0,
        help="Print the first N mismatches (0 disables).",
    )
    parser.add_argument(
        "--save-file",
        type=Path,
        default=None,
        help="Optional path to save an annotated copy of the prediction file with per-record 'correct' boolean.",
    )
    args = parser.parse_args()

    results, acc = score_file(
        args.pred_file,
        pred_key=args.pred_key,
        gold_key=args.gold_key,
        extraction_mode=args.extraction_mode,
        ignore_case=args.ignore_case,
        gold_solution_file=args.gold_solution_file,
        pred_format=args.pred_format,
    )

    total = len(results)
    correct = sum(1 for r in results if r.correct)
    invalid = sum(1 for r in results if r.invalid_pred)

    print(f"file: {args.pred_file}")
    print(f"total: {total}")
    print(f"correct: {correct}")
    print(f"invalid_pred: {invalid}")
    print(f"accuracy_exact_match: {acc:.6f}")

    if args.print_mismatches:
        shown = 0
        for i, r in enumerate(results):
            if r.correct:
                continue
            print("-" * 80)
            print(f"idx: {i}")
            print(f"gold_extracted: {r.gold_extracted!r} -> {r.gold_norm!r}")
            print(f"pred_extracted: {r.pred_extracted!r} -> {r.pred_norm!r}")
            shown += 1
            if shown >= args.print_mismatches:
                break

    if args.save_file is not None:
        _save_annotated_file(
            args.pred_file,
            args.save_file,
            pred_key=args.pred_key,
            gold_key=args.gold_key,
            extraction_mode=args.extraction_mode,
            ignore_case=args.ignore_case,
            regexes_to_ignore=DEFAULT_REGEXES_TO_IGNORE,  # Using constant for CLI
            gold_solution_file=args.gold_solution_file,
            pred_format=args.pred_format,
        )


if __name__ == "__main__":
    main()
