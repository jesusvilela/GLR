# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Shashi Kumar shashi.kumar@idiap.ch, shashi.kumar@epfl.ch
# SPDX-FileContributor: Yacouba Kaloga yacouba.kaloga@idiap.ch
#
# SPDX-License-Identifier: MIT

"""Convert an lm-evaluation-harness `--log_samples` file into this repo's eval JSONL format.

One record per document (`doc_id`): lm-eval logs a document once per answer filter
(GSM8K has two), so only the first occurrence is kept.
"""

import argparse
import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

parser = argparse.ArgumentParser()
parser.add_argument("--inp_path", required=True, help="lm-eval samples_<task>_<timestamp>.jsonl")
parser.add_argument("--out_path", required=True)
args = parser.parse_args()
inp_path = args.inp_path
out_path = args.out_path
inp_read = read_jsonl(inp_path)
print(f"Total samples read: {len(inp_read)}")
filt_data = []
seen_ids = set()

for line in inp_read:
    id_ = line["doc_id"]
    if id_ in seen_ids:
        continue
    seen_ids.add(id_)
    try:
        problem = line["doc"]["Problem"]
    except:
        try:
            problem = line["doc"]["problem"]
        except:
            problem = line["doc"]["question"]
    try:
        answer = line["doc"]["Answer"]
    except:
        answer = line["doc"]["answer"]
    target = line["target"]
    q = line["arguments"]["gen_args_0"]["arg_0"]
    try:
        solution = line["doc"]["Solution"]
    except:
        solution = ""
    to_dump = {
        "id": id_,
        "q": q,
        "a": answer,
        "target": target,
        "problem": problem,
        "solution": solution,
    }
    filt_data.append(to_dump)

print(f"Total samples after filtering: {len(filt_data)}")
with open(out_path, "w") as f_out:
    for item in filt_data:
        f_out.write(json.dumps(item) + "\n")