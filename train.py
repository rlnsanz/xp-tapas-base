import os
import shutil
from pathlib import Path, PurePath
import pandas as pd
from tqdm import tqdm
import json
import cloudpickle

from utils import parse_question
from transformers import TapasConfig, TapasForQuestionAnswering, TapasTokenizer

import torch
from torch.utils import data as torchdata


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

names = [
    "queries",
    "answer_coordinates",
    "answer_text",
    "float_values",
    "aggregation_functions",
]

data = {}
queries = {}
answer_coordinates = {}
answer_text = {}

float_values = {}
aggregation_functions = {}

checkpoints = Path("checkpoints")

if checkpoints.exists():
    for name in names:
        with open(checkpoints / PurePath(name).with_suffix(".pkl"), "rb") as f:
            locals()[name].update(cloudpickle.load(f))  # type: ignore
else:
    checkpoints.mkdir()
    for _, ut, ct, tv in tqdm(
        pd.read_csv("data/training.tsv", sep="\t", index_col=0).itertuples()
    ):
        try:
            parsed = parse_question(
                table=pd.read_csv(ct).astype(str),
                question=ut,
                answer_texts=tv.split("|"),
            )
        except Exception as e:
            # print(e, "::", ut, ct, tv)
            continue

        q, ans_txt, ans_coord, float_value, aggregation_function = parsed  # type: ignore

        if ct not in data:
            data[ct] = pd.read_csv(ct).astype(str)
            queries[ct] = []
            answer_coordinates[ct] = []
            answer_text[ct] = []
            float_values[ct] = []
            aggregation_functions[ct] = []

        queries[ct].append(q)
        answer_coordinates[ct].append(ans_coord)
        answer_text[ct].append(ans_txt)
        float_values[ct].append(float_value)
        aggregation_functions[ct].append(aggregation_function)

    for name in names:
        with open(checkpoints / PurePath(name).with_suffix(".pkl"), "wb") as f:
            cloudpickle.dump(locals()[name], f)


class TableDataset(torchdata.Dataset):
    def __init__(
        self,
        tokenizer,
        data,
        queries,
        answer_coordinates,
        answer_text,
        float_values,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.queries = queries
        self.answer_coordinates = answer_coordinates
        self.answer_text = answer_text
        self.float_values = float_values

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            table=self.data.iloc[idx],
            queries=self.queries.iloc[idx],
            answer_coordinates=self.answer_coordinates.iloc[idx],
            answer_text=self.answer_text.iloc[idx],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # remove the batch dimension which the tokenizer adds by default
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        # add the float_answer which is also required (weak supervision for aggregation case)
        encoding["float_answer"] = torch.tensor([float(r) for r in self.float_values])
        return encoding

    def __len__(self):
        return len(self.data)


# or, the base sized model with WTQ configuration
model_name = "google/tapas-base-finetuned-wtq"
config = TapasConfig.from_pretrained(model_name)
tokenizer = TapasTokenizer.from_pretrained(model_name)
model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)
assert isinstance(model, TapasForQuestionAnswering)
model = model.to(device)

tr_dataset = TableDataset(
    tokenizer, data, queries, answer_coordinates, answer_text, float_values
)

print("almost!")
