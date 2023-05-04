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

import flor
from flor import MTK as Flor

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

# or, the base sized model with WTQ configuration
model_name = "google/tapas-base-finetuned-wtq"
config = TapasConfig.from_pretrained(model_name)
tokenizer = TapasTokenizer.from_pretrained(model_name)
model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)
assert isinstance(model, TapasForQuestionAnswering)
model = model.to(device)


class TableDataset(torchdata.Dataset):
    def __init__(
        self,
        tokenizer=tokenizer,
        queries=queries,
        answer_coordinates=answer_coordinates,
        answer_text=answer_text,
        float_values=float_values,
    ):
        self.tokenizer = tokenizer
        self.queries = queries
        self.answer_coordinates = answer_coordinates
        self.answer_text = answer_text
        self.float_values = float_values

        self.locs = [k for k in queries.keys()]

    def __getitem__(self, idx):
        q = self.locs[idx]

        table = pd.read_csv(q).astype(str)
        queries = self.queries[q]
        answer_text = self.answer_text[q]
        float_values = [float(v) if v else float("nan") for v in self.float_values[q]]
        answer_coordinates = [c if c else [] for c in self.answer_coordinates[q]]

        encoding = self.tokenizer(
            table=table,
            queries=queries,
            answer_coordinates=answer_coordinates,
            answer_text=answer_text,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # remove the batch dimension which the tokenizer adds by default
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}

        # add the float_answer which is also required (weak supervision for aggregation case)
        encoding["float_answer"] = torch.tensor(float_values)

        return encoding

    def __len__(self):
        return len(self.locs)


def collate_fn(batch):
    """
    Unsequeeze
    device
    """

    for item in batch:
        if len(item["labels"].shape) < 2:
            for k in item:
                if k == "float_answer":
                    continue
                item[k] = item[k].unsqueeze(0)

    new_dict = {}
    for k in batch[0].keys():
        new_dict[k] = torch.cat([item[k] for item in batch], dim=0).to(device)
    return new_dict


tr_dataset = TableDataset()
train_dataloader = torchdata.DataLoader(
    tr_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

Flor.checkpoints(model, optimizer)
num_steps = len(train_dataloader)
for epoch in Flor.loop(range(3)):
    model.train()
    for i, batch in Flor.loop(enumerate(train_dataloader)):
        outputs = model(**batch)
        loss = outputs.loss

        if loss.item() > 5:
            for k in batch:
                batch[k].detach()
            continue

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(epoch + 1, f"({i + 1} / {num_steps})", flor.log("loss", loss.item()))

        for k in batch:
            batch[k].detach()
        loss.detach()
        if i >= 50:
            break
    torch.cuda.empty_cache()

print("All done!")
