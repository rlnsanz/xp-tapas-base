from utils import parse_question
from transformers import TapasConfig, TapasForQuestionAnswering, TapasTokenizer

import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# or, the base sized model with WTQ configuration
model_name = "google/tapas-base-finetuned-wtq"
config = TapasConfig.from_pretrained(model_name)
tokenizer = TapasTokenizer.from_pretrained(model_name)
model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)
assert isinstance(model, TapasForQuestionAnswering)
model = model.to(device)
