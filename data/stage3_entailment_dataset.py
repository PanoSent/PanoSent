import json
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer


class EntailmentDataset(Dataset):
    def __init__(self, json_path, tokenizer_name="google/flan-t5-xxl", max_input_len=512):
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.max_input_len = max_input_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        prompt = f"Context: {item['context']}\nClaim: {item['claim']}\nIs this claim entailed by the context? Answer yes or no."
        label_text = "yes" if item["label"] == 1 else "no"

        input_enc = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_input_len)
        label_enc = self.tokenizer(label_text, return_tensors="pt", padding="max_length", truncation=True, max_length=10)

        return {
            "input_ids": input_enc.input_ids.squeeze(0),
            "attention_mask": input_enc.attention_mask.squeeze(0),
            "labels": label_enc.input_ids.squeeze(0)
        }