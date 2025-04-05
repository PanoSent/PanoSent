# data/stage1_dataset.py
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer

class CaptionDataset(Dataset):
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        self.samples = [
            {"image": "example.jpg", "caption": "A person is standing on the beach."}
        ]

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = "Describe the image."
        input = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
        label = self.tokenizer(item["caption"], return_tensors="pt", padding="max_length", truncation=True, max_length=64)
        return {
            "input_ids": input["input_ids"].squeeze(0),
            "attention_mask": input["attention_mask"].squeeze(0),
            "labels": label["input_ids"].squeeze(0),
            "image": item["image"]
        }

    def __len__(self):
        return len(self.samples)

    def collate_fn(self, batch):
        keys = ["input_ids", "attention_mask", "labels"]
        return {
            key: torch.stack([b[key] for b in batch]) for key in keys
        } | {
            "image": [b["image"] for b in batch]
        }