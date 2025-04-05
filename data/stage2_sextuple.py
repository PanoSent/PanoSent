import json
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer


class SextupleDataset(Dataset):
    def __init__(self, json_path: str, tokenizer_name="google/flan-t5-xxl", max_input_len=512, max_output_len=256):
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.samples = self.flatten_dialogs(self.data)

    def flatten_dialogs(self, data):
        all_samples = []
        for sample in data:
            dialogue = sample["dialogue"]
            hexas = sample["hexatuple"]

            dialogue_text = ""
            speaker_map = {sp["id"]: sp["name"] for sp in sample.get("speakers", [])}
            for turn in dialogue:
                speaker = speaker_map.get(turn["speaker"], f"S{turn['speaker']}")
                dialogue_text += f"{speaker}: {turn['utterance']}\n"

            target_texts = []
            for h in hexas:
                target_texts.append(
                    f"Holder: {h['holder']['value']}; "
                    f"Target: {h['target']['value']}; "
                    f"Aspect: {h['aspect']['value']}; "
                    f"Opinion: {h['opinion']['value']}; "
                    f"Sentiment: {h['sentiment']}; "
                    f"Rationale: {h['rationale']['value']}"
                )

            full_target = "\n".join(target_texts)
            all_samples.append((dialogue_text.strip(), full_target.strip()))
        return all_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_text, target_text = self.samples[idx]

        input_enc = self.tokenizer(
            "Extract all opinion hexatuples from the following dialogue:\n" + input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_input_len,
            return_tensors="pt"
        )

        label_enc = self.tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_output_len,
            return_tensors="pt"
        )

        return {
            "input_ids": input_enc.input_ids.squeeze(0),
            "attention_mask": input_enc.attention_mask.squeeze(0),
            "labels": label_enc.input_ids.squeeze(0)
        }