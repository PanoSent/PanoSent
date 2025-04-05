import argparse
import torch
from transformers import TrainingArguments, Trainer
from sentica.model.flant5_model import FlanT5WithMM
from sentica.datasets.stage1_caption_dataset import CaptionDataset
from sentica.datasets.stage2_sextuple_dataset import SextupleDataset
from sentica.datasets.stage3_entailment_dataset import EntailmentDataset

def train_stage(stage):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FlanT5WithMM(device=device).train()

    if stage == 1:
        dataset = CaptionDataset()
        def collate(batch):
            input_ids = torch.stack([x["input_ids"] for x in batch])
            attention_mask = torch.stack([x["attention_mask"] for x in batch])
            labels = torch.stack([x["labels"] for x in batch])
            mm_paths = [x["mm_path"] for x in batch]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "mm_paths": mm_paths
            }
    elif stage == 2:
        dataset = SextupleDataset(json_path="./data/train.json")

        def collate(batch):
            return {k: torch.stack([x[k] for x in batch]) for k in ["input_ids", "attention_mask", "labels"]}
    elif stage == 3:
        dataset = EntailmentDataset(json_path="./data/PpV_train.json")

        def collate(batch):
            return {
                "input_ids": torch.stack([x["input_ids"] for x in batch]),
                "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
                "labels": torch.stack([x["labels"] for x in batch]),
            }
    else:
        raise ValueError("Invalid stage")

    args = TrainingArguments(
        output_dir=f"./checkpoints/stage{stage}",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=1e-4,
        fp16=True,
        save_strategy="epoch",
        logging_steps=10,
        report_to="none"
    )

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            mm_paths = inputs.pop("mm_paths", None)
            outputs = model(mm_paths=mm_paths, **inputs)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=model.tokenizer,
        data_collator=collate
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()
    train_stage(args.stage)