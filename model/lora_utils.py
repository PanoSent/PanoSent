from peft import get_peft_model, LoraConfig, TaskType

def apply_lora(model):
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    return get_peft_model(model, config)