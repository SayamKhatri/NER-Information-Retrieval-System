from transformers import AutoModelForTokenClassification
from dataset import id2label
from config import label2id
from peft import LoraConfig, get_peft_model


def load_model():
    model = AutoModelForTokenClassification.from_pretrained(
        "dmis-lab/biobert-base-cased-v1.1",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    return model

def peft_model():
    lora_config = LoraConfig(
        r=8,              # rank
        lora_alpha=16,    # scaling
        lora_dropout=0.1,
        bias="none",
        task_type="TOKEN_CLS"
    )

    model = load_model()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model

