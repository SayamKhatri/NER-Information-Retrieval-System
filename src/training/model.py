from transformers import AutoModelForTokenClassification
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer


def load_model(label2id, id2label):
    model = AutoModelForTokenClassification.from_pretrained(
        "dmis-lab/biobert-base-cased-v1.1",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    return model

def load_tokenizer():
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return tokenizer

def peft_model(label2id, id2label):
    lora_config = LoraConfig(
        r=8,              # rank
        lora_alpha=16,    # scaling
        lora_dropout=0.1,
        bias="none",
        task_type="TOKEN_CLS"
    )

    model = load_model(label2id, id2label)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model

