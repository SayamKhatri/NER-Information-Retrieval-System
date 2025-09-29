import os
from datasets import load_dataset
from transformers import AutoTokenizer

label2id = {
    "O": 0,
    "B-Chemical": 1,
    "B-Disease": 2,
    "I-Disease": 3,
    "I-Chemical": 4,
}
id2label = {v: k for k, v in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=256,
    )
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids, prev = [], None
        for w in word_ids:
            if w is None:
                label_ids.append(-100)
            elif w != prev:
                label_ids.append(label[w])
            else:
                curr = label[w]
                if curr == label2id["B-Chemical"]:
                    label_ids.append(label2id["I-Chemical"])
                elif curr == label2id["B-Disease"]:
                    label_ids.append(label2id["I-Disease"])
                else:
                    label_ids.append(curr)
            prev = w
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

# SageMaker paths
input_dir = "/opt/ml/processing/input"
output_dir = "/opt/ml/processing/output"

# Load parquet (mounted from S3)
train = load_dataset("parquet", data_files=os.path.join(input_dir, "train", "train_bc5cdr.parquet"))["train"]
test  = load_dataset("parquet", data_files=os.path.join(input_dir, "test",  "test_bc5cdr.parquet"))["train"]
valid = load_dataset("parquet", data_files=os.path.join(input_dir, "valid", "valid_bc5cdr.parquet"))["train"]

# Tokenize
train_tok = train.map(tokenize_and_align_labels, batched=True)
test_tok  = test.map(tokenize_and_align_labels, batched=True)
valid_tok = valid.map(tokenize_and_align_labels, batched=True)

# Save 
train_tok.save_to_disk(os.path.join(output_dir, "train"))
test_tok.save_to_disk(os.path.join(output_dir, "test"))
valid_tok.save_to_disk(os.path.join(output_dir, "valid"))

print("Preprocessing complete. Datasets saved to /opt/ml/processing/output/{train,test,valid}")
