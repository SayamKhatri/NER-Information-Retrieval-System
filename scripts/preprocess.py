from scripts.utils import download_data, save_data
import yaml
import os 
from transformers import AutoTokenizer
from datasets import load_dataset
import shutil

CONFIG_PATH = os.path.join('config', 'pipeline_config.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

bucket_name = config["bucket_name"]
train_path = config["train_path"]
test_path = config["test_path"]
valid_path = config["valid_path"]

def get_data():
    download_data(bucket_name, train_path)
    download_data(bucket_name, test_path)
    download_data(bucket_name, valid_path)


label2id = {
    "O": 0,
    "B-Chemical": 1,
    "B-Disease": 2,
    "I-Disease": 3,
    "I-Chemical": 4
}

id2label = {v: k for k, v in label2id.items()}

model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=256
    )

    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(-100)  # ignored by loss

            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  # first token of word

            else:

                current_label = label[word_idx]

                if current_label == label2id["B-Chemical"]:
                    label_ids.append(label2id["I-Chemical"])

                elif current_label == label2id["B-Disease"]:
                    label_ids.append(label2id["I-Disease"])

                else:
                    label_ids.append(current_label)

            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

get_data()
train_dataset = load_dataset('parquet', data_files='data/train_bc5cdr.parquet')
test_dataset = load_dataset('parquet', data_files='data/test_bc5cdr.parquet')
valid_dataset = load_dataset('parquet', data_files='data/valid_bc5cdr.parquet')

train_tokenized = train_dataset.map(tokenize_and_align_labels, batched=True)
test_tokenized = train_dataset.map(tokenize_and_align_labels, batched=True)
valid_tokenized = train_dataset.map(tokenize_and_align_labels, batched=True)

train_tokenized.save_to_disk(config['train_save_path'])
test_tokenized.save_to_disk(config['test_save_path'])
valid_tokenized.save_to_disk(config['valid_save_path'])

shutil.make_archive(config['train_save_path'], "zip", config['train_save_path'])
shutil.make_archive(config['test_save_path'], "zip", config['test_save_path'])
shutil.make_archive(config['valid_save_path'], "zip", config['valid_save_path'])

save_data(bucket_name, local_path=f"{config['train_save_path']}.zip")
save_data(bucket_name, local_path=f"{config['test_save_path']}.zip")
save_data(bucket_name, local_path=f"{config['valid_save_path']}.zip")
