import pandas as pd 
import boto3
from config import BUCKET_NAME, TRAIN_DATA, TEST_DATA, VALID_DATA, label2id
from transformers import AutoTokenizer
from datasets import Dataset

def download_data():
    s3 = boto3.client('s3')

    object_keys = [TRAIN_DATA, TEST_DATA, VALID_DATA]

    for key in object_keys:
        s3.download_file(
            BUCKET_NAME, key, f'data/{key}'
        )

def get_data():
    download_data()
    train_data = pd.read_parquet(f'data/{TRAIN_DATA}')
    test_data = pd.read_parquet(f'data/{TEST_DATA}')
    valid_data = pd.read_parquet(f'data/{VALID_DATA}')

    return train_data, test_data, valid_data


def load_tokenizer():
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    id2label = {v: k for k, v in label2id.items()}

    return tokenizer, id2label


def tokenize_and_align_labels(examples):
    tokenizer, id2label = load_tokenizer()

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


def make_dataset():
    train_df, test_df, valid_df = get_data()
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset   = Dataset.from_pandas(valid_df)
    test_dataset  = Dataset.from_pandas(test_df)

    train_tokenized = train_dataset.map(tokenize_and_align_labels, batched=True)
    val_tokenized   = val_dataset.map(tokenize_and_align_labels, batched=True)
    test_tokenized  = test_dataset.map(tokenize_and_align_labels, batched=True)

    return train_tokenized, test_tokenized, val_tokenized

