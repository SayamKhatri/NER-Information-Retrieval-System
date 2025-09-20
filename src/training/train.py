from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
import evaluate

def build_compute_metrics(id2label):
    def compute_metrics(p):
        seqeval = evaluate.load("seqeval")
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


def train(model, tokenizer, train_tokenized, val_tokenized, output_dir, id2label, epochs):
    training_args = TrainingArguments(
        output_dir="./ner_lora_biobert",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        report_to = "none"
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    processing_class=tokenizer,
    compute_metrics=build_compute_metrics(id2label),
    )


    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer