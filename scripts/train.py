import os
from transformers import AutoModelForTokenClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model
import evaluate
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import load_from_disk  
import json

# SageMaker paths
train_data_path = "/opt/ml/input/data/train"      
val_data_path = "/opt/ml/input/data/valid"   
test_data_path = "/opt/ml/input/data/test"
model_output_dir = "/opt/ml/model"       
evaluation_output_dir = "/opt/ml/output/data"  

seqeval = evaluate.load("seqeval")

label2id = {
    "O": 0,
    "B-Chemical": 1,
    "B-Disease": 2,
    "I-Disease": 3,
    "I-Chemical": 4
}

id2label = {v: k for k, v in label2id.items()}
base_model_name = "dmis-lab/biobert-base-cased-v1.1"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load model
model = AutoModelForTokenClassification.from_pretrained(
    base_model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Load preprocessed datasets 
print(f"Loading training data from {train_data_path}")
train_tokenized = load_from_disk(train_data_path)

print(f"Loading validation data from {val_data_path}")
val_tokenized = load_from_disk(val_data_path)

print(f"Loading test data from {test_data_path}")
test_tokenized = load_from_disk(test_data_path)

print(f"Train dataset size: {len(train_tokenized)}")
print(f"Validation dataset size: {len(val_tokenized)}")
print(f"Test dataset size: {len(test_tokenized)}")

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="TOKEN_CLS"
)

model = get_peft_model(model, lora_config)
print("LoRA adapters added to model")
model.print_trainable_parameters()  

# Metrics
def compute_metrics(p):
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

# Training arguments
training_args = TrainingArguments(
    output_dir=model_output_dir,  
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    logging_dir=f"{model_output_dir}/logs",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",  
    save_total_limit=2,
    load_best_model_at_end=True,  
    metric_for_best_model="f1",   
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# TRAIN THE MODEL
print("Starting training...")
trainer.train()
print("Training complete!")

# Evaluation On Test set
test_metrics = trainer.evaluate(test_tokenized)
print(test_metrics)

os.makedirs(evaluation_output_dir, exist_ok=True)
evaluation_file = os.path.join(evaluation_output_dir, 'evaluation_results.json')

evaluation_results = {
    "overall_precision": test_metrics['eval_precision'],
    "overall_recall": test_metrics['eval_recall'],
    "overall_f1": test_metrics['eval_f1'],
    "overall_accuracy": test_metrics['eval_accuracy'],
}

with open(evaluation_file, "w") as f:
    json.dump(evaluation_results, f, indent=2)

print(f"\nEvaluation results saved to {evaluation_file}")

# Merge LoRA weights back into base model
print("Merging LoRA adapters into base model...")
model = model.merge_and_unload()

# Save final model
print(f"Saving final model to {model_output_dir}")
model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)