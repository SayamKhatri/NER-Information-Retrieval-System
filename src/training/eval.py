import json
import logging
from src.training.train import build_compute_metrics

def evaluate_model(trainer, dataset, output_path="./eval_metrics.json"):

    results = trainer.evaluate(eval_dataset=dataset)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
