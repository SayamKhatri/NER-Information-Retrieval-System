from src.training.dataset import make_dataset
from src.training.model import peft_model, load_tokenizer
from src.training.train import train
from src.training.eval import evaluate_model


def run_pipeline(config):
    train_tokenized, test_tokenized, val_tokenized, label2id, id2label = make_dataset()
    print('Dataset step ran successful!')
    model = peft_model(label2id, id2label)
    tokenizer = load_tokenizer()
    trainer = train(model, tokenizer, train_tokenized, val_tokenized, 
                    config['output_dir'], id2label, config['epochs'])
    evaluate_model(trainer, test_tokenized)





if __name__ == "__main__":
    config = {
        "epochs": 1,
        "output_dir": "outputs/biobert_bc5cdr",
    }
    run_pipeline(config)