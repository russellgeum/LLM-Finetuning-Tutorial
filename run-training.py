import argparse
from module.model import *
from module.configuration import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model with specific configurations.")
    parser.add_argument('--dataset', type=str, default="dataset/kullm-v2/kullm-v2.jsonl", help='Path to the dataset')
    parser.add_argument('--model', type=str, default="beomi/gemma-ko-2b", help='Name of the base model')
    parser.add_argument('--adapter', type=str, default="model/beomi-gemma-2b-kullm-v2-adapter", help='Name of the adapter model')
    parser.add_argument('--checkpoint', type=str, default='model/beomi-gemma-2b-kullm-v2-checkpoint', help='Name of the checkpoint directory')
    parser.add_argument("--split", type=float, default=0.1, help='Name of the dataset split parameters')
    args = parser.parse_args()
    
    trainer = LLMTrainer(
        dataset=args.dataset,
        model=args.model,
        adapter=args.adapter,
        checkpoint=args.checkpoint,
        split=args.split
    )
    trainer.run()