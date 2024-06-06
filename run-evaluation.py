import argparse
from module.model import *
from module.configuration import *

# 상수는 모두 대문자 사용
BASE_MODEL           = "beomi/gemma-ko-2b"
BASE_MODEL_CACHE     = "model/beomi-gemma-2b/model"
BASE_TOKENIZER_CACHE = "model/beomi-gemma-2b/tokenizer"
ADAPTER_MODEL        = "model/beomi-gemma-2b-mil-adapter/"

def main(
        adapter: str,
        max_new_tokens: int,
        model_type: str
    ):
    """
    Main function to initialize model evlaution.
    """
    evaluation = LLMEvaluation(adapter, max_new_tokens, model_type)
    evaluation.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gemma-2B Model Inference with Gradio UI")
    parser.add_argument('--adapter', type=str, help='Path to the adapter model')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='Maximum number of new tokens to generate')
    parser.add_argument('--model_type', type=str, default="gemma", help='Respone type of Model')
    args = parser.parse_args()

    main(args.adapter, args.max_new_tokens, args.model_type)