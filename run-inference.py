import argparse
import torch
import gradio as gr
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
    Main function to initialize model inference and launch Gradio demo.
    """
    base = LLMInference(None, max_new_tokens, model_type)
    tune = LLMInference(adapter, max_new_tokens, model_type)

    # Gradio 블록 사용하여 두 인터페이스 동시에 띄우기
    with gr.Blocks() as demo:
        # 설명 추가
        gr.Markdown("""
        # Gemma-2B Model Inference
        Google Gemma 모델은 약 2B 스케일의 소형 언어 모델 (sLLM) 입니다.
        별도로 구축한 밀리터리 인스트럭션 데이터셋으로 파인튜닝해본 PoC 결과입니다.
                        
        이 페이지에서는 두 가지 모델을 사용할 수 있습니다:
        1. **Gemma-2B-base**: 기본 모델
        2. **Gemma-2B-mil**:  LoRA 어댑터가 적용된 모델

        아래의 입력 상자에 텍스트를 입력하고 결과를 확인해보세요.
        """)
        with gr.Column():
            base_interface = gr.Interface(
                fn=base.run,
                inputs=gr.Textbox(lines=2, placeholder="K-9 자주포에 대해서 알려주세요."),
                outputs="text",
                title="Gemma-2B-base"
            )
        
        with gr.Column():
            mil_interface = gr.Interface(
                fn=tune.run,
                inputs=gr.Textbox(lines=2, placeholder="K-9 자주포에 대해서 알려주세요."),
                outputs="text",
                title="Gemma-2B-mil"
            )

    demo.launch(share=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gemma-2B Model Inference with Gradio UI")
    parser.add_argument('--adapter', type=str, default=ADAPTER_MODEL, help='Path to the adapter model')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='Maximum number of new tokens to generate')
    parser.add_argument('--model_type', type=str, default="gemma", help='Respone type of Model')
    args = parser.parse_args()
    
    # # torch_dtype 인자를 실제 torch.dtype 객체로 변환
    # dtype_map = {
    #     'float16': torch.float16,
    #     'bfloat16': torch.bfloat16,
    #     'float32': torch.float32
    # }
    # torch_dtype = dtype_map.get(args.torch_dtype, torch.bfloat16)

    main(
        adapter=args.adapter,
        max_new_tokens=args.max_new_tokens,
        model_type=args.model_type
    )
