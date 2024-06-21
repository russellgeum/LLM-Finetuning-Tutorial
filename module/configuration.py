import os
import json
from tqdm import tqdm
from typing import List
from typing import Union
import gradio
import logging
import torch
from dataclasses import (
    field,
    asdict,
    dataclass,
)
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling
    )
from peft import (
    LoraConfig,
    PeftModel,
    AutoPeftModel,
    prepare_model_for_kbit_training
    )
from trl import (
    SFTTrainer, 
    DataCollatorForCompletionOnlyLM
    )
# import wandb
# from huggingface_hub import notebook_login


DEVICE_MAP = "auto"
GEMMA_2B_BASE_MODEL = "beomi/gemma-ko-2b"
GEMMA_7B_BASE_MODEL = None
EEVE_10B_BASE_MODEL = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"


# ================================================================
# Class: LoraConfiguration
# Purpose: This class holds the configuration settings for the
#          LoRA (Low-Rank Adaptation) method.
# ================================================================
@dataclass
class LoraConfiguration:
    r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: \
        ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"])
    task_type: str = "CAUSAL_LM"


# ================================================================
# Class: BitsAndBytesConfiguration
# Purpose: This class holds the configuration settings for the
#          bits and bytes (bnb) quantization method.
# ================================================================
@dataclass
class BitsAndBytesConfiguration:
    load_4bit: bool = True
    load_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"


# ================================================================
# Class: TrainingConfiguration
# Purpose: This class holds the configuration settings for the
#          training process.
# ================================================================
@dataclass
class TrainingConfiguration:
    """
    예시 설정:
    - 데이터셋 크기: 18,646
    - 배치 크기: 32
    - 체크포인트 저장 간격: 400 스텝
    - 로깅 간격: 40 스텝

    gradient_accumulation_steps = 1일 때:
    - 1 epoch 당 스텝 수: 582
    - 3 epoch 당 스텝 수: 1,746
    - 체크포인트 저장: 매 400 스텝마다
    - 로그 저장: 매 40 스텝마다
    - 로그 로깅: 매 40 스텝마다

    gradient_accumulation_steps = 2일 때:
    - 1 epoch 당 스텝 수: 291 (gradient_accumulation_steps로 나눈 값)
    - 3 epoch 당 스텝 수: 873 (gradient_accumulation_steps로 나눈 값)
    이유: 매 배치마다 optimizer.step()을 호출하는 것이 아니라, gradient_accumulation_steps마다 호출하기 때문
    - 체크포인트 저장: 400 스텝 기준으로 gradient_accumulation_steps을 반영하여 저장
    - 로그 저장: 40 스텝 기준으로 gradient_accumulation_steps을 반영하여 저장
    - 로그 로깅: 40 스텝 기준으로 gradient_accumulation_steps을 반영하여 로깅
    """
    save_steps: int = 200
    save_strategy: str = "steps"  # save_steps마다 저장
    num_train_epochs: int = 3  # 총 에포크 수
    logging_strategy: str = "steps"  # 학습 중 로그를 언제 저장할 것인지 지정
    logging_steps: int = 40  # 학습 중 로그를 언제 출력할 것인지 지정
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    optim: str = "paged_adamw_32bit"
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    gradient_checkpointing: bool = True
    group_by_length: bool = False
    disable_tqdm: bool = False
    report_to: str = "wandb"


# ================================================================
# Class: WrapperConfiguration
# Purpose: This class is a wrapper for the three configuration
#          classes: LoraConfiguration, BitsAndBytesConfiguration,
#          and TrainingConfiguration.
# ================================================================
class WrapperConfiguration:
    @staticmethod
    def create_configuration():
        return (
            asdict(LoraConfiguration()),
            asdict(BitsAndBytesConfiguration()),
            asdict(TrainingConfiguration())
        )


# ================================================================
# Function: check_GPU_state
# Purpose: This function checks the state of the GPU.
# ================================================================
def check_GPU_state():
    if torch.cuda.is_available():
        print(f"사용 가능한 GPU 장치: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} 정보:")
            print(f"  이름: {torch.cuda.get_device_name(i)}")
            print(f"  사용 중인 메모리: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")
            print(f"  예약된 메모리: {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB")
            print(f"  사용 가능한 메모리: {(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / 1024 ** 3:.2f} GB")
    else:
        print("GPU가 사용 불가능합니다.")


# ================================================================
# Class: DatasetLoader
# Purpose: This class is responsible for loading and splitting the
#          dataset into training and test sets.
# ================================================================
class DatasetLoader:
    def __init__(self, dataset, test_size=0.05, seed=42):
        self.dataset   = dataset
        self.test_size = test_size
        self.seed      = seed
        # Load the dataset
        self.dataset = load_dataset("json", data_files=self.dataset)
        print("*** Loaded json Dataset")
        print(self.dataset)
        
    def get_split_dataset(self):
        # Shuffle and split the dataset
        split_dataset = self.dataset["train"].shuffle(seed=self.seed)
        split_dataset = split_dataset.train_test_split(test_size=self.test_size)
        
        # Get train and test datasets
        train_dataset = split_dataset["train"]
        test_dataset  = split_dataset["test"]
        print("*** Train 데이터셋")
        print(train_dataset)
        print("*** Test 데이터셋")
        print(test_dataset)
        return train_dataset, test_dataset


# ================================================================
# Class: PromptGenerator
# Purpose: This class provides methods to generate prompts for
#          different types of models based on the dataset.
# ================================================================
class PromptGenerator:
    generators = {
        "kullm": "_PromptGenerator__kullm_generate_gemma_prompt",
        "custom": "_PromptGenerator__custom_generate_gemma_prompt",
        # "eeve": self.__generate_eeve_prompt
    }

    @classmethod
    def get_generator(cls, name):
        if name in cls.generators:
            return getattr(cls, cls.generators[name])
        else:
            raise ValueError(f"Generator {name} not found.")

    @staticmethod
    def __kullm_generate_gemma_prompt(dataset):
        prompt_list = []
        for idx in range(len(dataset["instruction"])):
            instruction = dataset["instruction"][idx]
            inputs      = dataset["input"][idx]
            outputs     = dataset["output"][idx]

            template = r"""<bos><start_of_turn>user
{} {}<end_of_turn>

<start_of_turn>model
{}<end_of_turn><eos>""".format(instruction, inputs, outputs)

            prompt_list.append(template)
        return prompt_list

    @staticmethod
    def __custom_generate_gemma_prompt(dataset):
        prompt_list = []
        for idx in range(len(dataset["prompt"])):
            prompt   = dataset["prompt"][idx]
            response = dataset["response"][idx]

            template = r"""<bos><start_of_turn>user
{}<end_of_turn>

<start_of_turn>model
{}<end_of_turn><eos>""".format(prompt, response)

            prompt_list.append(template)
        return prompt_list
    

# ================================================================
# Class: Query
# Purpose: This class handles the process of taking a query prompt,
#          tokenizing it, and generating a response using a model.
# ================================================================
class Query:
    def __init__(self, tokenizer, model, max_new_tokens, model_type):
        self.tokenizer = tokenizer
        self.model = model
        self.max_new_tokens = max_new_tokens

        # model_type에 따라 프롬프트 메소드를 설정
        if model_type == 'gemma':
            self.generate = self.__generate_gemma_response
        elif model_type == 'eeve':
            self.generate = self.__generate_eeve_response
        else:
            raise ValueError("Invalid response type. Choose 'gemma' or 'eeve'")
    
    def __generate_gemma_response(self, prompt):
        inputs = "<bos><start_of_turn>user\n" + prompt + "<end_of_turn> \n<start_of_turn>model"
        model_inputs = self.tokenizer(prompt, return_tensors='pt', padding = True).to("cuda")
        outputs      = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        output_text  = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return output_text

    # def __generate_phi3_response(self, prompt):
    #     inputs = 
        
    def __generate_eeve_response(self, prompt):
        inputs = "User:\n" + prompt
        model_inputs = self.tokenizer(prompt, return_tensors='pt', padding = True).to("cuda")
        outputs      = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        output_text  = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return output_text