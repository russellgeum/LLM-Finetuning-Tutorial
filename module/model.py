from .configuration import *


# =================================================================================================
# class Model:
# Purpose: This class defines the dataset, tokenizer, model, and LoRA adapter functions.
# =================================================================================================
class Model:
    def __init__(self, **kwargs):
        """
        이 클래스는 데이터셋, 토크나이저, 모델 및 LoRA 어댑터 함수를 정의합니다.
        이 클래스에서는 해당 함수들의 초기화나 실행을 하지 않고 선언만 합니다.
        def load_dataset
        def load_generator
        def load_tokenizer
        def load_model
        def merge_adaptor
        def save
        """
        # 없으면 기본값으로 대체
        self.dataset: str    = kwargs.get('dataset', "dataset/kullm-v2/kullm-v2.jsonl")
        self.model: str      = kwargs.get('model', "beomi/gemma-ko-2b")
        self.adapter: str    = kwargs.get('adapter', "model/beomi-gemma-2b-kullm-v2-adapter")
        self.checkpoint: str = kwargs.get('checkpoint', 'model/beomi-gemma-2b-kullm-v2-checkpoint')
        self.split: float    = kwargs.get('split', 0.01)
        self.log_file: str   = kwargs.get('log_file', 'app.log')
        self.dataset_dict    = {
            "dataset/kullm-v2/kullm-v2.jsonl": "kullm",
            "dataset/milstruction/milstruction.json": "custom",
        }
        self.cache_dir = "model/beomi-gemma-2b"
        self.logger = self.setup_logger()

    def setup_logger(self) -> logging.Logger:
        # Set up the logger for the training process
        logger = logging.getLogger("LLM mode")
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        return logger

    def load_dataset(self) -> DatasetLoader:
        # 데이터셋 로드
        return DatasetLoader(self.dataset, self.split, 42)

    def load_generator(self) -> callable:
        # 프롬프트 제네레이터 로드
        return PromptGenerator.get_generator(name=self.dataset_dict[self.dataset])

    def load_tokenizer(self) -> AutoTokenizer:
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def load_model(self, mode: bool, quantization_config) -> AutoModelForCausalLM:
        """
        Load the base model.
        self.mode == True
            quantization_config = self.config_BnB
            use_cache = False
        self.mode == False
            quantization_config = None
            use_cache = True
        """
        # 학습 모드
        if mode:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model,
                quantization_config=quantization_config,
                use_cache=False,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16,
                device_map='auto'
            )
            base_model.config.pretrainig_tp = 1
            base_model.gradient_checkpointing_enable()
            base_model = prepare_model_for_kbit_training(base_model)
            return base_model
        # 추론 모드
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model,
                quantization_config=None,
                use_cache=True, # 추론 성능 향상
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16,
                device_map='auto'
            )
            return base_model

    def merge_adapter(self, adapter: str) -> AutoModelForCausalLM:
        # 기본 모델에 어댑터 모델을 로드하고 머지한다.
        model = PeftModel.from_pretrained(
            model=self.model,
            model_id=adapter,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        return model.merge_and_unload()

    def save(self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, save_path: str) -> None:
        # Save the current state of the (merged) model to the specified path.
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)


# =================================================================================================
# class LLMTrainer(Model):
# Purpose: This class defines the LoRA training process.
# =================================================================================================
class LLMTrainer(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        1. 데이터셋을 로드합니다.
        2. LoRA, BnB, Training configuration을 생성합니다.
        3. GPU 상태를 확인하고 토크나이저, 제네레이터, 모델을 로드합니다.
        4. 트레이너에서는 base model에 adapter를 머지하고 학습을 진행합니다.
        """
        dataset = self.load_dataset()
        self.train_dataset, self.test_dataset = dataset.get_split_dataset()
        
        config_dict = self.create_config()
        self.lora_config_dict      = config_dict[0]
        self.precision_config_dict = config_dict[1]
        self.training_config_dict  = config_dict[2]
        self.config_LoRA     = LoraConfig(**self.lora_config_dict)
        self.config_BnB      = BitsAndBytesConfig(**self.precision_config_dict)
        self.config_Training = TrainingArguments(
            output_dir=self.checkpoint,
            **self.training_config_dict
        )

        check_GPU_state()
        self.tokenizer  = self.load_tokenizer()
        self.generator  = self.load_generator()
        self.base_model = self.load_model(mode=True, quantization_config=self.config_BnB)

        check_GPU_state()
        self.trainer = self.create_trainer()

    def create_config(self):
        # Create the configuration dictionaries for LoRA, precision, and training
        return WrapperConfiguration.create_configuration()

    def create_trainer(self) -> SFTTrainer:
        # Create the trainer with the loaded components.
        return SFTTrainer(
            model=self.base_model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            max_seq_length=600,
            args=self.config_Training,
            peft_config=self.config_LoRA,
            formatting_func=self.generator
        )

    def run(self):
        # Train the model and save the adapter.
        self.logger.info(f"Dataset path       : {self.dataset}")
        self.logger.info(f"Base model path    : {self.model}")
        self.logger.info(f"Adapter model path : {self.adapter}")
        self.logger.info(f"Checkpoint path    : {self.checkpoint}")

        self.logger.info('Starting model training')
        self.trainer.train()
        self.logger.info('Finished adapted model training')
        self.trainer.model.save_pretrained(self.adapter_name)
        self.logger.info('Saved adapted model')


# =================================================================================================
# class LLMInference(Model):
# Purpose: This class defines the inference process.
# =================================================================================================
class LLMInference(Model):
    def __init__(self, 
                 adapter: str, 
                 max_new_tokens: int,
                 model_type: str,
                 **kwargs):
        super().__init__(**kwargs)
        """
        1. 토크나이저를 로드합니다.
        2. 기본 모델을 로드합니다.
        3. 어댑터 모델을 로드하고 머지합니다.
        4. 쿼리 인스턴스를 초기화합니다.
        5. 인퍼런스에서는 임의 문장을 입력하면, 해당 문장에 대한 답변을 생성합니다.
        """
        print(self.cache_dir)
        # 토크나이저 로드
        self.tokenizer = self.load_tokenizer()

        # 기본 모델 로드
        self.model = self.load_model(mode=False, quantization_config=None)

        # 어댑터 모델을 로드하고 머지
        if adapter is None:
            self.model = self.model
        else:
            self.model = self.merge_adapter(adapter)

        # 인퍼런스를 위한 쿼리 인스턴스 초기화
        self.query = Query(self.tokenizer, self.model, max_new_tokens, model_type)

    def run(self, prompt: str) -> str:
        # 프롬프트를 입력받아서 응답 생성
        return self.query.generate(prompt)


# =================================================================================================
# class LLMEvaluation(Model):
# Purpose: This class defines the evaluation process.
# =================================================================================================
class LLMEvaluation(Model):
    def __init__(self, 
                 adapter: str, 
                 max_new_tokens: int,
                 model_type: str,
                 **kwargs):
        super().__init__(**kwargs)
        """
        1. 토크나이저를 로드합니다.
        2. 기본 모델을 로드합니다.
        3. 어댑터 모델을 로드하고 머지합니다.
        4. 쿼리 인스턴스를 초기화합니다.
        5. 인퍼런스에서는 임의 문장을 입력하면, 해당 문장에 대한 답변을 생성합니다.
        """
        # 데이터셋 로드
        dataset = self.load_dataset()
        self.train_dataset, self.test_dataset = dataset.get_split_dataset()

        # 토크나이저 로드
        self.tokenizer = self.load_tokenizer()
        # 프롬프트 제네레이터 로드
        self.generator  = self.load_generator()

        # 기본 모델 로드
        self.model = self.load_model(mode=False, quantization_config=None)

        # 어댑터 모델을 로드하고 머지
        if adapter is None:
            self.model = self.model
        else:
            self.model = self.merge_adapter(adapter)

    def run(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Evaluate the model on the test dataset
        for idx in tqdm(range(0, len(self.test_dataset), 1)): 
            batch  = self.test_dataset[idx:idx+1]
            batch  = self.generator(batch)
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

            input_ids      = inputs.input_ids.to("cuda")
            attention_mask = inputs.attention_mask.to("cuda")

            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss    = outputs.loss
                total_loss  += loss.cpu().item()
                num_batches += 1

        avg_loss   = total_loss / num_batches
        perpleixty = torch.exp(torch.tensor(avg_loss)).cpu().item()

        print(f"Average loss: {avg_loss}")
        print(f"Perplexity: {perpleixty}")