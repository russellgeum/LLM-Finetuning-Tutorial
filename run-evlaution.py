import json
from module.configuration import *

# torch_dtype 인자를 실제 torch.dtype 객체로 변환
dtype_map = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32
    }
torch_dtype = dtype_map.get("bfloat16", torch.bfloat16)

# 사용 예제
dataset_path = 'dataset/milstruction/milstruction.json'
dataset = DatasetLoader(dataset_path, 0.01, 42)
train_dataset, test_dataset = dataset.get_split_dataset()

# 상수는 모두 대문자 사용
BASE_MODEL           = "beomi/gemma-ko-2b"
BASE_MODEL_CACHE     = "model/beomi-gemma-2b/model"
BASE_TOKENIZER_CACHE = "model/beomi-gemma-2b/tokenizer"
ADAPTER_MODEL        = "model/beomi-gemma-2b-mil-adapter/"


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=BASE_MODEL_CACHE)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, 
    cache_dir=BASE_MODEL_CACHE, 
    torch_dtype=torch_dtype, 
    device_map='auto'
    )
model = PeftModel.from_pretrained(
    model=model, 
    model_id=ADAPTER_MODEL, 
    torch_dtype=torch_dtype, 
    device_map='auto'
    )
model.merge_and_unload()

q = Query(tokenizer, tokenizer, 512, "gemma")


def run(prompt: str) -> str:
    """Generate a response for the given prompt."""
    return q.generate_response(prompt)


def evaluate_model(model, tokenizer, texts):
    model.eval()
    total_loss = 0
    num_batches = 0
    batch_size = 0
    
    for idx in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[idx: idx + batch_size]
        