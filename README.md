# LLM-Finetuning-Tutorial
This repository is a tutorial explaining how to fine-tune and infer with large language models (LLM). 
The repository uses beomi-gemma-2b, a Korean-tuned version of Google's Gemma-2B, 
and KULLM-V2 from the Korea University NLP lab as the default settings. 
(There are traces of EEVE-10.8B, but it has not been implemented.)
- [Huggingface-NLPAI-kullm-v2](https://huggingface.co/datasets/nlpai-lab/kullm-v2)
- [Huggingface-beomi-gemma-ko-2b](https://huggingface.co/beomi/gemma-ko-2b)


## Folder Structure
```
LLM-Finetuning-Tutorial/
    ├── dataset/
    │ ├── kullm-v2
    ├── model/
    │ ├── beomi-gemma-2b
    ├── module/
    │ ├── init.py
    │ ├── configuration.py
    │ ├── model.py
    ├── scripts/
    │ ├── requirements.sh
    ├── run-evaluation.py
    ├── run-inference.py
    ├── run-training.py
    ├── README.md
```

## Requirements
The packages required to run the project are listed in the scripts/requirements.sh file. To install them, run the following command:
```sh
bash scripts/requirements.sh
```

## Usage
### Model Training  
To train the model, run the run-training.py script.
```sh
python run-training.py
```

### Model Inference
To perform inference using the trained model, run the run-inference.py script.
```sh
python run-inference.py
```

### Model Evaluation
To evaluate the model, run the run-evaluation.py script.
```sh
(base) python run-evaluation.py
(custom) python run-evaluation.py --adapter "model/{your-custom-model-path}"
```
|          | Loss     | Perplexity |
|----------|----------|----------|
| Gemma-2B-Ko          | 1.867    | 6.47     |
| Gemma-2B-Ko-custom   | 1.760    | 5.81     |

## Example
The following example shows the results of training on a custom instruction-tuning dataset with 30,000 instances, rather than kullm-v2.
### Gemma-2B Base
```
질문: K-9 자주포에 대해서 알려주세요.
답변: 자주포는 1970년대에 개발된 무기로서,  
1970년대에 개발된 무기는 1980년대에 개발된 무기보다 훨씬 더 뛰어난 무기입니다.  
1970년대에 개발된 무기는 1980년대에 개발된 무기보다 훨씬 더 뛰어난 무기입니다.  
1970년대에 개발된 무기는 1980년대에 개발된 무기보다 훨씬 더 뛰어난 무기입니다.  
1970년대에 개발된 무기는 1980년대에 개발된 무기보다 훨씬 더 뛰어난 무기입니다. ... ...
```
### Gemma-2B-custom
```
질문: K-9 자주포에 대해서 알려주세요.
답변: K-9 자주포에 대해서 알려주세요.  
K-9 자주포는 미국의 M777 견인포를 기반으로 개발된 자주포로, 155mm 곡사포를 장착하고 있습니다.  
이 자주포는 빠른 발사 속도와 높은 정확도를 자랑하며, 다양한 지형에서 운용이 가능합니다.  
또한, 자동화된 사격 통제 시스템을 갖추고 있어 신속한 사격 준비가 가능합니다.  ****
K-9 자주포는 한국군의 주요 자주포로 사용되고 있으며, 다양한 작전 환경에서 탁월한 성능을 발휘합니다.  
K-9 자주포는 한국군의 전력 증강에 크게 기여하고 있습니다.
```
