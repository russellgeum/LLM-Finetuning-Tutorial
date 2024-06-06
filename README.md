# LLM-Finetuning-Tutorial
이 레포지토리는 대형 언어 모델(LLM)을 파인튜닝하고 추론하는 방법을 설명하는 튜토리얼입니다.  
해당 레포지토리는 Google Gemma-2B을 한국어 튜닝한 beomi-gemma-2b 그리고 고려대학교 NLP 연구실의 kullm-v2를 기본 세팅으로 합니다. (EEVE-10.8B는 구현하지 않음)

- [Huggingface-beomi-gemma-ko-2b](https://huggingface.co/beomi/gemma-ko-2b)
- [Huggingface-NLPAI-kullm-v2](https://huggingface.co/datasets/nlpai-lab/kullm-v2)

## 기본 폴더 구조
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
프로젝트를 실행하기 위해 필요한 패키지는 `scripts/requirements.sh` 파일에 명시되어 있습니다.  
이를 설치하려면 다음 명령어를 실행하세요:

```sh
bash scripts/requirements.sh
```

## Usage
### 모델 학습  
모델을 학습시키리면 'run-training.py' 스크립트를 실행합니다.
```sh
python run-training.py
```

### 모델 추론
학습된 모델을 사용하여 추론을 수행하려면 run-inference.py 스크립트를 실행합니다.  
```sh
python run-inference.py
```

### 모델 평가
모델을 평가하려면 run-evaluation.py 스크립트를 실행합니다.  
현재 모델 평가는 아직 구현되지 않았습니다.
```sh
python run-evlaution.py
```