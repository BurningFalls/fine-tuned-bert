# 1. Introduction

## 1.1 model

* [Hugging Face (https://huggingface.co/burningfalls/my-fine-tuned-bert)](https://huggingface.co/burningfalls/my-fine-tuned-bert)

## 1.2 examples

![examples](https://github.com/BurningFalls/algorithm-study/assets/30232837/596e5010-53b6-4598-8dd3-4ef7fc65e60e)

## 1.3 f1-score

![bert_accuracy](https://github.com/BurningFalls/algorithm-study/assets/30232837/58830340-aebe-4dc2-85fa-313138ac3020)

---

# 2. Requirements
```python
# my env
python==3.11.3
tensorflow==2.12.0
transformers==4.29.2

# maybe you need to
python>=3.6
tensorflow>=2.0
transformers>=4.0
```

---

# 3. Load
```python
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import TextClassificationPipeline

BERT_PARH = "burningfalls/my-fine-tuned-bert"

def load_bert():
    loaded_tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    loaded_model = TFAutoModelForSequenceClassification.from_pretrained(BERT_PATH)

    text_classifier = TextClassificationPipeline(
        tokenizer=loaded_tokenizer,
        model=loaded_model,
        framework='tf',
        top_k=1
    )
```

---

# 4. Usage
```python
import re
import sentiments

def predict_sentiment(text):
    result = text_classifier(text)[0]
    feel_idx = int(re.sub(r'[^0-9]', '', result[0]['label']))
    feel = sentiments.Feel[feel_idx]["label"]

    return feel
```

---

# 5. sentiments.py
```python
Feel = [
    {"label": "가난한, 불우한", "index": 0},
    {"label": "감사하는", "index": 1},
    {"label": "걱정스러운", "index": 2},
    {"label": "고립된", "index": 3},
    {"label": "괴로워하는", "index": 4},
    {"label": "구역질 나는", "index": 5},
    {"label": "기쁨", "index": 6},
    {"label": "낙담한", "index": 7},
    {"label": "남의 시선을 의식하는", "index": 8},
    {"label": "노여워하는", "index": 9},
    {"label": "눈물이 나는", "index": 10},
    {"label": "느긋", "index": 11},
    {"label": "당혹스러운", "index": 12},
    {"label": "당황", "index": 13},
    {"label": "두려운", "index": 14},
    {"label": "마비된", "index": 15},
    {"label": "만족스러운", "index": 16},
    {"label": "방어적인", "index": 17},
    {"label": "배신당한", "index": 18},
    {"label": "버려진", "index": 19},
    {"label": "부끄러운", "index": 20},
    {"label": "분노", "index": 21},
    {"label": "불안", "index": 22},
    {"label": "비통한", "index": 23},
    {"label": "상처", "index": 24},
    {"label": "성가신", "index": 25},
    {"label": "스트레스 받는", "index": 26},
    {"label": "슬픔", "index": 27},
    {"label": "신뢰하는", "index": 28},
    {"label": "신이 난", "index": 29},
    {"label": "실망한", "index": 30},
    {"label": "악의적인", "index": 31},
    {"label": "안달하는", "index": 32},
    {"label": "안도", "index": 33},
    {"label": "억울한", "index": 34},
    {"label": "열등감", "index": 35},
    {"label": "염세적인", "index": 36},
    {"label": "외로운", "index": 37},
    {"label": "우울한", "index": 38},
    {"label": "자신하는", "index": 39},
    {"label": "조심스러운", "index": 40},
    {"label": "좌절한", "index": 41},
    {"label": "죄책감의", "index": 42},
    {"label": "질투하는", "index": 43},
    {"label": "짜증내는", "index": 44},
    {"label": "초조한", "index": 45},
    {"label": "충격 받은", "index": 46},
    {"label": "취약한", "index": 47},
    {"label": "툴툴대는", "index": 48},
    {"label": "편안한", "index": 49},
    {"label": "한심한", "index": 50},
    {"label": "혐오스러운", "index": 51},
    {"label": "혼란스러운", "index": 52},
    {"label": "환멸을 느끼는", "index": 53},
    {"label": "회의적인", "index": 54},
    {"label": "후회되는", "index": 55},
    {"label": "흥분", "index": 56},
    {"label": "희생된", "index": 57},
]
```

---

# 6. Reference

* BERT: [klue/bert-base](https://huggingface.co/klue/bert-base)

* Dataset: [AI-Hub 감성 대화 말뭉치](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86)
