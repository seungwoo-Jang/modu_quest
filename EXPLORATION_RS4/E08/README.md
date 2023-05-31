# Code Peer Review Templete
- 코더 : 장승우
- 리뷰어 : 이성주

# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 체크하고 확인하여 작성한 코드에 적용하세요.
- [⭕] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?

|평가문항|상세기준|완료여부|
|-------|---------|--------|
|한국어 전처리를 통해 학습 데이터셋을 구축하였다.|공백과 특수문자 처리, 토크나이징, 병렬데이터 구축의 과정이 적절히 진행되었다.|preprocess_sentence(),tokenize_and_filter() 등을 이용하여 적절히 진행되었습니다. |
|트랜스포머 모델을 구현하여 한국어 챗봇 모델 학습을 정상적으로 진행하였다.|구현한 트랜스포머 모델이 한국어 병렬 데이터 학습 시 안정적으로 수렴하였다.|![image](https://github.com/seungwoo-Jang/modu_quest/assets/29011595/35f6530e-0713-4758-8a84-537de9d58c5d)   loss는 감소하며, accuracy는 증가하는 방향으로 안정적으로 수렴 하였습니다.|
|한국어 입력문장에 대해 한국어로 답변하는 함수를 구현하였다.|한국어 입력문장에 맥락에 맞는 한국어로 답변을 리턴하였다. |![image](https://github.com/seungwoo-Jang/modu_quest/assets/29011595/f9a767c0-5924-4081-a9ca-dccd5fea56ad)      한국어 질문에 한국어로 답변을 리턴하였습니다.|

- [⭕] 2.주석을 보고 작성자의 코드가 이해되었나요?
``` python
# 하이퍼파라미터
NUM_LAYERS = 2 # 인코더와 디코더의 층의 개수
D_MODEL = 256 # 인코더와 디코더 내부의 입, 출력의 고정 차원
NUM_HEADS = 8 # 멀티 헤드 어텐션에서의 헤드 수 
UNITS = 512 # 피드 포워드 신경망의 은닉층의 크기
DROPOUT = 0.1 # 드롭아웃의 비율
```
> 위 주석을 보면 코드 이해하는 것에 도움이 많이 되었습니다.

- [❌] 3.코드가 에러를 유발한 가능성이 있나요?
``` MAX_SAMPLES = len(data) # 11823개 ```
 > 하드코딩보다 data의 크기등을 확인하여 코드를 작성하여 에러를 유발할 가능성을 줄여 작성하였습니다.

- [⭕] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
``` python
# 전처리 함수
def preprocess_sentence(sentence):
  # 입력받은 sentence를 소문자로 변경하고 양쪽 공백을 제거
  sentence = sentence.lower().strip()

  # 단어와 구두점(punctuation) 사이의 거리를 만듭니다.
  # 예를 들어서 "I am a student." => "I am a student ."와 같이
  # student와 온점 사이에 거리를 만듭니다.
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)
  # (a-z, A-Z, ".", "?", "!", ",")를 제외한 모든 문자를 공백인 ' '로 대체합니다.
  sentence = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z?.!,]+', " ", sentence)
  sentence = sentence.strip()
  return sentence
  ```
  ``` MAX_LENGTH = 25 ```
> 전처리 함수에 한글의 모음과 자음을 포함시킨점, MAX_LENGTH를 25로 수동으로 변경한점, 그리고 인터뷰를 통해 코드를 이해하고 작성한 것을 확인하였습니다.
- [⭕] 5.코드가 간결한가요?
``` python
# 질문과 답변의 쌍인 데이터셋을 구성하기 위한 데이터 로드 함수
def load_conversations():
  inputs = data['Q'].apply(lambda x: preprocess_sentence(x))
  outputs = data['A'].apply(lambda x: preprocess_sentence(x))

  return inputs, outputs
  ```
   > 코드를 적절한 함수화와 lambda를 이용하여 간결하게 작성하였습니다.
   
# 참고 링크 및 코드 개선 여부
