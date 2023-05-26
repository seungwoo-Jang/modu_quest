# Code Peer Review Templete
- 코더 : 장승우
- 리뷰어 : 이성주

# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 체크하고 확인하여 작성한 코드에 적용하세요.
- [⭕] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  - 1. Abstractive 모델 구성을 위한 텍스트 전처리 단계가 체계적으로 진행되었다
      - text와 headline의 전처리를 분석단계, 정제단계, 정규화와 불용어 제거, 데이터셋 분리, 인코딩 과정을 모두 진행 하고 그래프로 확인까지 진행하였습니다.
  - 2. 텍스트 요약모델이 성공적으로 학습되었음을 확인하였다.  
    -  train loss와 validation loss가 감소하는 경향을 그래프를 통해 확인하였습니다.
    -  ![image](https://github.com/seungwoo-Jang/modu_quest/assets/29011595/c6c9a629-257a-4632-9868-6aca4043b405)
  - 3. Extractive 요약을 시도해 보고 Abstractive 요약 결과과 함께 비교해 보았다.
- [⭕] 2.주석을 보고 작성자의 코드가 이해되었나요?
 - 각 문단별로 주석을 달아서 코드 이해에 도움이 되었습니다.
  ``` python
  
# 인코더 설계
encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h3, state_c3])

# 이전 시점의 상태들을 저장하는 텐서
decoder_state_input_h = Input(shape=(hidden_size,))
decoder_state_input_c = Input(shape=(hidden_size,))

dec_emb2 = dec_emb_layer(decoder_inputs)

# 문장의 다음 단어를 예측하기 위해서 초기 상태(initial_state)를 이전 시점의 상태로 사용. 이는 뒤의 함수 decode_sequence()에 구현
# 훈련 과정에서와 달리 LSTM의 리턴하는 은닉 상태와 셀 상태인 state_h와 state_c를 버리지 않음.
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

# 어텐션 함수
decoder_hidden_state_input = Input(shape=(text_max_len, hidden_size))
attn_out_inf = attn_layer([decoder_outputs2, decoder_hidden_state_input])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])
```
     
- [❌] 3.코드가 에러를 유발한 가능성이 있나요?
  - 없습니다.
 
- [⭕] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
  - ``` data['text'] = data['text'].apply(lambda x: x if text_min_len <= len(x.split()) <= text_max_len else '')```
    - max길이와 min길이의 제한을 두어 정규화를 진행 한점 등, 인터뷰를 통해 코드를 제대로 이해하고 작성한것으로 보입니다. 
- [⭕] 5.코드가 간결한가요?
  - range와 join을 이용하여 간결한 코드 구현을 하였습니다.
  ``` python
  for i in range(50,60,5): #온점이 있는 문장에 최소 길이 제한이 있어서 5개씩 잡고함
    print(i)
    test_data = '\n'.join(data['text'][(i-5):i])
    print('Headlines:',summarize(test_data)) 
  ```

# 참고 링크 및 코드 개선 여부
``` python
for i in range(50,60,5): #온점이 있는 문장에 최소 길이 제한이 있어서 5개씩 잡고함
    print(i)
    test_data = '\n'.join(data['text'][(i-5):i])
    print('Headlines:',summarize(test_data))
```
summarize의 ratio를 잘 조절하여 문장을 합치지 않고 구현하면 더 좋았을 것 같습니다.

``` print("예측 요약 : ", summarize(_text, ratio=0.35)) ```
    
 
