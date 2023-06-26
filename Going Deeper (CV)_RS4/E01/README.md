# Code Peer Review Templete
- 코더 : 장승우
- 리뷰어 : 김창완

# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 체크하고 확인하여 작성한 코드에 적용하세요.

1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?

- 정상적으로 동작하고 문제를 해결 했습니다.
- 다만 summury쪽이 주석 처리되어 있어서 이거만 풀면 제대로 나올 것 같습니다

2.주석을 보고 작성자의 코드가 이해되었나요?

- 전부 다는 아니지만 그래도 이해가 될 정도로 주석이 있었습니다.

```python
#블록 생성
def build_resnet_block(inputs,channel,is_50,is_plain,block_count):
  x = inputs
  for i in range(block_count):
    s = x # skip connection 지정
    if is_50 == False:
      if x.shape[-1] != channel:
        x = keras.layers.Conv2D(filters=channel,kernel_size=(3,3),strides=2,padding='same')(x)
        s = keras.layers.Conv2D(filters=channel,kernel_size=(1,1),strides=2,padding='same')(s) #스킵 커넥션 shape 같이 변경 
        s = keras.layers.BatchNormalization()(s)
      else:
        x = keras.layers.Conv2D(filters=channel,kernel_size=(3,3),padding='same')(x)
      x = keras.layers.BatchNormalization()(x)
      x = keras.layers.Activation('relu')(x)
```



3.코드가 에러를 유발한 가능성이 있나요?

- 에러를 유발하지 않습니다. 말끔하게 돌아갑니다

4.코드 작성자가 코드를 제대로 이해하고 작성했나요?

- 질문하신바에 전부 대답해 주셨습니다. 이해하고 짜셨습니다.

5.코드가 간결한가요?

- 코드가 간결하며 제가 더 간단하게 개선할 여지는 찾지 못했습니다.

# 참고 링크 및 코드 개선 여부

- Learning rate scheduling을 통해 학습률을 동적으로 조정해도 괜찮을것 같습니다
- 지금은 데이터 셋이 작지만 나중에 데이터가 커진다면 Learning Curve 확인 이후 EarlyStopping Callback을 이용해 조기종료 기법을 적용해도 좋을 것 같습니다.  제 경우에는 더 뛰어난 성능을 보여줬습니다.
