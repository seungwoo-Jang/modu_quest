# Code Peer Review Templete
- 코더 : 장승우
- 리뷰어 : 사재원

# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 체크하고 확인하여 작성한 코드에 적용하세요.
- [⭕] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  네 잘 해결하였습니다. 프로젝트의 요구사항들을 잘 충족시켰습니다.
```
Epoch 9/100
565/565 [==============================] - 86s 152ms/step - loss: 1.9161 - val_loss: 4.2283

Epoch 00009: val_loss improved from 4.26029 to 4.22828, saving model to /aiffel/aiffel/ocr/model_checkpoint.hdf5
loss 값이 안정적으로 최소값에 도달했을때 가중치를 저장시켰습니다

이후 test sample에 대해서도 잘 결과가 나왔습니다
```

- [⭕] 2.주석을 보고 작성자의 코드가 이해되었나요?
    단계별로 주석을 달아주셔서 전체적인 흐름을 파악하기 좋았습니다
  ```
  # 데이터셋과 모델을 준비합니다
  train_set = MJDatasetSequence(TRAIN_DATA_PATH, label_converter, batch_size=BATCH_SIZE, character=TARGET_CHARACTERS, is_train=True)
  val_set = MJDatasetSequence(VALID_DATA_PATH, label_converter, batch_size=BATCH_SIZE, character=TARGET_CHARACTERS)
  model = build_crnn_model()

  # 모델을 컴파일 합니다
  optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.1, clipnorm=5)
  model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
  ```
  
- [❌] 3.코드가 에러를 유발한 가능성이 있나요?
- 아니요 없어보입니다.

- [⭕] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
네 전체적인 모델 흐름을 파악하고있습니다
- 
- [⭕] 5.코드가 간결한가요?

  네 적절한 들여쓰기릍 통해 코드 읽기 수월했습니다.
```

checkpoint_path = HOME_DIR + '/model_checkpoint.hdf5'
ckp = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=False,
        save_weights_only=True,
        verbose=1,
        )
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


history = model.fit(train_set,
          steps_per_epoch=len(train_set),
          epochs=20,
          validation_data=val_set,
          validation_steps=len(val_set),
          # ModelCheckPoint와 EarlyStopping을 활용하는 경우 바로 아래 callbacks 옵션에 주석을 풀어주세요.
          callbacks=[ckp, earlystop]
)

```
- 

# 참고 링크 및 코드 개선 여부
