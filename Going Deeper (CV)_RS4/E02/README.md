# Code Peer Review Templete
- 코더 : 장승우
- 리뷰어 : 이동익

# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 체크하고 확인하여 작성한 코드에 적용하세요.
- [⭕] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
>  Augmentation, CutMix와 MixUp을 적용한 데이터셋으로 훈련한 각각의 ResNet 모델이 수렴하였고,
>  성능 비교시 CumMix를 적용한 모델이 좋은 일반화 성능을 보였습니다.
- [⭕] 2.주석을 보고 작성자의 코드가 이해되었나요?
>  주요 포인트마다 주석이 있어 코드 파악이 쉬웠습니다.
```python
as_supervised=True, #지도학습용으로 데이터에 레이블 값 포함되어있음
ds = ds.map(onehot,num_parallel_calls=4) # onehot을 하면 에러나서 sparse인 부분에선 제외
```
- [❌] 3.코드가 에러를 유발한 가능성이 있나요?
- [⭕] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
```python
def apply_normalize_on_dataset(ds, is_test=False, batch_size=batch_size,
                               with_aug=False, with_cutmix=False, with_mixup=False,is_sparse=False):
    ds = ds.map(normalize_and_resize_img, num_parallel_calls=4) # lms 코어수 4개  
```
```python
resnet50_sparse = keras.models.Sequential([
    keras.applications.resnet.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(*image_size,), #image_size 변수로 넘기면 None,None,3으로 넘어감;
        pooling='avg',
    ),
    keras.layers.Dense(num_classes, activation='softmax')
])
```
> 여러인자를 받을때 *arg를 쓸 수 있다는 사실,  lms 코어가 4개라는 사실을 알았습니다! 
- [⭕] 5.코드가 간결한가요?

# 참고 링크 및 코드 개선 여부
> 데이터 증강 tf core: https://www.tensorflow.org/tutorials/images/data_augmentation?hl=ko   
> 전이학습 및 미세조정 tf core: https://www.tensorflow.org/guide/keras/transfer_learning?hl=ko
