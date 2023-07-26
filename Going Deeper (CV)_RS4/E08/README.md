# Code Peer Review Templete
- 코더 : 장승우
- 리뷰어 : 이효준

# PRT(PeerReviewTemplate)
- [⭕] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  ![](https://velog.velcdn.com/images/joonlaxy/post/6663e354-c10f-476c-ac27-fde92830ecfb/image.png)
  > 네, 잘 해결되었습니다.

- [⭕] 2.주석을 보고 작성자의 코드가 이해되었나요?
```python
def extract_keypoints_from_heatmap(heatmaps): #히트맵을 기반으로 예측된 각 관절(키포인트)의 좌표를 추출하는 함수
    max_keypoints = find_max_coordinates(heatmaps)
    #히트맵을 모서리를 0으로 패딩하여, 히트맵 내에서 관절 좌표를 기반으로 하위 픽셀을 검사
    padded_heatmap = np.pad(heatmaps, [[1,1],[1,1],[0,0]], mode='constant')
    adjusted_keypoints = []
    for i, keypoint in enumerate(max_keypoints):
        max_y = keypoint[1]+1 #해당 키포인트를 중심으로 3x3 패치
        max_x = keypoint[0]+1
        
        patch = padded_heatmap[max_y-1:max_y+2, max_x-1:max_x+2, i] #3x3 패치를 추출
        patch[1][1] = 0 #패치 중앙의 값(해당 키포인트의 값)을 0으로 설정
        
        index = np.argmax(patch) #패치에서 가장 높은 값(다음 최대값)의 인덱스
        
        next_y = index // 3
        next_x = index - next_y * 3
        delta_y = (next_y - 1) / 4 #다음 최대값과 현재 최대값 사이의 y좌표 차이를 계산
        delta_x = (next_x - 1) / 4
        
        adjusted_keypoint_x = keypoint[0] + delta_x # 현재 키포인트의 x좌표에 x좌표 차이를 더하여 조정된 x좌표를 계산
        adjusted_keypoint_y = keypoint[1] + delta_y
        adjusted_keypoints.append((adjusted_keypoint_x, adjusted_keypoint_y))
        
    adjusted_keypoints = np.clip(adjusted_keypoints, 0, 64) #추정된 키포인트 좌표를 이미지 크기에 맞도록 0에서 64 사이의 값으로 클리핑
    normalized_keypoints = adjusted_keypoints / 64 #0에서 1 사이의 정규화된 좌표로 변환
    return normalized_keypoints #이미지 내 각 관절의 상대적인 위치
```
  > 친절한 주석덕에 잘 이해될 수 있었습니다.

- [❌] 3.코드가 에러를 유발한 가능성이 있나요?
  > 아니요, 없었씁니다.
  
- [⭕] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
```python
def Simplebaseline(input_shape=(256, 256, 3)):
    resnet = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet')

    def _make_deconv_layer(num_deconv_layers):
        seq_model = tf.keras.models.Sequential()

        for _ in range(num_deconv_layers):
            seq_model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', activation='relu'))
            seq_model.add(BatchNormalization(momentum=0.1))
            seq_model.add(ReLU())
        return seq_model

    upconv = _make_deconv_layer(3)

    final_layer = tf.keras.layers.Conv2D(16, kernel_size=1,strides=1,padding='same')  # Assuming 16 as the number of output classes

#     def Simplebaseline(input_shape=(256, 256, 3)): #scope 에러나서 함수 안에 넣으니 됨
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder (ResNet50)
    x = resnet(inputs, training=False) 

    # Decoder (Upconvolution layers)
    x = upconv(x)

    # Final output layer
    out = final_layer(x)

    model = tf.keras.Model(inputs, out, name='simple_baseline')
    return model

# Create the Simplebaseline model
sbl_model = Simplebaseline()

# Print model summary
sbl_model.summary()
```
  > 네, Simplebaseline()을 잘 이해하고 구현하셨습니다.

- [⭕] 5.코드가 간결한가요?
```python
sbl = Simplebaseline()
sbl.load_weights(SBL_WEIGHTS_PATH)

image, keypoints = predict(sbl, test_image)
draw_keypoints_on_image(image, keypoints)
draw_skeleton_on_image(image, keypoints)
```
  > 네! Simple baseline 모델과 weight파일을 불러와 image에 keypoint까지 간결하게 표현하였습니다.

# 참고 링크 및 코드 개선 여부
