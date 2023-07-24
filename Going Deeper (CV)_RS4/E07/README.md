# Code Peer Review Templete
- 코더 : 장승우
- 리뷰어 : 사재원

# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 체크하고 확인하여 작성한 코드에 적용하세요.
- [⭕] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  <br><br>네 ssd 모델 생성 및 학습 그리고 학습데이터 생성 까지 정상적으로 동작하였습니다.
```
Epoch: 1/1 | Batch 402/402 |         Batch time 0.079 || Loss: 6.224400 | loc loss:4.921696 |         class loss:1.302704  
```
  <br>```3. 이미지 속 다수의 얼굴에 스티커가 적용되었다.```<br> 위 평가 항목을 충족하였지만 landmark 추론과정에서 bbox를 ssd모델이 아닌 dlib모델을 이용한 부분이 아쉽습니다

- [⭕] 2.주석을 보고 작성자의 코드가 이해되었나요?
  <br><br>네 각 단계별로 주석을 작성해주셔서 코드가 쉽게 이해가 되었습니다
```
  PROJECT_PATH = os.getenv('HOME')+'/aiffel/face_detector'

# 이미지 경로 설정
img_path = os.path.join(PROJECT_PATH, 'sample.jpg')

# Dlib의 얼굴 검출기와 랜드마크 검출기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(PROJECT_PATH, 'shape_predictor_68_face_landmarks.dat'))

# 이미지를 불러온 후, 그레이스케일로 변환
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 검출
faces = detector(gray)

# 검출된 얼굴 랜드마크를 저장할 리스트 생성
landmark_list = []

for face in faces:
    # 랜드마크 검출
    landmarks = predictor(gray, face)
    
    # 랜드마크의 x, y 좌표를 리스트로 저장
    landmark_points = []
    for n in range(68):  # Dlib은 얼굴 랜드마크가 총 68개입니다.
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_points.append((x, y))
    
    landmark_list.append(landmark_points)

# 원본 이미지에 얼굴 랜드마크를 그려줍니다.
for landmarks in landmark_list:
    for point in landmarks:
        x, y = point
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

# 이미지를 화면에 보여줍니다. > 커널 죽음
# cv2.imshow("Facial Landmarks", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 이미지를 Matplotlib으로 표시합니다.
plt.imshow(image_rgb)
# plt.axis('off')  # 축 제거 (선택사항)
plt.show()
```
- [❌] 3.코드가 에러를 유발한 가능성이 있나요?
<br> 딱히 발견되지 않았습니다.<br> 
- [⭕] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
  <br> 네 각 단계별로 파이프라인을 설계하고 모델을 완성 시킨걸로보아 잘 이해한 것 같습니다 <br>
- [⭕] 5.코드가 간결한가요?
  <br><br> 네 적절한 들여쓰기로 인하여 가독성이 좋았습니다
```
model = SsdModel()

tf.keras.utils.plot_model(
    model,
    to_file=os.path.join(os.getcwd(), 'model.png'),
    show_shapes=True,
    show_layer_names=True
)

steps_per_epoch = DATASET_LEN // BATCH_SIZE
learning_rate = MultiStepWarmUpLR(
    initial_learning_rate=1e-2,
    lr_steps=[e*steps_per_epoch for e in [50, 70]],
    lr_rate=0.1,
    warmup_steps=5*steps_per_epoch,
    min_lr=1e-4
)
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)

multi_loss = MultiBoxLoss(len(IMAGE_LABELS), neg_pos_ratio=3)

img_raw = cv2.imread(os.path.join(os.getcwd(), 'model.png'))
plt.figure(figsize=(18, 16))
plt.imshow(img_raw)
plt.show()
```

# 참고 링크 및 코드 개선 여부
