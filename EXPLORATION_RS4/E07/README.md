# Code Peer Review Templete
- 코더 : 장승우
- 리뷰어 : 소용현

# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 체크하고 확인하여 작성한 코드에 적용하세요.
- [X] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
> sketch, colored를 실수하여 원본이미지가 인풋으로 주어지고, 스케치 이미지가 정답 데이터로 되어 학습되었다.
```
def load_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, 3) # 데이터를 디코딩하여 TensorFlow 텐서로 변환, 채널수
    
    w = tf.shape(img)[1] // 2 
    sketch = img[:, :w, :] 
    sketch = tf.cast(sketch, tf.float32) # TensorFlow의 데이터 타입 변환 함수 
    colored = img[:, w:, :] 
    colored = tf.cast(colored, tf.float32)
    return normalize(sketch), normalize(colored)
```
스케치를 이용하여 페이크 이미지를 생성하지 않고, 랜덤 노이즈를 이용하여 페이크 이미지를 만들었다.
```
@tf.function # TensorFlow의 그래프 모드에서 실행
def train_step(real_sketch, colored): # color 넣고 예측 스케치 생성 
    noise = tf.random.normal([real_sketch.shape[0], real_sketch.shape[1], 
                              real_sketch.shape[2], real_sketch.shape[3]])
    with tf.GradientTape() as gene_tape, tf.GradientTape() as disc_tape:
        fake_sketch = generator(noise)

        real_disc = discriminator(real_sketch, colored)
        fake_disc = discriminator(fake_sketch, colored)

        gene_loss, l1_loss = get_gene_loss(fake_sketch, real_sketch, fake_disc)
        disc_loss = get_disc_loss(fake_disc, real_disc)

    gene_gradient = gene_tape.gradient(gene_loss, generator.trainable_variables)
    disc_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gene_opt.apply_gradients(zip(gene_gradient, generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_gradient, discriminator.trainable_variables))
    return gene_loss, l1_loss, disc_loss
```
- [⭕] 2.주석을 보고 작성자의 코드가 이해되었나요?
```
@tf.function() # 빠른 텐서플로 연산을 위해 @tf.function()을 사용합니다. 
def apply_augmentation(sketch, colored):
    stacked = tf.concat([sketch, colored], axis=-1) #-1 차원 기준으로 합치기
    
    _pad = tf.constant([[30,30],[30,30],[0,0]]) #패딩 추가
    if tf.random.uniform(()) < .5: # 조건문을 사용하여 50%의 확률로 패딩 방식을 선택합니다.
        padded = tf.pad(stacked, _pad, "REFLECT") # "REFLECT" 패딩 방식은 이미지 경계에서 대칭적으로 값을 반사시키는 방식
    else:
        padded = tf.pad(stacked, _pad, "CONSTANT", constant_values=1.) # 주어진 상수 값으로 패딩을 추가

    out = image.random_crop(padded, size=[256, 256, 6]) # 이미지를 임의의 위치에서 크기 256x256으로 자르기
    
    out = image.random_flip_left_right(out) # 이미지를 좌우로 무작위로 뒤집기
    out = image.random_flip_up_down(out) # 이미지를 상하로 무작위로 뒤집기
    
    if tf.random.uniform(()) < .5: #  50%의 확률로 이미지를 임의로 회전
        degree = tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32)
        out = image.rot90(out, k=degree)
    
    return out[...,:3], out[...,3:]   # out[...,:3]는 증강된 스케치 이미지, out[...,3:]는 증강된 컬러 이미지
 ```
 상세한 주석으로 이해가 잘 되었다.
- [❌] 3.코드가 에러를 유발한 가능성이 있나요?
- [⭕] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
- [⭕] 5.코드가 간결한가요?

# 참고 링크 및 코드 개선 여부
