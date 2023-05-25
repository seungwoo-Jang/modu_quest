## **Code Peer Review Templete**
------------------
- 코더 : 장승우
- 리뷰어 : 김동규

## **PRT(PeerReviewTemplate)**
------------------  
- [O] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**

1. 아웃포커싱 효과가 적용된 인물 사진과 동물 사진, 그리고 크로마키 (성공)
2. 인물 사진에 대한 문제점 지적한 사진 (성공)
3. semantic segmentation mask의 오류 보와 방안 (없음)

- [O] **2. 주석을 보고 작성자의 코드가 이해되었나요?**

각 도구별 설명이 자세히 되어 있음
```python
# os: Operating System의 줄임말로, 운영체제에서 제공되는 여러 기능을 파이썬에서 사용할 수 있도록 함 (Ex. 디렉토리 경로 이동, 시스템 환경 변수 가져오기 등)
# urllib: URL 작업을 위한 여러 모듈을 모은 패키지. (Ex. urllib.request, urllib.parse, ...)
# cv2: OpenCV 라이브러리로, 실시간 컴퓨터 비전을 목적으로 한 프로그래밍 라이브러리
# numpy(NumPy): 행렬이나 대규모 다차원 배열을 쉽게 처리할 수 있도록 지원하는 라이브러리. 데이터 구조 외에도 수치 계산을 위해 효율적으로 구현된 기능을 제공
# pixellib: 이미지 및 비디오 segmentation을 수행하기 위한 라이브러리. 
# pixellib.semantic: segmentation 기법 중, semantic segmentation을 쉽게 사용할 수 있도록 만든 라이브러리
# matplotlib: 파이썬 프로그래밍 언어 및 수학적 확장 NumPy 라이브러리를 활용한 플로팅 라이브러리로, 데이터 시각화 도구
```

실행 단계별 설명이 잘 되어 있음
```python
 # 저장할 파일 이름을 결정합니다
    # 1. os.getenv(x)함수는 환경 변수x의 값을 포함하는 문자열 변수를 반환합니다. model_dir 에 "/aiffel/human_segmentation/models" 저장
    # 2. #os.path.join(a, b)는 경로를 병합하여 새 경로 생성 model_file 에 "/aiffel/aiffel/human_segmentation/models/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5" 저장
    # 1
    # model_dir = '/content/drive/MyDrive/Colab Notebooks/EXPLORATION_data/E05'
    # 2
    # model_file = '/content/drive/MyDrive/Colab Notebooks/EXPLORATION_data/E05/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5'
    
```

- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**

파일 다운로드 타깃의 존재 여부를 파악하여 런타임 에러를 사전에 방지
```python
if not os.path.exists(model_file):
        # PixelLib가 제공하는 모델의 url입니다
        model_url = 'https://github
```

- [O] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**

주석에 공부한 흔적이 자세히 드러나있다.
굳이 질문하지 않아도 얼마나 알아봤을지 보인다.
```python
# plt.imshow(): 저장된 데이터를 이미지의 형식으로 표시한다.
    # cv2.cvtColor(입력 이미지, 색상 변환 코드): 입력 이미지의 색상 채널을 변경
    # cv2.COLOR_BGR2RGB: 원본이 BGR 순서로 픽셀을 읽다보니
    # 이미지 색상 채널을 변경해야함 (BGR 형식을 RGB 형식으로 변경)   
    plt.imshow(cv2.cvtColor(img_orig_blur, cv2.COLOR_BGR2RGB))
    plt.show()

    # cv2.cvtColor(입력 이미지, 색상 변환 코드): 입력 이미지의 색상 채널을 변경
    # cv2.COLOR_BGR2RGB: 원본이 BGR 순서로 픽셀을 읽다보니
    # 이미지 색상 채널을 변경해야함 (BGR 형식을 RGB 형식으로 변경) 
    img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
```

- [O] **5. 코드가 간결한가요?**

한 셀 내에서 최대한 간결하게 표현된 것으로 보인다.
```python
 # 원본이미지를 img_show에 할당한뒤 이미지 사람이 있는 위치와 배경을 분리해서 표현한 color_mask 를 만든뒤 두 이미지를 합쳐서 출력
    img_show = img_orig.copy()

    # True과 False인 값을 각각 255과 0으로 바꿔줍니다
    img_mask = seg_map.astype(np.uint8) * 255

    # 255와 0을 적당한 색상으로 바꿔봅니다
    color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)

    # 원본 이미지와 마스트를 적당히 합쳐봅니다
    # 0.6과 0.4는 두 이미지를 섞는 비율입니다.
    img_show = cv2.addWeighted(img_show, 0.6, color_mask, 0.4, 0.0)

    plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
    plt.show()

```
위 코드의 경우에도 이미지를 복사해서 컬러맵을 바꾸는 등 작업하고 가중치를 통해 세그멘테이션과 섞어주는 코드로 보이는데
군더더기가 보이지 않는다.

코드의 대부분은 사람을 위한 주석뿐이다.

## **참고링크 및 코드 개선 여부**

코드가 너무 깔끔해서 깜짝 놀랐다.
다만, 아쉬운 것이 있다면 아래의 부분에 중복이 있었다.
```python
if 15 in segvalues['class_ids'] : #person
        #이미지 문제
        text = 'finger cut'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        color = (255, 0, 0)  # 텍스트 색상 (BGR 형식)

        # 텍스트를 그릴 위치 설정
        text_position = (450, 200)

        # 이미지에 텍스트 추가
        image_with_text = cv2.putText(img_concat, text, text_position, font, font_scale, color, thickness)
        plt.cla()
        plt.imshow(cv2.cvtColor(image_with_text, cv2.COLOR_BGR2RGB))
        plt.show()
        
        ...
```
이 부분까지 함수로 묶어줬다면, 간결성 및 가독성에서는 **완벽** 했을 것 같다.

    
