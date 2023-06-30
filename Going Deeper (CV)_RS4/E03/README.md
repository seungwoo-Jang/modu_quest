# Code Peer Review Templete
- 코더 : 장승우
- 리뷰어 : 

# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 체크하고 확인하여 작성한 코드에 적용하세요.
- [⭕] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?

  - 주어진 3가지의 문제를 전부 해결 했습니다\

    ```python
    item = get_one(ds_test)
    print(item['label'])
    plt.imshow(item['image'])
    plt.show()
    
    def get_bbox(cam_image, score_thresh=0.05):
        low_indicies = cam_image <= score_thresh #array에서 해당 조건에 따라 True,False 리턴
        cam_image[low_indicies] = 0 # True인 대상 0으로 변경 
        cam_image = (cam_image*255).astype(np.uint8) #기존 이미지로 복구
        
        #윤곽선 찾는 함수
        #cv2.RETR_TREE: 윤곽선 검출 모드로, 모든 윤곽선을 계층 구조로 반환
        #cv2.CHAIN_APPROX_SIMPLE: 윤곽선을 압축하여 저장하는 방식으로, 윤곽선을 구성하는 좌표 중 중요한 좌표만 저장
        #첫 번째 반환값은 윤곽선(contours) 정보이고, 두 번째 반환값은 계층 정보(hierarchy)
        contours,_ = cv2.findContours(cam_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #일반적으로 contours 리스트에는 여러 개의 윤곽선 정보가 포함될 수 있습니다. 
        #첫 번째 윤곽선은 대부분 가장 외곽에 있는 윤곽선이거나, 계층 구조에서 최상위에 있는 윤계층의 윤곽선
        cnt = contours[0] #[0] 해도 contours랑 같음; 다차원일 경우 각기 다른 윤곽선 정보 갖고있음
    
        rotated_rect = cv2.minAreaRect(cnt) # 윤곽선을 둘러싸는 최소 회전 사각형
        rect = cv2.boxPoints(rotated_rect) #회전 사각형의 꼭지점 반환
        rect = np.int0(rect) # 꼭지점 좌표를 정수형으로 변환
        return rect
    
    cam_image = generate_cam(cam_model, item)
    c_rect = get_bbox(cam_image) #shape (4,2)
    
    bbox_c_image = copy.deepcopy(item['image'])
    #이미지에 윤곽선을 그리기 위한 함수
    bbox_c_image = cv2.drawContours(bbox_c_image, [c_rect], 0, (0,0,255), 2) #이미지, 윤곽선 정보, 인덱스, 윤곽선 색상, 두께
    plt.imshow(bbox_c_image)
    plt.show()
    
    grad_cam_image = generate_grad_cam(cam_model, 'conv5_block3_out', item)
    gc_rect = get_bbox(grad_cam_image) #shape (4,2)
    
    bbox_gc_image = copy.deepcopy(item['image'])
    #이미지에 윤곽선을 그리기 위한 함수
    bbox_gc_image = cv2.drawContours(bbox_gc_image, [gc_rect], 0, (0,0,255), 2) #이미지, 윤곽선 정보, 인덱스, 윤곽선 색상, 두께
    plt.imshow(bbox_gc_image)
    plt.show()
    ```

    성공적으로 출력 확인

- [⭕] 2.주석을 보고 작성자의 코드가 이해되었나요?

  - 네 특히 cam 정리에서 잘 되어있었습니다

    ```python
    cam_model = tf.keras.models.load_model(cam_model_path)
    
    def generate_cam(model, item):
        item = copy.deepcopy(item)
        width = item['image'].shape[1] #가로
        height = item['image'].shape[0] #세로
        
        img_tensor, class_idx = normalize_and_resize_img(item)
        
        # 학습한 모델에서 원하는 Layer의 output을 얻기 위해서 모델의 input과 output을 새롭게 정의해줍니다.
        cam_model = tf.keras.models.Model([model.inputs], [model.layers[-3].output, model.output])
        #배치 축 추가로 형태 맞춰서 모델 전달 (layer result,softmax result)
        #마지막 acitvation의 output과 dense output
        conv_outputs, predictions = cam_model(tf.expand_dims(img_tensor, 0)) 
        conv_outputs = conv_outputs[0, :, :, :] # 첫번째 결과 선택 > 1,7,7,2048 > 7,7,2048
        
        # 모델의 weight activation은 마지막 layer에 있습니다. 
        # softmax층의 가중치 가져오기 2048,120, 0 = 가중치, 1 = 편향
        class_weights = model.layers[-1].get_weights()[0] 
        
        cam_image = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2]) # 7,7
        for i, w in enumerate(class_weights[:, class_idx]): # item 클래스 이미지의 채널 가중치 
            # conv_outputs의 i번째 채널과 i번째 weight를 곱해서 누적하면 활성화된 정도가 나타날 겁니다.
            cam_image += w * conv_outputs[:, :, i] # 빈 이미지에 채널별로 레이어에서 나온 값 곱해서 더하기
    
        cam_image /= np.max(cam_image) # activation score를 normalize합니다.
        cam_image = cam_image.numpy()
        cam_image = cv2.resize(cam_image, (width, height)) # 원래 이미지의 크기로 resize합니다.
        return cam_image
    
    cam_image = generate_cam(cam_model, item) 
    plt.imshow(cam_image) # 가중치가 높았던 픽셀의 경우 밝게 표시
    plt.show()
    ```

    

- [❌] 3.코드가 에러를 유발한 가능성이 있나요?

  -  WARNING도 없이 잘 돌아가서 딱히 에러 유발성은 보이지 않습니다

- [⭕] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?

  - 네 질문하신거에 전부 대답했습니다

    ```python
    conv_outputs, predictions = cam_model(tf.expand_dims(img_tensor, 0)) 
    ```

    - 이 코드에 대한 대답도 완벽하게 하셨습니다

- [⭕] 5.코드가 간결한가요?

  - 네 제가 더 줄일만한 부분은 찾지 못하였습니다
  - 모듈화도 잘 되어 있었습니다


# 참고 링크 및 코드 개선 여부

이번 프로젝트가 크게 어렵지  않아서 그런지 전체적으로 잘 작성된 느낌입니다

제 수준에서 딱히 개선할점은 찾지 못한거같습니다.
