# 쇼우나우
title: 쇼우나우
subtitle: shownow
categories: shownow
tags: 시각장애인을 위한 편의점 음료 구매앱
date: 2023-02-22 10:11:09 +0000
last_modified_at: 2023-02-22 10:11:09 +0000
---

<aside>
👉 **Basic Info.**
⌛ 2023.01.03 ~ 2023.02.17
https://github.com/dbtjr1103/mainpj
🛠️ Python, Pytorch, Java

🔫 Visual Studio Code, Colab(Pro/+), AWS, Git, Android Studio

</aside>

<aside>
💡 **Docs.**

[발표 PPT](%E1%84%89%E1%85%AD%E1%84%8B%E1%85%AE%E1%84%82%E1%85%A1%E1%84%8B%E1%85%AE%2052bb8b05d39e42ed896b667f9b179ee3/%25EC%2587%25BC%25EC%259A%25B0%25EB%2582%2598%25EC%259A%25B0_%25EC%25B5%259C%25EC%25A2%2585_7_230217.pdf)

</aside>

**목차**

# 개요

---

- 시각장애인을 위한 편의점 음료 인식 서비스 제공 어플 ‘쇼우나우’
- ‘쇼우나우’를 통한 기대효과

    - 상품 선택시 더욱 빨라지는 속도

    - 타인의 도움 없이 원활해지는 상품 구매 

### 👀 시각장애인의 시야

![2023-02-21 15 33 46.jpg](%E1%84%89%E1%85%AD%E1%84%8B%E1%85%AE%E1%84%82%E1%85%A1%E1%84%8B%E1%85%AE%2052bb8b05d39e42ed896b667f9b179ee3/2023-02-21_15_33_46.jpg)

### 🧃 시각장애인이 음료를 고르려면?

[<출처: **원샷한솔OneshotHansol  -**  [https://www.youtube.com/@OneshotHansol](https://www.youtube.com/@OneshotHansol)>](https://youtu.be/fq5xQaWaMO0?t=146)

<출처: **원샷한솔OneshotHansol  -**  [https://www.youtube.com/@OneshotHansol](https://www.youtube.com/@OneshotHansol)>

### 📱 시각장애인도 어플을 사용할 수 있습니다.

[<출처: **우령의 유디오** - [https://www.youtube.com/@Youdio-wooryeong](https://www.youtube.com/@Youdio-wooryeong)>](https://youtu.be/nr6O58qWRG8)

<출처: **우령의 유디오** - [https://www.youtube.com/@Youdio-wooryeong](https://www.youtube.com/@Youdio-wooryeong)>

# 수행 역할과 일정

---

### 👨‍👩‍👧‍👦 3조 구성 및 역할

| 이름 | 역할 | 개인 업무 | 공동 업무 |
| --- | --- | --- | --- |
| 이창재 | 팀장 | 모델 변환 | 모델 학습, 데이터 정제 |
| 김성모 | 팀원 | 웹 개발 시도 | 데이터 정제 |
| 박성혜 | 팀원 | Raw data 수집, 어플 UI 제작, PPT 제작 | 모델 학습, 데이터 정제 |
| 임보라 | 팀원 | 어플 개발, Raw data 수집, 발표 | 모델 학습, 데이터 정제 |
| 정유석 | 팀원 | 어플 개발, 모델 성능 개선 | 모델 학습, 데이터 정제 |

### 📆 수행일정

![수행일정.jpg](%E1%84%89%E1%85%AD%E1%84%8B%E1%85%AE%E1%84%82%E1%85%A1%E1%84%8B%E1%85%AE%2052bb8b05d39e42ed896b667f9b179ee3/%25EC%2588%2598%25ED%2596%2589%25EC%259D%25BC%25EC%25A0%2595.jpg)

# 데이터

---

### 🔎 데이터 수집

### 📷 직접 촬영

- 규모 : 약 700장
- 출처 : 알파코 근처 편의점 3곳

![                이마트24](%E1%84%89%E1%85%AD%E1%84%8B%E1%85%AE%E1%84%82%E1%85%A1%E1%84%8B%E1%85%AE%2052bb8b05d39e42ed896b667f9b179ee3/%25EA%25B7%25B8%25EB%25A6%25BC01.jpg)

                이마트24

![                   GS25](%E1%84%89%E1%85%AD%E1%84%8B%E1%85%AE%E1%84%82%E1%85%A1%E1%84%8B%E1%85%AE%2052bb8b05d39e42ed896b667f9b179ee3/%25EA%25B7%25B8%25EB%25A6%25BC02.jpg)

                   GS25

![                      CU](%E1%84%89%E1%85%AD%E1%84%8B%E1%85%AE%E1%84%82%E1%85%A1%E1%84%8B%E1%85%AE%2052bb8b05d39e42ed896b667f9b179ee3/%25EA%25B7%25B8%25EB%25A6%25BC03.jpg)

                      CU

### 💻 AI Hub

- 규모 : 약 800장
- 출처 : AI Hub 상품 이미지 데이터

[AI-Hub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=64)

![                     AI Hub 데이터 예시](%E1%84%89%E1%85%AD%E1%84%8B%E1%85%AE%E1%84%82%E1%85%A1%E1%84%8B%E1%85%AE%2052bb8b05d39e42ed896b667f9b179ee3/%25EA%25B7%25B8%25EB%25A6%25BC04.jpg)

                     AI Hub 데이터 예시

### 👩🏻‍💻 웹크롤링

- 규모 : 약 1800장
- 출처 : 다수의 검색 엔진

![그림05.jpg](%E1%84%89%E1%85%AD%E1%84%8B%E1%85%AE%E1%84%82%E1%85%A1%E1%84%8B%E1%85%AE%2052bb8b05d39e42ed896b667f9b179ee3/%25EA%25B7%25B8%25EB%25A6%25BC05.jpg)

### ✨ 데이터 규모와 정제 방식

### 데이터 규모(전처리 전)

| 종류 | 라벨 개수 | 전체 Bbox 개수 | 이미지 수 |
| --- | --- | --- | --- |
| 음료, 주류, 유제품 | 300개 | 9,280개 | 3,663장 |

### 데이터 규모(전처리 후)

| 종류 | 라벨 개수 | 전체 Bbox 개수 | 이미지 수 |
| --- | --- | --- | --- |
| 음료, 주류, 유제품 | 300개 | 27,840개 | 10,989장 |

### 데이터 정제 방식 (Roboflow로 진행)

1. Annotation : 수작업으로 객체 Bounding box 작업

![그림06.jpg](%E1%84%89%E1%85%AD%E1%84%8B%E1%85%AE%E1%84%82%E1%85%A1%E1%84%8B%E1%85%AE%2052bb8b05d39e42ed896b667f9b179ee3/%25EA%25B7%25B8%25EB%25A6%25BC06.jpg)

1. Augmentation : 각 이미지별 5% crop, 25% crop 진행

![그림07.jpg](%E1%84%89%E1%85%AD%E1%84%8B%E1%85%AE%E1%84%82%E1%85%A1%E1%84%8B%E1%85%AE%2052bb8b05d39e42ed896b667f9b179ee3/%25EA%25B7%25B8%25EB%25A6%25BC07.jpg)

# 모델

---

### 💼 YOLO Series?

- 가장 빠른 실시간 객체 인식 SOTA 모델 중 하나
- Papers with Code에서 COCO DataSet으로 Real-Time Object Detection 상위권 (2023.02.21)

![그림08.jpg](%E1%84%89%E1%85%AD%E1%84%8B%E1%85%AE%E1%84%82%E1%85%A1%E1%84%8B%E1%85%AE%2052bb8b05d39e42ed896b667f9b179ee3/%25EA%25B7%25B8%25EB%25A6%25BC08.jpg)

### ⚡ CPU 및 GPU에서 가장 빠른 YOLO 모델 → YOLOv5 Nano 선정

- 2022.11.29 LearnOpenCV 발표

[https://learnopencv.com/performance-comparison-of-yolo-models/](https://learnopencv.com/performance-comparison-of-yolo-models/)

### 📋 실제 3조 데이터로 모델 성능 비교

![그림09.jpg](%E1%84%89%E1%85%AD%E1%84%8B%E1%85%AE%E1%84%82%E1%85%A1%E1%84%8B%E1%85%AE%2052bb8b05d39e42ed896b667f9b179ee3/%25EA%25B7%25B8%25EB%25A6%25BC09.jpg)

- ‣

    - [https://github.com/ultralytics/yolov5/wiki](https://github.com/ultralytics/yolov5/wiki)

# 모델 평가 및 개선

---

### 1.  하이퍼 파라미터 핸들링

- YOLOv5 제작자 Ultralytics의 YOLOv5 nano 모델 학습 시 권장하는 하이퍼 파라미터 중, 
목적에 부합하지 않는 하이퍼 파라미터 세 가지 제거 및 수정

       ➀ Fliplr(좌우반전) : 0.5 → 0.0 (좌우반전 불필요)

       ➁ Mosaic(모자이크) : 1.0 → 0.0 (불규칙한 Crop을 막으며, 객체 크기 유지)

       ➂ Scale(확대/축소) : 0.5 → 0.2 (이미지 축소 방지 0.8~1.2배로 조정)

- 하이퍼 파라미터 조정 전/후 학습 데이터셋 비교 예시

![2023-02-21 23 14 04.jpg](%E1%84%89%E1%85%AD%E1%84%8B%E1%85%AE%E1%84%82%E1%85%A1%E1%84%8B%E1%85%AE%2052bb8b05d39e42ed896b667f9b179ee3/2023-02-21_23_14_04.jpg)

- 하이퍼 파라미터 조정 전/후 인퍼런스 비교 예시

![2023-02-21 23 15 23.jpg](%E1%84%89%E1%85%AD%E1%84%8B%E1%85%AE%E1%84%82%E1%85%A1%E1%84%8B%E1%85%AE%2052bb8b05d39e42ed896b667f9b179ee3/2023-02-21_23_15_23.jpg)

### 2.  데이터 핸들링

➀ Roboflow 내부 문제로 모델 학습을 위한 데이터 다운로드 과정에서 일부 라벨 소실 
    → 소실 데이터 제거

➁ AI Hub 원천 데이터 라벨링 오류 → Bounding box 및 라벨 수정

➂ 학습에 혼란을 줄 수 있는 가능성이 높은 이미지 삭제

![2023-02-21 23 19 57.jpg](%E1%84%89%E1%85%AD%E1%84%8B%E1%85%AE%E1%84%82%E1%85%A1%E1%84%8B%E1%85%AE%2052bb8b05d39e42ed896b667f9b179ee3/2023-02-21_23_19_57.jpg)

④ AP가 0.6 이상인 경우 인퍼런스에서 충분한 성능 확인
    → AP가 0.7 이하인 30개 객체에 대한 이미지 추가

![2023-02-21 23 21 57.jpg](%E1%84%89%E1%85%AD%E1%84%8B%E1%85%AE%E1%84%82%E1%85%A1%E1%84%8B%E1%85%AE%2052bb8b05d39e42ed896b667f9b179ee3/2023-02-21_23_21_57.jpg)

<aside>
💡 AP가 0.7 이하인 하위 30개 객체에 대해, 
이미지를 40장 추가시 전체 평균 mAP **6.6% 상승**, 80장 추가시 평균 mAP **8% 상승**함. 
또한, 최소 mAP를 0.24 에서 0.62 이상으로 향상시킴.

</aside>

# 어플

---

- ‘안드로이드 스튜디오’ 사용

### 어플 제작과정

![2023-02-21 22 33 38.jpg](%E1%84%89%E1%85%AD%E1%84%8B%E1%85%AE%E1%84%82%E1%85%A1%E1%84%8B%E1%85%AE%2052bb8b05d39e42ed896b667f9b179ee3/2023-02-21_22_33_38.jpg)

![2023-02-21 22 35 06.jpg](%E1%84%89%E1%85%AD%E1%84%8B%E1%85%AE%E1%84%82%E1%85%A1%E1%84%8B%E1%85%AE%2052bb8b05d39e42ed896b667f9b179ee3/2023-02-21_22_35_06.jpg)

![2023-02-21 22 35 29.jpg](%E1%84%89%E1%85%AD%E1%84%8B%E1%85%AE%E1%84%82%E1%85%A1%E1%84%8B%E1%85%AE%2052bb8b05d39e42ed896b667f9b179ee3/2023-02-21_22_35_29.jpg)

![2023-02-21 22 49 57.jpg](%E1%84%89%E1%85%AD%E1%84%8B%E1%85%AE%E1%84%82%E1%85%A1%E1%84%8B%E1%85%AE%2052bb8b05d39e42ed896b667f9b179ee3/2023-02-21_22_49_57.jpg)

### 시연영상1

[https://www.youtube.com/watch?v=KiOHH61wrDU](https://www.youtube.com/watch?v=KiOHH61wrDU)

### 시연영상2

[https://www.youtube.com/shorts/X6q9Tb-UsmQ](https://www.youtube.com/shorts/X6q9Tb-UsmQ)

# 평가와 확장

---

### 💥 한계점

1. 투명 용기 인식에 어려움
2. 상품의 패키징이 변화함에 따라 주기적인 업데이트 필요
3. 스마트폰 기종에 따른 UI와 인퍼런스 성능에 차이가 있어 개선 필요

### 🌱 추후 확장 방안

1. STT를 추가하여 소비자가 데이터 추가 요청 가능
2. 상품 별 이벤트 설명 가능
3. OCR을 추가하여 상품 별 유통기한 설명 가능
4. 상품군 확대 가능
5. 아이폰 어플 개발 가능

---

# References

1. https://github.com/ultralytics/yolov5
2. https://github.com/AarohiSingla/TFLite-Object-Detection-Android-App-Tutorial-Using-YOLOv5
3. [https://www.youtube.com/watch?v=ROn1_O2zEtk](https://www.youtube.com/watch?v=ROn1_O2zEtk)

---

# Presentation