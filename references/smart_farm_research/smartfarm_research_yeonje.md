# 토마토 익음도 데이터셋과 AI 활용 

## 문제배경 
스마트팜에서 토마토 수확 적기 판단은 농가 수익성과 직결되는 핵심 과제이다.  
기존에는 농부의 경험적 판단이나 단순 색상 기준에 의존했지만, 이는 **사람마다 기준이 달라** 주관적 편차가 발생하고, 대규모 재배 환경에서는 일관성을 유지하기 어렵다.  

## 문제점 
- **객관성 부족**: 동일한 익음 상태라도 농부마다 "수확 적기" 판단이 다를 수 있음.  
- **데이터 제약**: 공개된 데이터셋은 대부분 병해 진단용(PlantVillage 등)에 치중되어 있으며, **익음도 데이터셋은 제한적**임.  
- **경계 상태 부족**: 미숙 ↔ 반숙 ↔ 완숙 사이의 애매한 상태가 데이터에 충분히 반영되지 않아 모델 학습에 한계 발생.  

## 해결방법 
- **공개 데이터셋 활용 및 통합**  
  - PlantVillage, Kaggle Tomato Leaf Disease Dataset → 병해 진단용 이미지 확보  
  - Tomato Maturity Detection, tomatOD dataset → 익음 단계 라벨 포함된 데이터셋 활용  
  - FGrade dataset → 신선도·부패 정도 판단에 응용 가능  
- **sML/AI 기반 이상 탐지**  
  - 모델이 익음도 라벨 간 애매한 경계 데이터를 인식 → **사람 판단 필요 알림 시스템** 구축  
  - 다중 센서(이미지 + 환경데이터) 융합 → 정확도 향상  

## 기대효과 
- **수확 적기 예측 정밀화** → 농가 수익성 향상  
- **데이터 기반 일관성 확보** → 농부 간 기준 차이 해소  
- **노동 효율 증대** → 사람이 모든 토마토를 일일이 확인하지 않아도 됨  
- **연구 확장성** → 병해 탐지, 품질 등급화, 신선도 예측 등으로 확장 가능  

---

# 토마토 익음도 연구 현황 정리 

## 토마토 이미지 데이터셋 개요
- **tomatOD Dataset**  
  - 온실 촬영, 바운딩 박스 + 익음 단계(미숙/반숙/완숙) 라벨 포함  
- **Tomato Maturity Detection Dataset**  
  - 익음 3단계 라벨 제공, 약 800여 장 이미지  
- **FGrade Dataset**  
  - 신선도/부패 정도 10단계 라벨 포함 (6천여 장)  
- **Mendeley Real-world Tomato Dataset**  
  - 성장 단계·다양한 환경 조건 포함, 현실성 높음


# 토마토 이미지 데이터셋 개요 및 링크 🔗

| 데이터셋 이름 | 특징 / 포함 라벨 | 링크 / 출처 |
|---|---|---|
| **Tomato Maturity Detection & Quality Grading Dataset** | 익음 여부(immature / mature) + 신선/부패 구분 | Mendeley: *Tomato Maturity Detection and Quality Grading Dataset* :contentReference[oaicite:0]{index=0} |
| **Laboro Tomato** | 다양한 익음 단계 포함, 객체 탐지 및 분할 가능 | Kaggle: Laboro Tomato 데이터셋 :contentReference[oaicite:1]{index=1} |
| **KUTomaData (in “Tomato Maturity Recognition with Convolutional Transformers” 논문)** | 온실에서 찍은 익음 단계 이미지 약 700장 (Unripe / Half / Fully ripe) | 해당 논문: *Tomato Maturity Recognition with Convolutional Transformers* :contentReference[oaicite:2]{index=2} |
| **Tomato fruits dataset for binary & multiclass classification** | 익음 상태 + 불량 상태 포함 (binary, multiclass) | Mendeley: *Tomato fruits dataset* :contentReference[oaicite:3]{index=3} |
| **Healthy Tomato Image Dataset** | 건강한 토마토 이미지 위주 (품질/병해 판별 응용 가능) | Mendeley: *Healthy Tomato Image Dataset* :contentReference[oaicite:4]{index=4} |
| **TOMATO (good / bad 클래스)** | 토마토를 “좋음 / 나쁨”으로 구분 | Mendeley: *TOMATO* 데이터셋 :contentReference[oaicite:5]{index=5} |
| **Spoiled & fresh fruit inspection dataset** | 토마토 포함, 신선 vs 부패 상태 이미지 포함 | Mendeley: *Spoiled and fresh fruit inspection dataset* :contentReference[oaicite:6]{index=6} |


---

## 관련 연구 논문 분석 

### 1. *Tomato Maturity Recognition with Convolutional Transformers*  
- **주요 내용**  
  - CNN + Transformer 하이브리드 구조로 토마토 익음도 분류 제안  
  - `KUTomaData`라는 새 데이터셋 소개: 다양한 조명·카메라 조건 반영  
  - Laboro Tomato, Rob2Pheno Annotated Tomato 데이터셋과 비교 실험 수행  
  - 복잡 배경, 과실 겹침 상황에서도 robustness 확인  
- **의의**  
  - 인간이 판단하는 부분인 '익음도 분류'를 다룸
  - 다양한 환경 조건에서의 성능 검증은 실제 스마트팜 응용 가능성을 보여줌  
   

---

### 2. *LightMixer: A Novel Lightweight CNN for Tomato Leaf Diseases*  
- **주요 내용**  
  - `LightMixer`라는 경량 CNN 구조 제안 → 모바일/엣지 환경에 적합  
  - Depth Convolution + Light Residual 모듈 결합으로 연산량 최소화  
  - Leaf Disease(잎 병해) 데이터셋으로 검증했지만, 구조 자체는 익음도 분류에도 응용 가능  
- **의의**  
  - sML(경량화 모델) 관점에서 직접 참고할 수 있음  
  - 병해 진단 중심이지만, 경량 CNN 구조를 익음도 판별에 맞춰 변형 가능  
  - 스마트팜 환경에서 “엣지 디바이스”에 돌릴 수 있는 실험 설계에 도움  

---

(추가적으로, FGrade Dataset 기반 논문이나 Tomato Disease Detection 리뷰 논문도 참고 가능 → 이들은 익음도보다는 병해에 집중되어 있지만, 데이터셋 구축 및 평가 지표 설계 방식은 유용함.)

---

# 스마트팜 적용 시 고려사항 
1. **경계 상태 포함 여부** → 반숙·부분 숙성 데이터 필요  
2. **라벨링 기준 명확화** → 색상 비율, 당도 등 기준 확인 필요  
3. **현실 환경 반영** → 조명·배경·겹침·노출 등 잡음 포함 중요  
4. **데이터 균형** → 익음 단계별 샘플 수 편차 최소화  

---

# 결론 및 전망 
토마토 스마트팜에서 **익음도 AI 판별**은 농가의 수익성 및 노동 효율과 직결되는 핵심 요소이다.  
그러나 아직은 데이터셋 부족, 경계 상태 미포함, 라벨 기준 불명확 등 해결해야 할 과제가 많다.  
따라서 공개 데이터셋을 적극 활용하되, 실제 농장 데이터를 보강하여 **사람 + AI 협업 구조**로 발전시키는 것이 현실적인 방향.  


