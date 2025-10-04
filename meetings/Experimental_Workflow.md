# **Gemma-3-1B IT — 데이터 기반 프루닝 + LoRA + 양자화 파이프라인**

**목표**: google/gemma-3-1b-it(1B급)를 데이터(토마토 Q&A) 기반으로 경량화하고 도메인 특화(LoRA)한 뒤, 양자화하여 라즈베리파이/로컬 배포 가능한 모델을 만드는 절차 정리.

1. **데이터로 중요도(gradient/activation) 측정 → 1회 데이터 기반 프루닝(권장 30%)**
2. **프루닝된 모델에 LoRA로 도메인 파인튜닝 (rank=8~16 권장)**
3. **LoRA merge → (옵션) 지식증류 → 마지막으로 양자화(8bit/4bit)**
4. **라즈베리파이 배포용으로 추가 튜닝/검증**

---

## **1) 데이터 기반 프루닝**

- 실제 데이터(도메인 데이터)를 모델에 흘려서 **각 weight의 gradient 절댓값**(또는 activation 통계)을 누적 → 중요도 산정
- 중요도가 낮은 weight 순으로 전역 global_unstructured L1 pruning 적용
- 중요한 점: 중요도 텐서와 gradient 누적은 **CPU에 저장**해서 GPU 메모리 폭발 방지

## **2) LoRA 학습 (프루닝 후)**

- 프루닝된 모델에 LoRA 붙여 도메인 특화. LoRA는 파라미터 효율적이어서 메모리 부담 적음.
- rank r=8 또는 r=16 권장 (라즈베리파이 배포 목표라면 8 시작)
- LoRA 학습은 M3 Pro에서 충분히 가능(데이터 10k, batch 4~8로 조절 가능; MPS 메모리 확인 필요)

## **3) Merge (LoRA 합치기)**

- 학습된 adapter를 base에 merge: model = PeftModel.from_pretrained(base, adapter_dir); model = model.merge_and_unload(); model.save_pretrained("./merged_model")
- merge 후 adapter 파일 불필요 → 배포 모델이 하나의 weight로 통합됨

## **4) (옵션) 지식 증류 (성능 보강)**

- Teacher: 원본(또는 더 큰) 모델, Student: 프루닝+LoRA 합친 모델
- Loss = alpha * KL(student_probs || teacher_probs) + (1-alpha) * CE(student, true_label)
- 필요시 수행 — 배포 모델 품질을 더 끌어올림(시간·리소스 추가 필요)

## **5) 양자화 (Quantization)**

- 양자화는 마지막 단계. 8-bit → 4-bit 순으로 시도.
- **CUDA 서버(CUDA+bitsandbytes)**가 있으면 bitsandbytes로 8/4bit 양자화 권장(속도, 메모리 이득 큼).
- macOS MPS는 bitsandbytes 호환성이 약할 수 있어, 양자화는 원격서버에서 처리 후 결과만 로컬로 가져오는 게 가장 현실적.