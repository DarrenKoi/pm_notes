---
tags: [deep-learning, pytorch, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
document_type: index
---

# 딥러닝 읽기 순서

Tensor 연산과 gradient의 흐름을 익힌 뒤 학습 루프, 모델 구조, 사전학습 가중치의 적용 조건을 읽는다. 같은 학습 코드를 다루더라도 기초·재사용 템플릿·모델별 예제의 목적을 구분한다. 이 폴더는 교육용이며 실업무 데이터 성능을 검증한 자료가 아니다.

1. [PyTorch 기초](./pytorch-basics.md): Tensor → Autograd → Dataset/DataLoader → Module → 학습·복원. 2026-10-04 개별 검토; CPU 실행만 확인.
2. [학습 루프](./training-loop-template.md): early stopping·체크포인트·로깅의 재사용. 2026-10-04 개별 검토; 분류 CPU fixture/재개 상태 확인.
3. [CNN 이미지 분류](./cnn-image-classification.md): 이미지 입력·torchvision 모델. 2026-10-04 개별 검토; CPU fixture/가중치 미다운로드.
4. [시퀀스 모델](./sequence-models.md): 시간/순서 정보를 다루는 모델. 2026-10-04 개별 검토; 시간 분할·packing·CPU 2 epoch/한글 PNG 확인.
5. [전이 학습](./transfer-learning.md): 사전학습 모델의 재사용·미세조정. 2026-10-04 개별 검토; CPU 동결/해동/학습 fixture, 텍스트 API 공식 소스 대조.
6. [정리 기록](./organization-log.md): 수정·근거·실행 범위와 보류 사항.

> [!warning] 현재 확인 범위
> 원래5개를 모두 개별 검토했다. 가중치/CIFAR/BERT 다운로드·실제 데이터 성능은 검증하지 않았다. CUDA/MPS·멀티프로세스·배포 변환과 실제 Obsidian 읽기 화면은 미확인이다. 대표 예제 통합 판단은 Claude 연결 뒤 협의할 사항으로 보류한다.
