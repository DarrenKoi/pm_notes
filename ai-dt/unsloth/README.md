---
tags: [llm, sllm, finetuning, unsloth, qlora, lora]
level: intermediate
last_updated: 2026-03-11
status: in-progress
---

# Unsloth 기반 sLLM 파인튜닝 가이드

> 로컬 GPU 환경에서 작은 LLM(sLLM)을 빠르게 파인튜닝하고, GGUF/Ollama/vLLM까지 연결하는 실전 메모

## 왜 Unsloth를 보는가?

- 작은 GPU에서도 LoRA / QLoRA 파인튜닝을 비교적 현실적으로 돌릴 수 있다
- Hugging Face `transformers` / `trl` 생태계를 유지한 채 메모리와 속도를 개선할 수 있다
- 학습 후 GGUF, Ollama, llama.cpp, vLLM 등으로 이어지는 배포 경로가 좋다
- 로컬 API LLM을 teacher / judge로 쓰고, GPU에 올라간 sLLM을 student로 파인튜닝하는 워크플로우와 잘 맞는다

## 문서 구성

- [Unsloth 개요](./unsloth-overview.md) - Unsloth가 무엇인지, 무엇이 특별한지, 왜 쓰는지
- [로컬 sLLM 파인튜닝 워크플로우](./local-sllm-finetuning-workflow.md) - API 모델 + GPU student 모델을 활용한 단계별 진행법
- [데이터셋과 Chat Template 가이드](./dataset-and-chat-template-guide.md) - synthetic data, 형식 통일, template mismatch 방지
- [학습 및 배포 레시피](./training-and-deployment-recipe.md) - 설치, SFT 코드, 하이퍼파라미터, GGUF export

## 추천 학습 순서

1. [Unsloth 개요](./unsloth-overview.md)
2. [로컬 sLLM 파인튜닝 워크플로우](./local-sllm-finetuning-workflow.md)
3. [데이터셋과 Chat Template 가이드](./dataset-and-chat-template-guide.md)
4. [학습 및 배포 레시피](./training-and-deployment-recipe.md)

## 빠른 의사결정

### Unsloth가 잘 맞는 경우

- 1B ~ 14B 급 모델을 특정 태스크에 맞춰 빠르게 적응시키고 싶다
- 1대의 로컬 GPU 또는 소수 GPU에서 실험 속도를 높이고 싶다
- 전체 재학습보다는 LoRA / QLoRA 기반 미세조정이 목표다
- 학습 후 Ollama / llama.cpp / vLLM로 바로 넘기고 싶다

### Unsloth만으로 해결되지 않는 경우

- 최신 지식 자체를 지속적으로 넣고 싶다
- 출처 추적이 필수인 QA 시스템이 필요하다
- 모델 weights 없이 API만 있는 모델을 직접 파인튜닝하고 싶다

이 경우는 보통 `RAG + fine-tuning` 조합이 더 적절하다.

## 이 저장소 기준 권장 사용 방식

- 강한 로컬 API 모델: 데이터 생성, 데이터 정제 보조, 평가 judge
- GPU에 올린 sLLM: 실제 student 모델, Unsloth로 QLoRA/SFT 수행
- 결과물: adapter 또는 merged weights 저장 후 GGUF / Ollama / vLLM로 배포

## 참고 자료

- Unsloth Fine-Tuning Guide: <https://unsloth.ai/docs/get-started/fine-tuning-guide>
- Unsloth Install Guide: <https://docs.unsloth.ai/get-started/installing-%2B-updating>
- Unsloth Requirements: <https://docs.unsloth.ai/get-started/fine-tuning-for-beginners/unsloth-requirements>
- Unsloth Model Selection Guide: <https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/what-model-should-i-use>
- Unsloth Datasets Guide: <https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/datasets-guide>
- Unsloth LoRA Hyperparameters Guide: <https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide>
- Unsloth Chat Templates: <https://docs.unsloth.ai/basics/chat-templates>
- Unsloth Saving to GGUF: <https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-gguf>
- Hugging Face TRL Unsloth Integration: <https://huggingface.co/docs/trl/en/unsloth_integration>
