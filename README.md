# 🧠 ModuLog - 다국어 회의록 요약

## 팀 소개 👀
- Member : ([🐰조혜지](https://github.com/Hyeji-Jo), [🐑김수효](https://github.com/KimSooHyo), [🐰한종현](https://github.com/smilish67), [🐮김주보](https://github.com/winjujae) )
<img width="1507" alt="image" src="https://github.com/user-attachments/assets/287371c6-f867-40de-91b7-fedf645a6b40" />

## ✨ 프로젝트 개요

ModuLog는 음성 기반 다국어 회의 데이터를 자동으로 요약하고 번역하는 회의록 요약 시스템입니다.  
혼합 언어(Code Switching) 환경에서도 문맥 기반의 자연스러운 요약과 번역을 지원합니다.

## 프로젝트 소개 👩🏻‍🏫 
### 1️⃣  &nbsp; 배경 및 목적
- 글로벌 기업 및 다국적 협력 프로젝트의 증가로 인해 다양한 언어로 회의가 진행됨
- 수동 기록, 녹취 후 수기 정리 등 기존의 회의록 작성 방식은 시간과 비용이 많이 소요됨
- 본 프로젝트는 다국어 회의 내용을 인식하고 요약하여, **회의록 작성의 효율성을 높이고 언어 장벽을 줄이는 것이 목표**
- 기존 회의록 시스템의 경우 사용자가 음성 인식될 단일 언어를 선택해야하는 불편함 존재

### 2️⃣  &nbsp; 개요  
<img width="1415" alt="image" src="https://github.com/user-attachments/assets/0eb9425e-f458-4bf4-951a-92cc46e140ea" />

### 3️⃣  &nbsp; 진행 기간 : 25.03.10 ~ 25.03.28 (19일 = 약 3주)

## 🧩 담당 역할 ([🐮김주보](https://github.com/winjujae))

### 🔹 NLP 기반 Summarization & Translation Pipeline 개발

- **입력 처리**
  - 다국어 텍스트 회의록을 JSON 형식으로 파싱: `{화자: "발화 내용"}` 구조
  - 음성 텍스트 전환 후 회의 전체 내용 통합 처리

- **영문 변환 및 전처리**
  - 코드 스위칭된 다국어 문장을 문맥 기반으로 영어로 변환
  - 불필요한 기호, 감탄사 제거 등 전처리
  - 문맥 유지를 위한 Sliding Window 적용

- **요약 처리**
  - 모델: `google/long-t5-tglobal-base`
  - 학습 데이터: QMSum + AMI Corpus
  - 방식: Query-Answer 기반 요약 모델 파인튜닝 후 추론
  - 파라미터:
    - max token: 4096
    - optimizer: AdamW
    - beam search: 5
    - length penalty: 2

- **요약 결과 다국어 번역**
  - 모델: GPT-4 API 활용
  - Prompt Engineering 적용하여 자연스러운 번역 수행
  - 한국어/중국어로 사용자 맞춤형 출력 지원

### 🔹 추가 구현 기능

- **LangChain + RAG 통합**
  - 요약된 회의 정보 외에 문서(PDF 등)에서 청크 기반 정보 검색 가능
  - 사용자 질문에 대한 어시스턴스 기능 강화

## 🛠️ 기술 스택

- `FastAPI`, `LangChain`, `HuggingFace Transformers`
- `Google Long-T5`, `GPT-4 API`, `QMSum`, `AMI Corpus`
- `Python`, `Pandas`, `Sliding Window`, `Beam Search`

## 📌 참고 이미지/그래프
> 학습 그래프, 파이프라인 아키텍처, 모델 비교 성능 등은 아래 원본 리포지토리 참고  
> https://github.com/Hyeji-Jo/ModuLog





