## NLP based Summarization & Translation Pipeline
### 회의록 전문 입력
- 음성 데이터로부터 변환된 다국어 Text 전문 입력
- 입력 데이터는 Json 파일에서 처리되어 {화자:”발화내용”} 이 통합된 회의록 입력


### 전문 영문 변환
- 입력받은 다국어 회의록을 영어로 통합 변환
- 혼합 언어(Code Switching)도 문맥 기반으로 처리되어 일관된 번역 수행

### 영문 텍스트 전처리
- 불필요한 기호, 감탄사 등 노이즈 제거로 요약 품질 향상 
- 길이에 따라 Sliding Window 적용해 문맥 정보 유지 


### 영문 회의록 요약 실행
- 요약 모델을 통해 화자 정보 및 Query 기반 생성 요약 수행
- 요약된 영문 회의록을 번역 모델에 전달


### 요약 결과 다국어로 번역 출력
- 생성된 영어 요약을 사용자 언어 설정에 따라 번역
- 한국어/중국어 번역 시 GPT 기반 API 활용 

## 요약 관련
### 요약을 위해 사용한 모델, 데이터, 학습 방법
 - 모델 : Google/Long-T5-tglobal-base
    - 모델 학습 시 QMSum 데이터를 사용함
    - 일반적 요약 방식인 뉴스 기사 요약, 전체 문장 오염 후 복원하는 방식을 사용하기보다 T5 학습 방식에 어울리는 Query - Answer 형태의 데이터셋
    - Long-T5 파인 튜닝 단계를 거쳐 요약 모델 구성
    - 기본적으로 Task 별 요약을 위해 전처리하여 QMSum 데이터를 학습
    - 학습된 fine-tuned T5 모델에 대해 전문이 입력되어 추론 수행
 - 활용 데이터
    - QMSum : 100MB, AMI Meeting Corpus : 195MB
 - 학습 방법 및 최적화
    - Optimizer : AdamW
    - LearningRate : 1e-5
    - Batch size : 1
    - Max input token : 4096
    - Sliding Window : 적용
    - length penalty : 2
    - beam search : 5

## 번역 관련
### 번역을 위해 사용한 모델, 데이터, 학습 방법
 - 모델 : GPT 4 API
    - 사전 치환 방식(Word2Word), 통계 기반 번역(SMT), 신경망 기반 번역(NMT), Transformer 기반 LLM 중 LLM 선택
    - GPT-4 와 같은 대규모 언어모델은 Multi task에서 문맥 파악 능력, 문장 유창성, 복잡한 표현 처리에서 준수한 성능을 보임
    - 특정 도메인에 학습된 번역 모델이 아니라면 GPT는 합리적인 대안이라 판단함
 - 프로세스(입력 & 출력)
    - 번역 입력과 번역 출력의 단계에서 수행단계에 따른 일부 Prompt Engineering 적용

## 추가적인 기능 관련
### RAG를 활용한 전문 정보 Chat 어시스턴스
 - LangChain 을 활용하여 요약 정보에 추가적으로 RAG 기능을 활용하여 PDF, Documents 의 정보를 청크처리하여 읽어오는 프로세스를 기능하여 전체 파이프라인에 추가적으로 구축함