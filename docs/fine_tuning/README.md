# 1. 파인 튜닝 가이드

## 1) pyannote 파인튜닝 for speaker diarizaiton purpose
pyannote는 [pyannote.audio](https://github.com/pyannote/pyannote-audio/tree/main/tutorials)의 파인튜닝 가이드를 참고하였습니다. <br>
pyannote의 speaker diarization pipeline을 보유하신 데이터셋에 맞게 파인튜닝하는 과정은 다음과 같습니다.  
- Pyannote-segmentation 모델 파인튜닝
- Optimizing the pipeline hyper-parameters

본 가이드라인을 따르기 위해서는, 데이터셋을 다음과 같이 구성해야 합니다. ([pyannote.database](https://github.com/pyannote/pyannote-database) 참고) <br>
데이터셋 별로 포맷 변경에 필요한 작업이 다를 수 있기 때문에, 저희가 사용한 데이터셋을 포맷변환한 내용을 공유하겠습니다. (utils 폴더 내)

- dataset
    - train.txt (각 txt 파일에는 각 세트에 해당하는 데이터의 uri가 담깁니다.(파일명))
    - dev.txt
    - test.txt
    - train (학습 용도) 
        - wav (or else) folder
            - {uri}.wav
        - rttm folder
            - {uri}.rttm
        - uem folder
            - {uri}.uem
    - dev (pipeline hyper-parameters optimize 용도)
        - 동일
    - test (테스트)
        - 동일

## 2) NeMo 파인튜닝


# 2. 파인튜닝 성능  
- loss/val/segmentation
    - 검증 데이터에서 segmentation loss - 음성 분할 관련 손실
    - 낮을수록 모델이 더 정확하게 분할됨
- loss/val/vad
    - 검증 데이터에서 음성 활동 감지 (VAD) loss 값
- loss/val
    - 전체 검증 loss 값
    - 작을수록 모델이 더 정확함
- DiarizationErrorRate
    - 전체 화자인식 오류율(DER)
    - 작을수록 좋음
- DiarizationErrorRate/Confusion
    - 화자 간 혼동률 - 잘못된 화자 매칭 비율
    - 낮을수록 좋음
- DiarizationErrorRate/FalseAlarm
    - 음성이 없는데 있다고 감지한 비율
    - 낮을수록 좋음
- DiarizationErrorRate/Miss
    - 놓친 음성 비율
    - 낮을수록 좋음
- DiarizationErrorRate/Threshold
    - DER 계산 시 사용된 임계값

## 1) Adam + 20epoch
- Momentum + adaptive learning rate
- 빠른 수렴의 장점이 있지만, 과적합 위험 존재

|  Epoch | loss/val/segmentation | loss/val/vad | loss/val | DiarizationErrorRate | Confusion | FalseAlarm | Miss | Threshold |
|--------|----------------------|--------------|----------|----------------------|-----------|------------|------|-----------|
| 0      | 0.40065              | 0.488325     | 0.88895  | 0.6246               | 0.03265   | 0.05255    | 0.5398 | 0.6300    |
| 1      | 0.2192               | 0.24385      | 0.46305  | 0.2501               | 0.0352    | 0.0796     | 0.1353 | 0.6200    |
| 2      | 0.20745              | 0.2365       | 0.44385  | 0.2402               | 0.0311    | 0.0781     | 0.1311 | 0.5800    |
| 3      | 0.20395              | 0.23325      | 0.4372   | 0.2362               | 0.0297    | 0.0790     | 0.1274 | 0.5800    |
| 4      | 0.2004               | 0.2317       | 0.4321   | 0.2344               | 0.0289    | 0.0798     | 0.1258 | 0.5600    |
| 5      | 0.199                | 0.2294       | 0.42835  | 0.2310               | 0.0285    | 0.0771     | 0.1254 | 0.5600    |
| 6      | 0.198                | 0.22975      | 0.42775  | 0.2292               | 0.0278    | 0.0761     | 0.1253 | 0.5800    |
| 7      | 0.1965               | 0.2298       | 0.4263   | 0.2280               | 0.0273    | 0.0754     | 0.1253 | 0.5800    |
| 8      | 0.1957               | 0.2294       | 0.4251   | 0.2266               | 0.0268    | 0.0754     | 0.1244 | 0.5800    |
| 9      | 0.1942               | 0.2275       | 0.42175  | 0.2252               | 0.0265    | 0.0732     | 0.1256 | 0.5800    |
| 10     | 0.19305              | 0.22785      | 0.4209   | 0.2242               | 0.0260    | 0.0757     | 0.1225 | 0.5800    |
| 11     | 0.19265              | 0.22605      | 0.41865  | 0.2237               | 0.0264    | 0.0743     | 0.1230 | 0.5600    |
| 12     | 0.1926               | 0.226        | 0.41855  | 0.2227               | 0.0260    | 0.0738     | 0.1229 | 0.5800    |
| 13     | 0.1914               | 0.22605      | 0.41745  | 0.2223               | 0.0257    | 0.0727     | 0.1239 | 0.5800    |
| 14     | 0.18955              | 0.22485      | 0.41445  | 0.2217               | 0.0252    | 0.0777     | 0.1188 | 0.5600    |
| 15     | 0.189                | 0.22365      | 0.41265  | 0.2207               | 0.0253    | 0.0746     | 0.1208 | 0.5600    |
| 16     | 0.18815              | 0.223        | 0.4112   | 0.2196               | 0.0249    | 0.0768     | 0.1180 | 0.5600    |
| 17     | 0.18885              | 0.2245       | 0.4133   | 0.2191               | 0.0249    | 0.0759     | 0.1183 | 0.5800    |
| 18     | 0.1892               | 0.22435      | 0.4136   | 0.2191               | 0.0249    | 0.0729     | 0.1213 | 0.5800    |
| 19     | 0.18755              | 0.22245      | 0.41     | 0.2181               | 0.0244    | 0.0755     | 0.1182 | 0.5600    |


## 2) AdamW + 20epoch
- Adam + Decoupled Weight Decay
- L2 정규화로 일반화 성능이 향상 되지만 학습률 튜닝에 민감

## 3) RAdam (Rectified Adam)
- Adam의 빠른 수렴성과 SGD의 안정성을 절충
- 일반화 성능에서 Adam보다 좋다고 보고된 경우 많음
```py
from torch.optim import RAdam
def configure_optimizers(self):
    return RAdam(self.parameters(), lr=1e-4)

```

## 4) Lookahead + Adam (or RAdam)
- 두 개의 옵티마이저를 병렬로 사용해서 빠른 탐색 + 안정적 수렴
```py
pip install lookahead_pytorch

from lookahead import Lookahead
from torch.optim import RAdam

def configure_optimizers(self):
    base_opt = RAdam(self.parameters(), lr=1e-4)
    return Lookahead(base_opt, k=5, alpha=0.5)

```

## 5) RAdam + CosineAnnealing
- 정규화 + 학습률 스케줄링 실험
```py
from torch.optim import RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR

def configure_optimizers(self):
    optimizer = RAdam(self.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    return [optimizer], [scheduler]
```
