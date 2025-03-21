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

## 2) NeMo 파인튜닝 [[상세 기록 Link]](https://github.com/Hyeji-Jo/modu/blob/3d33cc6c82ddc20cbbf5d31615e04a3a545cbf26/docs/fine_tuning/Pyannote_%EC%84%B1%EB%8A%A5%EC%83%81%EC%84%B8.md)


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
| ...     | ...             | ...      | ...   | ...               | ...    | ...     | ... | ...    |
| 15     | 0.189                | 0.22365      | 0.41265  | 0.2207               | 0.0253    | 0.0746     | 0.1208 | 0.5600    |
| 16     | 0.18815              | 0.223        | 0.4112   | 0.2196               | 0.0249    | 0.0768     | 0.1180 | 0.5600    |
| 17     | 0.18885              | 0.2245       | 0.4133   | 0.2191               | 0.0249    | 0.0759     | 0.1183 | 0.5800    |
| 18     | 0.1892               | 0.22435      | 0.4136   | 0.2191               | 0.0249    | 0.0729     | 0.1213 | 0.5800    |
| 19     | 0.18755              | 0.22245      | 0.41     | 0.2181               | 0.0244    | 0.0755     | 0.1182 | 0.5600    |


## 2) AdamW + 20epoch
- Adam + Decoupled Weight Decay
- L2 정규화로 일반화 성능이 향상 되지만 학습률 튜닝에 민감  
  
|    |   loss/val/segmentation |   loss/val/vad |   loss/val |   DiarizationErrorRate |   DiarizationErrorRate/Confusion |   DiarizationErrorRate/FalseAlarm |   DiarizationErrorRate/Miss |   DiarizationErrorRate/Threshold |   loss/train/segmentation |   loss/train/vad |   loss/train |
|----|-------------------------|----------------|------------|------------------------|----------------------------------|-----------------------------------|-----------------------------|----------------------------------|---------------------------|------------------|--------------|
|  0 |                 0.39935 |       0.431175 |    0.8305  |                0.39435 |                          0.03865 |                           0.09995 |                      0.2557 |                             0.57 |                 nan       |        nan       |    nan       |
|  1 |                 0.2193  |       0.24265  |    0.46195 |                0.2545  |                          0.0359  |                           0.0849  |                      0.1337 |                             0.6  |                   0.28875 |          0.2774  |      0.56615 |
|  2 |                 0.21015 |       0.23805  |    0.44815 |                0.2464  |                          0.0329  |                           0.081   |                      0.1326 |                             0.58 |                   0.22295 |          0.2418  |      0.46475 |
| ...     | ...             | ...      | ...   | ...               | ...    | ...     | ... | ...    |
| 15 |                 0.18815 |       0.2228   |    0.4109  |                0.2219  |                          0.0253  |                           0.074   |                      0.1226 |                             0.56 |                   0.1765  |          0.20885 |      0.38535 |
| 16 |                 0.18895 |       0.22285  |    0.4118  |                0.2215  |                          0.0251  |                           0.0756  |                      0.1209 |                             0.58 |                   0.17485 |          0.2074  |      0.38225 |
| 17 |                 0.1881  |       0.22225  |    0.41035 |                0.2199  |                          0.0249  |                           0.0765  |                      0.1186 |                             0.58 |                   0.1736  |          0.2059  |      0.37955 |
| 18 |                 0.18835 |       0.2233   |    0.41165 |                0.2204  |                          0.0246  |                           0.0758  |                      0.12   |                             0.58 |                   0.17325 |          0.2063  |      0.3795  |
| 19 |                 0.1878  |       0.22205  |    0.4098  |                0.2198  |                          0.0244  |                           0.0771  |                      0.1183 |                             0.58 |                   0.17225 |          0.20485 |      0.3771  |

## 3) RAdam (Rectified Adam)
- Adam의 빠른 수렴성과 SGD의 안정성을 절충
- 일반화 성능에서 Adam보다 좋다고 보고된 경우 많음
```py
from torch.optim import RAdam
def configure_optimizers(self):
    return RAdam(self.parameters(), lr=1e-4)
```  
|    |   loss/val/segmentation |   loss/val/vad |   loss/val |   DiarizationErrorRate |   DiarizationErrorRate/Confusion |   DiarizationErrorRate/FalseAlarm |   DiarizationErrorRate/Miss |   DiarizationErrorRate/Threshold |   loss/train/segmentation |   loss/train/vad |   loss/train |
|----|-------------------------|----------------|------------|------------------------|----------------------------------|-----------------------------------|-----------------------------|----------------------------------|---------------------------|------------------|--------------|
|  0 |                 0.3967  |       0.439475 |   0.836175 |                 0.5042 |                           0.0335 |                            0.1178 |                     0.35285 |                             0.58 |                 nan       |        nan       |    nan       |
|  1 |                 0.2132  |       0.24265  |   0.45575  |                 0.2457 |                           0.0335 |                            0.0806 |                     0.1316  |                             0.6  |                   0.32425 |          0.3309  |      0.65515 |
|  2 |                 0.20605 |       0.23705  |   0.44305  |                 0.2396 |                           0.0315 |                            0.0791 |                     0.129   |                             0.58 |                   0.21145 |          0.24345 |      0.45495 |
| ...     | ...             | ...      | ...   | ...               | ...    | ...     | ... | ...    |
| 15 |                 0.18935 |       0.2221   |   0.41155  |                 0.2213 |                           0.0257 |                            0.0774 |                     0.1182  |                             0.56 |                   0.17605 |          0.2093  |      0.3854  |
| 16 |                 0.18845 |       0.2224   |   0.4109   |                 0.2212 |                           0.0252 |                            0.0749 |                     0.121   |                             0.58 |                   0.1745  |          0.2082  |      0.3827  |
| 17 |                 0.18865 |       0.2225   |   0.41115  |                 0.2203 |                           0.025  |                            0.0758 |                     0.1195  |                             0.58 |                   0.17365 |          0.2065  |      0.38015 |
| 18 |                 0.1898  |       0.2235   |   0.4133   |                 0.2205 |                           0.0252 |                            0.0776 |                     0.1177  |                             0.58 |                   0.173   |          0.2068  |      0.3798  |
| 19 |                 0.1877  |       0.2226   |   0.41035  |                 0.2199 |                           0.0247 |                            0.076  |                     0.1192  |                             0.58 |                   0.17215 |          0.2054  |      0.3776  |

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
