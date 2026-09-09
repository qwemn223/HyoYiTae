# HyoYiTae - 스트레스 지수 예측 AI 해커톤

데이콘 X 오즈코딩스쿨 해커톤 프로젝트 저장소입니다. 개인의 건강·생활 데이터를 활용해 스트레스 지수(stress_score)를 예측하는 회귀 모델을 개발합니다.

## 대회 개요

- **주제**: 스트레스 지수 예측 AI 알고리즘 개발
- **주최/주관**: 데이콘
- **참가 대상**: [초격차] AI 헬스케어 머신러닝 트랙 참여자
- **평가지표**: MAE (Mean Absolute Error, 낮을수록 좋음)
- **마감일**: 2026-09-17 (목)

## 데이터셋

| 컬럼 | 설명 |
|---|---|
| ID | 샘플별 고유 ID |
| gender | 성별 |
| age | 연령 |
| height | 키(cm) |
| weight | 몸무게(kg) |
| cholesterol | 콜레스테롤 수치 |
| systolic_blood_pressure | 수축기 혈압 |
| diastolic_blood_pressure | 이완기 혈압 |
| glucose | 혈당 수치(mg/dL) |
| bone_density | 골밀도(g/cm²) |
| activity | 생활시 운동 강도 |
| smoke_status | 흡연 상태 |
| medical_history | 만성질환 |
| family_medical_history | 가족력 |
| sleep_pattern | 수면패턴 |
| edu_level | 학력 |
| mean_working | 1주일당 평균 근로 시간 |
| stress_score | (TARGET) 스트레스 점수 |

데이터 파일(`train.csv`, `test.csv`, `sample_submission.csv`)은 용량 및 규정상 저장소에 올리지 않습니다. `.gitignore`에 등록되어 있으니, 로컬 `data/` 폴더에 직접 다운로드해서 사용하세요.

## 프로젝트 구조

```
HyoYiTae/
├── README.md
├── CONTRIBUTING.md          # 팀 그라운드 룰
├── .gitignore
├── docs/
│   └── data_preprocessing_notes.md   # 결측치 처리 / 파생변수 정리
├── notebooks/                # 각자 실험용 노트북
└── data/                      # train.csv, test.csv 등 (gitignore 처리됨)
```

## 팀원

- 김효민
- 김이슬
- 조태희

## 협업 규칙

브랜치 전략, 커밋 규칙, 실험 로그 작성법 등 자세한 개발 규칙은 [CONTRIBUTING.md](./CONTRIBUTING.md)를 참고하세요.

## 실행 방법

```bash
git clone https://github.com/qwemn223/HyoYiTae.git
cd HyoYiTae
pip install -r requirements.txt
```

데이터 파일을 `data/` 폴더에 넣은 뒤 노트북을 실행하세요.