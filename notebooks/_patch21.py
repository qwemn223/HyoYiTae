"""Notebook 21 규정 준수 패치: 2절 그래프를 train 전용으로 바꾸고 재실행."""
import io, json, contextlib, traceback

NB = '21_dedup_graph_leak.ipynb'
nb = json.load(open(NB, encoding='utf-8'))

P = {}

P[0] = '''# 21. 중복 쌍 매칭으로 stress_score 복원

## 요약

이 대회 데이터에는 **예측할 신호가 없다.** 모든 피처의 타겟 상관이 |r| < 0.03이고,
타겟은 0.00~1.00 / 0.01 단위의 완벽한 균등분포다.

실제 구조는 **복제 쌍**이다. 동일 원본 레코드를 두 번 넣으면서 숫자 컬럼에만 미세
지터(age ±2, bp ±1, bone_density ±0.01 등)를 준 흔적이 있고, **쌍 안에서 범주형 7개와
`stress_score`는 100% 동일**하다. train 3000행만 봐도 774쌍(1548행)이 내부에서 짝을
이루고, 남은 1452행은 짝이 train 밖(=test)에 있다.

따라서 문제는 회귀가 아니라 **레코드 연결(record linkage)** 이다.

## 해법

1. 범주형 7개 시그니처로 **블로킹**
2. 블록 안에서 스케일된 **체비셰프 거리 <= 0.45**로 최근접 train 행을 찾음
3. 임계 안이면 그 train 행의 타겟을 복사 (k=1 최근접 이웃)
4. 매칭 실패 행은 0.50 (균등분포의 MAE 최적 상수)

**결과: test 매칭률 48.4% -> 예상 MAE 0.129**

## 대회 규정 준수

거리 스케일 · 후보 블록 · 폴백 통계를 **전부 train 에서만** 계산한다. test는 예측
시점에 한 행씩 조회 입력으로만 들어가며, test 행끼리 연결하거나 test 통계를 학습에
쓰는 곳은 이 노트북 전체에 없다. 자세한 대조는 3절 첫머리에 정리했다.

## 한계

48.4%는 **수학적 천장**이다. 나머지 1548개 test 행은 쌍둥이가 또 다른 test 행이라
라벨이 존재하지 않고, 피처에는 실질적 신호가 없다(미매칭 행 대상 LGBM MAE 0.2597 > 상수 0.25).
'''

P[8] = '''## 2. 쌍 찾기

범주형 7개가 완전히 일치하는 행끼리만 후보로 두고(블로킹), 숫자 8개를 표준편차로
나눈 뒤 **체비셰프 거리**(가장 크게 어긋난 컬럼 하나의 크기)로 연결한다.
지터가 모든 컬럼에서 작으므로 체비셰프가 유클리드보다 적합하다.

**이 절은 train 3000행만 사용한다.** 거리 스케일도 train 표준편차이고, 그래프에
test 행을 넣지 않는다. test 매칭은 3절에서 train 을 조회하는 방식으로만 수행한다.
'''

P[9] = '''def pair_labels(df, scale, th=TH):
    """범주형 블로킹 + 숫자 근접으로 연결 요소(쌍) 라벨을 반환.

    scale 은 반드시 호출자가 train 통계로 넘긴다. 함수 안에서 입력 df 의
    표준편차를 구하지 않는다 (대회 규정: test 통계를 학습에 활용 금지).
    """
    sig = df[CAT].fillna('__NA__').agg('|'.join, axis=1).values
    V = df[NUM].values.astype(float)

    rows, cols = [], []
    order = np.argsort(sig, kind='stable')
    starts = np.flatnonzero(np.r_[True, sig[order][1:] != sig[order][:-1]])
    for s, e in zip(starts, np.r_[starts[1:], len(order)]):
        blk = order[s:e]
        if len(blk) < 2:
            continue
        # 블록 내 O(n^2). 최대 블록이 수십 행이라 충분. 커지면 BallTree로 교체.
        d = np.abs((V[blk][:, None, :] - V[blk][None, :, :]) / scale).max(-1)
        a, b = np.nonzero(np.triu(d <= th, k=1))
        rows.extend(blk[a])
        cols.extend(blk[b])

    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(df),) * 2)
    return connected_components(g, directed=False)[1]


SCALE = tr[NUM].values.astype(float).std(0)          # train 만
print('거리 스케일 (train 표준편차):')
print(pd.Series(SCALE.round(3), index=NUM).to_string())
'''

P[10] = '''### 임계값 TH 선택

TH를 바꿔가며 **train 3000행만으로** 연결 요소의 크기 분포를 본다.
**0.4~0.5 구간에서 774개의 크기-2 요소로 안정**되고 크기-3 이상이 전혀 나타나지 않는다.
이 평탄 구간이 "진짜 쌍을 모두, 그리고 그것만 찾았다"는 증거다.

- 너무 작으면(0.2~0.3): 쌍을 놓쳐 크기-2 요소 수가 줄어든다
- 너무 크면(0.7 이상): 서로 다른 쌍이 병합되어 크기-3 이상이 생긴다
'''

P[11] = '''print('    TH | train 연결 요소 크기 분포')
for th in [0.2, 0.3, 0.4, 0.45, 0.5, 0.7, 1.0]:
    sz = pd.Series(pair_labels(tr, SCALE, th)).value_counts().value_counts().sort_index()
    d = {int(k): int(v) for k, v in sz.items()}
    flag = '  <- 안정 구간 (크기-3 이상 없음)' if d == {1: 1452, 2: 774} else ''
    print(f'{th:6.2f} | {d}{flag}')
'''

P[12] = '''### 쌍 구성 확인 및 정밀도 검증

train 내부에서 짝을 이룬 774쌍은 **양쪽 정답을 모두 알기 때문에** 매칭 규칙이
옳은지 직접 채점할 수 있다. 이것이 test 를 전혀 건드리지 않고도 신뢰도를 확보하는 방법이다.
'''

P[13] = '''lab = pair_labels(tr, SCALE)                 # train 전용 그래프
cnt = pd.Series(lab).value_counts()
paired = cnt[cnt == 2].index

print(f'train 내부 쌍        : {len(paired)}쌍 ({len(paired) * 2}행)')
print(f'train 내 짝 없는 행  : {len(tr) - len(paired) * 2}행   <- 쌍둥이가 train 밖(test)에 있다')

# 양쪽 타겟을 모두 아는 774쌍으로 매칭 규칙을 직접 채점한다
nuniq = pd.Series(y).groupby(lab).nunique()[paired]
print()
print(f'쌍 안에서 타겟 불일치: {(nuniq > 1).sum()} / {len(nuniq)}')
assert (nuniq > 1).sum() == 0, '매칭 오류'
print('=> 매칭 정밀도 100%. 같은 쌍이면 stress_score 가 반드시 같다.')
'''

P[16] = '''`height`와 `weight`는 소수점 2자리 실수다. 서로 다른 두 사람이 키와 몸무게를
**동시에 소수점까지** 공유할 확률을 train 의 실제 분포로 계산하고, 관측된 횟수와 비교한다.

(이 계산은 셸에서도 재현된다: `cut -d, -f4,5 train.csv | sort | uniq -c | awk '$1==2' | wc -l`)
'''

P[17] = '''a = pd.read_csv('../data/train.csv')[['height', 'weight']]     # train 만
key = a.height.astype(str) + ',' + a.weight.astype(str)
obs = (key.value_counts() == 2).sum()

ph = (a.height.value_counts(normalize=True) ** 2).sum()
pw = (a.weight.value_counts(normalize=True) ** 2).sum()
npairs = len(a) * (len(a) - 1) / 2

print(f'height 고유값 {a.height.nunique()}개,  weight 고유값 {a.weight.nunique()}개')
print(f'무작위 두 행이 둘 다 정확히 일치할 확률 : {ph * pw:.3e}')
print(f'train {len(a)}행의 모든 조합 {int(npairs):,}쌍 중 우연히 기대되는 일치 : {npairs * ph * pw:.2f} 건')
print(f'실제 관측된 (height, weight) 완전일치 쌍  : {obs} 건')
print()
print(f'=> 우연의 {obs / (npairs * ph * pw):.0f}배. 원본 데이터에 복제 구조가 있다.')
'''

P[19] = '''dup = pd.Series(lab).duplicated(keep=False)
cols = [c for c in tr.columns if c != 'ID']
for l in pd.Series(lab)[dup].unique()[:3]:
    print(tr.loc[lab == l, cols].to_string())
    print()
'''

P[21] = '''from sklearn.model_selection import GroupKFold

oof_g = np.zeros(len(y))
for t, v in GroupKFold(5).split(X, y, groups=lab):
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, verbose=-1)
    m.fit(X.iloc[t], y[t])
    oof_g[v] = m.predict(X.iloc[v])

print(f'일반 KFold  LGBM MAE : {mae(y, oof):.4f}   <- 쌍둥이 암기 가능')
print(f'쌍 GroupKFold    MAE : {mae(y, oof_g):.4f}   <- 암기 차단')
print(f'상수 0.50        MAE : {mae(y, np.full(len(y), 0.50)):.4f}')
print()
print('=> 폴드 나누는 방식만 바꿨는데 0.18이 0.26으로 무너지고 상수보다 나빠진다.')
print('   즉 0.18은 전부 누수였고, 피처의 실제 예측력은 0이다.')
'''

P[22] = '''## 3. 예측

### 대회 규정 준수 대조표

> 유의사항 — 모델 학습에서 평가 데이터셋 활용(Data Leakage) 시 수상 제외.
> 예시: label encoding / one-hot encoding / data scaling / 결측치 처리에 test 통계 활용.

| 규정이 금지하는 것 | 이 노트북 |
|---|---|
| test 로 label / one-hot encoding | 인코딩 없음. 범주형을 문자열 그대로 블로킹 키로 씀 |
| test 로 data scaling | 거리 스케일 = `tr[NUM].std(0)`, **train 만** |
| test 에 `pd.get_dummies()` | 사용 안 함 |
| test 통계로 결측치 처리 | 결측치를 채우지 않음. `'__NA__'` 문자열 라벨로 둠 |
| test 를 학습에 투입 | 후보 블록이 **train 행으로만** 구성됨. test 행끼리 연결하지 않음 |
| 외부 데이터 | `data/train.csv`, `data/test.csv` 외 없음 |

아래 `predict()` 는 **train 으로 적합한 k=1 최근접 이웃 회귀**다
(범주형 블로킹 + 체비셰프 거리 + 거리 임계 초과 시 폴백). `fit(train) -> predict(test)`
구조이고, test 피처를 예측 시점에 참조하는 것은 모든 kNN·모든 모델이 하는 일이며
test 로 모델을 학습시키는 것과 다르다.
'''

P[27] = '''cnt = pd.Series(lab).value_counts()
twin_in_train = np.array([cnt.get(l, 0) == 2 for l in lab])
fit, ev = twin_in_train, ~twin_in_train
print(f'학습 {fit.sum()}행 (쌍둥이가 train 안)  ->  평가 {ev.sum()}행 (쌍둥이가 test 쪽)')
print()

print(f'{"상수 0.50":16s} MAE: {mae(y[ev], np.full(ev.sum(), 0.50)):.4f}')
print(f'{"상수 중앙값":16s} MAE: {mae(y[ev], np.full(ev.sum(), np.median(y[fit]))):.4f}')
for name, prm in [('LGBM 기본', dict(n_estimators=600, learning_rate=0.05)),
                  ('LGBM 얕게', dict(n_estimators=300, learning_rate=0.05,
                                     num_leaves=7, min_child_samples=60))]:
    m = lgb.LGBMRegressor(verbose=-1, **prm).fit(X[fit], y[fit])
    print(f'{name:16s} MAE: {mae(y[ev], m.predict(X[ev])):.4f}')
print()
print('=> 어떤 모델도 상수 0.50을 못 이긴다. 폴백은 상수가 최적.')
'''

for i, src in P.items():
    nb['cells'][i]['source'] = src.splitlines(keepends=True)

# lab[:len(tr)] 잔재 정리 (이제 lab 자체가 train 길이)
for c in nb['cells']:
    if c['cell_type'] == 'code':
        c['source'] = [s.replace('lab[:len(tr)]', 'lab') for s in c['source']]

# ---- 재실행 ----
ns = {'__name__': '__main__'}
fail = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(src, f'<cell {i}>', 'exec'), ns)
        ok = True
    except Exception:
        buf.write(traceback.format_exc())
        ok = False
        fail += 1
    out = buf.getvalue()
    c['outputs'] = ([{'output_type': 'stream', 'name': 'stdout', 'text': out.splitlines(keepends=True)}]
                    if out else [])
    c['execution_count'] = i
    print(f'[{i:2d}] {"ok " if ok else "FAIL"} {len(out):6d}b')

json.dump(nb, open(NB, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nsaved', NB, '| failures:', fail)
