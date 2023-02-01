import numpy as np
from pyqubo import Array, Constraint, Placeholder
from neal import SimulatedAnnealingSampler

'''
巡回セールスマン問題を解く量子アニーリング サンプルコード
経路最適化

i / j   都市A   都市B   都市C   都市D  都市E
都市A   -       20      20      50     40
都市B   30      -       10      30     20
都市C   20      10      -       30     20
都市D   50      30      20      -      10
都市E   40      20      20      10     -

最適解: A → B → E → D → C → A = 90
'''

# 都市数
N = 5

# 都市間ごとの移動距離
Q = np.array([[1000, 20, 20, 50, 40],
              [30, 1000, 10, 30, 20],
              [20, 10, 1000, 30, 20],
              [50, 30, 20, 1000, 10],
              [40, 20, 20, 10, 1000]])

# スピン定義
x = Array.create('x', shape=(N, N), vartype='BINARY')

# 目的関数 (訪れる移動距離の合計)
H_cost = 0
for t in range(N):
    for i in range(N):
        for j in range(N):
            H_cost += Q[i][j] * x[t][i] * x[(t+1) % N][j]

# 制約条件1 (各都市は1回は訪問すること) ソース先頭の表の行のビット合計が1のとき、式が最小
H_const1 = 0
for i in range(N):
    H_const1 += (np.sum(x[i]) - 1)**2

# 制約条件2 (1度に訪れる都市は1つであること)　ソース先頭の表の列のビット合計が1のとき、式が最小
H_const2 = 0
for i in range(N):
    H_const2 += (np.sum(x.T[i]) - 1)**2

H = H_cost + Placeholder("weight_const1")*Constraint(H_const1, label="H_const1") + \
    Placeholder("weight_const2")*Constraint(H_const2, label="H_const2")

# PyQUBOでモデルコンパイル＆QUBO作成
model = H.compile()
feed_dict = {"weight_const1": 1000.0, "weight_const2": 1000.0}
qubo, offset = model.to_qubo(feed_dict=feed_dict)

# Dwaveで疑似アニーリング
num_reads = 100
num_sweeps = 50000
sampler = SimulatedAnnealingSampler()
result = sampler.sample_qubo(
    qubo, num_reads=num_reads, num_sweeps=num_sweeps)

# 結果の加工
best_length = 90
valid = 0
for s, e, o in result.data(['sample', 'energy', 'num_occurrences']):
    length = 0
    for i in range(N):
        for j in range(N):
            spin_index = "x[" + str(i) + "][" + str(j) + "]"
            spin = s[spin_index]
            if i != j and spin == 1:
                length += Q[i][j]

    print("Length: ", length)
    # 最適解と求解の比較
    if best_length == length:
        valid += 1

print("Success count: ", valid, "/", num_reads)
