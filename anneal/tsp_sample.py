import numpy as np
from pyqubo import Array, Constraint, Placeholder
from neal import SimulatedAnnealingSampler
from time import time
import optuna

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

# 都市間ごとの移動距離(入力データ)
Q = np.array([[1000, 20, 20, 50, 40],
              [30, 1000, 10, 30, 20],
              [20, 10, 1000, 30, 20],
              [50, 30, 20, 1000, 10],
              [40, 20, 20, 10, 1000]])


def create_hamlitonian():
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
    return H


def exe(hamlitonian, weight_const1, weight_const2, num_sweeps):

    # PyQUBOでモデルコンパイル＆QUBO作成
    model = hamlitonian.compile()
    feed_dict = {"weight_const1": weight_const1,
                 "weight_const2": weight_const2}
    qubo, offset = model.to_qubo(feed_dict=feed_dict)

    # Dwaveで疑似アニーリング
    num_reads = 100
    num_sweeps = num_sweeps
    sampler = SimulatedAnnealingSampler()

    chk_time = time()
    result = sampler.sample_qubo(
        qubo, num_reads=num_reads, num_sweeps=num_sweeps)
    annealing_time = (time() - chk_time) / 1000000

    # 成功回数を取得
    correct_answer = success_counter(result)
    print("Success count: ", correct_answer, "/", num_reads)

    # ttsを計算
    pR = 0.99  # 一般的には確率定数を99%とするため
    ps = correct_answer / num_reads
    tau = annealing_time
    if ps == 0.0:
        tts = 0
    elif ps == 100:
        tts = tau
    else:
        tts = tau * (np.log(1 - pR) / np.log(1 - ps))

    return tts


# 制約充足したスピンをカウント
def success_counter(result):
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

    return valid


# Optunaでの最適パラメータ探索
def objective(trial):
    hamlitonian = create_hamlitonian()
    weight_const1 = trial.suggest_uniform('weight_const1', 1000, 10000)
    weight_const2 = trial.suggest_uniform('weight_const2', 1000, 10000)
    num_sweeps = trial.suggest_int('num_sweeps', 100000, 100000)
    tts = exe(hamlitonian=hamlitonian, weight_const1=weight_const1,
              weight_const2=weight_const2, num_sweeps=num_sweeps)
    return tts


study = optuna.create_study()
study.optimize(objective, n_trials=2)
best_trial = study.best_trial
best_params = best_trial.params
print(best_params)
