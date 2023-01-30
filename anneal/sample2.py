import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyqubo import Array, Placeholder
from neal import SimulatedAnnealingSampler

# 入力ファイル名
INPUT_FILE = "./school_lunch_sendai.csv"

# データの入力
df_data = pd.read_csv(INPUT_FILE)
D = df_data.values
N, M = D.shape
print(f'データの数: {N}, 食材の数: {M-1}')

# 定式化
x = Array.create('x', shape=(N, M-1), vartype='BINARY')

# 1. 主菜の調理方法(焼く、炒める、揚げる、煮る)・・・１つ選択
H1_cover = Placeholder('H1_cover')
H1 = H1_cover * sum((sum(x[i][j]
                    for j in range(0, 4)) - 1)**2 for i in range(N))

# 2. 副菜の調理方法(生、焼く、炒める、揚げる、和える、煮る、なし)・・・１つ選択
H2_cover = Placeholder('H2_cover')
H2 = H2_cover * sum((sum(x[i][j]
                    for j in range(4, 11)) - 1)**2 for i in range(N))

# 3. 汁物orデザート・・・１つ選択
H3_cover = Placeholder('H3_cover')
H3 = H3_cover * sum((sum(x[i][j]
                    for j in range(11, 12)) - 1)**2 for i in range(N))

# 4. 主食（ごはん、麦ごはん、パン）・・・１つ選択
H4_cover = Placeholder('H4_cover')
H4 = H4_cover * sum((sum(x[i][j]
                    for j in range(12, 15)) - 1)**2 for i in range(N))

H = H1 + H2 + H3 + H4

# PyQUBOで、モデルコンパイル＆QUBO生成
model = H.compile()
feed_dict = {'H1_cover': 1.0, 'H2_cover': 1.0,
             'H3_cover': 1.0, 'H4_cover': 1.0}
qubo, offset = model.to_qubo(feed_dict=feed_dict)

# Dwaveで疑似アニーリング
sampler = SimulatedAnnealingSampler()
result = sampler.sample_qubo(qubo)


# アニーリング結果の加工
for s, e, o in result.data(['sample', 'energy', 'num_occurrences']):

    spin_list = []
    for i in range(N):
        spin_tmp_list = []
        spin_tmp_list.append(D[i][0])  # 料理名を先頭に追加
        for j in range(0, M-2):
            spin_index = "x[" + str(i) + "][" + str(j) + "]"
            spin = s[spin_index]
            spin_tmp_list.append(spin)
        spin_list.append(spin_tmp_list)

    spin_list = np.array(spin_list).reshape(N, M-1)
    df = pd.DataFrame(spin_list)
    df.columns = ["焼く", "炒める", "揚げる", "煮る", "生", "焼く", "炒める", "揚げる",
                  "和える", "煮る", "なし", "汁物", "果物orゼリー", "ごはん", "麦ごはん", "パン"]
    print(df)
