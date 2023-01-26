
from time import time

import matplotlib.pyplot as plt
import neal
import numpy as np
from pyqubo import Binary, Placeholder


def create_qubo():
    """
    式・QUBOを作成する。

    Parameters
    ---------
    なし

    Returns
    ---------
    qubo : dict
        quboデータ
    offset : float
        offset値(イジングとQUBOモデルのエネルギー補正値)
    """

    # 制約: 隙間時間は、45分までとする
    sukima_jikan = 45
    anime = Binary("anime")          # 30分
    youtube = Binary("youtube")   # 5分
    amazon = Binary("amazon")   # 4分
    netfilx = Binary("netflix")   # 3分
    dorama = Binary("dorama")          # 60分

    # 制約条件を満たす時に、式の結果が最小値の結果となるようにする
    anime_time = 30
    youtube_time = 5
    amazon_time = 4
    netfilx_time = 3
    dorama_time = 60
    hamiltonian = Placeholder("sukima1")*(sukima_jikan - anime*anime_time - youtube*youtube_time - amazon*amazon_time - netfilx*netfilx_time - dorama*dorama_time)**2
    model = hamiltonian.compile()
    feed_dict = {"sukima1": 1.0}
    qubo, offset = model.to_qubo(feed_dict=feed_dict)
    return qubo, offset


def check_sukima(result):
    """
    制約充足を確認する。

    Parameters
    ----------
    result : dimod.sampleset.SampleSet
        Dwaveで取得したアニーリング結果のセット
    
    Returns
    ---------
    vaild : int
        制約充足した回数(成功回数)

    """

    valid = 0
    for s,e,o in result.data(['sample', 'energy', 'num_occurrences']):
        if s['anime'] == 1 and s['youtube'] == 1 and s['amazon'] == 1 and s['netflix'] == 1 :
            valid += 1
    
    return valid


def calc_tts(correct_answer, num_reads, annealing_time):
    """
    性能評価に使うTTSを計算する。

    Parameters
    ------------
    correct_answer : int
        制約充足した回数
    num_reads : int
        アニーリングの試行回数
    annealing_time : float
        1回のアニーリング時間

    Returns
    -------------
    tts : float
        TTS

    """


    pR = 0.99 # 一般的には確率定数を99%とするため
    ps = correct_answer / num_reads
    tau = annealing_time
    if ps == 0.0 :
        tts = 0
    elif ps == 100 :
        tts = tau
    else:
        tts = tau * (np.log(1 - pR) / np.log(1 - ps))

    return tts 

'''

'''
def tts_graph_view(tau_list, tts_list, success_count_list):
    """
    計算結果をグラフ化する

    Parameters
    ------------
    tau_list : list
        試行回数分のアニーリング時間の結果を格納したリスト
    tts_list : list
        試行回数分のTTSの結果を格納したリスト
    success_count_list
        試行回数分の成功回数を格納した結果を格納したリスト

    Returns
    ----------
    なし

    """

    fig = plt.figure()

    # アニーリング時間とTTSのグラフ生成
    plt.subplot(2, 1, 1)
    plt.plot(tau_list, tts_list, '-o')
    plt.xlabel("annealing time [sec]")
    plt.ylabel("TTS [sec]")

    # アニーリング時間と成功確率のグラフ生成
    plt.subplot(2, 1, 2)
    plt.plot(tau_list, success_count_list, '-o')
    plt.xlabel("annealing time [sec]")
    plt.ylabel("success rate [%]")

    # グラフ上下の余白設定
    plt.subplots_adjust(hspace=0.5)

    # 対数表示
    #plt.xscale("log")
    plt.show()


if __name__ == '__main__':
    """
    """

    # スキマ時間で量子アニーリング
    qubo, offset = create_qubo()
    num_reads = 100
    num_sweeps_list = [10, 100, 1000, 10000]
    tts_list = []
    tau_list = []
    success_count_list = []
    for num_sweeps in num_sweeps_list:
        
        # アニーリングAPIを指定
        sampler = neal.SimulatedAnnealingSampler()

        # Pythonのtime関数の差分からアニーリング時間を測定
        chk_time = time()
        result = sampler.sample_qubo(qubo, num_reads=num_reads, num_sweeps=num_sweeps)
        annealing_time = time() - chk_time
        tau = annealing_time / num_reads
        tau_list.append(tau)
       
        # 制約充足率を測定
        success_count = check_sukima(result)
        success_count_list.append(success_count)

        # TTSを測定
        tts = calc_tts(success_count, num_reads, tau)
        tts_list.append(tts)

        print("annealing_time = ", tau, "[sec]")
        print("success_rate = ", success_count, "%")
        print("TTS = ", tts, "[sec]")

    # 測定結果をmatplotlibでグラフ化
    tts_graph_view(tau_list, tts_list, success_count_list)
