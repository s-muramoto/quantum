
def bubble_sort(data):
    """
    バブルソート
    隣り合う値を比較して、大小を入れ替える

    """

    for i in range(len(data)):
        for j in range(len(data) - i -1):
            if data[j] > data[j+1]: #左の方が大きい場合
                data[j], data[j+1] = data[j+1], data[j] #前後入れ替え

    return data

def sellect_sort(arr):
    """
    選択ソート
    配列中の最大・最小を選択。それを端に移動させていく。

    """
    for ind, ele in enumerate(arr):
        min_ind = min(range(ind, len(arr)), key=arr.__getitem__)
        arr[ind], arr[min_ind] = arr[min_ind], ele
    return arr

if __name__ == '__main__':
    DATA = [6, 15, 4, 2, 8, 5, 11, 9, 7, 13]
    bubble_sorted_data = bubble_sort(DATA.copy())
    sellect_sorted_data = sellect_sort(DATA.copy())

    print("バブルソート")
    print(f"{DATA}  →  {bubble_sorted_data}")

    print("選択ソート")
    print(f"{DATA}  →  {sellect_sorted_data}")