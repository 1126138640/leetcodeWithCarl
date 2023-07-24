from typing import List


# 组合数
def combine(n: int, k: int) -> List[List[int]]:
    res = []
    backtrcking(n, k, 1, [], res)
    return res


def backtrcking(n: int, k: int, startIndex: int, path: [int], res: [int]):
    if len(path) == k:
        res.append(path.copy())
        return
    # n - (k - len(path)) + 2 剪枝，在已经选len(path)个数字的情况下，判断下面的数还够不够k个，左闭右开，最多到n - (k - len(path)) + 1
    for i in range(startIndex, n - (k - len(path)) + 2):
        path.append(i)
        backtrcking(n, k, i + 1, path, res)
        path.pop()


