from typing import List


# 递归语句执行树枝逻辑
# for循环 i > startIndex执行树层逻辑


# 组合数
def combine(n: int, k: int) -> List[List[int]]:
    res = []
    backtrcking_combine(n, k, 1, [], res)
    return res


def backtrcking_combine(n: int, k: int, startIndex: int, path: [int], res: [int]):
    if len(path) == k:
        res.append(path.copy())
        return
    # n - (k - len(path)) + 2 剪枝，在已经选len(path)个数字的情况下，判断下面的数还够不够k个，左闭右开，最多到n - (k - len(path)) + 1
    for i in range(startIndex, n - (k - len(path)) + 2):
        path.append(i)
        backtrcking_combine(n, k, i + 1, path, res)
        path.pop()


# 组合总和，找出所有相加之和为 n 的 k 个数的组合
def combinationSum3(k: int, n: int) -> List[List[int]]:
    res = []
    backTracking_combinationSum3(k, n, 1, [], res)
    return res


def backTracking_combinationSum3(k: int, n: int, startIndex: int, path: [int], res: [int]):
    if len(path) == k:
        if sum(path) == n:
            res.append(path.copy())
        return
    for i in range(startIndex, 11 - (k - len(path))):
        path.append(i)
        backTracking_combinationSum3(k, n, i + 1, path, res)
        path.pop()


# 电话按键，给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合，每一个数字表示一串字符
def letterCombinations(digits: str) -> List[str]:
    if len(digits) < 1: return []
    # 从0-9表示的字符串
    letter = ["", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]
    res = []
    backTracking_letterCombinations(digits, 0, "", res, letter)
    return res


def backTracking_letterCombinations(digits: str, index_nums: int, path: str, res: [str], letter: [str]):
    if index_nums == len(digits):
        res.append(path)
        return
    chars = letter[int(digits[index_nums])]
    for j in range(len(chars)):  # 同一层搜索
        path += chars[j]
        backTracking_letterCombinations(digits, index_nums + 1, path, res, letter)  # 深度优先，每次往下一层
        path = path[:-1]  # 回溯


# 组合求和, candidates中的数字可重复使用，无序[2,2,3] = [2,3,2]
def combinationSum(candidates: List[int], target: int) -> List[List[int]]:
    res = []
    candidates.sort()
    backTracking_combinationSum(candidates, target, [], res, 0, 0)
    return res


def backTracking_combinationSum(candidates: [int], target: int, path: [int], res: [[int]], currentSum: int,
                                startIndex: int):
    if currentSum > target: return  # 提前剪枝
    if currentSum == target:
        res.append(path[:])
        return
    for i in range(startIndex, len(candidates)):
        if currentSum + candidates[i] > target: break  # 提前剪枝
        path.append(candidates[i])
        currentSum += candidates[i]
        # 每次从startIndex开始，防止出现重复，从3开始取就不会再重复取到2了，防止出现[2,3,2]，这里i不+1，因为可重复选取
        # 此时这里面执行的树枝逻辑i = startIndex
        backTracking_combinationSum(candidates, target, path, res, currentSum, i)
        currentSum -= candidates[i]
        path.pop()


# 组合总和，每个数字在每个组合中只能使用 一次，candidates中会有重复数字，但是每个数字只能使用一次
def combinationSum2_withoutUsed(candidates: List[int], target: int) -> List[List[int]]:
    candidates.sort()
    res = []
    backTracking_combinationSum2_withoutUsed(candidates, target, 0, [], res, 0)
    return res


def backTracking_combinationSum2_withoutUsed(candidates: [int], target: int, startIndex: int, path: [int], res: [int],
                                             currentSum: int):
    if currentSum > target: return
    if currentSum == target and path not in res:
        res.append(path[:])
        return
    for i in range(startIndex, len(candidates)):
        if currentSum + candidates[i] > target: break
        # 当i > startIndex，执行的是树层逻辑，i == startIndex是树枝的逻辑
        if i > startIndex and candidates[i] == candidates[i - 1]: continue  # 树层剪枝
        currentSum += candidates[i]
        path.append(candidates[i])
        # 不可重复选取，i+1，此时这里面执行的树枝逻辑i = startIndex
        backTracking_combinationSum2_withoutUsed(candidates, target, i + 1, path, res, currentSum)
        currentSum -= candidates[i]
        path.pop()


# 增加一个状态数组标记是否已经选择了candidates[i]【便于理解】
def combinationSum2_withUsed(candidates: List[int], target: int) -> List[List[int]]:
    candidates.sort()
    res = []
    used = [False for _ in range(len(candidates))]
    backTracking_withUsed(candidates, target, 0, [], res, 0, used)
    return res


def backTracking_withUsed(candidates: [int], target: int, startIndex: int, path: [int], res: [int], currentSum: int,
                          used: [bool]):
    # 剪枝
    if currentSum > target: return
    if currentSum == target and path not in res:
        res.append(path[:])
        return
    for i in range(startIndex, len(candidates)):
        # 剪枝
        if currentSum + candidates[i] > target: break
        # 剪枝，当i > startIndex，执行的是树层逻辑【横着】，i == startIndex是树枝的逻辑【竖着】
        if i > startIndex and not used[i - 1] and candidates[i] == candidates[i - 1]: continue
        currentSum += candidates[i]
        path.append(candidates[i])
        used[i] = True
        backTracking_withUsed(candidates, target, i + 1, path, res, currentSum, used)
        currentSum -= candidates[i]
        used[i] = False
        path.pop()


# 切割回文串，给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是回文串。
def partition(s: str) -> List[List[str]]:
    res = []
    backTracking_Palindrome(s, 0, [], res)
    return res


def backTracking_Palindrome(s: str, startIndex: int, path: [str], res: [[str]]):
    if startIndex == len(s):
        res.append(path[:])
        return
    # 这是树层的逻辑
    for i in range(startIndex, len(s)):
        if is_Palindrome(s[startIndex:i + 1]):
            path.append(s[startIndex:i + 1])
        else:
            continue
        backTracking_Palindrome(s, i + 1, path, res)
        path.pop()


def is_Palindrome(s: str) -> bool:
    if len(s) == 1: return True
    for i in range(len(s) // 2 + 1):
        if s[i] != s[len(s) - i - 1]:
            return False
        return True


# 给定一个只包含数字的字符串，复原它并返回所有可能的 IP 地址格式
def restoreIpAddresses(s: str) -> List[str]:
    res = []
    backTracking_restoreIpAddresses(s, 0, [], res)
    return res


def backTracking_restoreIpAddresses(s: str, startIndex: int, path: [], res: [str]):
    if startIndex == len(s) and len(path) == 4:
        res.append('.'.join(path))
    for i in range(startIndex, len(s)):
        if len(path) < 4 and (len(s) - startIndex - 1) // (4 - len(path)) > 3: break
        if i > startIndex and s[startIndex] != '0' and int(s[startIndex:i + 1]) <= 255:
            path.append(s[startIndex:i + 1])
        elif i == startIndex:
            path.append(s[startIndex])
        else:
            continue
        backTracking_restoreIpAddresses(s, i + 1, path, res)
        path.pop()


# 求子集,给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）【包括空集】
# 结果在每个子节点中，不止是叶子节点
def subsets(nums: List[int]) -> List[List[int]]:
    res = []
    backTracking_subsets(nums, 0, [], res)
    return res


def backTracking_subsets(nums: [int], startIndex: int, path: [int], res: [[int]]):
    res.append(path[:])
    for i in range(startIndex, len(nums)):
        path.append(nums[i])
        backTracking_subsets(nums, i + 1, path, res)
        path.pop()


# 求子集,给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集(幂集)
def subsetsWithDup(nums: List[int]) -> List[List[int]]:
    res = []
    nums.sort()  # 排序以去重，不然就得每次循环都要判断是否已存在
    backTracking_subsetsWithDup(nums, 0, [], res)
    return res


def backTracking_subsetsWithDup(nums: [int], startIndex: int, path: [int], res: [[int]]):
    res.append(path[:])
    for i in range(startIndex, len(nums)):
        if i > startIndex and nums[i] == nums[i - 1]: continue
        path.append(nums[i])
        backTracking_subsetsWithDup(nums, i + 1, path, res)
        path.pop()


# 求递增子序列
def findSubsequences(nums: List[int]) -> List[List[int]]:
    res = []
    backTracking_findSubsequences(nums, 0, [], res)
    return res


def backTracking_findSubsequences(nums: [int], startIndex: int, path: [int], res: [[int]]):
    if len(path) > 1:
        res.append(path[:])
    uset = set()  # 只在同一树层中有效，因为本题不能排序，只能通过在数层中设置集合来去重
    for i in range(startIndex, len(nums)):
        # 去重，1是非递增去掉，2是去掉同一树层重复元素
        if (path and nums[i] < path[-1]) or nums[i] in uset: continue
        path.append(nums[i])
        uset.add(nums[i])  # 只记录同一层的元素
        backTracking_findSubsequences(nums, i + 1, path, res)
        path.pop()


# 全排列,给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。
def permute(nums: List[int]) -> List[List[int]]:
    res = []
    # 记录元素是否被使用
    used = [False for _ in range(len(nums))]
    backTracking_permute(nums, used, [], res)
    return res


def backTracking_permute(nums: [int], used: [bool], path: [int], res: [[int]]):
    if len(path) == len(nums):
        res.append(path[:])
        return
    for i in range(len(nums)):
        if used[i]: continue
        path.append(nums[i])
        used[i] = True
        backTracking_permute(nums, used, path, res)
        used[i] = False
        path.pop()


# 全排列，含重复元素
# 在每个树层中设置一个集合用于记录已遍历的元素，比较是否有重复
def permuteUnique_withoutSort(nums: List[int]) -> List[List[int]]:
    res = []
    used = [False for _ in range(len(nums))]
    backTracking_permuteUnique_withoutSort(nums, used, [], res)
    return res


def backTracking_permuteUnique_withoutSort(nums: [int], used: [bool], path: [int], res: [[int]]):
    if len(path) == len(nums):
        res.append(path[:])
    uset = set()
    for i in range(len(nums)):
        if used[i] or nums[i] in uset: continue
        path.append(nums[i])
        used[i] = True
        uset.add(nums[i])
        backTracking_permuteUnique_withoutSort(nums, used, path, res)
        used[i] = False
        path.pop()


# 去重，先排序，再比较相邻元素【效率更高】
def permuteUnique_withSort(nums: List[int]) -> List[List[int]]:
    res = []
    nums.sort()
    used = [False for _ in range(len(nums))]
    backTracking_permuteUnique_withSort(nums, used, [], res)
    return res


def backTracking_permuteUnique_withSort(nums: [int], used: [bool], path: [int], res: [[int]]):
    if len(path) == len(nums):
        res.append(path[:])
    for i in range(len(nums)):
        # used[i-1]=True是树枝上的去重【第一个值取了后变为True往下走，遇到第二个值此时还是True】
        # used[i-1]=False才是树层上的去重【第一值走完树枝后又从True置为False了】
        # 树层去重效率更高
        if used[i] or (i > 0 and nums[i] == nums[i-1] and not used[i-1]): continue
        path.append(nums[i])
        used[i] = True
        backTracking_permuteUnique_withSort(nums, used, path, res)
        used[i] = False
        path.pop()


from collections import defaultdict


# 重新安排行程，欧拉路径
def findItinerary(tickets: List[List[str]]) -> List[str]:
    path = defaultdict(list)
    for i in tickets:
        path[i[0]].append(i[1])
    for j in path.keys():
        path[j].sort(reverse=True)
    res = []

    def dfs(node):
        while path[node]:
            # 每次pop掉最小的，即下次出发的起点，如果没有对应的路径，则认为到达终点
            # 即跳出while，开始回溯，在回溯过程中发现新的路径
            dfs(path[node].pop())
        res.append(node)

    dfs('JFK')
    return res[::-1]


# n皇后
def solveNQueens_carl(n: int) -> List[List[str]]:
    result = []  # 存储最终结果的二维字符串数组
    chessboard = ['.' * n for _ in range(n)]  # 初始化棋盘
    backtracking_solveNQueens_carl(n, 0, chessboard, result)  # 回溯求解
    return [[''.join(row) for row in solution] for solution in result]  # 返回结果集


def backtracking_solveNQueens_carl(n: int, row: int, chessboard: List[str], result: List[List[str]]) -> None:
    if row == n:
        result.append(chessboard[:])  # 棋盘填满，将当前解加入结果集
        return

    for col in range(n):
        if isValid(row, col, chessboard):  # 合理才递归
            chessboard[row] = chessboard[row][:col] + 'Q' + chessboard[row][col + 1:]  # 放置皇后
            backtracking_solveNQueens_carl(n, row + 1, chessboard, result)  # 递归到下一行
            chessboard[row] = chessboard[row][:col] + '.' + chessboard[row][col + 1:]  # 回溯，撤销当前位置的皇后


def isValid(row: int, col: int, chessboard: List[str]) -> bool:
    # 检查列
    for i in range(row):
        if chessboard[i][col] == 'Q':
            return False  # 当前列已经存在皇后，不合法

    # 检查 45 度角是否有皇后
    i, j = row - 1, col - 1
    while i >= 0 and j >= 0:
        if chessboard[i][j] == 'Q':
            return False  # 左上方向已经存在皇后，不合法
        i -= 1
        j -= 1


# n皇后，记录已选状态，便于判断
def solveNQueens(n: int) -> List[List[str]]:
    res = []
    path = [['.' for _ in range(n)] for _ in range(n)]
    backTracking_solveNQueens(n, path, res, 0, [])
    return res


def backTracking_solveNQueens(n: int, path: [], res: [[str]], row: int, record: []):
    if row == n:
        res.append([''.join(i) for i in path])
        return
    for i in range(n):
        if is_valid(i, row, n, record):
            path[row][i] = 'Q'
            record.append(i)
            backTracking_solveNQueens(n, path, res, row + 1, record)
            record.pop()
            path[row][i] = '.'


def is_valid(self, column: int, row: int, n, record: [int]):
    for i in range(row):
        if column in record or column == record[i] + row - i or column == record[i] - row + i:
            return False
    return True


# n皇后，总共有几种方案
result = 0


def totalNQueens(n: int) -> int:
    backTracking_totalNQueens(n, [], 0)
    return result


def backTracking_totalNQueens(n: int, record: [int], row: int):
    if row == n:
        global result
        result += 1
        return
    for i in range(n):
        if is_valid_totalNQueens(i, row, record):
            record.append(i)
            backTracking_totalNQueens(n, record, row + 1)
            record.pop()


def is_valid_totalNQueens(column: int, row: int, record: []):
    for i in range(row):
        if column in record or column == record[i] - row + i or column == record[i] + row - i: return False
    return True


# 解数独, 二维递归
def solveSudoku(board: List[List[str]]) -> None:
    """
    Do not return anything, modify board in-place instead.
    """
    backTracking_solveSudoku(board)


def backTracking_solveSudoku(board: [[str]]) -> bool:
    # 若有解，返回True；若无解，返回False
    for i in range(len(board)):  # 遍历行
        for j in range(len(board[0])):  # 遍历列
            # 若空格内已有数字，跳过
            if board[i][j] != '.': continue
            for k in range(1, 10):
                if is_valid_solveSudoku(i, j, k, board):
                    board[i][j] = str(k)
                    if backTracking_solveSudoku(board): return True
                    board[i][j] = '.'
            # 若数字1-9都不能成功填入空格，返回False无解
            return False
    return True  # 有解


def is_valid_solveSudoku(row: int, col: int, val: int, board: List[List[str]]) -> bool:
    # 判断同一行是否冲突
    for i in range(9):
        if board[row][i] == str(val):
            return False
    # 判断同一列是否冲突
    for j in range(9):
        if board[j][col] == str(val):
            return False
    # 判断同一九宫格是否有冲突
    start_row = (row // 3) * 3
    start_col = (col // 3) * 3
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == str(val):
                return False
    return True