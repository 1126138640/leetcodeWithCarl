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
        backTracking_letterCombinations(digits, index_nums+1, path, res, letter)  # 深度优先，每次往下一层
        path = path[:-1]  # 回溯


# 组合求和, candidates中的数字可重复使用，无序[2,2,3] = [2,3,2]
def combinationSum(candidates: List[int], target: int) -> List[List[int]]:
    res = []
    candidates.sort()
    backTracking_combinationSum(candidates, target, [], res, 0, 0)
    return res


def backTracking_combinationSum(candidates: [int], target: int, path: [int], res: [[int]], currentSum: int, startIndex: int):
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
        backTracking_combinationSum2_withoutUsed(candidates, target, 0, [],res, 0)
        return res


def backTracking_combinationSum2_withoutUsed(candidates: [int], target: int, startIndex: int, path: [int], res: [int], currentSum: int):
    if currentSum > target: return
    if currentSum == target and path not in res:
        res.append(path[:])
        return
    for i in range(startIndex, len(candidates)):
        if currentSum + candidates[i] > target: break
        # 当i > startIndex，执行的是树层逻辑，i == startIndex是树枝的逻辑
        if i > startIndex and candidates[i] == candidates[i-1]: continue  # 树层剪枝
        currentSum += candidates[i]
        path.append(candidates[i])
        # 不可重复选取，i+1，此时这里面执行的树枝逻辑i = startIndex
        backTracking_combinationSum2_withoutUsed(candidates, target, i+1, path, res, currentSum)
        currentSum -= candidates[i]
        path.pop()


# 增加一个状态数组标记是否已经选择了candidates[i]【便于理解】
def combinationSum2_withUsed(candidates: List[int], target: int) -> List[List[int]]:
    candidates.sort()
    res = []
    used = [False for _ in range(len(candidates))]
    backTracking_withUsed(candidates, target, 0, [], res, 0, used)
    return res


def backTracking_withUsed(candidates: [int], target: int, startIndex: int, path: [int], res: [int], currentSum: int, used: [bool]):
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

