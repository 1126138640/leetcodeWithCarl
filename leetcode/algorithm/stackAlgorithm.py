import collections
import heapq
from operator import add, sub, mul


# 判断括号序列是否合法
def isValid(s: str) -> bool:
    stackArr = []
    for i in s:
        if i in '([{':
            stackArr.append(i)
        else:
            if len(stackArr) < 1 or i == ')' and stackArr[-1] != '(' or i == ']' and stackArr[-1] != '[' or i == '}' and stackArr[-1] != '{':
                return False
            else:
                stackArr.pop()
    return True if len(stackArr) < 1 else False


# 逆波兰表达式
op_map = {'+': add, '-': sub, '*': mul, '/': lambda x, y: int(x / y)}


def evalRPN(self, tokens: [str]) -> int:
    stack = []
    for token in tokens:
        if token not in {'+', '-', '*', '/'}:
            stack.append(int(token))
        else:
            op2 = stack.pop()
            op1 = stack.pop()
            stack.append(self.op_map[token](op1, op2))  # 第一个出来的在运算符后面
    return stack.pop()


# 滑动窗口最大值
def maxSlidingWindow(nums: [int], k: int) -> [int]:
    deque = collections.deque()
    # 未形成窗口
    for i in range(k):
        while deque and deque[-1] < nums[i]:
            deque.pop()
        deque.append(nums[i])
    res = [deque[0]]
    # 形成窗口
    for i in range(k, len(nums)):
        print(deque, i)
        if deque[0] == nums[i - k]:  # i-最大值的下标=k，此时窗口满
            deque.popleft()
        while deque and deque[-1] < nums[i]:
            deque.pop()
        deque.append(nums[i])
        res.append(deque[0])
    return res


# 求出现频率前k高的数
def topKFrequent(nums: [int], k: int) -> [int]:
    # 要统计元素出现频率
    map_ = {}  # nums[i]:对应出现的次数
    for i in range(len(nums)):
        map_[nums[i]] = map_.get(nums[i], 0) + 1  # hash存储键值对

    # 对频率排序
    # 定义一个小顶堆，大小为k
    pri_que = []  # 小顶堆

    # 用固定大小为k的小顶堆，扫描所有频率的数值
    for key, freq in map_.items():
        heapq.heappush(pri_que, (freq, key))
        if len(pri_que) > k:  # 如果堆的大小大于了K，则队列弹出，保证堆的大小一直为k
            heapq.heappop(pri_que)  # 每次pop的都是最小的

    # 找出前K个高频元素，因为小顶堆先弹出的是最小的，所以倒序来输出到数组
    result = [0] * k
    for i in range(k - 1, -1, -1):
        result[i] = heapq.heappop(pri_que)[1]
    return result
