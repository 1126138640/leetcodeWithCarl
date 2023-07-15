import collections
from typing import List
# 连续子数组的和 >= target，求长度最短子数组

def minSubArrayLen(self, target: int, nums: List[int]) -> int:
    slow = fast = sum = 0
    res = []
    while fast < len(nums) or sum >= target:
        if sum < target:
            sum += nums[fast]
            fast = fast + 1 if fast < len(nums) else len(nums) - 1
        else:
            res.append(fast - slow)
            sum = sum - nums[slow]
            slow += 1
    return min(res) if len(res) > 0 else 0


 # 滑动窗口求最大值
def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
    if not nums or k == 0: return []
    deque = collections.deque()
    # 未形成窗口
    for i in range(k):
        while deque and deque[-1] < nums[i]:
            deque.pop()  # 保证deque[0]是最大值
        deque.append(nums[i])
    res = [deque[0]]
    # 形成窗口后
    for i in range(k, len(nums)):
        if deque[0] == nums[i - k]:
            deque.popleft()  # 一个窗口内有两个最大值，把下标小的pop出
        while deque and deque[-1] < nums[i]:
            deque.pop()
        deque.append(nums[i])
        res.append(deque[0])
    return res
