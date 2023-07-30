from typing import List


# 单调栈，求左边第一个小于本元素的值或右边第一个大于本元素的值
def dailyTemperatures(temperatures: List[int]) -> List[int]:
    deque = []
    r = [0 for _ in range(len(temperatures))]
    for i in range(len(temperatures)):
        while len(deque) != 0 and temperatures[i] > temperatures[deque[-1]]:
            r[deque[-1]] = i - deque[-1]
            deque.pop()
        deque.append(i)
    return r


# 下一个更大元素，nums1 中数字 x 的 下一个更大元素 是指 x 在 nums2 中对应位置 右侧 的 第一个 比 x 大的元素。
# 【不是下标，是值】
def nextGreaterElement(nums1: List[int], nums2: List[int]) -> List[int]:
    r = [-1 for _ in range(len(nums1))]
    deque = []
    for i in range(len(nums2)):
        while len(deque) != 0 and nums2[i] > nums2[deque[-1]]:
            if nums2[deque[-1]] in nums1:
                r[nums1.index(nums2[deque[-1]])] = nums2[i]
            deque.pop()
        deque.append(i)
    return r


# 给定一个循环数组（最后一个元素的下一个元素是数组的第一个元素），输出每个元素的下一个更大元素。
def nextGreaterElements(nums: List[int]) -> List[int]:
    r = [-1 for _ in range(len(nums))]
    deque = []
    for i in range(2 * len(nums)):
        t = i % len(nums)
        while len(deque) != 0 and nums[t] > nums[deque[-1]]:
            r[deque[-1]] = nums[t]
            deque.pop()
        deque.append(t)
    return r


# 接雨水
def trap(height: List[int]) -> int:
    res, deque = 0, [0]
    for i in range(1, len(height)):
        while deque and height[i] > height[deque[-1]]:
            tmp = deque.pop()
            if deque:
                # 因为递增栈，deque[-2]一定>=deque[-1]
                res += (min(height[deque[-1]], height[i]) - height[tmp]) * (i - deque[-1] - 1)
        deque.append(i)
    return res


# 柱状图中的最大矩形
def largestRectangleArea(heights: List[int]) -> int:
    deque, res = [], 0
    heights.insert(0, 0)
    heights.append(0)
    for i in range(len(heights)):
        while deque and heights[i] < heights[deque[-1]]:
            tmp = deque.pop()
            if deque:
                res = max(res, heights[tmp] * (i - deque[-1] - 1))
        deque.append(i)
    return res