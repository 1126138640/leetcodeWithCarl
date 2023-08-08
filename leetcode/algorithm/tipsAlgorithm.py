from typing import List


# 空间换时间
# 统计数组中每个元素的比它小的所有数字的数目
def smallerNumbersThanCurrent(nums: List[int]) -> List[int]:
    hashArr = dict()
    res = nums[:]
    res.sort()
    for i in range(len(res)):
        if res[i] not in hashArr.keys():
            hashArr[res[i]] = i
    for i in range(len(nums)):
        res[i] = hashArr[nums[i]]
    return res


# 有效的山脉数组
# 1 双指针法，若是山脉数组，左右指针最终会相遇在山峰
def validMountainArray(arr: List[int]) -> bool:
    left, right = 0, len(arr) - 1
    while left < (len(arr) - 1) and arr[left + 1] > arr[left]: left += 1
    while right > 0 and arr[right - 1] > arr[right]: right -= 1
    # 注意有可能是单调数组，所以要判断指针都有没有走
    return left == right and left != 0 and right != len(arr) - 1


# 如果数组中有出现频次相同的数字，则返回False，否则返回True
# hashmap法
def uniqueOccurrences(self, arr: List[int]) -> bool:
    hashArr = dict()
    for i in range(len(arr)):
        if arr[i] not in hashArr.keys():
            hashArr[arr[i]] = 1
        else:
            hashArr[arr[i]] += 1
    return len(hashArr.values()) == len(set(hashArr.values()))


# 将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。双指针法
def moveZeroes(nums: List[int]) -> None:
    slow, fast = 0, 0
    while fast < len(nums):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1  # 保持[0, slow)区间是没有0的
        fast += 1


# 轮转数组
# 右旋：nums = [1,2,3,4,5,6,7], k = 3  --》[5,6,7,1,2,3,4]【从尾部旋转】【所以需要先总体倒序】
def rotate_right(nums: List[int], k: int) -> None:
    def reverse(i: int, j: int):
        while i < j:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1
    k %= len(nums)
    reverse(0, len(nums) - 1)
    reverse(0, k - 1)
    reverse(k, len(nums) - 1)


# 左旋:nums = [1, 2, 3, 4, 5, 6, 7], k = 3  --》[4, 5, 6, 7, 1, 2, 3]【从头部旋转】【最后再总体倒序】
def rotate_left(nums: List[int], k: int) -> None:
    def reverse(i: int, j: int):
        while i < j:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1
    k %= len(nums)
    reverse(0, k-1)
    reverse(k, len(nums) - 1)
    reverse(0, len(nums) - 1)


# 寻找数组中心下标
# 数组 中心下标 是数组的一个下标，其左侧所有元素相加的和等于右侧所有元素相加的和。【不包含当前下表元素】
# [1,7,3,6,5,6]======[1,7,3],[5,6]，中心下标为3，左边+中心左边值+右边 = sum(arr)
def pivotIndex(nums: List[int]) -> int:
    left_sum = 0
    nums_sum = sum(nums)
    for i in range(len(nums)):
        right_sum = nums_sum - left_sum - nums[i]  # 把当前元素的值也得减去
        if right_sum == left_sum:
            return i
        left_sum += nums[i]
    return -1


# 双指针法，二分搜索
def binaryInsert(nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        else:
            return mid
    return -1


# 双指针发，搜索插入位置
def searchInsert(nums: List[int], target: int) -> int:
    # 初始化的时候就决定了，区间是左闭右毕，还是左闭右开
    # 不同的区间，初始化、while终止条件、left/right重新赋值都不同
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        else:
            return mid
    return left


# 寻找元素的起始位置和终止位置
def searchRange(self, nums: List[int], target: int) -> List[int]:
    left, right = 0, len(nums) - 1
    start_index = end_index = -2
    # left永远比right大1
    # 求右边界，=放在<上，此时right是右边界的准确值，left比真实右边界right大1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] <= target:
            left = mid + 1
            end_index = left
        else:
            right = mid - 1

    # 求左边界，=放在>上，此时left是左边界的准确值，right比真实左边界left小1
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
            start_index = right
    # target不在nums中
    if start_index == -2 or end_index == -2: return [-1, -1]
    if end_index - start_index > 1: return [start_index + 1, end_index - 1]
    return [-1, -1]


# 偶数在偶位置，奇数在奇位置， 双指针
def sortArrayByParity_while(nums: List[int]) -> List[int]:
    # left只走奇数位置，right只走偶数位置，所以位置奇偶不用判断
    left, right = 0, 1
    while left < len(nums) and right < len(nums):
        if nums[left] % 2 != 0:
            while nums[right] % 2 != 0:
                right += 2
            nums[left], nums[right] = nums[right], nums[left]
        else:
            left += 2
    return nums


# for循环效率更高
def sortArrayByParity_for(nums: List[int]) -> List[int]:
    oddIndex = 1
    for i in range(0, len(nums), 2):  # 步长为2
        if nums[i] % 2:  # 偶数位遇到奇数
            while nums[oddIndex] % 2:  # 奇数位找偶数
                oddIndex += 2
            nums[i], nums[oddIndex] = nums[oddIndex], nums[i]
    return nums

