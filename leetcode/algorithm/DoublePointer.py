'''
双指针解法
'''


# 原地删除数组, 双指针法，通用解法


def delElement(arr: [], val: int):
    slow = fast = 0
    for i in range(len(arr)):
        if arr[fast] != val:
            arr[slow] = arr[fast]
            slow += 1
        fast += 1

    return arr[0:slow]


# python 中可用的方法
def delElement1(arr: [], val: int):
    index = 0
    while index < len(arr):
        if arr[index] == val:
            del arr[index]
        else:
            index += 1
    return arr


# 输入一个非递减顺序数组，返回数组值的平方且顺序仍未非递减
def sortedSquares(nums: []) -> []:
    slow = 0
    fast = len(nums) - 1
    res = [0] * len(nums)
    index = len(nums) - 1
    while index >= 0:
        if pow(nums[slow], 2) <= pow(nums[fast], 2):
            res[index] = pow(nums[fast], 2)
            fast -= 1
        else:
            res[index] = pow(nums[slow], 2)
            slow += 1
        index -= 1
    return res


# 链表定义
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# 判断两个链表是否有交点【双指针解法】
def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
    tmp1 = headA
    tmp2 = headB
    while tmp1 != tmp2:
        tmp1 = tmp1.next if (tmp1 != None) else headB
        tmp2 = tmp2.next if (tmp2 != None) else headA
    return tmp1


