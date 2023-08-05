import collections
from typing import List, Optional


class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    # 复制复杂链表【复制节点】
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head: return
        tmp = head
        while tmp:
            tmpNode = Node(tmp.val)
            tmpNode.next = tmp.next
            tmp.next = tmpNode
            tmp = tmpNode.next
        cur = head
        while cur:
            if cur.random:
                cur.next.random = cur.random.next
            cur = cur.next.next
        ann = res = head.next
        pre = head
        while ann.next:
            pre.next = pre.next.next
            ann.next = ann.next.next
            pre = pre.next
            ann = ann.next
        pre.next = None
        return res

    # 根据值删除链表结点【双指针】
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        # 单指针解法
        # if not head: return head
        # if head.val == val: return head.next
        # tmp = head
        # while tmp:
        #     if not tmp.next:
        #         return head
        #     if tmp.next.val == val:
        #         tmp.next = tmp.next.next
        #     tmp = tmp.next
        # return head
        # 双指针解法
        if head.val == val: return head.next
        pre, cur = head, head.next
        while cur and cur.val != val:
            pre, cur = cur, cur.next
        if cur: pre.next = cur.next
        return head

    # 链表倒数第k个值【快慢指针】
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        # 计数【单指针计数的方法】
        # count = 0
        # tmp = head
        # while tmp:
        #     count += 1
        #     tmp = tmp.next
        # res = head
        # while res:
        #     if count == k:
        #         return res
        #     count -= 1
        #     res = res.next
        # return res
        former, latter = head, head
        for _ in range(k):
            former = former.next
        while former:
            former, latter = former.next, latter.next
        return latter

    # 判断两个链表是否有交点【双指针解法】
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        tmp1, tmp2 = headA, headB
        while tmp1 != tmp2:
            tmp1 = tmp1.next if tmp1 else headB
            tmp2 = tmp2.next if tmp2 else headA
        return tmp1

    # 字符串反转"the sky is blue"----->"blue is sky the"
    def reverseWords(self, s: str) -> str:
        # 掌握函数 ‘ ’.join()，s.split(),s.strip()
        s = s.strip()
        res = ''
        end_index = len(s) - 1
        inter_index = len(s)
        space = 0
        while end_index >= 0:
            if (s[end_index] == ' ') | (end_index == 0):
                if end_index == 0: space = 0
                if space == 0:
                    if end_index == 0:
                        res += s[end_index:inter_index]
                    else:
                        res += s[end_index + 1:inter_index] + ' '
                    inter_index = end_index
                    end_index = end_index - 1
                    space = 1
                else:
                    inter_index = end_index
                    end_index -= 1
            else:
                end_index -= 1
                space = 0
        # return res  #【暴力破解版】
        return " ".join(s.split()[::-1])  # [取巧版]

    # 滑动窗口求最大值
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        deque = collections.deque()
        res, n = [], len(nums)
        for i, j in zip(range(1 - k, n + 1 - k), range(n)):
            # 删除 deque 中对应的 nums[i-1]
            if i > 0 and deque[0] == nums[i - 1]:
                deque.popleft()
            # 保持 deque 递减
            while deque and deque[-1] < nums[j]:
                deque.pop()
            deque.append(nums[j])
            # 记录窗口最大值
            if i >= 0:
                res.append(deque[0])
        return res

    # 滑动窗口求最大值，解法2
    def maxSlidingWindow1(self, nums: List[int], k: int) -> List[int]:
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

    # 顺时针打印矩阵
    def spiralOrder(self, matrix: [[int]]) -> [int]:
        if not matrix: return []
        l, r, t, b, res = 0, len(matrix[0]) - 1, 0, len(matrix) - 1, []
        while True:
            for i in range(l, r + 1): res.append(matrix[t][i])  # left to right
            t += 1
            if t > b: break
            for i in range(t, b + 1): res.append(matrix[i][r])  # top to bottom
            r -= 1
            if l > r: break
            for i in range(r, l - 1, -1): res.append(matrix[b][i])  # right to left
            b -= 1
            if t > b: break
            for i in range(b, t - 1, -1): res.append(matrix[i][l])  # bottom to top
            l += 1
            if l > r: break
        return res

    # 双指针法反转链表 【递归】
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        return self.reverseL(None, head)

    def reverseL(self, pre: Optional[ListNode], cur: Optional[ListNode]) -> Optional[ListNode]:
        if not cur: return pre
        tmp = cur.next
        cur.next = pre
        return self.reverseL(cur, tmp)

    # 双指针法
    def reverseList1(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head
        pre = None
        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        return pre


    # 双指针法简洁版
    def reverseList2(self, head: Optional[ListNode]) -> Optional[ListNode]:
        node = None
        while head:
            head.next, head, node = node, head.next, head
        return node

    # 两两交换相邻节点 [1, 2, 3, 4]---[2, 1, 4, 3]
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummyHead = ListNode(0, head)
        cur = dummyHead
        # 交换1，2必须知道1之前的节点，不然节点2就会丢失
        while cur.next and cur.next.next:
            tmp = cur.next  # 1
            cur.next = cur.next.next  # 0--2
            tmp.next = cur.next.next  # 1--3
            cur.next.next = tmp  # 2-1
            cur = tmp
        return dummyHead.next

    # 删除链表中倒数第n个元素
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummyHead = ListNode(0, head)
        slow = fast = dummyHead
        count = 0
        while fast.next:
            if count < n:
                fast = fast.next
            else:
                fast = fast.next
                slow = slow.next
            count += 1
        slow.next = slow.next.next
        return dummyHead.next

    # 求有环链表的环入口，快慢指针
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                slow = head
                while slow != fast:
                    slow = slow.next
                    fast = fast.next
                return slow
        return None

    # 重排链表 L0 → L1 → … → Ln - 1 → Ln ------》L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
    # 先用快慢指针截断链表，跨指针每次走两步，慢指针走一步
    # 再把后半部分链表倒序重排，重排后的链表逐个插入
    def reorderList(self, head: Optional[ListNode]) -> None:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        node = None
        while slow:
            slow.next, slow, node = node, slow.next, slow
        dummyHead = head
        while node.next:  # node.next 否则会有循环
            tmp1 = node.next
            node.next = dummyHead.next
            dummyHead.next = node
            node, dummyHead = tmp1, dummyHead.next.next

    # 判断链表是否有环
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False

    #

