from typing import Optional, List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.left = left
        self.right = right
        self.val = val


# 迭代的统一写法
# 迭代前序遍历树
def preorderTraversal(root: TreeNode) -> [int]:
    result = []
    st = []
    if root:
        st.append(root)
    while st:
        node = st.pop()
        if node is not None:
            if node.right:  # 右
                st.append(node.right)
            if node.left:  # 左
                st.append(node.left)
            st.append(node)  # 中
            st.append(None)
        else:
            node = st.pop()
            result.append(node.val)
    return result


# 迭代中序遍历树
def inorderTraversal(root: TreeNode) -> [int]:
    result = []
    st = []
    if root:
        st.append(root)
    while st:
        node = st.pop()
        if node is not None:
            if node.right:  # 添加右节点（空节点不入栈）
                st.append(node.right)

            st.append(node)  # 添加中节点
            st.append(None)  # 中节点访问过，但是还没有处理，加入空节点做为标记。

            if node.left:  # 添加左节点（空节点不入栈）
                st.append(node.left)
        else:  # 只有遇到空节点的时候，才将下一个节点放进结果集
            node = st.pop()  # 重新取出栈中元素
            result.append(node.val)  # 加入到结果集
    return result


# 迭代后序遍历树，与前序遍历类似，交换left和right结点的append顺序，再将结果逆序
def postorderTraversal(root: TreeNode) -> [int]:
    result = []
    st = []
    if root:
        st.append(root)
    while st:
        node = st.pop()
        if node is not None:
            st.append(node)  # 中
            st.append(None)

            if node.right:  # 右
                st.append(node.right)
            if node.left:  # 左
                st.append(node.left)
        else:
            node = st.pop()
            result.append(node.val)
    return result


# 反转二叉树 递归法
def invertTree(root: TreeNode) -> TreeNode:
    if not root:
        return None
    root.left, root.right = root.right, root.left
    invertTree(root.left)
    invertTree(root.right)
    return root


# 反转二叉树 迭代法
def invertTree1(root: TreeNode) -> TreeNode:
    if not root:
        return None
    stack = [root]
    while stack:
        node = stack.pop()
        node.left, node.right = node.right, node.left
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    return root


# 根节点到叶子节点的数字之和
def sumNumbers(root: Optional[TreeNode]) -> int:
    res = []
    treverse_sumNumbers(root, "", res)
    return sum(res)


def treverse_sumNumbers(root: Optional[TreeNode], path: str, res: [int]):
    if not root:
        return
    path += str(root.val)
    if root.left is None and root.right is None:
        res.append(int(path))
    treverse_sumNumbers(root.left, path, res)
    treverse_sumNumbers(root.right, path, res)
    path[::-1]


# 判断是否为二叉搜索树,二叉搜索树的中序遍历为递增序列
maxVal = float('inf')


# 递归法
def isValidBST(root: Optional[TreeNode]) -> bool:
    global maxVal
    if not root: return True
    left = isValidBST(root.left)
    if root.val <= maxVal:
        return False
    else:
        maxVal = root.val
    right = isValidBST(root.right)
    return left and right


# 迭代法
def isValidBST_dd(root):
    stack = []
    cur = root
    pre = None  # 记录前一个节点
    while cur is not None or len(stack) > 0:
        if cur is not None:
            stack.append(cur)
            cur = cur.left  # 左
        else:
            cur = stack.pop()  # 中
            if pre is not None and cur.val <= pre.val:
                return False
            pre = cur  # 保存前一个访问的结点
            cur = cur.right  # 右
    return True


# 有序数组转二叉搜索树
def sortedArrayToBST(nums: List[int]) -> Optional[TreeNode]:
    return creatTree(nums, 0, len(nums) - 1)


def creatTree(nums: List[int], left: int, right: int) -> Optional[TreeNode]:
    if left > right:
        return None
    root_index = left + (right - left) // 2
    root = TreeNode(nums[root_index])
    root.left = creatTree(nums, left, root_index - 1)
    root.right = creatTree(nums, root_index + 1, right)
    return root


# 将二叉搜索树变平衡
def balanceBST(root: TreeNode) -> TreeNode:
    res = []
    treverse_balanceBST(root, res)
    return createBST(res, 0, len(res) - 1)


def treverse_balanceBST(root: TreeNode, arr: [int]):
    if not root:
        return
    treverse_balanceBST(root.left, arr)
    arr.append(root.val)
    treverse_balanceBST(root.right, arr)


def createBST(nums: [int], left: int, right: int):
    if left > right:
        return None
    root_index = left + (right - left) // 2
    root = TreeNode(nums[root_index])
    root.left = createBST(nums, left, root_index - 1)
    root.right = createBST(nums, root_index + 1, right)
    return root


# 填充每个节点的下一个右侧指针
def connect(root: 'Optional[Node]') -> 'Optional[Node]':
    if not root: return root
    deque = [root]
    while deque:
        size = len(deque)
        for i in range(size):
            tmp = deque[0]
            deque.pop(0)
            if i < size - 1:
                tmp.next = deque[0]
            if tmp.left: deque.append(tmp.left)
            if tmp.right: deque.append(tmp.right)
    return root
