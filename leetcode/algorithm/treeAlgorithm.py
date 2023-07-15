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
