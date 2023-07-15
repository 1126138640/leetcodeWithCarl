# 原地交换 字符串；使用range或直接使用切片s[:] = s[::-1]
def reverseString(s: [str]) -> None:
    for i in range(len(s) // 2):
        s[i], s[len(s) - i - 1] = s[len(s) - i - 1], s[i]


# 反转字符串Ⅱ， 反转前k个字符  https://leetcode.cn/problems/reverse-string-ii/submissions/
def reverseStr(s: str, k: int) -> str:
    res = ""
    nums = 0
    k_point = 0
    for i in range(0, len(s)):
        if (i + 1) % k == 0:
            if nums % 2 == 0:
                res += s[k_point:i + 1][::-1]
            else:
                res += s[k_point:i + 1]
            nums += 1
            k_point = i + 1
    return res + s[k_point:] if nums % 2 == 1 else res + s[k_point:][::-1]


# 反转字符串中的word
def reverseWords(self, s: str) -> str:
    # 删除前后空白
    s = s.strip()
    # 反转整个字符串
    s = s[::-1]
    # 将字符串拆分为单词，并反转每个单词
    s = ' '.join(word[::-1] for word in s.split())
    return s


# 字符串匹配 KMP算法，即实现字符串find函数，寻找第一个复合的结果，返回首字母下标，底层逻辑是KMP
def getNext(next: [int], s: str) -> None:  # 求前缀表
    j = 0
    next[0] = 0
    for i in range(1, len(s)):
        while j > 0 and s[i] != s[j]:
            j = next[j - 1]
        if s[i] == s[j]:
            j += 1
        next[i] = j


def strStr(haystack: str, needle: str) -> int:  # find函数
    if len(needle) == 0:
        return 0
    next = [0] * len(needle)
    getNext(next, needle)
    j = 0
    for i in range(len(haystack)):
        while j > 0 and haystack[i] != needle[j]:
            j = next[j - 1]
        if haystack[i] == needle[j]:
            j += 1
        if j == len(needle):
            return i - len(needle) + 1
    return -1


# 判断一个字符串是否由多个小字符串单元组成， KMP法
# nxt[-1] 表示最长前缀的长度=最长后缀的长度，最长前缀+最小单元 = 最长后缀+最小单元 = 字符串【若字符串由最小单元组成】
# 得到最小单元长度，若能被字符串长度整除，则字符串由最小单元组成
def repeatedSubstringPattern(self, s: str) -> bool:
    if len(s) == 0:
        return False
    nxt = [0] * len(s)
    self.getNext(nxt, s)
    if nxt[-1] != 0 and len(s) % (len(s) - nxt[-1]) == 0:
        return True
    return False


# 判断一个字符串是否由多个小字符串单元组成， 查找法
# s[1:] + s[:-1]中若包含原字符串，则该字符串由多个小字符串单元组成
def repeatedSubstringPattern1(self, s: str) -> bool:
    n = len(s)
    if n <= 1:
        return False
    ss = s[1:] + s[:-1]
    print(ss.find(s))
    return ss.find(s) != -1

