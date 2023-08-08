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


# 长键按入, typed中要包括name，且顺序相同，不能有额外字母，name和naamee符合，naame和namme不符合
def isLongPressedName_my(name: str, typed: str) -> bool:
    ptr_name = ptr_typed = 0
    while ptr_name < len(name) and ptr_typed < len(typed):
        # 相等均向前，否则typed向前
        if name[ptr_name] == typed[ptr_typed]:
            ptr_typed += 1
            ptr_name += 1
        else:
            if ptr_typed == 0 or typed[ptr_typed] != name[ptr_name - 1]:
                return False
            ptr_typed += 1
    # name比typed长
    if ptr_name < len(name):
        return False
    # typed比name长
    for i in range(ptr_typed, len(typed), 1):
        if typed[i] != name[-1]:
            return False
    return True


def isLongPressedName_Carl(name: str, typed: str) -> bool:
    i = j = 0
    while i < len(name) and j < len(typed):
        # If the current letter matches, move as far as possible
        if typed[j] == name[i]:
            # 把两边重复的走完
            while j + 1 < len(typed) and typed[j] == typed[j + 1]:
                j += 1
                # special case when there are consecutive repeating letters
                if i + 1 < len(name) and name[i] == name[i + 1]:
                    i += 1
            j += 1
            i += 1
        else:
            return False
    return i == len(name) and j == len(typed)


# 比较含删除键的字符串
# https://leetcode.cn/problems/backspace-string-compare/description/
def backspaceCompare(s: str, t: str) -> bool:
    # 倒序
    s_index, t_index = len(s) - 1, len(t) - 1
    s_backspace, t_backspace = 0, 0  # 记录s,t的#数量
    while s_index >= 0 or t_index >= 0:  # 使用or，以防长度不一致
        while s_index >= 0:  # 从后向前，消除s的#
            if s[s_index] == '#':
                s_index -= 1
                s_backspace += 1
            else:
                if s_backspace > 0:
                    s_index -= 1
                    s_backspace -= 1
                else:
                    break
        while t_index >= 0:  # 从后向前，消除t的#
            if t[t_index] == '#':
                t_index -= 1
                t_backspace += 1
            else:
                if t_backspace > 0:
                    t_index -= 1
                    t_backspace -= 1
                else:
                    break
        if s_index >= 0 and t_index >= 0:  # 后半部分#消除完了，接下来比较当前位的值
            if s[s_index] != t[t_index]:
                return False
        # 长度不一
        elif s_index >= 0 or t_index >= 0:  # 一个字符串找到了待比较的字符，另一个没有，返回False
            return False
        s_index -= 1
        t_index -= 1
    return True
