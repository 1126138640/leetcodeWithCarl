# 判断是否为字母异位词
import collections


def isAnagram(s: str, t: str) -> bool:
    hashArr = [0] * 26
    for i in s:
        hashArr[ord(i) - ord('a')] += 1
    for i in t:
        hashArr[ord(i) - ord('a')] -= 1
    for i in hashArr:
        if i != 0:
            return False
    return True


# 求两个数组的交集，无序，数组元素大小小于1000
def intersection(self, nums1: [int], nums2: [int]) -> [int]:
    count1 = [0] * 1001
    count2 = [0] * 1001
    result = []
    for i in range(len(nums1)):
        count1[nums1[i]] += 1
    for j in range(len(nums2)):
        count2[nums2[j]] += 1
    for k in range(1001):
        if count1[k] * count2[k] > 0:
            result.append(k)
    return result


# 求快乐数，数的各个位的平方之和等于1，则为快乐数
def isHappy(n: int) -> bool:
    res = []
    while n not in res:  # 如果出现循环，则跳出
        res.append(n)
        if n == 1:
            return True
        sum = 0
        for i in str(n):
            sum += pow(int(i), 2)
        n = sum
    return False


# 两数之和， 同一数组，仅一种结果
def twoSum(nums: [int], target: int) -> [int]:
    hashDict = {}
    for k, v in enumerate(nums):
        if (target - v) in hashDict.values():
            k1 = nums.index(target - v)
            return [k, k1]
        hashDict.update({k: v})
    return []


# 四数相加，四个数组中各取一个数，使其和为0
def fourSumCount(nums1: [int], nums2: [int], nums3: [int], nums4: [int]) -> int:
    hashArr1 = {}
    res = 0
    for k, v in enumerate(nums1):
        for k1, v1 in enumerate(nums2):
            if v + v1 in hashArr1.keys():
                hashArr1[v + v1] += 1
            else:
                hashArr1[v + v1] = 1
    for k, v in enumerate(nums3):
        for k1, v1 in enumerate(nums4):
            if -(v + v1) in hashArr1.keys():
                res += hashArr1[-(v + v1)]
    return res


# 赎金信，ransomNote 和 magazine，ransomNote要是magazine的子集； 用hashArr要比hashMap更高效
def canConstruct(ransomNote: str, magazine: str) -> bool:
    ransom_count = [0] * 26
    magazine_count = [0] * 26
    for c in ransomNote:
        ransom_count[ord(c) - ord('a')] += 1
    for c in magazine:
        magazine_count[ord(c) - ord('a')] += 1
    return all(ransom_count[i] <= magazine_count[i] for i in range(26))


# 三数相加，多三步剪枝，效率更高，固定一个值，剩下用双指针
# 剪枝注意，target可能为负数
def threeSum(nums: [int]) -> [[int]]:
    result = []
    nums.sort()
    for i in range(len(nums)):
        # 如果第一个元素已经大于0，不需要进一步检查， 剪枝
        if nums[i] > 0:
            return result

        # 跳过相同的元素以避免重复， 剪枝1
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left = i + 1
        right = len(nums) - 1

        while right > left:
            sum_ = nums[i] + nums[left] + nums[right]
            if sum_ < 0:
                left += 1
            elif sum_ > 0:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                # 跳过相同的元素以避免重复， 剪枝2
                while right > left and nums[right] == nums[right - 1]:
                    right -= 1
                while right > left and nums[left] == nums[left + 1]:
                    left += 1
                right -= 1
                left += 1
    return result


# 四数相加，核心还是三数相加，多一层循环，多固定v一个数，剩下还是双指针
# 剪枝注意target可能为负
def fourSum(nums: [int], target: int) -> [[int]]:
    nums.sort()
    res = []
    for i in range(len(nums) - 3):
        if i > 0 and nums[i] == nums[i - 1]: continue
        if nums[i] > target and nums[i] > 0:
            break
        for j in range(i + 1, len(nums) - 2):
            if nums[j] > target - nums[i] and nums[j] > 0:
                break
            if j > i + 1 and nums[j] == nums[j - 1]: continue
            left = j + 1
            right = len(nums) - 1
            while left < right:
                sum = nums[i] + nums[j] + nums[left] + nums[right]
                if sum > target:
                    right -= 1
                elif sum < target:
                    left += 1
                else:
                    if [nums[i], nums[j], nums[left], nums[right]] not in res:
                        res.append([nums[i], nums[j], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
    return res


# 求出现频率前k高的数字，使用hash法，重点掌握字典排序方法
def topKFrequent(nums: [int], k: int) -> [int]:
    hashDict = {}
    res = []
    for i in nums:
        if i not in hashDict.keys():
            hashDict[i] = 1
        else:
            hashDict[i] += 1
    for i in sorted(hashDict.items(), key=lambda x: x[1], reverse=True)[0:k]:
        res.append(i[0])
    return res


# 同构字符串，egg//app；title//paper
def isIsomorphic(s: str, t: str) -> bool:
    if len(s) != len(t): return False
    hasharr = {}
    for i in range(len(s)):
        if s[i] in hasharr.keys():
            if hasharr[s[i]] != t[i]:
                return False
        else:
            if t[i] in hasharr.values():
                return False
            hasharr[s[i]] = t[i]
    return True


# 查找共用字符 ["bella","label","roller"]---['e', 'l', 'l']
def commonChars(words: [str]) -> [str]:
    hasharr = [[0 for _ in range(26)] for _ in range(len(words))]
    res = []
    for i in range(len(words)):
        for j in range(len(words[i])):
            hasharr[i][ord(words[i][j]) - ord('a')] += 1
    for j in range(len(hasharr[0])):
        tmp = 100
        for i in range(len(hasharr)):
            # 找最小值
            tmp = min(hasharr[i][j], tmp)
        for k in range(tmp):
            res.append(chr(j + ord('a')))
    return res
