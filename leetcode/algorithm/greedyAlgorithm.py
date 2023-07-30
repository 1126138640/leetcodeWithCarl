# 小于n且数字单调递增的最大值
import collections
from typing import List


def ace_max_number(a: int):
    b = str(a)
    c = []
    for i in b:
        c.append(int(i))
    index = len(c)-1
    for i in range(len(c)-1, 0, -1):  # start， end ，reverse
        if c[i] < c[i-1]:
            index = i-1
            c[i-1] -= 1
    res = 0
    # 复原
    for i in range(len(b)):
        if i < index:
            res += int(b[i])*pow(10, len(b)-i-1)
        elif i > index:
            res += 9*pow(10, len(b)-i-1)
        else:
            if i != len(b)-1:
                res += (int(b[i])-1)*pow(10, len(b)-i-1)
            else:
                res += int(b[i]) * pow(10, len(b) - i - 1)
    return res


# 分饼干【遍历胃口，大饼干优先】【遍历饼干，小饼干优先】
# 遍历胃口，大饼干优先
def findContentChildren_children(self, g, s):
    g.sort()  # 将孩子的贪心因子排序
    s.sort()  # 将饼干的尺寸排序
    index = len(s) - 1  # 饼干数组的下标，从最后一个饼干开始
    result = 0  # 满足孩子的数量
    for i in range(len(g) - 1, -1, -1):  # 遍历胃口，从最后一个孩子开始
        if index >= 0 and s[index] >= g[i]:  # 遍历饼干
            result += 1
            index -= 1
    return result

# 遍历饼干，小饼干优先
def findContentChildren_cookie(self, g, s):
    g.sort()  # 将孩子的贪心因子排序
    s.sort()  # 将饼干的尺寸排序
    index = 0
    for i in range(len(s)):  # 遍历饼干
        if index < len(g) and g[index] <= s[i]:  # 如果当前孩子的贪心因子小于等于当前饼干尺寸
            index += 1  # 满足一个孩子，指向下一个孩子
    return index  # 返回满足的孩子数目


# 摆动序列【贪心解法，要考虑首尾元素、单调平坡、非单调平坡等状态】
def wiggleMaxLength(nums):
    if len(nums) <= 1:
        return len(nums)  # 如果数组长度为0或1，则返回数组长度
    curDiff = 0  # 当前一对元素的差值
    preDiff = 0  # 前一对元素的差值
    result = 1  # 记录峰值的个数，初始为1（默认最右边的元素被视为峰值）
    for i in range(len(nums) - 1):
        curDiff = nums[i + 1] - nums[i]  # 计算下一个元素与当前元素的差值
        # 如果遇到一个峰值
        if (preDiff <= 0 and curDiff > 0) or (preDiff >= 0 and curDiff < 0):
            result += 1  # 峰值个数加1
            preDiff = curDiff  # 注意这里，只在摆动变化的时候更新preDiff
    return result  # 返回最长摆动子序列的长度


# 摆动序列【动态规划】
def wiggleMaxLength_dp(nums):
    # 设 dp 状态dp[i][0]，表示考虑前 i 个数，第 i 个数作为山峰的摆动子序列的最长长度
    # 设 dp 状态dp[i][1]，表示考虑前 i 个数，第 i 个数作为山谷的摆动子序列的最长长度
    dp = [[0, 0] for _ in range(len(nums))]  # 创建二维dp数组，用于记录摆动序列的最大长度
    dp[0][0] = dp[0][1] = 1  # 初始条件，序列中的第一个元素默认为峰值，最小长度为1
    for i in range(1, len(nums)):
        dp[i][0] = dp[i][1] = 1  # 初始化当前位置的dp值为1
        for j in range(i):
            if nums[j] > nums[i]:
                dp[i][1] = max(dp[i][1], dp[j][0] + 1)  # 如果前一个数比当前数大，可以形成一个上升峰值，更新dp[i][1]
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i][0] = max(dp[i][0], dp[j][1] + 1)  # 如果前一个数比当前数小，可以形成一个下降峰值，更新dp[i][0]
    return max(dp[-1][0], dp[-1][1])  # 返回最大的摆动序列长度


# 最大连续子序列和，具有最大和的连续子数组（子数组最少包含一个元素）
def maxSubArray(nums: [int]) -> int:
    count = 0
    result = -pow(10, 4)
    for i in range(len(nums)):
        count += nums[i]
        if count > result:
            result = count
        # 复原
        if count < 0:
            count = 0
    return result


# 买卖股票的最佳时机【不能同时持有多只股票】
def maxProfit(prices: [int]) -> int:
    result = 0
    for i in range(1, len(prices)):
        result += max(0, prices[i] - prices[i - 1])
    return result


# 等同于下面的思想
def maxProfit_m(prices: [int]) -> int:
    dp = [0 for _ in range(len(prices))]
    for i in range(1, len(prices)):
        dp[i] = max(dp[i-1], dp[i-1] + prices[i] - prices[i-1])
    return dp[-1]


# 跳跃，从第i个位置跳，最远可以跳到nums[i]+i的位置
def canJump_while(nums: [int]) -> bool:
    if len(nums) == 1: return True
    cover = nums[0]
    index = 0
    while index <= cover:
        cover = max(cover, index + nums[index])
        if cover >= len(nums) - 1:
            return True
        index += 1
    return False


def canJump_for(nums: [int]) -> bool:
    if len(nums) == 1: return True
    cover = nums[0]
    for i in range(len(nums)):
        if i <= cover:
            cover = max(cover, nums[i]+i)
            if cover >= len(nums)-1:
                return True
    return False


# 跳跃2 跳跃步数最少
def jump(self, nums: [int]) -> int:
    # 起点位置就是终点位置
    if len(nums) == 1: return 0
    cover = steps = index = 0
    for i in range(len(nums)):
        cover = max(nums[i]+i, cover)
        if i == index:
            steps += 1
            index = cover
            if cover >= len(nums)-1:
                break
    return steps


# 代码简洁版
def jump_sim(nums: [int]) -> int:
    if len(nums) == 1: return 0
    cover = index = steps = 0
    for i in range(len(nums)-1):
        cover = max(nums[i]+i, cover)
        if i == index:
            steps += 1
            index = cover
    return steps


# 给定数组nums和k，将数组中的数字取反k次，使数组和最大
def largestSumAfterKNegations(A: [int], K: int) -> int:
    A.sort(key=lambda x: abs(x), reverse=True)  # 第一步：按照绝对值降序排序数组A
    for i in range(len(A)):  # 第二步：执行K次取反操作【将负数取反】
        if A[i] < 0 < K:  # A[i] < 0 and k > 0
            A[i] *= -1
            K -= 1
    if K % 2 == 1:  # 第三步：如果K还有剩余次数，将绝对值最小的元素取反【此时数组中全是正数,只需要再执行一次】
        A[-1] *= -1
    result = sum(A)  # 第四步：计算数组A的元素和
    return result


# 加油站，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。初始值为0，顺序绕路环行，每个站点加油gas[i]
def canCompleteCircuit(gas: [int], cost: [int]) -> int:
    start = 0
    currentSum = 0
    totalSum = 0
    for i in range(len(gas)):
        currentSum += gas[i] - cost[i]
        totalSum += gas[i] - cost[i]
        if currentSum < 0:
            currentSum = 0
            start = i + 1  # 当前面总剩余油量为负，起始点定为i+1
    if totalSum < 0: return -1  # 若总的油量小于消耗油量，则不可能环行一周
    return start


# 分发糖果，给定ratings数组，求最少需要的糖果
# 每个孩子至少分配到 1 个糖果。
# 相邻两个孩子评分更高的孩子会获得更多的糖果。
def candy(ratings: [int]) -> int:
    candyArr = [1 for _ in range(len(ratings))]
    for i in range(len(ratings) - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candyArr[i] = candyArr[i + 1] + 1
    for i in range(1, len(ratings)):
        if ratings[i] > ratings[i - 1]:
            # 求max为了避免类似的情况
            # [1,6,10,8,7,3,2]
            # 第一轮变为 [1,1,5,4,3,2,1]
            # 若不是求max只是+1的话会变成[1,2,3,4,3,2,1]
            candyArr[i] = max(candyArr[i - 1] + 1, candyArr[i])
    return sum(candyArr)


# 找零，按序找零
def lemonadeChange(self, bills: [int]) -> bool:
    if bills[0] > 5: return False
    five = ten = twenty = 0
    for i in range(len(bills)):
        if bills[i] == 5:
            five += 1
        if five < 1: return False
        if bills[i] == 10:
            ten += 1
            five -= 1
        if bills[i] == 20:
            twenty += 1
            if ten > 0:
                ten -= 1
                five -= 1
            elif five > 2:
                five -= 3
            else:
                return False
    return True


# 身高排序，people[i] = [hi, ki] 表示第 i 个人的身高为 hi ，前面 正好 有 ki 个身高大于或等于 hi 的人。
def reconstructQueue(self, people: [[int]]) -> [[int]]:
    # 先按照h维度的身高顺序从高到低排序。确定第一个维度
    # lambda返回的是一个元组：当-x[0](维度h）相同时，再根据x[1]（维度k）从小到大排序，-x[0]是因为要从大到小排序，h逆序，k正序
    people.sort(key=lambda x: (-x[0], x[1]))
    que = []
    # 根据每个元素的第二个维度k，贪心算法，进行插入
    # people已经排序过了：同一高度时k值小的排前面。
    for p in people:
        que.insert(p[1], p)
    return que

