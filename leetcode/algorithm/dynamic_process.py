import copy
import time
from math import floor

'''
动态规划的核心是找到子问题
优化的方式是加一个缓存，不重复执行子问题
1.dp数组的定义和下标。
2.递推公式。
3.dp数组如何初始化，初始化也需要注意。
4.遍历顺序，比较考究，01 先遍历背包，后遍历物品。
    4.1排列和组合的遍历顺序是不相同的。
    4.1.1 排列：背包在外 物品在内。（322）
    4.1.2 组合：物品在外，背包在内。（518）
5.（出现问题）打印dp数组。（打印dp数组，检查是否有问题，检验1 2 3 4 步骤）
'''


# 斐波那契数列， dp五步走，dp数组--递推公式--初始化--遍历顺序
def fib(n: int) -> int:
    dp = [i for i in range(n + 1)]
    for i in range(n + 1):
        if i > 1:
            dp[i] = dp[i - 1] + dp[i - 2]
    return dp[-1]


origin_price = [4, 5, 8, 9, 10, 17, 17, 20, 24, 30, 33]


# 使用最小花费爬楼梯
# 到达第i位置花费为dp[i]
# cost[0,1,2,2], dp[0,0,0,1,2]
def minCostClimbingStairs(cost: [int]) -> int:
    dp = [0 for _ in range(len(cost) + 1)]
    for i in range(2, len(cost) + 1):
        dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
    return dp[len(cost)]


# 机器人从左上角走到左下角一共有几条路径
def uniquePaths(m: int, n: int) -> int:
    dp = [[1 for i in range(n)] for j in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[-1][-1]


# 数论的方法 C(m+n-1)^(m-2) 排列组合
def uniquePaths1(self, m: int, n: int) -> int:
    numerator = 1  # 分子
    denominator = m - 1  # 分母
    count = m - 1  # 计数器，表示剩余需要计算的乘积项个数
    t = m + n - 2  # 初始乘积项
    while count > 0:
        numerator *= t  # 计算乘积项的分子部分
        t -= 1  # 递减乘积项
        while denominator != 0 and numerator % denominator == 0:
            numerator //= denominator  # 约简分子
            denominator -= 1  # 递减分母
        count -= 1  # 计数器减1，继续下一项的计算
    return numerator  # 返回最终的唯一路径数


# 存在障碍物的情况下，机器人从左上角走到左下角一共有几条路径
def uniquePathsWithObstacles(obstacleGrid: [[int]]) -> int:
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    dp = [[1 for i in range(n)] for j in range(m)]
    for i in range(m):
        for j in range(n):
            if obstacleGrid[i][j] == 1:
                dp[i][j] = 0
                if i == 0:
                    for c in range(j, n):
                        dp[i][c] = 0
                if j == 0:
                    for v in range(i, m):
                        dp[v][j] = 0
            else:
                if i > 0 and j > 0:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[-1][-1]


# 将整数n拆分成k个数，求这k个数乘积的最大值
def integerBreak(n: int) -> int:
    dp = [1 for _ in range(n + 1)]
    # 从3开始，n=2、1结果均为1
    for i in range(3, n + 1):
        summary = 1
        for j in range(1, i // 2 + 1):
            summary = max([j * dp[i - j], j * (i - j), summary])
        dp[i] = summary
    return dp[n]


# 输入n，求由节点1-n组成的二叉搜索树共有多少种，按左右子树结点个数来判断可组成二叉搜索树的个数
def numTrees(n: int) -> int:
    dp = [1 for i in range(n + 1)]  # do[i]表示i个节点可组成二叉搜索树的个数
    for i in range(2, n + 1):
        dp[i] = 0
        for j in range(i):
            # j和i-j-1分别代表左右子树节点的个数
            dp[i] += dp[j] * dp[i - j - 1]
    return dp[-1]


'''
背包问题：0-1背包，完全背包
'''


# 标准0-1背包,每种物品只有0和1两种状态，放或不放；二维dp数组
def bagProblem2D(weight: [int], value: [int], space: int):
    # i表示物品，j表示背包重量
    # dp[i-1][j]表示背包容量为j的情况下不放第i个物品
    # dp[i-1][j-weight[i]+value[i] 表示放第i个物品
    dp = [[0 for _ in range(space + 1)] for _ in range(len(weight))]
    for i in range(len(weight)):
        for j in range(space + 1):
            if i == 0 and j >= weight[i]:
                dp[i][j] = value[i]
            if i > 0 and j > 0:
                if j < weight[i]:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])
    return dp[-1][-1]


# 标准0-1背包,每种物品只有0和1两种状态，放或不放；一维dp数组--滚动数组
def bagProblem1D(weight: [int], value: [int], space: int):
    # 初始化
    dp = [0] * (space + 1)  # 容量为j的背包所能装的最大价值
    for i in range(len(weight)):  # 遍历物品
        for j in range(space, weight[i] - 1, -1):  # 遍历背包容量
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
    return dp[-1]


# 分割等和子集--可以抽象成0-1背包，其中重量和价值都是nums，当最大重量为sum//2是，可以找到等和子集
def canPartition(nums: [int]) -> bool:
    weights = values = nums
    summary = 0
    for i in nums:
        summary += i
    if summary % 2 != 0:
        return False
    dp = [0 for i in range(summary // 2 + 1)]
    for j in range(len(weights)):
        for i in range(summary // 2, weights[j] - 1, -1):  # 注意是倒叙遍历，不然会重复放入
            dp[i] = max(dp[i], dp[i - weights[j]] + values[j])
    return dp[-1] == summary // 2


# 研磨石头，即将石头分成两堆，使其差最小
# 此时的stones[i] = weight[i] = value[i]
def lastStoneWeightII(stones: [int]) -> int:
    weights = values = stones
    summary = sum(stones)
    dp = [0 for i in range(summary // 2 + 1)]
    for i in range(len(weights)):
        for j in range(summary // 2, weights[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    return summary - dp[-1] * 2


# 背包放满有几种放法
# 此时的nums[i] = weight[i] = value[i]
def findTargetSumWays(nums: [int], target: int) -> int:
    weights = nums
    if abs(target) > sum(nums):
        return 0
    if (sum(nums) + target) % 2 != 0:
        return 0
    length = (sum(nums) + target) // 2
    dp = [1] + [0 for i in range(1, length + 1)]  # 背包容量为i的时候，有几种放法
    for i in range(len(weights)):
        for j in range(length, weights[i] - 1, -1):  # 左闭右开，所以-1，倒序是为了避免重复
            dp[j] += dp[j - weights[i]]  # 累加
    return dp[-1]


# 装满背包最多能装几个物品
# 二维背包最多能背几个物品
def findMaxForm(strs: [str], m: int, n: int) -> int:
    weights = []
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in strs:
        v1 = v2 = 0
        for j in i:
            if j == '0': v1 += 1
            if j == '1': v2 += 1
        weights.append((v1, v2))
    for k in weights:
        for i in range(m, k[0] - 1, -1):
            for j in range(n, k[1] - 1, -1):
                dp[i][j] = max(dp[i][j], dp[i - k[0]][j - k[1]] + 1)
    return dp[-1][-1]


'''
完全背包,背包遍历正序，纯完全背包可以调换物品背包遍历顺序
'''


# 标准完全背包【放满背包，背包能放得最大价值】
def perBagAlgorithm(weights: [], values: [], space: int):
    dp = [0 for i in range(space + 1)]
    for i in range(len(weights)):
        for j in range(weights[i], space + 1):
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    return dp[-1]


# 可以凑成总金额的硬币组合数，即完全背包的几种放法【放满背包几种放法】
def change(amount: int, coins: [int]) -> int:
    dp = [1] + [0 for i in range(amount)]  # 表示背包空间为n时，有几种放法
    for i in range(len(coins)):
        for j in range(coins[i], amount + 1):  # 正序遍历，物品可使用多次
            dp[j] += dp[j - coins[i]]
    return dp[-1]


# 从 nums 中找出并返回总和为 target 的元素组合的个数，有序【排列】，(3,2)与(2,3)是不同的
# 如果求组合数就是外层for循环遍历物品，内层for遍历背包。【无序】
# 如果求排列数就是外层for遍历背包，内层for循环遍历物品。【有序】【爬楼梯问题可转化为有序完全背包】
def combinationSum4(nums: [int], target: int) -> int:
    dp = [1] + [0 for _ in range(target)]
    for i in range(1, target + 1):  # 遍历背包
        for j in range(len(nums)):  # 遍历物品
            if i - nums[j] >= 0:
                dp[i] += dp[i - nums[j]]
    return dp[-1]


# 进阶版爬楼梯, 爬n阶楼梯，每次可以爬m层【放满背包有几种放法】
def climbStairs(n: int, m: int) -> int:
    nums = [i for i in range(m + 1)]
    dp = [1] + [0 for _ in range(n)]
    for i in range(1, n + 1):
        for j in range(len(nums)):
            if i >= nums[j]:
                dp[i] += dp[i - nums[j]]
    return dp[-1]


# 装满容量为j得背包，最少物品数【放满背包最少物品数量】
def coinChange(coins: [int], amount: int) -> int:
    dp = [0] + [pow(2, 31) for i in range(amount)]  # 当容量为0时，有0个，当容量大于0时，初始化一个大数，因为求得是最小值
    for j in range(len(coins)):
        for i in range(coins[j], amount + 1):
            dp[i] = min(dp[i], dp[i - coins[j]] + 1)
    return dp[-1] if dp[-1] != pow(2, 31) else -1  # 当装不满时，背包还是初始化的数


# 给定一个数n，最少需要几个完全平方数使其和为n【放满背包最少物品数量】
def numSquares(n: int) -> int:
    nums = []
    dp = [0] + [n for _ in range(n)]  # 要初始化为一个大数
    # 求完全平方数
    for i in range(floor(pow(n, 0.5)) + 1):
        nums.append(pow(i, 2))
    # 装满背包最少需要几样物品
    for i in range(len(nums)):
        for j in range(nums[i], n + 1):
            dp[j] = min(dp[j], dp[j - nums[i]] + 1)
    return dp[-1]


# 给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s
# 每次判断len(wordDict[i])长度的，判断0-j和j-j+len(wordDict[i])是否都为True
def wordBreak(s: str, wordDict: [str]) -> bool:
    dp = [True] + [False for _ in range(len(s))]
    # 排列顺序不同，组成的串不一样，因此求得是排列数，先背包再物品
    # 遍历背包
    for j in range(1, len(s) + 1):
        # 遍历单词
        for word in wordDict:
            if j >= len(word):
                dp[j] = dp[j] or (dp[j - len(word)] and word == s[j - len(word):j])
    return dp[len(s)]


'''
多重背包：即每个物品可以用m次，将物品展开，即变成了0-1背包
'''


# 标准多重背包--方法1将物品展开【改变物品个数】
def test_multi_pack():
    weight = [1, 3, 4]
    value = [15, 20, 30]
    nums = [2, 3, 2]
    bagWeight = 10

    # 将数量大于1的物品展开
    for i in range(len(nums)):
        while nums[i] > 1:
            weight.append(weight[i])
            value.append(value[i])
            nums[i] -= 1

    dp = [0] * (bagWeight + 1)
    for i in range(len(weight)):  # 遍历物品
        for j in range(bagWeight, weight[i] - 1, -1):  # 遍历背包容量
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
        for j in range(bagWeight + 1):
            print(dp[j], end=" ")


# 标准多重背包--方法2遍历物品可用个数【改变遍历个数】
def test_multi_pack1():
    weight = [1, 3, 4]
    value = [15, 20, 30]
    nums = [2, 3, 2]
    bagWeight = 10
    dp = [0] * (bagWeight + 1)

    for i in range(len(weight)):  # 遍历物品
        for j in range(bagWeight, weight[i] - 1, -1):  # 遍历背包容量
            # 以上为01背包，然后加一个遍历个数
            for k in range(1, nums[i] + 1):  # 遍历个数
                if j - k * weight[i] >= 0:
                    dp[j] = max(dp[j], dp[j - k * weight[i]] + k * value[i])

        # 打印一下dp数组
        for j in range(bagWeight + 1):
            print(dp[j], end=" ")


'''
打家劫舍
'''


# 求数组序列的最大和，不能取相邻值
# 第i个最大值只与i-1和i-2有关
def rob1(nums: [int]) -> int:
    dp = [nums[0]] + [0 for _ in range(1, len(nums))]  # dp[i]表示nums[0:i]内的最大值
    for i in range(1, len(nums)):
        if i < 2:
            dp[i] = max(dp[i - 1], nums[i])  # 避免数组越界
        else:
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])  # dp[i] 与 dp[i-1]、dp[i-2]的关系
    return dp[-1]


# 求数组序列最大和，不能取相邻值，不能同时取首尾
# 第i个最大值只与i-1和i-2有关
# 考虑不取首位的两种情况，1、[0:n-2]【去尾】；2、[1:n-1]【去首】；对这两种序列求最大和【不取相邻值】
def rob2(nums: [int]) -> int:
    if len(nums) < 3: return max(nums)
    dp2 = [nums[0]] + [max(nums[0], nums[1])] + [0 for i in range(2, len(nums) - 1)]
    dp1 = [nums[1]] + [max(nums[1], nums[2])] + [0 for i in range(3, len(nums))]
    for i in range(3, len(nums)):
        dp1[i - 1] = max(dp1[i - 2], dp1[i - 3] + nums[i])
    for i in range(2, len(nums) - 1):
        dp2[i] = max(dp2[i - 1], dp2[i - 2] + nums[i])
    return max(dp1[-1], dp2[-1])


# 小区分布呈二叉树结构，相邻节点不可同时取
class TreeNode:
    def __init__(self, val: int, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def rob(self, root: [TreeNode]) -> int:
    # dp数组（dp table）以及下标的含义：
    # 1. 下标为 0 记录 **不偷该节点** 所得到的的最大金钱
    # 2. 下标为 1 记录 **偷该节点** 所得到的的最大金钱
    dp = self.traversal(root)
    return max(dp)


# 要用后序遍历, 因为要通过递归函数的返回值来做下一步计算【树形DP，存两个状态值】
def traversal(self, node):
    # 递归终止条件，就是遇到了空节点，那肯定是不偷的
    if not node:
        return 0, 0
    left = self.traversal(node.left)
    right = self.traversal(node.right)
    # 不偷当前节点, 偷子节点
    val_0 = max(left[0], left[1]) + max(right[0], right[1])
    # 偷当前节点, 不偷子节点
    val_1 = node.val + left[0] + right[0]
    return val_0, val_1


'''
买卖股票最佳时机
'''


# 只能买卖一次
# 何时买，何时卖能达到最大收益
def maxProfit(prices: [int]) -> int:
    dp = [0 for _ in range(len(prices))]
    for i in range(1, len(prices)):
        dp[i] = max(0, prices[i] - prices[i - 1] + dp[i - 1])
    return max(dp)


# carl动态规划，二维DP数组
def maxProfit_carl(prices: [int]) -> int:
    dp = [[-prices[0], 0]] + [[0, 0] for _ in range(1, len(prices))]
    for i in range(1, len(prices)):
        dp[i][0] = max(dp[i - 1][0], -prices[i])  # 每次都是0-price[i]是为了求最小值，因为只能买卖一次
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i])
    return max(dp[-1][1], dp[-1][0])


# 可以多次买卖，同一天只能持有一支股票,求可多次买卖的条件下，可获得的最大利润，可在同一天进行买入、卖出
# 可以买卖多次
def maxProfit1(prices: [int]) -> int:
    dp = [0 for _ in range(len(prices))]
    for i in range(1, len(prices)):
        dp[i] = max(dp[i - 1], dp[i - 1] + prices[i] - prices[i - 1])
    return dp[-1]


# carl动态规划，二维DP数组
def maxProfit1_carl(prices: [int]) -> int:
    dp = [[-prices[0], 0]] + [[0, 0] for _ in range(1, len(prices))]
    for i in range(1, len(prices)):
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] - prices[i])  # 每次都是之前卖出的钱-price[i]，因为可以买卖多次
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i])
    return max(dp[-1][0], dp[-1][1])


# 最多可买卖两次,同一天只能持有一支股票,求可多次买卖的条件下，可获得的最大利润，可在同一天进行买入、卖出
# 可买卖两次
def maxProfit2_carl(prices: [int]) -> int:
    dp = [[0, -prices[0], 0, -prices[0], 0]] + [[0, 0, 0, 0, 0] for _ in range(1, len(prices))]
    for i in range(1, len(prices)):
        dp[i][0] = dp[i - 1][0]  # 不操作/保持状态
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])  # 第一次买入状态
        dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] + prices[i])  # 第一次卖出状态
        dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] - prices[i])  # 第二次买入状态
        dp[i][4] = max(dp[i - 1][4], dp[i - 1][3] + prices[i])  # 第二次卖出状态
    return dp[-1][4]


# 最多可以买卖k次，同一天只能持有一支股票,求可多次买卖的条件下，可获得的最大利润，可在同一天进行买入、卖出
# 可买卖k次
def maxProfit3_carl(k: int, prices: [int]) -> int:
    # 买卖两次有5个状态，买卖k次有2k+1个状态，构造长度为2k+1的状态数组dp[i]
    # 其中dp[i][0]表示不操作
    # dp[i][j]j为奇数时表示买入，j为偶数时表示卖出
    dp = [[0] + [0 if i % 2 == 0 else -prices[0] for i in range(1, 2 * k + 1)]] + [[0 for _ in range(2 * k + 1)] for _ in range(1, len(prices))]
    for i in range(1, len(prices)):
        for j in range(0, 2 * k - 1, 2):
            # dp[i - 1][j + 1]上一天本次买入；dp[i - 1][j] - prices[i]上一次卖出-prices[i]
            dp[i][j + 1] = max(dp[i - 1][j + 1], dp[i - 1][j] - prices[i])
            # dp[i - 1][j + 2]上一天本次卖出；dp[i - 1][j + 1] + prices[i]本次买入+prices[i]
            dp[i][j + 2] = max(dp[i - 1][j + 2], dp[i - 1][j + 1] + prices[i])
    return max(dp[-1])


# 含冷冻期，买卖股票，同一天只能持有一只股票
# 可多次买卖
def maxProfit4(prices: [int]) -> int:
    dp = [[-prices[0], 0]] + [[0, 0] for _ in range(1, len(prices))]
    # 状态压缩了，只保持了持有股票和不持有股票两种状态
    for i in range(1, len(prices)):
        if i == 1:
            dp[i][0] = max(dp[i - 1][0], -prices[i])
        else:
            # 买入与前一天卖或不卖两种状态有关，要么保持买入，要么冷冻期过了买入
            dp[i][0] = max(dp[i - 1][0], dp[i - 2][1] - prices[i])
        # 卖出只有前一天卖出或者前一天买入有关，要么保持，要么当天卖
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i])
    return max(dp[-1][0], dp[-1][1])


def maxProfit4_carl(prices: [int]) -> int:
    # 四维数组表示四种状态
    # 状态一：持有股票状态（今天买入股票，或者是之前就买入了股票然后没有操作，一直持有）
    # 不持有股票状态，这里就有两种卖出股票状态
    # 状态二：保持卖出股票的状态（两天前就卖出了股票，度过一天冷冻期。或者是前一天就是卖出股票状态，一直没操作）
    # 状态三：今天卖出股票
    # 状态四：今天为冷冻期状态，但冷冻期状态不可持续，只有一天！
    dp = [[-prices[0], 0, 0, 0]]+[[0, 0, 0, 0] for _ in range(1, len(prices))]
    for i in range(1, len(prices)):
        # 买入状态，与前一天是买入状态、前一天是冷冻期状态、前一天是保持卖出的状态有关
        dp[i][0] = max([dp[i-1][0], dp[i-1][3]-prices[i], dp[i-1][1]-prices[i]])
        # 保持卖出状态，与前一天是保持卖出状态、前一天是冷冻期状态有关
        dp[i][1] = max(dp[i-1][1], dp[i-1][3])
        # 当天卖出状态，只与前一天是买入状态有关
        dp[i][2] = dp[i-1][0] + prices[i]
        # 当天为冷冻期状态，至于前一天是卖出状态有关
        dp[i][3] = dp[i-1][2]
    return max(dp[-1])


# 每笔交易都要支付一笔手续费
# 可以无限次买卖，从买入到卖出只需要支付一次手续费
def maxProfit5(prices: [int], fee: int) -> int:
    dp = [[-prices[0], 0]] + [[0, 0] for _ in range(1, len(prices))]
    for i in range(1, len(prices)):
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] - prices[i])
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i] - fee)
    return max(dp[-1])


'''
子序列系列
'''


# 最长递增子序列
def lengthOfLIS(nums: [int]) -> int:
    dp = [1 for _ in range(len(nums))]  # i之前（包括i）最长递增子序列的长度为dp[i]
    result = 1
    for i in range(1, len(nums)):
        for j in range(0, i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
        result = max(dp[i], result)  # 求值最大的dp[i]
    return result


# 最长递增子序列的个数
def findNumberOfLIS(nums: List[int]) -> int:
    if len(nums) <= 1: return len(nums)
    dp = [1 for _ in range(len(nums))]  # i之前（包括i）最长递增子序列的长度为dp[i]
    count = [1 for _ in range(len(nums))]  # 以nums[i]为结尾的最长递增子序列的长度
    sum = maxCount = 0
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                # 出现新的最长递增子序列
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    count[i] = count[j]
                # 出现相同长度的另一个递增子序列
                elif dp[j] + 1 == dp[i]:
                    count[i] += count[j]
            maxCount = max(dp[i], maxCount)
    for i in range(len(nums)):
        if maxCount == dp[i]: sum += count[i]
    return sum


# 最长连续递增子序列
def findLengthOfLCIS(self, nums: [int]) -> int:
    dp = [1 for _ in range(len(nums))]
    result = 1
    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:
            dp[i] = dp[i - 1] + 1
        result = max(result, dp[i])
    return result


# 最长连续公共子序列，和最长公共子序列不同
def findLength(nums1: [int], nums2: [int]) -> int:
    # 以nums1[i-1]，nums2[j-1]结尾的最长连续子序列的长度
    # dp[-1][-1]不一定是最大值
    dp = [[0 for _ in range(len(nums1) + 1)] for _ in range(len(nums2) + 1)]
    result = 0
    for i in range(1, len(nums2) + 1):
        for j in range(1, len(nums1) + 1):
            if nums1[j - 1] == nums2[i - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            result = max(result, dp[i][j])
    return result


# 最长公共子序列，不要求连续,但要求有序
def longestCommonSubsequence(text1: str, text2: str) -> int:
    # 以text1[i-1]，text2[j-1]结尾的最长子序列的长度
    # dp[-1][-1]一定为最大值,因为累加了之前的状态
    dp = [[0 for _ in range(len(text1) + 1)] for _ in range(len(text2) + 1)]
    for i in range(1, len(text2) + 1):
        for j in range(1, len(text1) + 1):
            if text1[j - 1] == text2[i - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
    return dp[-1][-1]


# 具有最大和的连续子序列
def maxSubArray(nums: [int]) -> int:
    dp = [nums[0]] + [0 for _ in range(1, len(nums))]
    result = dp[0]
    for i in range(1, len(nums)):
        dp[i] = max(nums[i], dp[i - 1] + nums[i])
        result = max(dp[i], result)
    return result


# 判断s是否为t的子序列，不连续,但有序;即从t中删除几个元素可得到s【模拟删除问题】
def isSubsequence_DP(s: str, t: str) -> bool:
    # dp[i][j] 表示以下标i-1为结尾的字符串s，和以下标j-1为结尾的字符串t，相同子序列的长度为dp[i][j]
    dp = [[0] * (len(t) + 1) for _ in range(len(s) + 1)]
    for i in range(1, len(s) + 1):
        for j in range(1, len(t) + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # 即i继续往前走，j停一步
                dp[i][j] = dp[i][j-1]  # 模拟删除，不考虑t[j-1]，t是母串
    if dp[-1][-1] == len(s):
        return True
    return False


# 计算s中有多少回文子串
def countSubstrings(s: str) -> int:
    # 以s[i]为头s[j]为尾的字符串是否为回文串
    dp = [[False for _ in range(len(s))] for _ in range(len(s))]
    result = 0
    # 从左下角向右上角遍历
    for i in range(len(s) - 1, -1, -1):
        for j in range(i, len(s)):
            if s[i] == s[j] and (j - i <= 1 or dp[i + 1][j - 1]):
                dp[i][j] = True
                result += 1
    return result


# 判断是否为回文串，双指针法
def countSubstrings_DoublePointer(self, s: str) -> int:
    result = 0
    for i in range(len(s)):
        result += self.extend(s, i, i, len(s))  # 以i为中心  一个点为中心
        result += self.extend(s, i, i + 1, len(s))  # 以i和i+1为中心，两个点为中心
    return result


# 最长回文子序列的长度
def longestPalindromeSubseq(s: str) -> int:
    # 以s[i]为头s[j]为尾的最长回文子串的长度
    dp = [[1 if i == j else 0 for i in range(len(s))] for j in range(len(s))]
    for i in range(len(s) - 1, -1, -1):
        for j in range(i, len(s)):
            if s[i] == s[j]:
                if j - i <= 1:
                    dp[i][j] = j - i + 1
                else:
                    # 前后各+1
                    dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][-1]


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


def extend(self, s, i, j, n):
    res = 0
    while i >= 0 and j < n and s[i] == s[j]:
        i -= 1
        j += 1
        res += 1
    return res


# 求最长回文子串的长度



def isSubsequence_doublePointer(s: str, t: str) -> bool:
    # 双指针解法
    slow = fast = 0
    while slow < len(s) and fast < len(t):
        if s[slow] == t[fast]:
            slow += 1
            fast += 1
        else:
            fast += 1
    return slow == len(s)


# 给你两个字符串 s 和 t ，统计并返回在 s 的 子序列 中 t 出现的个数
# 即，长序列s删除其中的元素，有几种删除方式能得到t【模拟删除问题】
def numDistinct(s: str, t: str) -> int:
    # 当 j = 0 时,表示t为空串，即s有几种删除方式能得到空串，即dp[0][i] = 1
    # dp[i][j]：以i-1为结尾的s子序列中出现以j-1为结尾的t的个数为dp[i][j]
    dp = [[1 for _ in range(len(s) + 1)]] + [[0 for _ in range(len(s) + 1)] for _ in range(1, len(t) + 1)]
    for i in range(1, len(t) + 1):
        for j in range(1, len(s) + 1):
            if t[i - 1] == s[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + dp[i-1][j]  # 考虑s[i-1]和都不考虑两种情况的总和
            else:
                dp[i][j] = dp[i-1][j]  # 两者不相同考虑s[i-1]的情况，因为s为母串
    return dp[-1][-1]


# 求由word1变为word2要执行几步删除操作【模拟删除问题】
def minDistance(word1: str, word2: str) -> int:
    # 当i为0是表示空串，dp[0][j]要执行j步删除才能为空，即dp[0][j] = j；同理，dp[i][0] = i
    # dp[i][j]：以i-1为结尾的字符串word1，和以j-1位结尾的字符串word2，想要达到相等，所需要删除元素的最少次数
    dp = [[i for i in range(len(word1) + 1)]] + [[j for _ in range(len(word1) + 1)] for j in range(1, len(word2) + 1)]
    for i in range(1, len(word2) + 1):
        for j in range(1, len(word1) + 1):
            if word2[i - 1] == word1[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 不执行删除
            else:
                # 当同时删word1[i - 1]和word2[j - 1]，dp[i][j-1] 本来就不考虑 word2[j - 1]了，
                # 那么我在删 word1[i - 1]，是不是就达到两个元素都删除的效果，即 dp[i][j-1] + 1。
                # 考虑一方+1，两方都考虑+2，取最小
                dp[i][j] = min([dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 2])  # 删除1个或两个都删取最小
    return dp[-1][-1]


# 编辑距离，将word1通过几次增删改可以变为word2
def minDistance_th(word1: str, word2: str) -> int:
    # 当i为0是表示空串，dp[0][j]要执行j步删除才能为空，即dp[0][j] = j；同理，dp[i][0] = i
    # dp[i][j]：以i-1为结尾的字符串word1，和以j-1位结尾的字符串word2，想要达到相等，所需要删除元素的最少次数
    dp = [[i for i in range(len(word1) + 1)]] + [[j for _ in range(len(word1) + 1)] for j in range(1, len(word2) + 1)]
    for i in range(1, len(word2) + 1):
        for j in range(1, len(word1) + 1):
            if word2[i - 1] == word1[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # 增删改均只需要一次操作
                dp[i][j] = min([dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]]) + 1
    return dp[-1][-1]


def cache_func(func):
    tmp = {}

    def wrapper(*args, **kwargs):
        n = args[0]
        if n not in tmp.keys():
            tmp[n] = func(n)
        return func(n)

    return wrapper


@cache_func
def cal_number(length):
    if length == 1:
        return origin_price[length - 1]
    max_prince = origin_price[length - 1]
    for i in range(1, length - 1):
        max_prince = max(max_prince, origin_price[i - 1] + cal_number(length - i))
        print(origin_price[i - 1])
    return max_prince


# 跳格子
@cache_func
def jumpStage(n):
    if n == 1 or n == 2:
        return n
    return jumpStage(n - 1) + jumpStage(n - 2)


# 输出数字 非递归
def numberOut(n):
    res = []
    start = 1
    while start <= n:
        tmp = copy.deepcopy(res)
        if len(tmp) >= 1:
            for i in tmp:
                res.append(i + [start])

        res.append([start])
        start += 1
    return res


# 深度优先搜索 DFS, 指数
st = [0] * 15  # 记录每一步的状态


def numberOut1(n, m):
    if n > m:
        for i in range(n):
            if st[i] == 1:
                print(i)
        print('/')
        return
    # 先不选
    st[n] = 2
    numberOut1(n + 1, m)
    st[n] = 0

    # 先选
    st[n] = 1
    numberOut1(n + 1, m)
    st[n] = 0


# 深度优先搜索 全排列
res = []


def allList(x: [], n):
    if len(x) == n:
        res.append(x)
    for i in range(1, n + 1):
        if i not in x:
            allList(x + [i], n)
    return res


# 深度优先搜索 组合输出


def combineOut(x: [], n, m, start):
    if len(x) == m:
        res.append(x)
    for i in range(start, n + 1):
        if i not in x:
            combineOut(x + [i], n, m, i + 1)
    return res


# 深度优先搜索 选数
def chooseNumber(x: [], sum, k, start, count):
    # 剪枝
    if len(x) - start + 1 + count < k: return res
    if count == k and is_PlainNumber(sum):
        res.append(sum)
    for i in range(start, len(x)):
        # count表示执行加法得次数
        chooseNumber(x, sum + x[i], k, i + 1, count + 1)
    return len(res), res


# 判断是不是素数
def is_PlainNumber(n):
    print(n, n // 2)
    for i in range(2, n // 2):
        if n % i == 0:
            return False
    return True


# 深度优先遍历 烤鸡
resN = 0


def turkey(start: int, sum: int, n: int, tmp: []):
    if sum > n:
        return  # 剪枝
    if start > 9:
        if sum == n:
            res.append(tmp.copy())
            print(tmp)
        return  # 剪枝
    for i in range(1, 4):
        tmp[start - 1] = i
        turkey(start + 1, sum + i, n, tmp)
        tmp[start - 1] = 0  # 恢复现场

# turkey(0, 0, 11, [0 for i in range(10)])
