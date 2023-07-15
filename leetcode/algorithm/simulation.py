# 创建螺旋矩阵【输入数字n，创建n阶的螺旋矩阵】
# tips 保证左闭右开，四条边统一原则
# 中心点，奇偶不同
# 正确创建n维list[[1 for _ in range(4)] for i in range (3)]；不能直接用[[]*n]*n，这样每层地址相同
def generateMatrix(n: int) -> [[int]]:
    if n == 1: return [[1]]
    res = [[0 for _ in range(n)] for i in range(n)]
    row = 0
    column = 0
    count = 1
    m = n
    while n//2:
        # right
        for i in range(row, m - row - 1):
            if count >= pow(n, 2): break
            res[column][i] = count
            count += 1
            row += 1
        # down
        for i in range(column, m - column - 1):
            if count >= pow(n, 2): break
            res[i][row] = count
            count += 1
            column += 1
        # left
        for i in range(row, m - row - 1, -1):
            if count >= pow(n, 2): break
            res[column][i] = count
            count += 1
            row -= 1
        # up
        for i in range(column, m - column - 1, -1):
            if count >= pow(n, 2): break
            res[i][row] = count
            count += 1
            column -= 1
        if count >= pow(n, 2): break
        column += 1
        row += 1
    if m % 2 == 0:
        res[column][row] = count
    else:
        res[column + 1][row + 1] = count
    return res


print(generateMatrix(5))