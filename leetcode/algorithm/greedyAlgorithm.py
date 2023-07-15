# 小于n且数字单调递增的最大值
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


print(ace_max_number(10))