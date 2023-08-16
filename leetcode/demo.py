index = 0
res = []
while 1:
    s = input().strip().split()
    if index == 0:
        res += s
    print(res)
    if index == 1:
        l = len(res) // 2
        res = res[:l] + s + res[l:]
    index += 1
    if index == 2:
        index %= 2
        res = []