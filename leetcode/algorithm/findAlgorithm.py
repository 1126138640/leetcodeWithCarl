def halveFind(a: [], n):
    left = 0
    right = len(a)-1
    while left <= right:
        middle = int(left + (right - left) / 2)  # 避免大数溢出
        if a[middle] < n:
            left = middle + 1
        elif a[middle] > n:
            right = middle - 1
        else:
            return middle
    return -1



print(halveFind([1, 2, 4, 6, 9, 10], 2))