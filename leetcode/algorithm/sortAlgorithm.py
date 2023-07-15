arr = [9, 8, 7, 5, 6, 4, 2]

# 冒泡排序 [9, 8, 7, 6, 5, 4]


def bubbleSort(arr: []) -> []:
    for i in range(len(arr)):
        for j in range(1, len(arr)-i):
            if arr[j-1] > arr[j]:
                arr[j-1], arr[j] = arr[j], arr[j-1]
    return arr


# 选择排序 [9, 8, 7, 6, 5, 4]


def selectSort(arr: []) -> []:
    for i in range(len(arr)-1):
        for j in range(i+1, len(arr)):
            if arr[j] < arr[i]:
                arr[i], arr[j] = arr[j], arr[i]
    return arr


# 快速排序 [9, 8, 7, 6, 5, 4, 2], 分治法


def fastSort(arr: []) -> []:
    n = len(arr)
    if n <= 1: return arr
    middle = arr[0]
    left = [arr[i] for i in range(1, len(arr)) if arr[i] < middle]
    right = [arr[i] for i in range(1, len(arr)) if arr[i] >= middle]
    return fastSort(left) + [middle] + fastSort(right)


# 归并排序 [9, 8, 7, 6, 5, 4, 2]

print(fastSort(arr))