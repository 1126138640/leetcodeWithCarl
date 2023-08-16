import numpy as np
from scipy.sparse import coo_matrix


def sparse_tensor(tensor):
    # 获取非零元素的索引和值
    indices = np.nonzero(tensor)
    values = tensor[indices]
    print(type(indices))
    # 处理高于二维的张量
    if tensor.ndim > 2:
        # 递归处理每个维度
        sub_tensors = np.split(indices, indices.shape[1], axis=1)
        sub_tensors_values = np.split(values, indices.shape[1])
        sub_sparse_tensors = [sparse_tensor(sub_tensor) for sub_tensor in sub_tensors]

        # 合并稀疏矩阵
        row_indices = np.concatenate([sub_sparse.row for sub_sparse in sub_sparse_tensors])
        col_indices = np.concatenate([sub_sparse.col for sub_sparse in sub_sparse_tensors])
        values = np.concatenate([sub_sparse.data for sub_sparse in sub_sparse_tensors])
        shape = tensor.shape
        sparse_tensors = coo_matrix((values, (row_indices, col_indices)), shape=shape)
    else:
        # 创建稀疏矩阵
        sparse_tensors = coo_matrix((values, indices), shape=tensor.shape)

    return sparse_tensors


def sparse_dot_product(tensor1, tensor2):
    # 将tensor1稀疏化
    values1 = tensor1[np.nonzero(tensor1)]
    indices1 = np.nonzero(tensor1)
    sparse_tensor1 = coo_matrix((values1, indices1), shape=tensor1.shape)

    # 将tensor2稀疏化
    values2 = tensor2[np.nonzero(tensor2)]
    indices2 = np.nonzero(tensor2)
    sparse_tensor2 = coo_matrix((values2, indices2), shape=tensor2.shape)

    # 计算稀疏张量的点乘
    dot_product = sparse_tensor1.multiply(sparse_tensor2).sum()
    return dot_product


if __name__ == '__main__':
    # 生成一个随机的高维张量
    tensor = np.random.rand(100, 100, 100)
    # 设置稀疏化阈值
    threshold = 0.5

    # 将小于阈值的元素置为零
    tensor[tensor < threshold] = 0

    # 打印稀疏化后的张量
    print("稀疏化后的张量：")
    print(tensor)
    print(sparse_dot_product(tensor, tensor))
