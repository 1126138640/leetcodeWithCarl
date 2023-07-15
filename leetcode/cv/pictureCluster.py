import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

Cus_data = pd.read_table('data.txt', sep=',', header=None)
Cus_data.info()     # 查看是否存在缺失值，与形状有存在差异，则存在缺失值

x1 = Cus_data.iloc[:, :2].values  # 使用最后两列作为分群依据
x2 = Cus_data.iloc[:, 2:].values
kmeans_model = KMeans(n_clusters=3, init='k-means++', random_state=0)   # 模型创建
y_kmeans = kmeans_model.fit_predict(x2)  # 进行聚类处理,这里数据已经是array形式
Cus_data['聚类结果'] = kmeans_model.labels_
# 聚类结果可视化
# 颜色/标签/形状列表
colors_list = ['teal', 'skyblue', 'tomato']
labels_list = ['id_1', 'id_2', 'id_3']
markers_list = ['o', '*', 'D']  # 分别为圆、星型、菱形, 四边形，六边形

# 进行x[y_kmeans==i,0]
for i in range(3):
    plt.scatter(x1[y_kmeans == i, 0], x1[y_kmeans == i, 1], s=10, c=colors_list[i], label=labels_list[i], marker=markers_list[i])

# # 设置聚类中心点，颜色设置为黄色
# plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], s=10, c='yellow', label='centroids')
plt.legend(loc=1)  # 图例位置放在第二象限
plt.xlabel('')
plt.ylabel('')
# plt.xlim(-100, 100)
# plt.ylim(-200, 200)
plt.show()