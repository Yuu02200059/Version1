import os

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import h5py
import random
import warnings
import scipy.io as sio
warnings.filterwarnings("ignore")


# 需要h5py读取
ALL_data = dict(
    Caltech101_7= {1: 'Caltech101_7', 'N': 1400, 'K': 7, 'V': 5, 'n_input': [1984, 512, 928, 254, 40]},
    HandWritten = {1: 'handwritten1031_v73', 'N': 2000, 'K': 10, 'V': 6, 'n_input': [240, 76, 216, 47, 64, 6]},
    Caltech101_20={1: 'Caltech101_20', 'N': 2386, 'K': 20, 'V': 6, 'n_input': [48, 40, 254, 1984, 512, 928]},
    LandUse_21 = {1: 'LandUse_21_v73','N': 2100, 'K': 21, 'V': 3, 'n_input': [20,59,40]},
    Scene_15 = {1: 'Scene_15','N': 4485, 'K': 15, 'V': 3, 'n_input': [20,59,40]},
    ALOI_100 = {1: 'ALOI_100_7', 'N': 10800, 'K': 100, 'V': 4, 'n_input': [77, 13, 64, 125]},
    YouTubeFace10_4Views={1: 'YTF10_4', 'N': 38654, 'K': 10, 'V': 4, 'n_input': [944, 576, 512, 640]},
    AWA={1: 'AWA_73', 'N': 10158, 'K': 50, 'V': 7, 'n_input': [2688, 2000, 2000, 2000, 2000, 4096,4096]},
    # AWA={1: 'AWA_73', 'N': 30735, 'K': 50, 'V': 6, 'n_input': [2688, 2000, 252, 2000, 2000, 2000]},
    EMNIST_digits_4Views={1: 'EMNIST_digits_4Views_v73', 'N': 280000, 'K': 10, 'V': 4, 'n_input': [944, 576, 512, 640]}
)
# ALL_data =Caltech101_7 = {1: 'Caltech101_7', 'N': 1474, 'K': 6, 'V': 56, 'n_input': [1984, 512, 928, 254, 40]}

path = './Dataset/'


# def get_mask(view_num, alldata_len, missing_rate):
#     '''生成缺失矩阵：
#     view_num为视图数
#     alldata_len为数据长度
#     missing_rate为缺失率
#     return 缺失矩阵 alldata_len*view_num大小的0和1矩阵
#     '''
#     missindex = np.ones((alldata_len, view_num))
#     b=((10 - 10*missing_rate)/10) * alldata_len
#     miss_begin = int(b)  #将b转换成整数 作为缺失开始的索引
#     for i in range(miss_begin, alldata_len):
#         missdata = np.random.randint(0, high=view_num,
#                                      size=view_num - 1)
#
#         missindex[i, missdata] = 0
#
#     return missindex
#
def get_mask(view_num, alldata_len, missing_rate):
    """
    生成缺失矩阵：
    - (1-missing_rate) 的数据是 *完整* 的 (全 1)
    - (missing_rate) 的数据是 *不完整* 的
    - 不完整的数据 *至少保留 2 个视图*
    """
    missindex = np.ones((alldata_len, view_num))

    # 计算不完整数据开始的索引
    complete_len = int(alldata_len * (1 - missing_rate))

    # 遍历不完整样本
    for i in range(complete_len, alldata_len):
        # 1. 决定要 *丢弃* 多少个视图
        #    (至少丢 1 个, 最多丢 V-2 个, 保证至少 2 个保留)

        if view_num <= 2:
            # 如果 V=2, 只能丢 1 个 (但这样只剩 1 个了)
            # 为了 L_align, 我们假设 V > 2
            num_to_drop = 1
        else:
            # 丢弃 1 到 V-2 个
            num_to_drop = np.random.randint(1, view_num - 1)  # high 是 V-1 (不包含)

        # 2. 选择要丢弃的视图索引
        drop_indices = np.random.choice(view_num, num_to_drop, replace=False)

        # 3. 在掩码中设置为 0
        missindex[i, drop_indices] = 0

    return missindex


# --- [修复结束] ---
def Form_Incomplete_Data(missrate=0.5, X = [], Y = []):


    size = len(Y[0])
    view_num = len(X)
    index = [i for i in range(size)]
    np.random.shuffle(index)
    for v in range(view_num):
        X[v] = X[v][index]
        Y[v] = Y[v][index]

    ##########################获取缺失矩阵###########################################
    missindex = get_mask(view_num, size, missrate)

    index_complete = []
    index_partial = []
    for i in range(view_num):
        index_complete.append([])
        index_partial.append([])
    for i in range(missindex.shape[0]):
        for j in range(view_num):
            if missindex[i, j] == 1:
                index_complete[j].append(i)
            else:
                index_partial[j].append(i)

    filled_index_com = []
    for i in range(view_num):
        filled_index_com.append([])
    max_len = 0
    for v in range(view_num):
        if max_len < len(index_complete[v]):
            max_len = len(index_complete[v])
    for v in range(view_num):
        if len(index_complete[v]) < max_len:
            diff_len = max_len - len(index_complete[v])

            diff_value = random.sample(index_complete[v], diff_len)
            filled_index_com[v] = index_complete[v] + diff_value
        elif len(index_complete[v]) == max_len:
            filled_index_com[v] = index_complete[v]

    filled_X_complete = []
    filled_Y_complete = []
    for i in range(view_num):
        filled_X_complete.append([])
        filled_Y_complete.append([])
        filled_X_complete[i] = X[i][filled_index_com[i]]
        filled_Y_complete[i] = Y[i][filled_index_com[i]]
    for v in range(view_num):

        X[v] = torch.from_numpy(X[v])
        filled_X_complete[v] = torch.from_numpy(filled_X_complete[v])

    return X, Y, missindex, filled_X_complete, filled_Y_complete, index_complete, index_partial

# def load_data(dataset, missrate):
#     data = h5py.File(path + dataset[1] + ".mat")
#     X = []
#     Y = []
#     Label = np.array(data['Y']).T
#     print(data.keys())
#     Label = Label.reshape(Label.shape[0])
#     mm = MinMaxScaler()
#     for i in range(data['X'].shape[1]):
#         diff_view = data[data['X'][0, i]]
#         diff_view = np.array(diff_view, dtype=np.float32).T
#         std_view = mm.fit_transform(diff_view)
#         X.append(std_view)
#         Y.append(Label)
#     X, Y, missindex, X_com, Y_com, index_com, index_incom = Form_Incomplete_Data(missrate=missrate, X=X, Y=Y)
#
#     return X, Y, missindex, X_com, Y_com, index_com, index_incom
def load_data(dataset, missrate):
    file_path = path + dataset[1] + ".mat"

    X = []
    Y = []

    try:
        # -------------------------------------------
        # 尝试 1: 使用 h5py 读取 (针对 MATLAB v7.3)
        # -------------------------------------------
        data = h5py.File(file_path, 'r')

        # 读取标签
        # 注意：h5py 读取时通常需要转置 (.T)
        Label = np.array(data['Y']).T
        Label = Label.reshape(Label.shape[0])

        # 读取视图数据
        # v7.3 中 X 通常是一个包含引用的 Cell Array
        for i in range(data['X'].shape[1]):
            # 获取引用，再通过引用拿到真实数据
            diff_view = data[data['X'][0, i]]
            diff_view = np.array(diff_view, dtype=np.float32).T

            # 归一化
            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            X.append(std_view)
            Y.append(Label)

        print(f"成功使用 h5py (v7.3) 加载: {dataset[1]}")

    except OSError:
        # -------------------------------------------
        # 尝试 2: 使用 scipy 读取 (针对 MATLAB v5/v7)
        # -------------------------------------------
        print(f"h5py 加载失败，尝试使用 scipy.io 加载: {dataset[1]} ...")
        try:
            data = sio.loadmat(file_path)

            # 读取标签
            # scipy 读取通常不需要转置，或者维度已经是正确的 (N, 1)
            Label = data['Y']
            if Label.shape[0] < Label.shape[1]:
                Label = Label.T  # 确保是 (N, 1)
            Label = Label.reshape(Label.shape[0])

            # 读取视图数据
            # scipy 读取 Cell Array 通常是 Object Array
            # data['X'] 形状可能是 (1, V) 或 (V, 1)
            X_cells = data['X']
            if X_cells.shape[0] > X_cells.shape[1]:
                X_cells = X_cells.T  # 确保是 (1, V) 以便遍历

            for i in range(X_cells.shape[1]):
                diff_view = X_cells[0, i]  # 直接取内容，无需引用
                diff_view = np.array(diff_view, dtype=np.float32)

                # 检查是否需要转置 (N, D)
                # 通常 HandWritten 类数据行是样本，列是特征
                # 如果发现行数 != 标签数，可能需要转置
                if diff_view.shape[0] != Label.shape[0]:
                    diff_view = diff_view.T

                mm = MinMaxScaler()
                std_view = mm.fit_transform(diff_view)
                X.append(std_view)
                Y.append(Label)

            print(f"成功使用 scipy (v7) 加载: {dataset[1]}")

        except Exception as e:
            print(f"严重错误: 无法加载数据集 {file_path}")
            print("可能原因: 1. 文件损坏 2. 路径错误 3. 格式不支持")
            raise e

    # 处理缺失
    X, Y, missindex, X_com, Y_com, index_com, index_incom = Form_Incomplete_Data(missrate=missrate, X=X, Y=Y)

    return X, Y, missindex, X_com, Y_com, index_com, index_incom

