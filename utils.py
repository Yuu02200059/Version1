# utils.py
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment  # 用于 ACC
import torch.nn.functional as F

# --- 1. 评估指标 (Stage 5) ---

def cluster_acc(y_true, y_pred):
    """
    使用匈牙利算法 (linear_sum_assignment) 计算聚类准确率 (ACC)
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # 使用 linear_sum_assignment (匈牙利算法) 找到最优匹配
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size


def evaluate_metrics(y_pred, y_true):
    """
    计算 ACC 和 NMI
    """
    # 确保是 numpy 数组
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    acc = cluster_acc(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='geometric')
    return acc, nmi


# --- 2. K-Means 操作 (Stage 2 & 4B) ---

@torch.no_grad()
def update_prototypes_by_kmeans(
        model,
        all_xs_list,
        all_mask_tensor,
        N_local_prototypes,
        device
):
    """
    执行 "Stage 2: 初始化局部原型" 或 "Stage 4B: 更新局部原型"。

    1. 冻结 E_v (model.eval())
    2. 将 *所有* 真实可见数据送入 E_v 得到最新的 H_all^v
    3. 在每个 H_all^v 上 *独立* 运行 K-Means
    4. 返回新的簇中心

    Args:
        model (nn.Module): 包含 E_v 的 MultiViewModel
        all_xs_list (list): 包含 *所有* 样本的 X 列表
        all_mask_tensor (tensor): 包含 *所有* 样本的 (N, V) 掩码
        N_local_prototypes (int): 局部原型数 (P_v)
        device: 'cuda' or 'cpu'

    Returns:
        torch.Tensor: 新的 (V, N_local, D) 原型张量
    """
    model.eval()  # 冻结编码器 E_v
    V = len(all_xs_list)
    feature_dim = model.encoders[0].encoder[-1].out_features
    new_prototypes = torch.zeros(V, N_local_prototypes, feature_dim).to(device)

    print("  (K-Means Update) 开始更新局部原型...")

    for v in range(V):
        # 1. 为视图 v 提取所有可见数据
        visible_indices = all_mask_tensor[:, v].nonzero(as_tuple=True)[0]

        # 必须从CPU加载（假设 all_xs_list 在 CPU 上）
        x_vis = all_xs_list[v][visible_indices].to(device)

        if x_vis.shape[0] == 0:
            print(f"  警告: 视图 {v} 在 K-Means 更新时没有可见数据。")
            # 保留旧的原型或随机原型
            new_prototypes[v] = model.proto_module.P[v]  # (假设)
            continue

        # 2. 获取潜在特征 H_all^v (分批处理防 OOM)
        H_v_list = []
        with torch.no_grad():
            for x_batch in torch.split(x_vis, 512):  # 512 批大小
                z_batch = model.encoders[v](x_batch)
                H_v_list.append(z_batch.cpu())

        H_v = torch.cat(H_v_list, dim=0).numpy()  # (N_vis, D)

        # 3. 运行 K-Means
        print(f"  (K-Means Update) 视图 {v}: 在 {H_v.shape[0]} 个样本上运行 K-Means...")
        kmeans = KMeans(
            n_clusters=N_local_prototypes,
            random_state=0,
            n_init=10  # 运行10次 K-Means (n_init=10) 确保稳定性
        )
        kmeans.fit(H_v)

        # 4. 将簇中心赋值给 P^v
        new_prototypes[v] = torch.tensor(
            kmeans.cluster_centers_,
            dtype=torch.float32
        ).to(device)

    print("  (K-Means Update) 局部原型更新完成。")
    return new_prototypes


# @torch.no_grad()
# def initialize_alignment_matrix(
#         proto_module,
#         N_global_clusters,
#         device
# ):
#     """
#     执行 "Stage 3: (可选) 初始化对齐矩阵 B^v"。
#     在 *局部原型* P^v 上再次运行 K-Means，
#     并构建一个稀疏的 B^v 矩阵。
#
#     Args:
#         proto_module (nn.Module): 已被 (Stage 2) 初始化的原型模块
#         N_global_clusters (int): 全局簇 K
#
#     Returns:
#         torch.Tensor: 新的 (V, P_v, K) 对齐矩阵 B
#     """
#     # (V, P_v, D)
#     P_all = proto_module.P.cpu().numpy()
#     V, P_v, K = P_all.shape[0], P_all.shape[1], N_global_clusters
#
#     new_B = torch.zeros(V, P_v, K).to(device)
#
#     print("  (Init B^v) 开始初始化对齐矩阵...")
#
#     for v in range(V):
#         P_v_data = P_all[v]  # (P_v, D)
#
#         # 在 P_v (例如100个) 原型上运行 K-Means (K=15)
#         kmeans = KMeans(
#             n_clusters=N_global_clusters,
#             random_state=0,
#             n_init=10
#         )
#         kmeans.fit(P_v_data)
#
#         # labels_ 数组的索引 i 对应局部原型 P_v[i]
#         # labels_ 数组的值 k 对应它被分配到的全局簇 K
#         labels = kmeans.labels_  # (P_v,)
#
#         # 构建稀疏矩阵 B^v:
#         # 若 P_v[i] 属于 K[k]，则 B^v[i, k] = 1
#         for i in range(P_v):
#             k = labels[i]
#             new_B[v, i, k] = 1.0
#
#     print("  (Init B^v) 对齐矩阵初始化完成。")
#     return new_B

@torch.no_grad()
def initialize_alignment_matrix(
        proto_module,
        N_global_clusters,
        device,
        high_logit=10.0, # [新] 正确类的 logit
        low_logit=-10.0  # [新] 错误类的 logit
):
    """
    [!!! 已修复 !!!]
    执行 "Stage 3: 初始化对齐矩阵 B^v"。
    在局部原型 P^v 上运行 K-Means，
    并构建一个 *Logits* 矩阵 B^v。
    """
    # (V, P_v, D)
    P_all = proto_module.P.cpu().numpy()
    V, P_v, K = P_all.shape[0], P_all.shape[1], N_global_clusters

    # [新] 初始化为 -10.0 (低 logit)
    new_B = torch.full((V, P_v, K), low_logit).to(device)

    print("  (Init B^v) 开始初始化对齐矩阵...")

    for v in range(V):
        P_v_data = P_all[v]  # (P_v, D)

        # 在 P_v (例如100个) 原型上运行 K-Means (K=15)
        kmeans = KMeans(
            n_clusters=N_global_clusters,
            random_state=0,
            n_init=10
        )
        kmeans.fit(P_v_data)

        # labels_ 数组的索引 i 对应局部原型 P_v[i]
        # labels_ 数组的值 k 对应它被分配到的全局簇 K
        labels = kmeans.labels_  # (P_v,)

        # [新] 构建 Logits 矩阵 B^v:
        # 若 P_v[i] 属于 K[k]，则 B^v[i, k] = 10.0
        for i in range(P_v):
            k = labels[i]
            new_B[v, i, k] = high_logit # [新]

    print("  (Init B^v) 对齐矩阵初始化完成。")
    return new_B # [新] 返回的是 Logits 矩阵
# utils.py

# ... (你原来的 cluster_acc, evaluate_metrics, update_prototypes_by_kmeans, initialize_alignment_matrix 函数保持不变) ...


# === [新] 辅助函数 1: 获取所有 Z 和 H 特征 ===
# 作用：获取所有 N 个样本在所有 V 个视图上的 Z (结构) 和 H (语义) 特征
@torch.no_grad()
def get_all_features(model, all_xs_list, all_mask_tensor, device, batch_size, get_z=True, get_h=True):
    """
    Args:
        model (Network): 已经包含 Encoder 和 Projector 的模型
        get_z (bool): 是否需要 Z^v 特征 (来自 Encoder)
        get_h (bool): 是否需要 H^v 特征 (来自 Projector)
    Returns:
        dict: {"zs": zs_full_list, "hs": hs_full_list}
    """
    V = len(all_xs_list)
    N = all_xs_list[0].shape[0]

    # 动态获取 Z 和 H 的维度
    D_z = model.encoders[0].encoder[-1].out_features
    # H 的维度来自 Projector 的最后一层
    D_h = model.projectors[0][-1].out_features

    model.eval()  # 确保在评估模式

    zs_full_list = [torch.zeros(N, D_z).to(device) for _ in range(V)] if get_z else None
    hs_full_list = [torch.zeros(N, D_h).to(device) for _ in range(V)] if get_h else None

    # 分视图处理，避免 OOM
    for v in range(V):
        vis_indices_v = all_mask_tensor[:, v].nonzero(as_tuple=True)[0]
        x_vis_v = all_xs_list[v][vis_indices_v].to(device)

        z_vis_v_list = []
        h_vis_v_list = []

        # 分批处理
        for x_batch in torch.split(x_vis_v, batch_size):
            # 1. Z^v (来自 Encoder)
            z_batch = model.encoders[v](x_batch)

            if get_z:
                z_vis_v_list.append(z_batch)

            if get_h:
                # 2. H^v (来自 Projector)
                h_batch_unnorm = model.projectors[v](z_batch)
                h_batch = F.normalize(h_batch_unnorm, p=2, dim=1)
                h_vis_v_list.append(h_batch)

        # 将分批结果放回 (N, D) 的完整张量中
        if get_z:
            zs_full_list[v][vis_indices_v] = torch.cat(z_vis_v_list, dim=0)
        if get_h:
            hs_full_list[v][vis_indices_v] = torch.cat(h_vis_v_list, dim=0)

    return {"zs": zs_full_list, "hs": hs_full_list}


# === [新] 辅助函数 2: PyTorch 版特征融合 ===
# 作用：将 H^v 列表融合成一个 Fused H，用于 K-Means
@torch.no_grad()
def get_fea_com_torch(hs_full_list, mask_tensor):
    """
    Args:
        hs_full_list (list): V 个 (N, D_h) 的 H^v 特征张量
        mask_tensor (torch.Tensor): (N, V) 的掩码
    Returns:
        fused_H (torch.Tensor): (N, D_h) 的融合后特征
    """
    V = len(hs_full_list)
    N, D = hs_full_list[0].shape
    device = hs_full_list[0].device

    # (N, V, 1) * (N, V, D) -> (N, V, D)
    mask_expanded = mask_tensor.unsqueeze(2).to(device)  # (N, V, 1)
    hs_stacked = torch.stack(hs_full_list, dim=1)  # (N, V, D)

    # 掩码求和 (N, D)
    masked_hs_sum = torch.sum(hs_stacked * mask_expanded, dim=1)

    # 计算每个样本的视图数 (N, 1)
    view_counts = torch.sum(mask_tensor, dim=1, keepdim=True).to(device)  # (N, 1)

    # (N, D) / (N, 1) -> 得到平均融合特征
    fused_H = masked_hs_sum / (view_counts + 1e-10)
    return fused_H


# === [新] 步骤 C: 更新全局原型 C (在 H^v 上 K-Means) ===
# 作用：这是我们的新 (步骤 C)，用于更新“老师” C_global
@torch.no_grad()
def update_global_prototypes(
        model,
        all_xs_list,
        all_mask_tensor,
        K_global,
        estimator_global,
        device,
        batch_size
):
    """
    1. 获取所有 H^v (语义) 特征
    2. 融合它们得到 Fused H
    3. 在 Fused H 上运行 K-Means
    4. 返回新的全局簇中心 C
    """
    print("  (步骤 C) 更新全局原型 C (K-Means on Fused H) ...")
    model.eval()  # 确保模型在评估模式

    # 1. 获取所有 H^v 特征
    all_hs_list_tensor = get_all_features(
        model, all_xs_list, all_mask_tensor, device, batch_size,
        get_z=False, get_h=True  # 我们只关心 H^v
    )["hs"]

    # 2. 融合 H^v 得到 Fused H
    fused_H_tensor = get_fea_com_torch(all_hs_list_tensor, all_mask_tensor)
    fused_H_np = fused_H_tensor.cpu().numpy()

    # 3. 在融合后的 H 特征上运行 K-Means
    estimator_global.fit(fused_H_np)
    centroids_C = estimator_global.cluster_centers_

    print("  (步骤 C) 全局原型 C 更新完成。")

    # 4. 返回新原型
    return torch.tensor(centroids_C).float().to(device)