import torch
import torch.nn.functional as F


# 1. 重构损失 (不变)
def reconstruction_loss(xs, xrs, mask, verbose=False):
    loss = 0.0
    valid_views = 0
    V = len(xs)
    loss_details = []
    for v in range(V):
        mask_v = mask[:, v]
        idx = mask_v.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            loss_details.append(0.0)
            continue
        mse = F.mse_loss(xrs[v][idx], xs[v][idx], reduction='mean')
        loss += mse
        valid_views += 1
        loss_details.append(mse.item())

    if verbose:
        print(f"    [L_rec] Views: {[f'{l:.4f}' for l in loss_details]}")
    return loss / (valid_views + 1e-8)


# 2. 对比损失 (不变)
def contrastive_loss(hs_list, mask, temperature=0.5, verbose=False):
    device = mask.device
    V = len(hs_list)
    total_loss = 0.0
    pairs_count = 0

    for i in range(V):
        for j in range(i + 1, V):
            common_mask = (mask[:, i] > 0) & (mask[:, j] > 0)
            idx = common_mask.nonzero(as_tuple=True)[0]
            if len(idx) < 2: continue

            z_i = hs_list[i][idx]
            z_j = hs_list[j][idx]
            sim_matrix = torch.matmul(z_i, z_j.T) / temperature
            labels = torch.arange(z_i.shape[0]).to(device)
            loss = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
            total_loss += loss
            pairs_count += 1

    if pairs_count == 0: return torch.tensor(0.0, device=device)
    return total_loss / pairs_count


# 3. [新增] 全局引导损失 (ITGG Loss)
def global_guidance_loss(S, hs_list, mask, temperature=0.5, verbose=False):
    """
    最大化 S 与 H^v 的互信息 (通过 InfoNCE 实现)
    """
    total_loss = 0.0
    valid_views_count = 0
    device = mask.device

    # S 必须归一化
    S = F.normalize(S, p=2, dim=1)

    debug_loss = []

    for v, h in enumerate(hs_list):
        if h is None: continue

        # 只计算该视图存在的样本
        mask_v = mask[:, v]
        idx = mask_v.nonzero(as_tuple=True)[0]
        if len(idx) == 0: continue

        s_sub = S[idx]  # 全局
        h_sub = h[idx]  # 局部

        # InfoNCE: 正样本是对角线 (同一个样本的 S 和 H)
        logits = torch.matmul(s_sub, h_sub.T) / temperature
        labels = torch.arange(s_sub.shape[0]).to(device)

        l_v = F.cross_entropy(logits, labels)
        total_loss += l_v
        valid_views_count += 1
        debug_loss.append(l_v.item())

    if verbose and debug_loss:
        # 打印一下 S 和各个视图的对齐程度
        print(f"    [L_global] Align S with Views: {[f'{l:.2f}' for l in debug_loss]}")

    if valid_views_count == 0:
        return torch.tensor(0.0, device=device)

    return total_loss / valid_views_count


# 4. [新增] 结构一致性损失 (Structural Consistency)
def structural_consistency_loss(S, hs_list, mask, verbose=False):
    """
    不再强迫特征值对齐，而是强迫"样本间的相似度关系"对齐。
    解决 Scene-15 异质视图无法点对点匹配的问题。
    """
    device = S.device
    total_loss = 0.0
    valid_views = 0

    # 1. 计算全局 S 的关系矩阵 P (作为 Target)
    S = F.normalize(S, p=2, dim=1)
    S_sim = torch.matmul(S, S.t())
    # 使用 Softmax 变成概率分布，温度系数设为 1.0
    P_target = F.softmax(S_sim, dim=1).detach()  # Detach! 我们希望 H 向 S 学习

    # 2. 遍历每个视图
    loss_details = []
    for v, h in enumerate(hs_list):
        if h is None: continue

        # 只对存在的样本计算
        mask_v = mask[:, v]
        if mask_v.sum() == 0: continue

        h = F.normalize(h, p=2, dim=1)
        H_sim = torch.matmul(h, h.t())
        Q_pred = F.softmax(H_sim, dim=1)

        # KL 散度: KL(P || Q)
        # 意思：S 认为这两个样本很像，H 也必须认为它们很像
        kl_div = F.kl_div(Q_pred.log(), P_target, reduction='none')

        # Masking: 只计算有效样本的行
        kl_per_sample = kl_div.sum(dim=1)
        valid_loss = (kl_per_sample * mask_v).sum() / (mask_v.sum() + 1e-8)

        total_loss += valid_loss
        valid_views += 1
        loss_details.append(valid_loss.item())

    if verbose and loss_details:
        print(f"    [L_struct] KL Divs: {[f'{l:.4f}' for l in loss_details]}")

    if valid_views == 0:
        return torch.tensor(0.0).to(device)

    return total_loss / valid_views
# loss.py (追加内容)

def diversity_loss(H, gamma=1.0):
    """
    基于方差和协方差的多样性正则化 (VICReg style)
    对应上传图片中的 Eq. 6, 7, 8
    """
    # H: (Batch, Dim)
    batch_size = H.shape[0]
    dim = H.shape[1]
    device = H.device

    # 1. 计算标准差 (Standard Deviation)
    # epsilon 防止 sqrt(0)
    std_H = torch.sqrt(H.var(dim=0) + 1e-4)  # (Dim,)

    # 2. 方差损失 (Variance Loss) - Eq. 6
    # 强迫每个维度的标准差至少为 gamma
    # relu(gamma - std)
    std_loss = torch.mean(F.relu(gamma - std_H))

    # 3. 计算协方差矩阵 (Covariance Matrix) - Eq. 7
    # 先去均值
    H_centered = H - H.mean(dim=0)
    # Cov = (H.T @ H) / (N - 1)
    cov_H = (H_centered.t() @ H_centered) / (batch_size - 1)

    # 4. 协方差损失 (Covariance Loss) - Eq. 8
    # 惩罚非对角线元素 (off-diagonal)
    # 也就是把对角线变成0，然后算平方和
    off_diag_mask = ~torch.eye(dim, dtype=torch.bool, device=device)
    cov_loss = (cov_H[off_diag_mask] ** 2).mean()

    # 总损失 = 方差项 + 协方差项
    # 通常这两个项的权重是 1:1，或者协方差项权重大一点
    return std_loss + cov_loss