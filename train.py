import os
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import random as rd
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import pandas as pd
from datetime import datetime
import csv
# 引用项目模块
from load_data import load_data, ALL_data
from datasets import TrainDataset_All
from network import Network
# 引入新的损失函数
from loss import reconstruction_loss, structural_consistency_loss, global_guidance_loss, diversity_loss
from configs import get_config
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from logger import TrainerLogger
logger = TrainerLogger(log_dir="logs_scene15", dataset="Scene15")
def visualize_tsne(features, labels, epoch, save_path="tsne_results"):
    """
    画 t-SNE 图
    features: (N, D) 也就是 S
    labels: (N,) 真实标签
    """
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("  > 正在生成 t-SNE 可视化图...")

    # 1. 降维: 64维 -> 2维
    # init='pca' 可以让结果更稳定
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    X_embedded = tsne.fit_transform(features)

    # 2. 绘图
    plt.figure(figsize=(10, 8))
    # 使用不同颜色标记不同类别 (c=labels)
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='tab20', s=10, alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f't-SNE Visualization of S (Epoch {epoch})')

    # 3. 保存
    filename = f"{save_path}/tsne_epoch_{epoch}.png"
    plt.savefig(filename)
    plt.close()
    print(f"  > t-SNE 图已保存至: {filename}")

def create_eval_csv(dataset, missrate, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset}_miss{missrate}_{timestamp}.csv"
    filepath = os.path.join(save_dir, filename)

    # 初始化表头
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "ACC", "NMI"])  # 表头

    return filepath


def append_eval_result(csv_path, epoch, acc, nmi):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{acc:.4f}", f"{nmi:.4f}"])
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    rd.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size


def check_feature_stats(name, tensor):
    if tensor is None: return
    with torch.no_grad():
        t = tensor.float()
        mean = t.mean().item()
        std = t.std().item()
        min_v = t.min().item()
        max_v = t.max().item()
        dead = (torch.abs(t) < 1e-6).float().mean().item() * 100
        print(
            f"      |-> {name:15s}: Mean={mean:6.4f} | Std={std:6.4f} | Range=[{min_v:6.3f}, {max_v:6.3f}] | Dead={dead:4.1f}%")
def get_warmup_and_step_scheduler(optimizer, total_epochs, warmup_epochs=3, step_size=40, gamma=0.5):
    def lr_lambda(epoch):
        # warmup
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        # step decay
        return gamma ** ((epoch - warmup_epochs) // step_size)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler



def main(args):
    setup_seed(args.seed)

    # 加载配置
    config = get_config(args.dataset_name)

    args.feature_dim = config['feature_dim']
    args.semantic_dim = config['semantic_dim']
    args.hidden_dim = config['hidden_dim']
    args.lambda_rec = config['lambda_rec']
    args.lambda_struct = config['lambda_struct']  # 这里其实控制的是 Struct Loss
    args.lambda_global = config['lambda_global']
    args.lambda_div = config['lambda_div']
    args.lr = config['lr']
    args.batch_size = config['batch_size']
    args.epochs = config['epochs']

    print("=" * 80)
    print(f"启动 [结构一致性版] 训练 | 数据集: {args.dataset_name}")
    print(f"网络: Hidden={args.hidden_dim} -> Feature={args.feature_dim}")
    print(f"权重: Rec(自重构)={args.lambda_rec} | Struct(结构)={args.lambda_struct} | Global={args.lambda_global} | Div={args.lambda_div} ")
    print(f"LR: {args.lr}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_config = ALL_data[args.dataset_name]
    X, Y, missindex, _, _, _, _ = load_data(dataset_config, args.missrate)
    dataset = TrainDataset_All(X, Y, missindex)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = Network(
        views=dataset_config['V'],
        input_size=dataset_config['n_input'],
        feature_dim=args.feature_dim,
        semantic_dim=args.semantic_dim,
        hidden_dim=args.hidden_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0.0
    best_nmi = 0.0  # [新增] 跟踪最佳 NMI
    start_time = time.time()  # [新增] 记录开始时间
    csv_path = create_eval_csv(args.dataset_name, args.missrate)
    for epoch in range(args.epochs):
        model.train()
        total_loss_accum = 0

        verbose_epoch = (epoch < 5) or ((epoch + 1) % 10 == 0)
        if verbose_epoch: print(f"\n--- Epoch {epoch + 1} ---")

        for batch_idx, (xs, _, mask) in enumerate(loader):
            xs = [x.to(device) for x in xs]
            mask = mask.to(device)

            # 前向传播 (5个返回值)
            zs, hs, xrs, S, weights = model(xs, mask)

            loss_verbose = (verbose_epoch and batch_idx == 0)

            # 1. 自重构损失 (恢复使用)
            l_rec = reconstruction_loss(xs, xrs, mask, verbose=loss_verbose)

            # 2. [核心] 结构一致性损失 (替代原来的 Contrastive Loss)
            l_struct = structural_consistency_loss(S, hs, mask, verbose=loss_verbose)

            # 3. 全局引导损失
            l_global = global_guidance_loss(S, hs, mask, temperature=args.tau, verbose=loss_verbose)
            l_div = diversity_loss(S)
            loss = args.lambda_rec * l_rec + args.lambda_struct * l_struct + args.lambda_global * l_global + args.lambda_div * l_div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_accum += loss.item()
            if loss_verbose:
                print(
                    f"  [Loss] Total: {loss.item():.4f} | Rec: {l_rec.item():.4f} | Struct: {l_struct.item():.4f} | Global: {l_global.item():.4f} | div: {l_div.item():.4f}")
                avg_w = weights.mean(dim=0).detach().cpu().numpy()
                w_str = ", ".join([f"{w:.3f}" for w in avg_w])
                print(f"  [Weights]: [{w_str}]")
                check_feature_stats("S (Fused)", S)

        print(f"  > Epoch {epoch + 1} Avg Loss: {total_loss_accum / len(loader):.4f}", end="\r")

        if (epoch + 1) % 10 == 0:
            print("")
            acc, nmi = evaluate_final(model, X, missindex, Y[0], device, args, epoch=epoch+1)
            append_eval_result(csv_path, epoch + 1, acc, nmi)
            if acc > best_acc:
                best_acc = acc
                best_nmi = nmi  # [关键] ACC 更新时，同时更新 NMI
                print(f"  >>> New Best ACC: {best_acc * 100:.2f}% (Saved!)")
            deep_diagnosis(model, xs, mask, zs, hs, S, weights)
    print(f"\n训练结束，最佳 ACC: {best_acc * 100:.2f}%")
    logger.save()

def calculate_entropy(weights):
    """计算注意力权重的熵 (越小越好，越大说明越平均)"""
    # weights: (B, V)
    # epsilon 防止 log(0)
    entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1).mean()
    return entropy.item()


def calculate_matrix_correlation(h1, h2):
    """计算两个特征矩阵对应的'相似度矩阵'之间的皮尔逊相关系数"""
    # h1, h2: (B, D)
    # 1. 计算相似度矩阵 (B, B)
    h1 = F.normalize(h1, p=2, dim=1)
    h2 = F.normalize(h2, p=2, dim=1)
    sim1 = torch.matmul(h1, h1.t())
    sim2 = torch.matmul(h2, h2.t())

    # 2. 展平为向量 (B*B)
    vec1 = sim1.view(-1)
    vec2 = sim2.view(-1)

    # 3. 归一化去均值
    vec1 = vec1 - vec1.mean()
    vec2 = vec2 - vec2.mean()

    # 4. 计算相关系数 Cosine
    # Pearson Correlation = Cosine Similarity of centered vectors
    denom = (vec1.norm() * vec2.norm()) + 1e-8
    corr = (vec1 * vec2).sum() / denom
    return corr.item()


def estimate_effective_rank(features):
    """估计特征的有效秩 (看看64维里到底用了几维)"""
    # features: (B, D)
    # 奇异值分解
    try:
        _, S, _ = torch.svd(features)
        # 计算多少个奇异值占据了 99% 的能量
        total_energy = torch.sum(S)
        cum_energy = torch.cumsum(S, dim=0)
        effective_k = torch.searchsorted(cum_energy, total_energy * 0.99).item() + 1
        return effective_k, S[0].item()  # 返回有效维数和最大奇异值
    except:
        return 0, 0


def deep_diagnosis(model, xs, mask, zs, hs, S, weights):
    """
    综合诊断函数，打印所有深层指标
    """
    print("\n[Deep Diagnosis Report]")

    # 1. 注意力健康度
    avg_w = weights.mean(dim=0).detach().cpu().numpy()
    entropy = calculate_entropy(weights)
    print(f"  1. [Attention] Entropy: {entropy:.4f} (Max for 3 views is 1.09)")
    print(f"     Weights Dist: {[f'{w:.3f}' for w in avg_w]}")

    # 2. 结构对齐度 (Scene-15 成功的关键)
    # 选取前两个存在的视图计算相关性
    valid_views = []
    for v, h in enumerate(hs):
        if h is not None: valid_views.append(v)

    if len(valid_views) >= 2:
        v1, v2 = valid_views[0], valid_views[1]
        corr_v1_v2 = calculate_matrix_correlation(hs[v1], hs[v2])
        corr_v1_S = calculate_matrix_correlation(hs[v1], S)
        print(f"  2. [Structure] Correlation V{v1}-V{v2}: {corr_v1_S:.4f}")
        print(f"     Correlation V{v1}-S: {corr_v1_S:.4f}")
        if corr_v1_v2 < 0.1:
            print("     ⚠️ 警告: 视图间结构未对齐！结构损失可能权重过低。")

    # 3. 特征维度健康度 (S)
    eff_rank, max_sing = estimate_effective_rank(S)
    feat_std = S.std().item()
    print(f"  3. [Feature S] Std: {feat_std:.4f} | Effective Rank: {eff_rank}/64")
    if feat_std < 0.01:
        print("     ⚠️ 警告: 特征发生坍塌 (Collapse)！所有输出都一样了。")
    if eff_rank < 5:
        print("     ⚠️ 警告: 维度坍塌！特征主要集中在极少数维度。")

    print("-" * 40)
def evaluate_final(model, X_all, mask_all, Y_true, device, args,epoch=0):
    model.eval()
    N = X_all[0].shape[0]
    final_S = torch.zeros(N, args.semantic_dim).to(device)
    mask_tensor = torch.tensor(mask_all).to(device)
    batch_size = args.batch_size

    with torch.no_grad():
        num_batches = int(np.ceil(N / batch_size))
        for i in range(num_batches):
            start = i * batch_size;
            end = min((i + 1) * batch_size, N)
            idx = np.arange(start, end)
            xs_batch = [torch.tensor(X_all[v][idx], dtype=torch.float32).to(device) for v in range(len(X_all))]
            mask_batch = mask_tensor[idx]

            _, _, _, S_batch, _ = model(xs_batch, mask_batch)
            if S_batch is not None: final_S[start:end] = S_batch

    feats = final_S.cpu().numpy()
    # if epoch == args.epochs or epoch % 50 == 0:
    #     visualize_tsne(feats, Y_true, epoch)
    kmeans = KMeans(n_clusters=len(np.unique(Y_true)), n_init=20, random_state=args.seed)
    y_pred = kmeans.fit_predict(feats)
    acc = cluster_acc(Y_true, y_pred)
    nmi = nmi_score(Y_true, y_pred)
    print(f"  > [Eval] ACC: {acc * 100:.2f}%, NMI: {nmi * 100:.2f}%")
    return acc, nmi


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Caltech101_20')
    parser.add_argument('--missrate', type=float, default=0.1)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)

    # 占位符
    parser.add_argument('--feature_dim', type=int, default=0)
    parser.add_argument('--semantic_dim', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0)
    parser.add_argument('--lambda_rec', type=float, default=0)
    parser.add_argument('--lambda_struct', type=float, default=0)
    parser.add_argument('--lambda_global', type=float, default=0)
    parser.add_argument('--lambda_div', type=float, default=0)

    args = parser.parse_args()
    main(args)