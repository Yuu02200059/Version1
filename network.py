# network.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim=512):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # [新增] 稳住分布
            nn.ReLU(),
            nn.Dropout(0.2),  # 必须保留 Dropout 防止死记硬背
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # [新增]
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim=512):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Projector(nn.Module):
    def __init__(self, feature_dim, semantic_dim):
        super(Projector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, semantic_dim)
        )

    def forward(self, x):
        h = self.net(x)
        return F.normalize(h, p=2, dim=1)


# === ITGG 融合模块 (注意力机制) ===
class ITG_FusionModule(nn.Module):
    def __init__(self, feature_dim):
        super(ITG_FusionModule, self).__init__()
        # 一个轻量级的打分网络
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, hs_list, mask):
        batch_size = mask.shape[0]
        device = mask.device
        scores_list = []
        valid_hs_list = []

        # 获取特征维度
        D = 0
        for h in hs_list:
            if h is not None:
                D = h.shape[1];
                break
        if D == 0: return None, None

        # 计算每个视图的分数
        for v, h in enumerate(hs_list):
            if h is None:
                dummy_h = torch.zeros(batch_size, D).to(device)
                dummy_score = torch.full((batch_size, 1), -1e9).to(device)
                valid_hs_list.append(dummy_h)
                scores_list.append(dummy_score)
            else:
                score = self.attention_net(h)
                scores_list.append(score)
                valid_hs_list.append(h)

        # 拼接与掩码处理
        scores = torch.cat(scores_list, dim=1)  # (B, V)
        stacked_hs = torch.stack(valid_hs_list, dim=1)  # (B, V, D)

        mask_bool = (mask > 0)
        scores = scores.masked_fill(~mask_bool, -1e9)  # 缺失视图分数归零

        # Softmax 归一化
        weights = F.softmax(scores, dim=1)  # (B, V)
        weights_expanded = weights.unsqueeze(2)

        # 加权融合
        S = torch.sum(stacked_hs * weights_expanded, dim=1)  # (B, D)

        return S, weights


class Network(nn.Module):
    def __init__(self, views, input_size, feature_dim, semantic_dim, hidden_dim=512):
        super(Network, self).__init__()
        self.views = views
        # 动态构建 Encoder/Decoder
        self.encoders = nn.ModuleList([Encoder(input_size[v], feature_dim, hidden_dim) for v in range(views)])
        self.decoders = nn.ModuleList([Decoder(input_size[v], feature_dim, hidden_dim) for v in range(views)])
        self.projectors = nn.ModuleList([Projector(feature_dim, semantic_dim) for v in range(views)])
        self.semantic_dim = semantic_dim

        self.fusion_module = ITG_FusionModule(semantic_dim)

    def forward(self, xs, mask=None):
        zs, hs, xrs = [], [], []
        for v in range(self.views):
            x = xs[v]
            if x is None or x.numel() == 0:
                zs.append(None);
                hs.append(None);
                xrs.append(None)
                continue

            z = self.encoders[v](x)
            h = self.projectors[v](z)
            xr = self.decoders[v](z)

            zs.append(z);
            hs.append(h);
            xrs.append(xr)

        S, weights = None, None
        if mask is not None:
            S, weights = self.fusion_module(hs, mask)

        # 返回 5 个值
        return zs, hs, xrs, S, weights