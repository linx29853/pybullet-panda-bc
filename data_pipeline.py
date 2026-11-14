"""配套数据接口：严格按轨迹划分，仅用训练数据计算归一化统计。"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ACTION_SCALE = 0.04


def load_demonstrations(path):
    with np.load(path, allow_pickle=False) as data:
        obs = data['observations'].copy()
        actions = data['actions'].copy()
        episodes = data['episode_id'].copy()
        success = data['episode_success'].copy()
    n = len(obs)
    if obs.shape != (n, 13) or actions.shape != (n, 7):
        raise ValueError('预期 observations=[N,13]、actions=[N,7]')
    if episodes.shape != (n,) or success.shape != (n,):
        raise ValueError('episode_id / episode_success 形状不匹配')
    if not np.isfinite(obs).all() or not np.isfinite(actions).all():
        raise ValueError('数据包含 NaN/Inf')
    if np.max(np.abs(actions), initial=0) > ACTION_SCALE + 1e-6:
        raise ValueError('动作超出预期 ±0.04 rad；请核对数据动作定义')
    return obs[success].astype(np.float32), actions[success].astype(np.float32), episodes[success]


def prepare_data(path, seed=0, val_fraction=0.2, batch_size=64):
    obs, actions, ep = load_demonstrations(path)
    unique = np.unique(ep)
    if len(unique) < 2:
        raise ValueError('至少需要两条成功轨迹才能划分训练与验证集')
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = min(len(unique)-1, max(1, int(round(len(unique)*val_fraction))))
    val_ids, train_ids = unique[:n_val], unique[n_val:]
    val_mask = np.isin(ep, val_ids)
    train_mask = ~val_mask
    mean = obs[train_mask].mean(axis=0)
    std = obs[train_mask].std(axis=0)
    # 用下限避免常量特征或极小方差放大数值。
    std = np.maximum(std, 0.01).astype(np.float32)
    x = (obs - mean) / std
    y = actions / ACTION_SCALE
    train = TensorDataset(torch.from_numpy(x[train_mask]), torch.from_numpy(y[train_mask]))
    val = TensorDataset(torch.from_numpy(x[val_mask]), torch.from_numpy(y[val_mask]))
    generator = torch.Generator().manual_seed(seed)
    loaders = (DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator),
               DataLoader(val, batch_size=batch_size, shuffle=False))
    metadata = dict(mean=mean.tolist(), std=std.tolist(), action_scale=ACTION_SCALE,
                    train_episode_ids=train_ids.tolist(), val_episode_ids=val_ids.tolist(),
                    train_samples=len(train), val_samples=len(val),
                    data_path=str(Path(path).resolve()))
    return *loaders, metadata
