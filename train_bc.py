"""已接好的训练入口。核心实现在 policy.py 和 learning.py。"""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn

from data_pipeline import prepare_data
from policy import BCPolicy
from learning import train_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/expert100.npz'))
    parser.add_argument('--run-dir', type=Path, default=Path('runs/bc_new'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--check-data', action='store_true', help='仅验证数据接口，无需填写 TODO')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden_dim) < 1 or args.lr <= 0:
        parser.error('epochs、batch-size、hidden-dim 和 lr 必须为正')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    train_loader, val_loader, meta = prepare_data(args.data, args.seed, batch_size=args.batch_size)
    print(f"训练：{meta['train_samples']} 帧，轨迹 {meta['train_episode_ids']}")
    print(f"验证：{meta['val_samples']} 帧，轨迹 {meta['val_episode_ids']}")
    x, y = next(iter(train_loader))
    print(f'归一化输入 {tuple(x.shape)}；缩放后的动作标签 {tuple(y.shape)}')
    if args.check_data:
        return
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('当前 PyTorch 无可用 CUDA；这个小网络可以用 --device cpu')
    policy = BCPolicy(hidden_dim=args.hidden_dim).to(device)
    # 未填写时提供清楚提示，不创建空的训练目录。
    with torch.no_grad():
        pred = policy(x.to(device))
    if pred.shape != y.shape:
        raise ValueError(f'策略输出应为 {tuple(y.shape)}，实际 {tuple(pred.shape)}')
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.run_dir/'config.json').write_text(json.dumps(dict(config=config, data=meta), indent=2))
    best = float('inf')
    with (args.run_dir/'loss.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['epoch', 'train_scaled_mse', 'val_scaled_mse', 'val_rmse_rad'])
        for epoch in range(1, args.epochs+1):
            policy.train()
            total, count = 0., 0
            for x, y in train_loader:
                loss = train_step(policy, x.to(device), y.to(device), optimizer)
                if not isinstance(loss, (int, float)) or not math.isfinite(loss):
                    raise ValueError('train_step 必须返回有限的 Python float loss.item()')
                total += loss * len(x); count += len(x)
            policy.eval()
            val_total, val_count = 0., 0
            with torch.no_grad():
                for x, y in val_loader:
                    pred = policy(x.to(device))
                    if pred.shape != y.shape or not torch.isfinite(pred).all():
                        raise ValueError('验证时策略输出形状错误或包含 NaN/Inf')
                    val_total += nn.functional.mse_loss(pred, y.to(device)).item()*len(x)
                    val_count += len(x)
            train_mse, val_mse = total/count, val_total/val_count
            rmse = math.sqrt(val_mse)*meta['action_scale']
            writer.writerow([epoch, train_mse, val_mse, rmse]); handle.flush()
            checkpoint = dict(model=policy.state_dict(), hidden_dim=args.hidden_dim,
                              normalization=meta, epoch=epoch, val_scaled_mse=val_mse)
            if val_mse < best:
                best = val_mse
                torch.save(checkpoint, args.run_dir/'best.pt')
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(f'epoch {epoch:03d} train={train_mse:.6f} val={val_mse:.6f} RMSE={rmse:.5f} rad')
        torch.save(checkpoint, args.run_dir/'last.pt')
    print(f'已保存 {args.run_dir}/best.pt、last.pt、loss.csv')


if __name__ == '__main__':
    main()
