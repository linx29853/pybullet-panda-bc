import torch
from torch import nn


class BCPolicy(nn.Module):
    def __init__(self, obs_dim=13, action_dim=7, hidden_dim=128):
        super().__init__()
        # 隐藏层用 ReLU，输出层不加激活（动作可以为负）。
        # 输入已归一化；输出为缩放后的动作，约在 [-1, 1]。
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observations):
        # 将 observations 送入 self.net 并返回结果。
        return self.net(observations)


if __name__ == '__main__':
    model = BCPolicy()
    output = model(torch.zeros(4, 13))
    assert output.shape == (4, 7), output.shape
    assert torch.isfinite(output).all()
    print('前向接口通过：4 × 13 -> 4 × 7')
