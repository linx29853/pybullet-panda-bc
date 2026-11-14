
import torch
from torch import nn


def train_step(policy, observations, expert_actions, optimizer):
    """执行一次训练步骤。"""
    # 1. 前向传播：将 observations 送入 policy 得到预测动作。
    predicted_actions = policy(observations)
    # 2. 计算 MSE 损失。
    loss = torch.nn.functional.mse_loss(predicted_actions, expert_actions)
    # 3. 梯度清零、反向传播、参数更新。
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

