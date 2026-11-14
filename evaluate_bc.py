"""模型闭环评估：不调用专家，只借用与采集一致的观察/执行接口。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import pybullet as p
import pybullet_data
from collect_expert import HOME, JOINTS, MAX_DELTA, observe, execute
from policy import BCPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/bc_100/best.pt'))
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=10000, help='默认不同于专家采集的 seed')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('runs/eval_01.json'))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes 必须为正')
    if args.output.exists():
        parser.error('结果文件已存在，请更换 --output')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    policy = BCPolicy(hidden_dim=checkpoint['hidden_dim'])
    policy.load_state_dict(checkpoint['model']); policy.eval()
    torch.set_num_threads(2)
    meta = checkpoint['normalization']
    mean, std = np.asarray(meta['mean'], dtype=np.float32), np.asarray(meta['std'], dtype=np.float32)
    rng = np.random.default_rng(args.seed)
    reports = []
    p.connect(p.GUI if args.gui else p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81); p.setTimeStep(1/240)
        p.loadURDF('plane.urdf')
        robot = p.loadURDF('franka_panda/panda.urdf', useFixedBase=True)
        shape = p.createVisualShape(p.GEOM_SPHERE, radius=.015, rgbaColor=[1, .1, .1, 1])
        marker = p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape)
        p.resetDebugVisualizerCamera(1.3, 45, -30, [.4, 0, .3])
        for ep in range(args.episodes):
            for j, q in enumerate(HOME): p.resetJointState(robot, j, q)
            for j in [9, 10]:
                p.resetJointState(robot, j, .04)
                p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=.04, force=20)
            goal = rng.uniform([.35, -.2, .25], [.65, .2, .55])
            p.resetBasePositionAndOrientation(marker, goal, [0, 0, 0, 1])
            stable = 0
            for step in range(200):
                obs = observe(robot, goal)
                x = torch.from_numpy((obs-mean)/std).unsqueeze(0)
                with torch.no_grad(): scaled_action = policy(x).squeeze(0).numpy()
                if scaled_action.shape != (7,) or not np.isfinite(scaled_action).all():
                    raise ValueError('模型动作应为有限的 7 维向量')
                action = np.clip(scaled_action*meta['action_scale'], -MAX_DELTA, MAX_DELTA)
                execute(robot, obs, action, args.gui)
                nxt = observe(robot, goal)
                distance = float(np.linalg.norm(nxt[7:10]-goal))
                stable = stable+1 if distance < .015 else 0
                if stable >= 5: break
            reports.append(dict(episode=ep, success=stable >= 5, steps=step+1,
                                goal=goal.tolist(), final_distance_m=distance))
            print(f'episode {ep:03d}: success={stable>=5}, steps={step+1}, distance={distance:.4f} m')
    finally:
        p.disconnect()
    result = dict(checkpoint=str(args.checkpoint), seed=args.seed, episodes=args.episodes,
                  success_rate=sum(r['success'] for r in reports)/args.episodes, reports=reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"成功率 {result['success_rate']:.1%}；结果已保存到 {args.output}")


if __name__ == '__main__':
    main()
