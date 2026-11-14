"""Step 1: scripted Panda expert -> state/action demonstrations (no neural net)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

HOME = [0, -0.5, 0, -2.0, 0, 1.5, 0.8]
JOINTS = list(range(7))
EE = 11  # panda_grasptarget, fixed link at the gripper center
SUBSTEPS = 12
MAX_DELTA = 0.04


def observe(robot, goal):
    q = np.array([p.getJointState(robot, j)[0] for j in JOINTS])
    ee = np.array(p.getLinkState(robot, EE, computeForwardKinematics=True)[4])
    return np.concatenate([q, ee, goal]).astype(np.float32)


def expert(robot, obs):
    # Re-query from the current robot state; also usable later for DAgger labels.
    solution = p.calculateInverseKinematics(
        robot, EE, obs[10:13].tolist(), maxNumIterations=100,
        residualThreshold=1e-5,
    )
    return np.clip(np.asarray(solution[:7]) - obs[:7], -MAX_DELTA, MAX_DELTA).astype(np.float32)


def execute(robot, obs, action, gui):
    # action is a joint-position increment in radians, NOT joint velocity.
    target = obs[:7] + action
    p.setJointMotorControlArray(robot, JOINTS, p.POSITION_CONTROL,
                               targetPositions=target.tolist(), forces=[87]*7)
    for _ in range(SUBSTEPS):
        p.stepSimulation()
        if gui:
            time.sleep(1 / 240)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('data/expert.npz'))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    if args.output.exists():
        parser.error(f'{args.output} already exists; choose a new --output')
    rng = np.random.default_rng(args.seed)
    p.connect(p.GUI if args.gui else p.DIRECT)
    rows, actions, next_rows, episode_ids, successes, ends = [], [], [], [], [], []
    reports = []
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        p.loadURDF('plane.urdf')
        robot = p.loadURDF('franka_panda/panda.urdf', useFixedBase=True)
        marker_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.015,
                                          rgbaColor=[1, 0.1, 0.1, 1])
        marker = p.createMultiBody(baseMass=0, baseVisualShapeIndex=marker_shape)
        p.resetDebugVisualizerCamera(1.3, 45, -30, [0.4, 0, 0.3])
        for ep in range(args.episodes):
            for j, q in enumerate(HOME):
                p.resetJointState(robot, j, q)
            for j in [9, 10]:
                p.resetJointState(robot, j, 0.04)
                p.setJointMotorControl2(robot, j, p.POSITION_CONTROL,
                                       targetPosition=0.04, force=20)
            goal = rng.uniform([0.35, -0.2, 0.25], [0.65, 0.2, 0.55])
            p.resetBasePositionAndOrientation(marker, goal, [0, 0, 0, 1])
            transitions = []
            stable = 0
            for step in range(200):
                obs = observe(robot, goal)
                action = expert(robot, obs)
                execute(robot, obs, action, args.gui)
                nxt = observe(robot, goal)
                distance = float(np.linalg.norm(nxt[7:10] - goal))
                stable = stable + 1 if distance < 0.015 else 0
                transitions.append((obs, action, nxt))
                if stable >= 5:
                    break
            success = stable >= 5
            for i, (obs, action, nxt) in enumerate(transitions):
                rows.append(obs); actions.append(action); next_rows.append(nxt)
                episode_ids.append(ep); successes.append(success)
                ends.append(i == len(transitions) - 1)
            reports.append(dict(episode=ep, goal=goal.tolist(), success=success,
                                steps=len(transitions), final_distance_m=distance))
            print(f'episode {ep:03d}: success={success}, steps={len(transitions)}, distance={distance:.4f} m')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, observations=np.asarray(rows),
                            actions=np.asarray(actions), next_observations=np.asarray(next_rows),
                            episode_id=np.asarray(episode_ids, dtype=np.int32),
                            episode_success=np.asarray(successes, dtype=bool),
                            episode_end=np.asarray(ends, dtype=bool))
        summary = dict(seed=args.seed, episodes=args.episodes,
                       successful_episodes=sum(r['success'] for r in reports),
                       transitions=len(rows), observation_columns=['q0:q7', 'ee_xyz', 'goal_xyz'],
                       action='7 joint position increments, radians',
                       physics_hz=240, action_hz=240/SUBSTEPS,
                       max_delta_rad=MAX_DELTA, reports=reports)
        args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2))
        print(f'Saved {len(rows)} transitions to {args.output}')
    finally:
        p.disconnect()


if __name__ == '__main__':
    main()
