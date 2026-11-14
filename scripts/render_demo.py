"""Render the saved policy with PyBullet TinyRenderer; no desktop window."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import pybullet as p
import pybullet_data
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from collect_expert import HOME,observe,execute,MAX_DELTA
from policy import BCPolicy

def rollout(checkpoint,goal):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    net=BCPolicy(hidden_dim=ck['hidden_dim']);net.load_state_dict(ck['model']);net.eval()
    meta=ck['normalization'];mean=np.asarray(meta['mean'],np.float32);std=np.asarray(meta['std'],np.float32)
    p.connect(p.DIRECT)
    frames=[]
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath());p.setGravity(0,0,-9.81);p.setTimeStep(1/240)
        p.loadURDF('plane.urdf');robot=p.loadURDF('franka_panda/panda.urdf',useFixedBase=True)
        shape=p.createVisualShape(p.GEOM_SPHERE,radius=.015,rgbaColor=[1,.1,.1,1]);p.createMultiBody(baseMass=0,baseVisualShapeIndex=shape,basePosition=goal)
        for j,q in enumerate(HOME):p.resetJointState(robot,j,q)
        for j in [9,10]:
            p.resetJointState(robot,j,.04);p.setJointMotorControl2(robot,j,p.POSITION_CONTROL,targetPosition=.04,force=20)
        view=p.computeViewMatrixFromYawPitchRoll([.4,0,.3],1.25,45,-28,0,2)
        projection=p.computeProjectionMatrixFOV(50,480/360,.05,3)
        stable=0;done=False;distance=float(np.linalg.norm(observe(robot,goal)[7:10]-goal));success_step=None
        for step in range(201):
            if step%2==0:
                rgba=p.getCameraImage(480,360,view,projection,renderer=p.ER_TINY_RENDERER)[2]
                frame=Image.fromarray(np.asarray(rgba,dtype=np.uint8).reshape(360,480,4)[:,:,:3])
                frames.append((frame,distance,done,step))
            if step==200:break
            if not done:
                obs=observe(robot,goal)
                with torch.no_grad():raw=net(torch.from_numpy((obs-mean)/std).unsqueeze(0)).squeeze(0).numpy()
                execute(robot,obs,np.clip(raw*meta['action_scale'],-MAX_DELTA,MAX_DELTA),False)
                distance=float(np.linalg.norm(observe(robot,goal)[7:10]-goal));stable=stable+1 if distance<.015 else 0
                if stable>=5:done=True;success_step=step+1
    finally:p.disconnect()
    return frames,success_step

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'assets/rollout.gif');args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; choose a new --output')
    torch.set_num_threads(2)
    goal=np.random.default_rng(10000).uniform([.35,-.2,.25],[.65,.2,.55])
    right,success=rollout(ROOT/'runs/bc_100/best.pt',goal)
    frames=[]
    for entry in right[:max(12,(success or 200)//2+10)]:
        canvas=Image.new('RGB',(480,430),'#101c2c');canvas.paste(entry[0],(0,65));draw=ImageDraw.Draw(canvas)
        draw.text((12,10),'Panda reaching | BC: 100 demonstrations',fill='white')
        draw.text((12,30),f"t={entry[3]*.05:.1f}s  distance={entry[1]:.3f}m  {'SUCCESS (held)' if entry[2] else 'RUNNING'}",fill='#73e3ba' if entry[2] else '#ffca85')
        draw.rectangle((0,405,480,430),fill='#101c2c')
        draw.text((12,412),'PyBullet TinyRenderer | seed 10000, episode 0 | 1x',fill='white');frames.append(canvas)
    args.output.parent.mkdir(parents=True,exist_ok=True);frames[0].save(args.output,save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=False)
    frames[min(10,len(frames)-1)].save(args.output.with_suffix('.png'))
    print(f'{args.output}: {len(frames)} frames; expanded policy success at step {success}')
if __name__=='__main__':main()
