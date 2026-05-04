"""
Usage:
(robodiff)$ python eval_real_robot_carrot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_flexiv> --camera_serials <serial1> <serial2> ...


================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and exit program.
"""

import time
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform

OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', default='192.168.2.100', help="Flexiv robot's IP address e.g. 192.168.2.100")
@click.option('--camera_serials', '-cs', multiple=True, default=['135122075425'], help="RealSense serial numbers")
@click.option('--vis_camera_idx', default=0, type=int, help="Which camera index to visualize.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=30, type=float, help="Control frequency in Hz.")
def main(input, output, robot_ip, camera_serials, vis_camera_idx,
         steps_per_inference, max_duration, frequency):
    
    # 加载 checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # 策略类型适配
    action_offset = 0
    delta_action = False
    if 'diffusion' in cfg.name:
        policy: BaseImagePolicy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        device = torch.device('cuda')
        policy.eval().to(device)
        policy.num_inference_steps = 16
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    elif 'robomimic' in cfg.name:
        policy = workspace.model
        device = torch.device('cuda')
        policy.eval().to(device)
        steps_per_inference = 1
        action_offset = cfg.n_latency_steps
        delta_action = cfg.task.dataset.get('delta_action', False)
    elif 'ibc' in cfg.name:
        policy = workspace.model
        policy.pred_n_iter = 5
        policy.pred_n_samples = 4096
        device = torch.device('cuda')
        policy.eval().to(device)
        steps_per_inference = 1
        action_offset = 1
        delta_action = cfg.task.dataset.get('delta_action', False)
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    # 控制参数
    dt = 1 / frequency
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    from diffusion_policy.device.agent import EvalAgent
    # 初始化 EvalAgent
    agent = EvalAgent(robot_ip=robot_ip, camera_serials=list(camera_serials))
    # 可选：设置相机曝光等（根据实际需求）
    # for cam in agent.camera:
    #     cam.set_exposure(120, gain=0)
    #     cam.set_white_balance(5900)

    # 准备保存视频（简单保存为 mp4，每个 episode 一个文件）
    output_path = pathlib.Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    video_writer = None
    episode_idx = 0

    # 预热推理
    print("Warming up policy inference")
    obs_sample = agent.get_observation()  # list of (rgb, depth)
    rgb_sample = obs_sample[0][0]  # 取第一个相机
    tcp_sample = agent.get_tcp_pose()   # (7,)
    # 构造观测字典（与 shape_meta 匹配）
    obs_dict_np = {
        'camera_0': rgb_sample.transpose(2,0,1).astype(np.float32),
        'tcp': tcp_sample.astype(np.float32)
    }
    with torch.no_grad():
        policy.reset()
        obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
        result = policy.predict_action(obs_dict)
        action = result['action'][0].detach().cpu().numpy()
        assert action.shape[-1] == 8
        del result

    print('Ready! Starting policy control loop...')

    # ---------- 策略控制循环 ----------
    try:
        while True:
            policy.reset()
            start_delay = 1.0
            eval_t_start = time.time() + start_delay
            t_start = time.monotonic() + start_delay
            precise_wait(eval_t_start - 0.03)   # 等待启动时刻
            print(f"Episode {episode_idx} started.")

            # 准备视频写入
            video_path = output_path / f"episode_{episode_idx}_camera{vis_camera_idx}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_size = (obs_res[1], obs_res[0]) if obs_res else (1280, 720)
            video_writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, frame_size)

            iter_idx = 0
            term_area_start_timestamp = float('inf')
            prev_target_pose = None

            while True:
                t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                # 获取观测
                obs_list = agent.get_observation()          # list of (rgb, depth)
                rgb = obs_list[vis_camera_idx][0]           # (H, W, 3) BGR? 需转RGB
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # 转为RGB
                tcp_pose = agent.get_tcp_pose()             # (7,)

                # 构建观测字典
                obs_dict_np = {
                    'camera_0': rgb.transpose(2,0,1).astype(np.float32),
                    'tcp': tcp_pose.astype(np.float32)
                }

                # 推理
                with torch.no_grad():
                    s_time = time.time()
                    obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                    result = policy.predict_action(obs_dict)
                    action = result['action'][0].detach().cpu().numpy()   # (T, 8)
                    print(f'Inference latency: {time.time() - s_time:.3f}s')

                # 处理动作（delta 或 absolute）
                if delta_action:
                    # delta 模式：动作是位置增量 + 夹爪变化
                    assert len(action) == 1
                    if prev_target_pose is None:
                        prev_target_pose = tcp_pose.copy()
                    # 前2维为位置增量（假设只控制XY），其他保持
                    new_pose = prev_target_pose.copy()
                    new_pose[:2] += action[-1][:2]
                    prev_target_pose = new_pose
                    target_poses = np.expand_dims(new_pose, axis=0)
                    gripper_cmds = action[-1][7:8]   # 夹爪
                else:
                    # 绝对位姿模式：动作前7维为目标TCP，第8维为夹爪
                    target_poses = action[:, :7]      # (T,7)
                    gripper_cmds = action[:, 7]       # (T,)

                # 决定执行哪一步（取第一步或最后一步，这里取第一步并忽略多步，简化）
                # 若 steps_per_inference>1，可循环执行多步，但为保持稳定，只执行第一个动作并等待总时长
                target_pose = target_poses[0]
                gripper_cmd = gripper_cmds[0]

                # 夹爪命令映射：假设策略输出范围 [-1,1] 或 [0,1]，映射到 0~1000 mm
                # 根据训练时归一化方式调整，这里假设输出为 [0,1]
                gripper_width = np.clip(gripper_cmd, 0, 1) * 1000.0

                # 执行动作
                agent.set_tcp_pose(target_pose, rotation_rep='quaternion', blocking=False)
                agent.set_gripper_width(gripper_width, blocking=False)

                # 可视化与记录
                vis_img = rgb.copy()
                episode_time = time.monotonic() - t_start
                text = f'Episode: {episode_idx}, Time: {episode_time:.1f}s'
                cv2.putText(vis_img, text, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.imshow('Policy Control', vis_img[:,:,::-1])
                if video_writer is not None:
                    video_writer.write(cv2.resize(vis_img, frame_size))

                key = cv2.pollKey()
                if key == ord('s'):
                    print("Stopped by user.")
                    break

                # 终止条件：超时或到达终止区域
                terminate = False
                if episode_time > max_duration:
                    terminate = True
                    print('Terminated by timeout.')

                # 终止区域判断（示例，需根据实际任务定义）
                term_pose = np.array([0.34, 0.22, 0.045, 2.22, -2.22, -0.0004, 0.0])  # 位置+四元数
                curr_pos = tcp_pose[:3]
                term_pos = term_pose[:3]
                dist = np.linalg.norm(curr_pos - term_pos)
                if dist < 0.03:
                    curr_time = time.time()
                    if term_area_start_timestamp == float('inf'):
                        term_area_start_timestamp = curr_time
                    elif curr_time - term_area_start_timestamp > 0.5:
                        terminate = True
                        print('Terminated by reaching goal area.')
                else:
                    term_area_start_timestamp = float('inf')

                if terminate:
                    break

                # 等待至本控制周期结束（保证频率）
                precise_wait(t_cycle_end)
                iter_idx += steps_per_inference

            # 结束 episode，释放资源
            if video_writer is not None:
                video_writer.release()
            episode_idx += 1
            print(f"Episode {episode_idx-1} finished.\n")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        agent.stop()
        cv2.destroyAllWindows()
        print("Robot stopped, exit.")

if __name__ == '__main__':
    main()
