import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import os
import imageio
from pathlib import Path
from argparse import ArgumentParser
import pygame
# from matplotlib import pyplot as plt # 移除 matplotlib

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def get_xbox_command(joystick, max_cmd):
    # 注意：如果开启了 Pygame 显示窗口，这里 event.pump 也是必要的
    pygame.event.pump()
    dead_zone = 0.1
    if joystick is not None:
        lx = joystick.get_axis(0)
        ly = joystick.get_axis(1)
        rx = joystick.get_axis(3)
        if abs(lx) < dead_zone: lx = 0
        if abs(ly) < dead_zone: ly = 0
        if abs(rx) < dead_zone: rx = 0
        cmd_x = -ly * max_cmd[0]
        cmd_y = -lx * max_cmd[1]
        cmd_yaw = -rx * max_cmd[2]
        return np.array([cmd_x, cmd_y, cmd_yaw], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)

def draw_moe_weights(screen, weights, width, height):
    """使用 Pygame 绘制 MoE 权重"""
    screen.fill((255, 255, 255)) # 白底
    
    num_experts = len(weights)
    if num_experts == 0:
        return

    # 设置边距
    margin = 5
    bar_width = (width - 2 * margin) / num_experts
    max_bar_height = height - 2 * margin

    for i, w in enumerate(weights):
        # 限制 w 在 [0, 1] 之间用于显示
        w_clamped = max(0.0, min(1.0, w))
        bar_height = int(w_clamped * max_bar_height)
        
        # 计算矩形位置 (Pygame 坐标原点在左上角)
        # left, top, width, height
        x = margin + i * bar_width
        y = height - margin - bar_height # 从底部向上长
        
        # 绘制矩形 (蓝色)
        # 在 bar 之间留一点空隙 (width - 2)
        pygame.draw.rect(screen, (50, 100, 255), (x, y, bar_width - 2, bar_height))
        
    pygame.display.flip()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-video", action="store_true", help="Whether to save video of the simulation.")
    parser.add_argument("--visualize-moe-weights", action="store_true", help="Whether to visualize mixture of experts weights.")
    args = parser.parse_args()
    save_video = args.save_video
    visualize_moe_weights = args.visualize_moe_weights
    config_file = "go2.yaml"

    # Pygame 初始化
    pygame.init()
    
    use_joystick = False
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        use_joystick = True
        print(f"Detected Joystick: {joystick.get_name()}")
    else:
        print("No Joystick detected. Using default commands from config.")

    # 如果需要可视化权重，设置 Pygame 窗口
    screen = None
    win_width, win_height = 400, 200
    if visualize_moe_weights:
        # 创建一个独立的窗口用于显示权重
        screen = pygame.display.set_mode((win_width, win_height))
        pygame.display.set_caption("MoE Weights Visualization")

    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]

        cmd = np.array(config["cmd_init"], dtype=np.float32)

        idx_model2mj = idx_mj2model = list(range(num_actions))
        if 'mujoco_joint_names' in config and 'model_joint_names' in config:
            mujoco_joint_names = config["mujoco_joint_names"]
            model_joint_names = config["model_joint_names"]
            idx_model2mj = [model_joint_names.index(joint) for joint in mujoco_joint_names]
            idx_mj2model = [mujoco_joint_names.index(joint) for joint in model_joint_names]

    video_save_dir = str(Path(__file__).parent / "videos")
    os.makedirs(video_save_dir, exist_ok=True)

    model_name = os.path.basename(policy_path).split('.')[0]
    cmd_str = f"cmd_{cmd[0]}_{cmd[1]}_{cmd[2]}"
    video_filename = f"{model_name}_{cmd_str}.mp4"
    video_path = os.path.join(video_save_dir, video_filename)
    print(f"Video recording will be saved to: {video_path}")

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    last_action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    renderer = mujoco.Renderer(m, height=360, width=640)
    
    # load policy
    policy = torch.jit.load(policy_path)

    if save_video:
        video_fps = 50
        sim_fps = 1.0 / m.opt.timestep
        frame_skip = int(sim_fps / video_fps)
        if frame_skip < 1:
            frame_skip = 1
        writer = imageio.get_writer(video_path, fps=video_fps)
        print(f"Sim FPS: {sim_fps:.2f}, Video FPS: {video_fps}, Frame Skip: {frame_skip}, Save at: {video_path}")

    # 移除了 plt 初始化逻辑

    with mujoco.viewer.launch_passive(m, d) as viewer:

        # set viewer.camera to follow robot
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -30.0
        viewer.cam.azimuth = 0.0

        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            if use_joystick and counter % control_decimation == 0:
                cmd = get_xbox_command(joystick, config["max_cmd"])
                print(f"Cmd: Vx={cmd[0]:.2f}, Vy={cmd[1]:.2f}, Wz={cmd[2]:.2f}", end='\r')
            elif visualize_moe_weights and counter % control_decimation == 0:
                 # 如果没有手柄但开了可视化，也需要 pump 事件，防止窗口卡死
                 pygame.event.pump()

            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            
            mujoco.mj_step(m, d)

            if save_video and counter % frame_skip == 0:
                try:
                    renderer.update_scene(d, camera=viewer.cam)
                    frame = renderer.render()
                    writer.append_data(frame)
                except Exception as e:
                    print(f"Error rendering frame: {e}")

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                lin_vel = d.qvel[:3]
                ang_vel = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale

                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                lin_vel = lin_vel * lin_vel_scale
                ang_vel = ang_vel * ang_vel_scale

                obs[:3] = ang_vel
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj[idx_mj2model]
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj[idx_mj2model]
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action[idx_mj2model]
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                last_action = action
                result = policy(obs_tensor)
                
                # 处理 MoE 和 绘图
                if isinstance(result, tuple):
                    action, (weights, latent) = result  # moe
                    action = action.detach().numpy().squeeze()[idx_model2mj]
                    weights = weights.detach().numpy().squeeze()
                    
                    if visualize_moe_weights and screen is not None:
                        draw_moe_weights(screen, weights, win_width, win_height)
                        
                else:
                    action = result.detach().numpy().squeeze()[idx_model2mj]
                    
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            
            # 如果需要严格同步时间，可以解开下面的注释
            # time_until_next_step = m.opt.timestep - (time.time() - step_start)
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)

    if save_video:
        writer.close()
    
    # 退出时清理 Pygame
    pygame.quit()
    print(f"Video saved successfully to {video_path}")