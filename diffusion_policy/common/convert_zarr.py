import os
import numpy as np
import cv2
import zarr
from replay_buffer import ReplayBuffer

def read_image(path):
    """读取 PNG 图像并返回 RGB 数组"""
    img = cv2.imread(path)                 # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为 RGB
    return img

def load_episode_data(episode_dir):
    """
    加载单个 episode 的所有数据，返回字典：
    {
        'action': (T,8) float32,
        'joint': (T,7) float32,
        'tcp': (T,7) float32,
        'image': (T,H,W,3) uint8
    }
    """
    # 各模态文件夹路径
    action_dir = os.path.join(episode_dir, 'action')
    gripper_dir = os.path.join(episode_dir, 'gripper_command')
    joint_dir = os.path.join(episode_dir, 'joint')
    tcp_dir = os.path.join(episode_dir, 'tcp')
    image_dir = os.path.join(episode_dir, 'cam_750612070265', 'color')

    # 获取所有时间戳（按文件名排序）
    action_files = sorted([f for f in os.listdir(action_dir) if f.endswith('.npy')])
    timestamps = [f[:-4] for f in action_files]   # 去除 '.npy'

    T = len(timestamps)
    if T == 0:
        raise ValueError(f"No data found in {episode_dir}")

    # 收集每个时间步的数据
    action_list, gripper_list, joint_list, tcp_list, image_list = [], [], [], [], []
    for ts in timestamps:
        action_list.append(np.load(os.path.join(action_dir, ts + '.npy')).flatten())
        gripper_list.append(np.load(os.path.join(gripper_dir, ts + '.npy')).flatten())
        joint_list.append(np.load(os.path.join(joint_dir, ts + '.npy')).flatten())
        tcp_list.append(np.load(os.path.join(tcp_dir, ts + '.npy')).flatten())
        image_list.append(read_image(os.path.join(image_dir, ts + '.png')))

    # 堆叠为数组
    actions = np.stack(action_list, axis=0).astype(np.float32)        # (T,7)
    grippers = np.stack(gripper_list, axis=0).astype(np.float32)      # (T,1)
    new_action = np.concatenate([actions, grippers], axis=-1)         # (T,8)
    joints = np.stack(joint_list, axis=0).astype(np.float32)          # (T,7)
    tcps = np.stack(tcp_list, axis=0).astype(np.float32)              # (T,7)
    images = np.stack(image_list, axis=0).astype(np.uint8)            # (T,H,W,3)

    return {
        'action': new_action,
        'joint': joints,
        'tcp': tcps,
        'image': images
    }

def main(root_dir, output_zarr_path):
    # 获取所有 episode 子文件夹（假设直接位于 root_dir 下）
    episode_dirs = [
        os.path.join(root_dir, d) for d in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    print(f"Found {len(episode_dirs)} episodes.")

    # 创建 Zarr 存储（直接写入磁盘）
    store = zarr.DirectoryStore(output_zarr_path)
    buffer = ReplayBuffer.create_empty_zarr(storage=store)

    # 第一个 episode 用于确定分块和压缩器（后续自动沿用）
    for i, ep_dir in enumerate(episode_dirs):
        print(f"Processing episode {i+1}/{len(episode_dirs)}: {ep_dir}")
        episode_data = load_episode_data(ep_dir)

        if i == 0:
            # 从第一帧图像获取尺寸，手动设置分块（时间维度分块，空间保持完整）
            H, W, C = episode_data['image'].shape[1:]
            chunks = {
                'image': (10, H, W, C),        # 每10帧一个块，空间不分块
                'action': (100, 8),             # 动作时间块大小100
                'joint': (100, 7),
                'tcp': (100, 7)
            }
            # 所有字段使用 Blosc 压缩（这里使用 'disk' 策略，也可用 'default'）
            compressors = {
                'image': 'disk',
                'action': 'disk',
                'joint': 'disk',
                'tcp': 'disk'
            }
            buffer.add_episode(episode_data, chunks=chunks, compressors=compressors)
        else:
            # 后续 episode 沿用已存在的数组的 chunks 和 compressor
            buffer.add_episode(episode_data)

    print("All episodes added.")
    print(buffer.tree())
    print("Chunks:", buffer.get_chunks())
    print("Compressors:", buffer.get_compressors())

if __name__ == '__main__':
    # 请根据实际路径修改
    root_dir = '/data/liuyitang/data/carrot/train'          # 包含50个子文件夹的目录
    output_zarr = '/data/liuyitang/data/carrot/train/carrot_dataset_2.zarr'   # 输出的 Zarr 路径
    main(root_dir, output_zarr)