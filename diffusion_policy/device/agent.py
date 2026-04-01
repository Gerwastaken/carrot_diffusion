"""
Evaluation Agent.
"""

import time
import numpy as np
from PIL import Image
from diffusion_policy.device.robot import FlexivRobot, FlexivGripper
from diffusion_policy.device.camera import CameraD400
from utils.transformation import xyz_rot_transform


class Callable:
    def __call__(self, *args, **kwds):
        pass


class DoNothing:
    def __getattribute__(self, name: str):
        return Callable()


BLOCK_TIME = 0.1 - 0.03


class EvalAgent:
    """
    Evaluation agent with Flexiv arm, Dahuan gripper and Intel RealSense RGB-D camera.

    Follow the implementation here to create your own real-world evaluation agent.
    """
    rot90 = False

    def __init__(
        self,
        robot_ip='192.168.2.100',
        camera_serials=['135122075425'],
        **kwargs
    ):

        print("Init robot, gripper, and camera.")
        self.robot = FlexivRobot(robot_ip)
        self.robot.send_tcp_pose(self.ready_pose, slow=True)
        time.sleep(2)
        self.gripper = FlexivGripper(self.robot)

        self.camera = [CameraD400(camera_serial, fps=30)
                       for camera_serial in camera_serials]
        self.camera_serials = camera_serials
        self.use_hand = False
        # self.use_hand = True
        if self.use_hand:
            self.camera_h = CameraD400('104122061850', res=(640, 480))
            self.intrinsics_h = self.camera_h.getIntrinsics()
        print('Init done')

    @property
    def intrinsics(self):
        raise DeprecationWarning()
        return np.array(
            [[898.3217688885679804, 362.02886876018886],
             [0, 0, 1]]
        )

    @property
    def ready_pose(self):
        return np.array([0.6, 0, 0.3, 0, 0.5**0.5, -0.5**0.5, 0])
        return np.array([0.4, 0, 0.3, 0, 0, 1, 0])

    @property
    def ready_rot_6d(self):
        # return np.array([1, 0, 0, 0, 1, 0.3]) # TODO
        from utils.transformation import xyz_rot_transform
        return xyz_rot_transform(
            self.ready_pose,
            from_rep='quaternion',
            to_rep="rotation_6d",
        )[3:]
        return np.array([-1, 0, 0, 0, 1, 0])

    def _get_observation(self, i):
        # colors, depths = self.camera[i].get_data()
        colors, depths = self.camera[i].get_data()
        import cv2
        colors = cv2.cvtColor(colors, cv2.COLOR_RGB2BGR)
        colors = np.clip(colors + np.random.normal(0, 40, colors.shape), 0,255).astype(np.uint8)
        return colors, depths / 1000.

    def get_observation(self):
        return [self._get_observation(i) for i in range(len(self.camera))]

    def get_observation_h(self):
        # colors, depths = self.camera_h.get_data()
        colors, depths = self.camera_h.get_data()
        import cv2
        colors = cv2.cvtColor(colors, cv2.COLOR_RGB2BGR)
        return colors, depths / 1000.

    def get_tcp_pose(self):
        tcp_pose = self.robot.get_tcp_pose()
        tcp_pose = xyz_rot_transform(
            tcp_pose,
            from_rep="quaternion",
            to_rep="matrix",
        )
        # import pdb;pdb.set_trace()
        if self.rot90:
            tcp_pose[:3, :3] = tcp_pose[:3, :3] @ np.array([[0, -1, 0],
                                                            [1, 0, 0],
                                                            [0, 0, 1]])
        tcp_pose = xyz_rot_transform(
            tcp_pose,
            from_rep='matrix',
            to_rep="quaternion",
        )
        return tcp_pose

    def set_tcp_pose(self, pose, rotation_rep, rotation_rep_convention=None, blocking=False, slow=False):
        tcp_pose = xyz_rot_transform(
            pose,
            from_rep=rotation_rep,
            to_rep="matrix",
            from_convention=rotation_rep_convention
        )
        # import pdb;pdb.set_trace()
        if self.rot90:
            tcp_pose[:3, :3] = tcp_pose[:3, :3] @ np.array([[0, 1, 0],
                                                            [-1, 0, 0],
                                                            [0, 0, 1]])
        tcp_pose = xyz_rot_transform(
            tcp_pose,
            from_rep='matrix',
            to_rep="quaternion",
            from_convention=rotation_rep_convention
        )
        self.robot.send_tcp_pose(tcp_pose, slow=slow)
        if blocking:
            time.sleep(BLOCK_TIME)

    def set_joint_pose(self, joint_pose, blocking=False):
        self.robot.send_joint_pose(joint_pose)
        if blocking:
            time.sleep(BLOCK_TIME)

    def set_gripper_width(self, width, force=30, blocking=False):
        # width = 1000 if width > 500 else 0
        print(width)
        self.gripper.move(width, force=force)
        if blocking:
            time.sleep(2)
        # import ipdb
        # ipdb.set_trace()

    def stop(self):
        self.robot.stop()

    def get_transform_h(self, tcp_pose, cam_to_base):
        from dataset.constants import INHAND_CAM_TCP
        from utils.transformation import xyz_rot_to_mat, mat_to_xyz_rot

        current_pose = xyz_rot_to_mat(tcp_pose, rotation_rep='quaternion')

        return np.linalg.inv(cam_to_base) @ current_pose @ INHAND_CAM_TCP
        camT = np.eye(4)
        return np.linalg.inv(cam_to_base) @ camT


class FakeRobot:
    def __init__(self, ready_pose):
        self.ready_pose = ready_pose

    def get_tcp_pose(self):
        return self.ready_pose


class FakeGripper:
    def get_gripper_state(self):
        return 0.8


class FakeAgent:
    def __init__(
        self,
        camera_serial='135122075425',
        **kwargs
    ):
        self.robot = FakeRobot(self.ready_pose)
        self.gripper = FakeGripper()
        self.use_hand = False
        self.camera_serial = camera_serial
        print("Fake! Just for Debug")

    @property
    def intrinsics(self):
        return np.array(
            [[898.3217688363682, 0, 652.8827921243005],
             [0, 899.0708885679804, 362.02886876018886],
             [0, 0, 1]]
        )

    @property
    def ready_pose(self):
        return np.array([0.6, 0, 0.2, 0, 0.5**0.5, 0.5**0.5, 0])
        return np.array([0.6, 0, 0.2, 0, 0, 1, 0])

    @property
    def ready_rot_6d(self):
        # return np.array([1, 0, 0, 0, 1, 0.3]) # TODO
        from utils.transformation import xyz_rot_transform
        return xyz_rot_transform(
            self.ready_pose,
            from_rep='quaternion',
            to_rep="rotation_6d",
        )[3:]
        return np.array([-1, 0, 0, 0, 1, 0])

    def get_observation(self):
        colors = np.array(Image.open('./rgb_test.png'), dtype=np.float32)
        depths = np.array(Image.open('./depth_test.png'), dtype=np.float32)
        return colors, depths / 1000.

    def get_observation_h(self):
        raise NotImplementedError

    def set_tcp_pose(self, pose, rotation_rep, rotation_rep_convention=None, blocking=False):
        if blocking:
            time.sleep(0.1)

    def set_gripper_width(self, width, blocking=False):
        if blocking:
            time.sleep(0.5)

    def stop(self):
        pass

    def get_transform_h(self, tcp_pose, cam_to_base):
        raise NotImplementedError


if __name__ == '__main__':
    agent = EvalAgent(camera_serials=[])
    obv = agent.get_observation()
    tcp = agent.get_tcp_pose()
    agent.set_gripper_width(500)
    agent.set_tcp_pose(
        tcp, 
        rotation_rep="quaternion",
        blocking=True
    )
    import ipdb;ipdb.set_trace()
