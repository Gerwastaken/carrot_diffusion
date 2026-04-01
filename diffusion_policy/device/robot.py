import time
import numpy as np
import logging

import flexivrdk


class ModeMap:
    idle = "IDLE"
    cart_impedance_online = "NRT_CARTESIAN_MOTION_FORCE"
    joint = "NRT_JOINT_POSITION"


class FlexivRobot:

    """
    Flexiv Robot Control Class.
    """

    logger_name = "FlexivRobot"

    def __init__(self, *args, **kwargs):
        self.robot_states = flexivrdk.RobotStates()
        self.log = logging.getLogger(__name__)
        self.mode = flexivrdk.Mode
        # self.robot = flexivrdk.Robot(robot_ip_address, pc_ip_address)
        self.robot = flexivrdk.Robot("Rizon4-062713")
        self.init_robot()
        self.init_pose = self.get_tcp_pose()

    def init_robot(self):
        log = self.log
        mode = self.mode
        robot = self.robot

        # Clear fault on robot server if any
        if robot.fault():
            log.warning("Fault occurred on robot server, trying to clear ...")
            # Try to clear the fault
            robot.ClearFault()
            time.sleep(2)
            # Check again
            if robot.fault():
                log.error("Fault cannot be cleared, exiting ...")
                return
            log.info("Fault on robot server is cleared")

        # Enable the robot, make sure the E-stop is released before enabling
        log.info("Enabling robot ...")
        robot.Enable()

        # Wait for the robot to become operational
        while not robot.operational():
            time.sleep(1)

        log.info("Robot is now operational")

        robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
        robot.ExecutePrimitive("ZeroFTSensor", {}, block_until_started=False)

        # WARNING: during the process, the robot must not contact anything, otherwise the result
        # will be inaccurate and affect following operations
        log.warning(
            "Zeroing force/torque sensors, make sure nothing is in contact with the robot"
        )

        # Wait for primitive completion
        while robot.busy():
            time.sleep(1)
        log.info("Sensor zeroing complete")

        # Move robot to home pose
        # log.info("Moving to home pose")
        # self.send_tcp_pose((0.6,0,0.2,0,0,1,0))
        # time.sleep(1)

    def enable(self, max_time=10):
        """Enable robot after emergency button is released."""
        self.robot.Enable()
        tic = time.time()
        while not self.is_operational():
            if time.time() - tic > max_time:
                return "Robot enable failed"
            time.sleep(0.01)
        return

    def _get_robot_status(self):
        self.robot_states = self.robot.states()
        return self.robot_states

    def mode_mapper(self, mode):
        assert mode in ModeMap.__dict__.keys(), "unknown mode name: %s" % mode
        return getattr(self.mode, getattr(ModeMap, mode))

    def get_control_mode(self):
        return self.robot.mode()

    def set_control_mode(self, mode):
        control_mode = self.mode_mapper(mode)
        self.robot.SwitchMode(control_mode)

    def switch_mode(self, mode, sleep_time=0.01):
        """switch to different control modes.

        Args:
            mode: 'idle', 'cart_impedance_online'
            sleep_time: sleep time to control mode switch time

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        if self.get_control_mode() == self.mode_mapper(mode):
            return

        while self.get_control_mode() != self.mode_mapper("idle"):
            self.set_control_mode("idle")
            time.sleep(sleep_time)
        while self.get_control_mode() != self.mode_mapper(mode):
            self.set_control_mode(mode)
            time.sleep(sleep_time)

        print("[Robot] Set mode: {}".format(str(self.get_control_mode())))

    def clear_fault(self):
        self.robot.clearFault()

    def is_fault(self):
        """Check if robot is in FAULT state."""
        return self.robot.fault()

    def is_stopped(self):
        """Check if robot is stopped."""
        return self.robot.stopped()

    def is_connected(self):
        """return if connected.

        Returns: True/False
        """
        return self.robot.connected()

    def is_operational(self):
        """Check if robot is operational."""
        return self.robot.operational()

    def get_tcp_pose(self):
        """get current robot's tool pose in world frame.

        Returns:
            7-dim list consisting of (x,y,z,rw,rx,ry,rz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().tcp_pose)

    def get_tcp_vel(self):
        """get current robot's tool velocity in world frame.

        Returns:
            7-dim list consisting of (vx,vy,vz,vrw,vrx,vry,vrz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().tcp_vel)

    def get_joint_pos(self):
        """get current joint value.

        Returns:
            7-dim numpy array of 7 joint position

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().q)

    def get_joint_vel(self):
        """get current joint velocity.

        Returns:
            7-dim numpy array of 7 joint velocity

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().dq)

    def stop(self):
        """Stop current motion and switch mode to idle."""
        self.robot.Stop()
        while self.get_control_mode() != self.mode_mapper("idle"):
            time.sleep(0.005)

    def set_max_contact_wrench(self, max_wrench):
        self.switch_mode('cart_impedance_online')
        self.robot.setMaxContactWrench(max_wrench)

    def send_impedance_online_pose(self, tcp):
        """make robot move towards target pose in impedance control mode,
        combining with sleep time makes robot move smmothly.

        Args:
            tcp: 7-dim list or numpy array, target pose (x,y,z,rw,rx,ry,rz) in world frame
            wrench: 6-dim list or numpy array, max moving force (fx,fy,fz,wx,wy,wz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        self.switch_mode('cart_impedance_online')
        self.robot.SendCartesianMotionForce(np.array(tcp))

    def send_tcp_pose(self, tcp, slow=False):
        """
        Send tcp pose.
        """
        if not slow:
            self.send_impedance_online_pose(tcp)
        else:
            self.switch_mode('cart_impedance_online')
            self.robot.SendCartesianMotionForce(np.array(tcp), [0] * 6, 0.1)
            for i in range(10):
                time.sleep(0.1)
                tcpPose, jointPose, tcpVel, jointVel = self.get_robot_state()
                diff = np.linalg.norm(
                    np.array(tcpPose[:3]) - np.array(tcp[:3]))
                from scipy.spatial.transform import Rotation as R
                diff2 = (R.from_quat(
                    np.array(tcpPose[3:])) * R.from_quat(np.array(tcp[3:])).inv()).magnitude()

                print(diff, diff2)
                if (diff < 0.1 and diff2 < 0.3):
                    break

    def send_joint_pose(self, q):
        """
        Send joint pose.
        """
        self.switch_mode('joint')
        DOF = len(q)
        target_vel = [0.0] * DOF
        target_acc = [0.0] * DOF
        MAX_VEL = [1.0] * DOF
        MAX_ACC = [1.0] * DOF
        self.robot.SendJointPosition(
            np.array(q), target_vel, target_acc, MAX_VEL, MAX_ACC)

    def get_robot_state(self):
        raw = self._get_robot_status()
        tcpPose = raw.tcp_pose
        tcpVel = raw.tcp_vel
        jointPose = raw.q
        jointVel = raw.dq
        return tcpPose, jointPose, tcpVel, jointVel


class FlexivGripper:
    def __init__(self, r: FlexivRobot) -> None:
        self.gripper_state = flexivrdk.GripperStates()
        self.gripper = flexivrdk.Gripper(r.robot)
        self.gripper.Enable("Robotiq-2F-85")
        self.gripper_state = self.gripper.states()
        self.max_width = 0.085
        print('gripper maxWidth:', self.max_width)
        self.move(1000)
        # time.sleep(1)

    def move(self, width, force=40):
        self.gripper.Move(self.max_width * width / 1000, 0.1, force)
        # self.gripper.Move(self.max_width * width / 1000, 0.1, force)

    def get_gripper_state(self):
        # raise NotImplementedError
        self.gripper_state = self.gripper.states()
        return self.gripper_state.width


if __name__ == '__main__':
    # import ipdb;ipdb.set_trace()
    robot = FlexivRobot()
    robot.init_pose = (0.6, 0, 0.5, 0, 0, 1, 0)
    robot.send_tcp_pose(robot.init_pose)
    gripper = FlexivGripper(robot)
    print(robot.get_robot_state())
    print(gripper.get_gripper_state())
    flag = 0
    while True:
        input('input to switch gripper')
        gripper.move(flag)
        flag = 1000 - flag
    diff_p = robot.init_pose[:3]

    from scipy.spatial.transform import Rotation as R
    while True:
        oa, ob, og = [float(x) for x in input('input a, b, g:').split()]
        r = R.from_euler('XYZ', np.array([oa, ob, og]), degrees=False)
        diff_r = r * R.from_quat(robot.init_pose[3:], scalar_first=True)

        robot.send_tcp_pose(np.concatenate(
            (diff_p, diff_r.as_quat(scalar_first=True)), 0))
