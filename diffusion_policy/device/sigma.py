import numpy as np
from scipy.spatial.transform import Rotation as R

# fmt: off
import sys
sys.path.insert(0, "sigma_sdk")
import sigma7
# fmt: on


def parse(px, py, pz, oa, ob, og):
    # return np.array([py, -px, pz]), np.array([oa, ob, og])
    # return np.array([px, py, pz]), R.from_euler('xyz', np.array([-og, ob, -oa]), degrees=False)
    return np.array([px, py, pz]), R.from_euler('XYZ', np.array([oa, ob, og]), degrees=False)
    # return np.array([-px, -py, pz]), np.array([ob, -oa, og])


class Sigma7:
    def __init__(self, pos_scale=5, width_scale=1000) -> None:
        self.pos_scale = pos_scale
        self.width_scale = width_scale
        self.start_sigma()
        self.init_p, self.init_r, _ = self.get_current()

    def start_sigma(self):
        sigma7.drdOpen()
        sigma7.drdAutoInit()
        print('starting sigma')
        sigma7.drdStart()
        sigma7.drdRegulatePos(on=False)
        sigma7.drdRegulateRot(on=False)
        sigma7.drdRegulateGrip(on=False)
        print('sigma ready')

    def get_current(self):
        sig, px, py, pz, oa, ob, og, pg, matrix = sigma7.drdGetPositionAndOrientation()
        # x，y, z，绕x旋转，绕y旋转，绕z旋转
        curr_p, curr_r = parse(px, py, pz, oa, ob, og)
        return curr_p, curr_r, pg

    def get_control(self):
        curr_p, curr_r, pg = self.get_current()

        diff_p = curr_p - self.init_p
        diff_r = curr_r * self.init_r.inv()
        diff_p = diff_p * self.pos_scale
        width = pg / -0.027 * self.width_scale
        return diff_p, diff_r, width

    def detach(self):
        self._prev_p, self._prev_r, _ = self.get_current()

    def detach_init(self):
        self.init_p = self._prev_p
        self.init_r = self._prev_r

    def resume(self):
        curr_p, curr_r, _ = self.get_current()
        self.init_p = self.init_p - self._prev_p + curr_p
        self.init_r = self.init_r * self._prev_r.inv() * curr_r


class Sigma7RPC:
    def __init__(self, pos_scale=5, width_scale=1000) -> None:
        import rpyc
        self.c = rpyc.connect("100.119.231.81", 18861)

        self.pos_scale = pos_scale
        self.width_scale = width_scale
        self.start_sigma()
        sig, px, py, pz, oa, ob, og, pg, matrix = self.c.root.drdGetPositionAndOrientation()
        self.init_p = np.array([-py, px, pz])
        self.init_r = np.array([-oa, -ob, og])

    def start_sigma(self):
        self.c.root.start_sigma()
        # sigma7.drdOpen()
        # sigma7.drdAutoInit()
        # print('starting sigma')
        # sigma7.drdStart()
        # sigma7.drdRegulatePos(on = False)
        # sigma7.drdRegulateRot(on = False)
        # sigma7.drdRegulateGrip(on = False)
        print('sigma ready')

    def get_control(self):
        sig, px, py, pz, oa, ob, og, pg, matrix = self.c.root.drdGetPositionAndOrientation()
        curr_p = np.array([-py, px, pz])
        curr_r = np.array([-oa, -ob, og])

        diff_p = curr_p - self.init_p
        diff_r = curr_r - self.init_r
        diff_p = diff_p * self.pos_scale
        width = pg / -0.027 * self.width_scale
        diff_r = R.from_euler('yzx', -diff_r, degrees=False)
        return diff_p, diff_r, width

import time
if __name__ == '__main__':
    sigma = Sigma7()
    time.sleep(1)
    while True:
        sigma.get_control()
        time.sleep(1)