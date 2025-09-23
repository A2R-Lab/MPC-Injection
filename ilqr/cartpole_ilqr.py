import mujoco
import numpy as np
from mujoco import viewer
import time
from matplotlib import pyplot as plt

class Cartpole_iLQR:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        try:
            print("Launching viewer...")
            self.viewer = viewer.launch_passive(self.model, self.data)
        except Exception as e:
            print("Viewer could not be launched. Continuing without viewer.")
            self.viewer = None

        self.nq = self.model.nq     # number of position variables
        self.nv = self.model.nv     # number of velocity variables
        self.nx = self.nq + self.nv # total state dimension
        self.nu = self.model.nu     # number of control inputs

        self.dt = self.model.opt.timestep # time step, TODO: verify if this is correct from the xml
        self.horizon = 10 # time horizon for iLQR
        self.max_iter = 100

        # Regularization params
        self.reg_min = 1e-6
        self.reg_max = 1e6
        self.reg_factor = 10
        self.reg = 1.0

        # Weights for cost functions
        self.W_pos_err_weight = np.diag([5., 5., 5.]) # TODO: determine why this size?
        self.Q = np.diag([1.0] * self.nq + [0.1] * self.nv) # state weight
        self.R = np.diag([0.01] * self.nu)                  # control weight
        self.Qf = self.Q * 10.                              # final state weight

        self.trajectories = []
    
    def get_state(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        return np.concatenate([qpos[:self.nq], qvel[:self.nv]]) # NOTE: probably don't need these slices
    
    def set_state(self, x):
        self.data.qpos[:self.nq] = x[:self.nq]
        self.data.qvel[:self.nv] = x[self.nq:]
        mujoco.mj_forward(self.model, self.data)

    def step(self, u):
        self.data.ctrl[:self.nu] = np.clip(u, -self.model.actuator_ctrlrange[:,0],
                                               self.model.actuator_ctrlrange[:,1])
        mujoco.mj_step(self.model, self.data)
        return self.get_state()
    
    def iLQR(self, target_state):
        # TODO: Implement the iLQR algorithm here!
        print("Running backwards pass...")
    
    def close_viewer(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    
if __name__ == "__main__":
    controller = Cartpole_iLQR("ilqr/xmls/cartpole.xml")

    print(controller.get_state())

    # Make sure to close the viewer properly
    controller.close_viewer()