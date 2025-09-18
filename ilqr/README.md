# iLQR solver

Writing our own iLQR solver in mujoco, based on this [project](https://github.com/proxymallick/ur5e_mujoco_control/). The motivation behind writing our own is that we don't want to explicitly write the dynamics of systems and wish to be able to just use the mujoco models. Thus, we are using iLQR with mujoco's model derivative functions.

The xml files for quick reference are pulled from [mujoco_playground](https://github.com/google-deepmind/mujoco_playground/tree/main/mujoco_playground/_src/dm_control_suite/xmls).