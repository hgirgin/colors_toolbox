#!/usr/bin/env python
import numpy as np
from baxter_interface import *
from baxter_pykdl import *
from data_utils import quat2euler, euler2quat
import copy


def array2dict(angles, limb_side):
    joint_angles = {limb_side + '_s0': angles[0],
                    limb_side + '_s1': angles[1],
                    limb_side + '_e0': angles[2],
                    limb_side + '_e1': angles[3],
                    limb_side + '_w0': angles[4],
                    limb_side + '_w1': angles[5],
                    limb_side + '_w2': angles[6]}

    return joint_angles


def get_wrench(limb):
    wr = limb.endpoint_effort()
    fr = wr['force']
    tr = wr['torque']

    return np.array([fr.x, fr.y, fr.z, tr.x, tr.y, tr.z])


def get_joint_efforts(limb, limb_side):
    eff = limb.joint_efforts()
    current_efforts = [eff[limb_side + "_s0"],
                       eff[limb_side + "_s1"],
                       eff[limb_side + "_e0"],
                       eff[limb_side + "_e1"],
                       eff[limb_side + "_w0"],
                       eff[limb_side + "_w1"],
                       eff[limb_side + "_w2"]]

    return np.array(current_efforts)


def get_joint_positions(limb, limb_side):
    pos = limb.joint_angles()
    current_angles = [pos[limb_side + "_s0"],
                      pos[limb_side + "_s1"],
                      pos[limb_side + "_e0"],
                      pos[limb_side + "_e1"],
                      pos[limb_side + "_w0"],
                      pos[limb_side + "_w1"],
                      pos[limb_side + "_w2"]]

    return np.array(current_angles)


def get_joint_velocities(limb, limb_side):
    pos = limb.joint_velocities()
    current_velocities = [pos[limb_side + "_s0"],
                          pos[limb_side + "_s1"],
                          pos[limb_side + "_e0"],
                          pos[limb_side + "_e1"],
                          pos[limb_side + "_w0"],
                          pos[limb_side + "_w1"],
                          pos[limb_side + "_w2"]]

    return np.array(current_velocities)


def get_end_pose(limb):
    pose = limb.endpoint_pose()
    end_pose = np.array([pose["position"].x,
                         pose["position"].y,
                         pose["position"].z,
                         pose["orientation"].x,
                         pose["orientation"].y,
                         pose["orientation"].z,
                         pose["orientation"].w])
    return end_pose


def get_end_pose_euler(limb):
    end_pose = get_end_pose(limb)
    end_pose_euler = np.append(end_pose[:3], quat2euler(end_pose[3:]))

    return end_pose_euler


def inverse_kinematics(solver, desired_end_pose, current_joint_positions):
    desired_joint_positions = solver.inverse_kinematics(desired_end_pose[:3].tolist(), \
                                                        desired_end_pose[3:].tolist(), \
                                                        current_joint_positions)
    if desired_joint_positions == None:
        return current_joint_positions
    else:
        return desired_joint_positions


def compute_force_coupling_term(current_forces, desired_forces, K1, K2, Jacobian):
    zeta = np.linalg.multi_dot((K1, Jacobian, K2, (current_forces - desired_forces).T))

    return zeta


def poseEuler2pose(desired_pose_euler):
    p = copy.deepcopy(desired_pose_euler)
    if len(p.shape) == 1:
        desired_pose = np.append(p[:3], euler2quat(p[3:]))
    else:
        desired_pose = np.column_stack((p[:, :3], euler2quat(p[:, 3:])))

    return desired_pose


def save_data(time, limb_side, current_ff, current_ep, fname):
    with open(fname, 'w') as f:
        f.write('time,')
        f.write(limb_side + '_force_x,' +
                limb_side + '_force_y,' +
                limb_side + '_force_z,' +
                limb_side + '_torque_x,' +
                limb_side + '_torque_y,' +
                limb_side + '_torque_z,')
        f.write(limb_side + '_endp_x,' +
                limb_side + '_endp_y,' +
                limb_side + '_endp_z,' +
                limb_side + '_endq_x,' +
                limb_side + '_endq_y,' +
                limb_side + '_endq_z,' +
                limb_side + '_endq_w' + '\n')
        for i in range(current_ff.shape[0]):
            f.write(str(time[i]) + ',')
            for j in range(6):
                f.write(str(current_ff[i, j]) + ',')
            for j in range(6):
                f.write(str(current_ep[i, j]) + ',')

            f.write(str(current_ep[i, 6]) + '\n')


def save_data_without_force(time, limb_side, current_ep, fname):
    with open(fname, 'w') as f:
        f.write('time,')
        f.write(limb_side + '_endp_x,' +
                limb_side + '_endp_y,' +
                limb_side + '_endp_z,' +
                limb_side + '_endq_x,' +
                limb_side + '_endq_y,' +
                limb_side + '_endq_z,' +
                limb_side + '_endq_w' + '\n')
        for i in range(current_ep.shape[0]):
            f.write(str(time[i]) + ',')
            for j in range(6):
                f.write(str(current_ep[i, j]) + ',')

            f.write(str(current_ep[i, 6]) + '\n')
