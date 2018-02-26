#!/usr/bin/env python

# Python
import sys
sys.path.insert(0, '/home/hakan/ros_ws/src/baxter_examples/scripts/ASM')

import numpy as np
import argparse
import matplotlib.pyplot as plt
import rospy

from data_utils import *
from baxter_utils import *
from scipy.stats import multivariate_normal
from baxter_interface import Gripper
import copy

class Loop:
	def __init__(self):

		abs_path = "/home/hakan/ros_ws/src/baxter_examples/scripts/ASM/"

		with open(abs_path+'models/dmp_model.pkl', 'rb') as input:
			[ self.dmp_data, self.DMPObject] = pickle.load(input)

		# Uncomment this if force data available.
		with open(abs_path + "models/force_model.pkl", 'rb') as input :
			self.HMMObject = pickle.load(input)

		
		self.Kx = 0
		self.Ky = 80
		self.alpha = 0.98

		self.slow = 1.0
		self.limb_side = 'right'

		# Start ROS
		self.limb    = Limb(self.limb_side)
		self.kin     = baxter_kinematics(self.limb_side)
		self.gripper = Gripper(self.limb_side)
		self.gripper.calibrate()
		self.gripper.close()


		# DMP Parameters initialization
		self.dt = 1.0/100  # denominator is recording rate of this file

		self.current_end_pose  = copy.deepcopy(self.dmp_data.end_pose[0,:])
		self.initial_end_pose  = copy.deepcopy(self.current_end_pose[:])
		self.goal_end_pose     = copy.deepcopy(self.dmp_data.end_pose[-1,:])
		self.desired_end_pose  = copy.deepcopy(self.current_end_pose[:])
		self.des_tau           = copy.deepcopy(self.dmp_data.time[-1])

		self.current_end_velocities = np.zeros(7)
		self.desired_end_velocities = copy.deepcopy( self.current_end_velocities[:])

		self.zeta = np.zeros(7)

		# Going to initial position by Inverse Kinematics
		# self.desired_end_pose = poseEuler2pose(self.desired_end_pose)
		# self.desired_joint_positions = self.inverse_kinematics(self.desired_end_pose)
		self.desired_joint_positions = copy.deepcopy(self.dmp_data.joint_positions[0,:])
		self.current_joint_positions = copy.deepcopy(self.desired_joint_positions[:])
		self.limb.move_to_joint_positions(array2dict(self.desired_joint_positions, self.limb_side))

		# Variables to feed the other functions
		self.t = 0 # to feed GMR
		self.step = 0 # to feed savgol_filter

		# Plotting variables holders initialization
		self.current_joint_position_holder = [self.desired_joint_positions]
		self.desired_joint_position_holder = [self.desired_joint_positions]

		self.current_end_pose_holder = [get_end_pose(self.limb)]
		self.desired_end_pose_holder = [get_end_pose(self.limb)]

		self.time_holder = []

		self.current_forces_holder = []
		self.desired_forces_holder = []

		self.error_prev = 0


	def inverse_kinematics(self, desired_end_pose):
		current_joint_positions = get_joint_positions(self.limb, self.limb_side).tolist()
		desired_joint_positions = self.kin.inverse_kinematics(desired_end_pose[:3].tolist(),\
															desired_end_pose[3:].tolist())

		if desired_joint_positions == None:
			return current_joint_positions
		else:
			return desired_joint_positions


	def compute_zeta(self):


		self.desired_forces_holder.append(gmr(self.t,self.HMMObject.mu, self.HMMObject.sigma).reshape((6,)))

		error_now = (get_wrench(self.limb) - gmr(self.t,self.HMMObject.mu, self.HMMObject.sigma).reshape((6,)))

		error = (1-self.alpha)*error_now + self.alpha*self.error_prev
		self.error_prev = error

		self.zeta[0]  = self.Kx*error[0]
		self.zeta[1]  = self.Ky*error[1]
		

	def loop(self):

		# Real-time part
		while rospy.get_time() == 0:
			pass

		rate = rospy.Rate(1/self.dt)

		# while not rospy.is_shutdown() and self.t <= self.des_tau:
		while not rospy.is_shutdown() and self.t <= 9.45:

			self.time_holder.append(self.t)
			self.current_forces_holder.append(get_wrench(self.limb))
			self.compute_zeta() # uncomment this if force data available.
			self.desired_end_pose, self.desired_end_velocities = \
			self.DMPObject.dmp.execute(self.t, self.dt/self.slow, self.des_tau,
							  self.initial_end_pose,
							  self.goal_end_pose,
							  self.desired_end_pose,
							  self.desired_end_velocities,
							  zeta=self.zeta)
			desired_end_pose = np.append(self.desired_end_pose[:3], self.dmp_data.end_pose[self.step,3:])
			self.desired_joint_positions = self.inverse_kinematics(desired_end_pose)

			self.limb.set_joint_positions(array2dict(self.desired_joint_positions,self.limb_side))
			self.t += self.dt/self.slow

			self.step += 1

			self.current_joint_position_holder.append(get_joint_positions(self.limb, self.limb_side))
			self.desired_joint_position_holder.append(self.desired_joint_positions)
			self.current_end_pose_holder.append(get_end_pose(self.limb))
			self.desired_end_pose_holder.append(self.desired_end_pose)

			rate.sleep()



def main():

	rospy.init_node("PushASM")
	loop_object = Loop()
	loop_object.loop()
	# loop_object.gripper.open()
	current_jp = np.array(loop_object.current_joint_position_holder)
	desired_jp = np.array(loop_object.desired_joint_position_holder)
	time       = np.array(loop_object.time_holder)
	current_ep = np.array(loop_object.current_end_pose_holder)
	desired_ep = np.array(loop_object.desired_end_pose_holder)
	current_f  = np.array(loop_object.current_forces_holder)
	desired_f  = np.array(loop_object.desired_forces_holder)

	# save_data(time, loop_object.limb_side, current_f, current_ep,  fname = 'beta3_Ky80.csv')


	# print time.shape, loop_object.dmp_data.end_pose[0][:,0].shape, current_ep.shape
	# plt.figure(figsize=(30, 25))
	names = ['x(m)','y(m)','z(m)',r'$q_x$',r'$q_y$',r'$q_z$',r'$q_w$']

	for i in range(3):
		plt.figure()
		# plt.subplot(4, 2, i+1)
		# plt.plot(time, current_ep[:-1,i],label = 'actual')
		# plt.plot(time, desired_ep[:-1,i], label = 'desired')
		plt.plot(time, (current_f-desired_f)[:,i], label = 'actual')
		plt.ylabel(names[i])
		# plt.plot(sample_data(time,loop_object.dmp_data.end_pose[:,i].shape[0]), loop_object.true_data.end_pose[:,i],label='true',linestyle = '--')
		plt.legend()

	# plt.show()

	return 0


if __name__ == "__main__":
	sys.exit(main())
