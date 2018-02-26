#!/usr/bin/env python
import numpy as np
from hmm import hmm
from DMP import DMP
from gmr import gmr
import glob
from GaussianEllipses import plot_gauss_ellipse
import matplotlib.pyplot as plt
import pickle
from scipy.signal  import savgol_filter
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import copy

from numpy import (
	arctan2,
	arcsin,
	cos,
	sin
)


# dataset assumed to be (datalength x dimension) if only demonstrations of one task
# or 	  (task_number x datalength x dimension) if demonstrations of more than one task (after sampling)

class data:

	def __init__(self, filename = None, abs_path = None):

		if filename:
			self.dataset = self.from_file(filename)
		elif abs_path :
			self.dataset = self.from_folder(abs_path)

	def from_file(self, filename):

		text_file = open(filename,"r")
		lines     = text_file.readlines()
		matrix      = np.array([lines[i].split(",") for i in range(len(lines))])
		self.names  = matrix[0]
		dataset = matrix[1:,:].astype(float)

		return np.array(dataset)

	def from_folder(self, abs_path):

		dataset = []
		task_number = 0
		self.filenames =[]

		for filename in np.sort(glob.glob(abs_path+"*.csv")):


			matrix = self.from_file(filename)
			dataset.append(matrix)
			task_number += 1
			self.filenames.append(filename.replace(abs_path, ''))

		self.task_number = task_number

		## cannot create numpy array dataset since each demo may have
		## different datasizes

		return dataset

	def prepare(self, time_interval=None, sample_size = None, window_size = None, dimensions = None):

		# time_interval is a list : [T0, TN] where T0 and TN are initial and final time of original data -- list
		# sample_size is the number of sample datapoint to take from the task demo -- int
		# window_size is the window average's window size -- int
		# dimensions is the dimensions on which to apply window average -- list

		# Check if we have one csv file read or more
		if type(self.dataset) == list:
			dataset = []
			for task in range(self.task_number):
				matrix = self.dataset[task]

				# First : apply time interval and start time from 0 seconds
				time = matrix[:, 0] - matrix[0,0]

				if time_interval :
					if time_interval[1] == -1:
						t = range(find_index(time, time_interval[0]),matrix.shape[0])
					else :
						t = range(find_index(time, time_interval[0]),\
								  find_index(time, time_interval[1]))

				else :
					t = range(0,matrix.shape[0])

				matrix = matrix[t]

				# Second : sample data
				if sample_size :
					matrix = sample_data(matrix, sample_size)

				# Final : take window average to smooth data on required dimensions
				if window_size :
					matrix[:,dimensions] = savgol_filter(matrix[:,dimensions], window_size,10, axis = 0)

					# matrix = window_average(matrix, window_size, dimensions ).T

				dataset.append(matrix)

			self.dataset = dataset # Not numpy array if no sampling

		else :
			matrix = self.dataset
			# First : apply time interval and start time from 0 seconds
			time = matrix[:, 0] - matrix[0,0]

			if time_interval :
				if time_interval[1] == -1:
					t = range(find_index(time, time_interval[0]),matrix.shape[0])
				else :
					t = range(find_index(time, time_interval[0]),\
							  find_index(time, time_interval[1]))
			else :
				t = range(0,matrix.shape[0])

			matrix = matrix[t]

			# Second : sample data
			if sample_size :
				matrix = sample_data(matrix, sample_size)

			# Final : take window average to smooth data on required dimensions
			if window_size :
				matrix[:,dimensions] = savgol_filter(matrix[:,dimensions], window_size, 2, axis = 0)

			self.dataset = matrix

	def extract_features(self):
		print "Extracting features..."
		if type(self.dataset) == list:
			self.task_number = len(self.dataset)


			self.time                 = []
			self.data_length          = []
			self.joint_positions      = []
			self.end_positions        = []
			self.end_quaternions      = []
			self.end_pose             = []
			self.end_pose_euler       = []
			self.wrench               = []
			self.end_linear_velocity  = []
			self.end_angular_velocity = []
			self.end_velocity         = []
			self.joint_velocities     = []

			for task in range(self.task_number) :

				matrix  = self.dataset[task]
				self.data_length.append(matrix.shape[0]-1)

				self.time.append(matrix[1:, 0]-matrix[0,0])

				self.joint_positions.append(matrix[1:, 1:8])

				self.end_positions.append(matrix[1:, 9:12])
				self.end_quaternions.append(matrix[1:, 12:16])
				self.end_pose.append(matrix[1:, 9:16])
				self.end_pose_euler.append(quatrep2eulrep(matrix[1:, 9:16]))

				self.wrench.append(matrix[1:, 16:22])

				self.end_linear_velocity.append(matrix[1:, 22:25])
				self.end_angular_velocity.append(matrix[1:, 25:28])
				self.end_velocity.append(matrix[1:, 22:28])
				self.joint_velocities.append(matrix[1:, 28:35])
			# this gives lists, not numpy.arrays because the datalengths of
			# each task may not be equal. Consider sampling first or convert
			# to numpy arrays after getting the data. !! may be corrected !!

		else :
			self.task_number = 1
			matrix = self.dataset
			self.data_length = matrix.shape[0]-1
			self.time = matrix[1:, 0] - matrix[0,0]
			self.joint_positions = matrix[1:, 1:8]
			self.end_positions   = matrix[1:, 9:12]
			self.end_quaternions = matrix[1:, 12:16]
			self.end_pose        = matrix[1:, 9:16]
			self.end_pose_euler  = quatrep2eulrep(matrix[1:, 9:16])

			self.wrench          = matrix[1:, 16:22]

			self.end_linear_velocity  = matrix[1:, 22:25]
			self.end_angular_velocity = matrix[1:, 25:28]
			self.end_velocity         = matrix[1:, 22:28]
			self.joint_velocities     = matrix[1:, 28:35]
			# self.end_pose_euler  = np.array([quat2euler(matrix[i,[15,12,13,14]]) for i in range(len(t))])

	def apply_dtw(self,features):
		print "Applying Dynamic Time Warping, this can take a while..."
		# features = list of strings ex: ["joint_positions"] or ["joint_positions", "end_pose"]
		# requires more than one task , run it after calling extract_features() method
		for string in features:
			if string == "joint_positions":
				self.joint_positions,base_index = dtw(self.task_number, self.joint_positions, self.time)
			if string == "end_pose":
				self.end_pose,base_index = dtw(self.task_number, self.end_pose, self.time)
			if string == "end_pose_euler":
				self.end_pose_euler,base_index = dtw(self.task_number, self.end_pose_euler, self.time)

		self.time = np.tile(self.time[base_index],(self.task_number,1))
		self.time = np.reshape(self.time, (self.time.shape[0],self.time.shape[1], ))

def quatrep2eulrep(end_pose):

	end_pose_euler = np.column_stack((end_pose[:,:3], quat2euler(end_pose[:,3:])))
	return end_pose_euler

class model:
	def __init__(self, model, dataset, model_parameters):
		### model : string 'PHMM' ,'HMM' , 'DMP'
		### dataset : (datalength x dimension) for all models
		if dataset.ndim >2 :
			dataset = dataset.reshape((dataset.shape[0]*dataset.shape[1],dataset.shape[2]))
		if model == 'PHMM':
			raise Exception("Not ready yet!")
		if model == 'HMM':
			self.hmm = hmm(model_parameters, dataset)
			self.prior,self.a,self.mu,self.sigma,self.loglikelihood = self.hmm.Baum_Welch()
		if model == 'DMP':
			# model_parameters = [bf_number, K, task_number]
			# K 		       = vector
			self.dmp = DMP(model_parameters[0], model_parameters[1],model_parameters[2], dataset)
			b = False
			

def find_index(time_array, moment):
	index = 0
	while time_array[index] <= moment:
		index += 1
	return index

def dtw(task_number, jp, t):
	## it says jp but it can be applied to endpose or other as well
	datasizes = np.zeros(task_number)
	for i in range(task_number):
		datasizes[i] = np.array(t[i][:]).shape[0]
		# print np.array(t[i][:]).shape[0]
	base_index = np.argmax(datasizes)
	base_datasize = np.array(t[base_index]).shape[0]

	joint_positions = np.zeros((task_number,base_datasize,jp[base_index].shape[1]))

	for j in range(jp[base_index].shape[1]):
		# plt.figure()

		for i in range(task_number):
			# print np.array(jp[i][:,j]).shape, np.array(jp[base_index][:,j]).shape
			d,path = fastdtw(jp[i][:,j],jp[base_index][:,j], dist=euclidean)
			# print np.array(path).shape
			p = np.array(jp[i])
			path = np.array(path)
			joint_positions[i,:,j] = sample_data(p[path[:,0],j], base_datasize)
			# plt.plot(t[i],jp[i][:,j],label = 'demo', linestyle = '--')
			# plt.plot(t[base_index], sample_data(p[path[:,0],j], base_datasize))
	return joint_positions,base_index

def window_average(dataset, window_size, dimensions):
	# Window average to smooth data
	dataLength = dataset.shape[0]
	D = copy.deepcopy(dataset)
	data       = copy.deepcopy(dataset[:,dimensions])
	windowed = []
	n = 0
	somme = np.zeros(len(dimensions))

	while n < dataLength:
		if n < window_size:
			somme = somme + data[n,:]
			avg   = somme / (n+1)
			windowed.append( avg )
		else :
			windowed.append(  np.sum(data[n-window_size:n,:], axis = 0) / window_size )

		n = n + 1

	windowed = np.array(windowed)
	D[:, dimensions] = windowed


	return D


def sample_data(data, sample_size):
	dataLength   = data.shape[0]

	indices      = np.linspace(0, dataLength-1, num=sample_size, dtype = np.int16)

	sampled_data = data[indices]

	return np.array(sampled_data)

def get_sampled_features(dataset, sample_size):

	if type(dataset) == list:
		task_number = len(dataset)
		sampled_fatures = []
		for i in range(task_number):
			data = dataset[i]
			sampled_fatures.append(sample_data(data,sample_size))
		sampled_fatures = np.array(sampled_fatures)
	else:
		sampled_fatures = sample_data(dataset,sample_size)

	return sampled_fatures




def plot_gaussian_phmm(mu, sigma, inp, out):

	hiddenStateNumber = sigma.shape[0]
	task_number  	  = mu.shape[0]
	for i in range(hiddenStateNumber):
		for k in range(task_number):
			if k == 0:
				a = 'r'
			elif k==1:
				a = 'b'
			elif k==2:
				a = 'g'
			plot_gauss_ellipse(sigma[i][[inp,out]][:,[inp,out]],\
							   mu[k][i][[inp,out]],\
							   nstd = 0.5, alpha= 0.5, color = a)
def plot_data(data, inp, out):
	input_data = data[0]
	output_data = data[1]

	if type(input_data) == list:
		task_number = len(input_data)
		for i in range(task_number):
			input_  = input_data[i]
			output_ = output_data[i]
			if input_.ndim != 2:
				input_ = np.reshape(input_, (input_.shape[0],1))
			if output_.ndim != 2:
				output_ = np.reshape(output_, (output_.shape[0],1))

			plt.plot(input_[:,inp], output_[:,out])
	else:
		if output_data.ndim>2 or input_data.ndim >2:
			task_number = input_data.shape[0]
			if output_data.ndim > input_data.ndim:
				input_data = np.reshape(input_data, (input_data.shape[0],input_data.shape[1], 1))

			elif output_data.ndim < input_data.ndim:
				output_data = np.reshape(output_data, (output_data.shape[0],output_data.shape[1], 1))

			for i in range(task_number):
				plt.plot(input_data[i][:,inp], output_data[i][:,out])

		else :
			if input_data.ndim == 1:
				input_data  = np.reshape(input_data,  (input_data.shape[0],1))
			if output_data.ndim == 1:
				output_data = np.reshape(output_data, (output_data.shape[0],1))
			plt.plot(input_data[inp], output_data[out])

def plot_gaussian_hmm(mu, sigma, inp, out):
	hiddenStateNumber = sigma.shape[0]

	for i in range(hiddenStateNumber):
		plot_gauss_ellipse(sigma[i][[inp,out]][:,[inp,out]],mu[i][[inp,out]],  nstd = 0.5, alpha= 0.5 )

def concatenate_tasks(data):
	task_number = data.shape[0]
	task_length = data.shape[1]
	try :
		dimensions  = data.shape[2]
		return np.reshape(data,(task_number*task_length, dimensions))
	except IndexError:

		return np.reshape(data, (task_number*task_length, 1))

def concatenate_dimensions(dim1, dim2):
	dim1_ = concatenate_tasks(dim1)
	dim2_ = concatenate_tasks(dim2)
	return np.column_stack((dim1_, dim2_))

def save_object(obj, filename):
	with open(filename, 'wb') as output:
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)



def quat2euler(quaternion): #x,y,z,w order

	if type(quaternion[0]) ==  np.ndarray or type(quaternion[0]) == list: #if more than one conversion required
		quaternion = np.array(quaternion)
		q0 = quaternion[:,3]
		q1 = quaternion[:,0]
		q2 = quaternion[:,1]
		q3 = quaternion[:,2]

		# roll = arctan2(2*( np.dot(q0,q1)+ np.dot(q2,q3)), 1-2*(np.dot(q1,q1)+np.dot(q2,q2))   )
		# pitch = arcsin(2*(np.dot(q0,q2)-np.dot(q3,q1)))
		# yaw = arctan2(2*(np.dot(q2,q3)+ np.dot(q1,q2)), 1-2*(np.dot(q2,q2)+np.dot(q3,q3)) )
		roll = arctan2(2*(q0*q1+q2*q3),1-2*(q1**2+q2**2))
		pitch = arcsin(2*(q0*q2-q3*q1))
		yaw = arctan2(2*(q0*q3+q1*q2), 1-2*(q2**2+q3**2))

	else: # if only one conversion required

		q0 = quaternion[3]
		q1 = quaternion[0]
		q2 = quaternion[1]
		q3 = quaternion[2]

		roll = arctan2(2*(q0*q1+q2*q3),1-2*(q1**2+q2**2))
		pitch = arcsin(2*(q0*q2-q3*q1))
		yaw = arctan2(2*(q0*q3+q1*q2), 1-2*(q2**2+q3**2))


	euler = np.array([roll,pitch,yaw])
	# print np.transpose(euler).shape
	return np.transpose(euler)


def euler2quat(euler) : #roll, pitch, yaw order (x,y,z)

	if type(euler[0]) == np.ndarray  or type(euler[0]) == list:
		euler = np.array(euler)

		phi   = euler[:,0] #roll
		theta = euler[:,1] #pitch
		psi   = euler[:,2] #yaw

	else:
		phi   = euler[0] #roll
		theta = euler[1] #pitch
		psi   = euler[2] #yaw

	half_phi = phi/2
	half_theta = theta/2
	half_psi = psi/2


	q0 = cos(half_phi)*cos(half_theta)*cos(half_psi) + sin(half_phi)*sin(half_theta)*sin(half_psi)
	q1 = sin(half_phi)*cos(half_theta)*cos(half_psi) - cos(half_phi)*sin(half_theta)*sin(half_psi)
	q2 = cos(half_phi)*sin(half_theta)*cos(half_psi) + sin(half_phi)*cos(half_theta)*sin(half_psi)
	q3 = cos(half_phi)*cos(half_theta)*sin(half_psi) - sin(half_phi)*sin(half_theta)*cos(half_psi)


	quaternion = np.array([q1,q2,q3,q0]) #x,y,z,w order

	return np.transpose(quaternion)


