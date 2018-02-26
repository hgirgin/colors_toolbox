#!/usr/bin/env python

# Python
import sys
sys.path.insert(0, '/home/hakan/ros_ws/src/baxter_examples/scripts/ASM')

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from data_utils import *
from gmr import gmr
np.set_printoptions(precision=3)



# Reading Force Data
abs_path = "/home/hakan/ros_ws/src/baxter_examples/scripts/ASM/"
force_data  = data(abs_path = abs_path + 'experiments/force/force_new/')
# Data preprocessing and feature extraction
time_list     = []
wrench_list   = []
end_pose_list = []
for i in range(force_data.task_number):

	time_list.append(    force_data.dataset[i][:,0])
	wrench_list.append(  force_data.dataset[i][:, 1:7])
	end_pose_list.append(force_data.dataset[i][:, 7:])

training_data = concatenate_dimensions(np.array(time_list),np.array(wrench_list))

# Encode forces by HMM
hiddenStateNumber = 10
print "Encoding Forces..."
HMMObject = model("HMM", training_data, hiddenStateNumber)

## plot
# plt.figure()
# covs      = HMMObject.sigma
# means     = HMMObject.mu
# number_of_tasks = 3
# plot_dim = 1
# data_length = training_data.shape[0]/number_of_tasks
# for i in range(number_of_tasks):
# 	j = range(i*data_length,(i+1)*data_length)
# 	if i == 0:
# 		plt.plot(training_data[j,0], training_data[j,plot_dim], color = 'r', linewidth = 0.5,  label = 'demo')
# 	plt.plot(training_data[j,0], training_data[j,plot_dim], linewidth = 0.5)
# plt.plot(training_data[:data_length,0], gmr(np.reshape(training_data[:data_length,0], (data_length, 1)), means, covs)[:,plot_dim-1], color = 'b', linewidth = 2.0,label = 'reproduction')
# plt.legend()
# plt.xlabel('time (s)')
# plt.ylabel('Force_x (N)')
# plt.title('Learning and Reproduction of Forces in x-direction')
# plot_gaussian_hmm(means, covs,0,plot_dim )
#
# plt.show()
## Trajectory Encoding by DMP


objects_to_save = HMMObject
path_to_save = abs_path +  'models/'

save_object(objects_to_save, path_to_save +'journal_forces2.pkl')
