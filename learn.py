#!/usr/bin/env python

# Python
import sys
sys.path.insert(0, '/home/hakan/ros_ws/src/baxter_examples/scripts/ASM')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from data_utils import *

# np.set_printoptions(precision=3)


# Encode Trajectory Data
abs_path = "/home/hakan/ros_ws/src/baxter_examples/scripts/ASM/JOURNAL2018/"
dmp_data = data(filename = abs_path + "experiments/push2.csv")
dmp_data.extract_features()
print dmp_data.task_number

print "Features are extracted."

bf_number = 100
K         = np.ones(7)*5000
X         = np.column_stack([dmp_data.time,dmp_data.end_pose])


## For HMM and LWR
DMPObject = model("DMP", X, [bf_number, K,dmp_data.task_number])
print "DMP encoded."
path_to_save = abs_path +  'models/'
save_object([dmp_data,DMPObject], path_to_save + 'journal2.pkl')
print "DMP object saved."
