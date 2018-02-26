# colors_toolbox
A toolbox of robot learning for the CoLoRs (Cognition, Learning and Robotics) Group of AILAB ( Artificial Intelligence Laboratory) , Bogazici University 

## Data Utilities
- data_utils.py contains some utilities for file reading, sampling, modeling etc. Created for the simplicity of the main codes.

## Baxter Utilities
- baxter_utils.py contains some utilities for Baxter robot. Created for the ease of use of the baxter interface.

## DMP - HMM - GMR
- DMP.py creates (Dynamic Movement Primitives) DMP model of the trajectory. A Jupyter Notebook explaining how to use DMP.py is given by DMP_documentation.ipynb
- hmm.py creates (Hidden Markov Model) HMM for encoding trajectory or force/torque feedback from the robot.
- gmr.py has the implementation of Gaussian Mixture Regression (GMR) for reproducing the trajectory or force/torque feedback encoded by HMM.

## Trajectory Encoding
- learn.py uses DMP.py for encoding the trajectory data obtained from Baxter robot.

## Force/Torque Encoding
- encode_forces.py encode forces using the data obtained by without force feedback execution ( zeta = 0) of the DMP model. 

## Execution
- execute.py executes the trajectory using the DMP model and with/without force feedback.
