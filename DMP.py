# Copyright (C) 2017 CoLoRs-AILAB, Bogazici University
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Authors : Hakan Girgin <hakangirgin21@gmail.com>

from __future__ import division

import numpy as np


class DMP:

    def __init__(self, BF_number, K, task_number, X):
         self.task_number = task_number
         self.BF_number   = BF_number             # Basis function number
         self.t           = X[:, 0] - X[0, 0]     # Making the time start from 0s.
         self.x           = X[:, 1:]              # Demonstrated trajectory

         self.dataLength  = X.shape[0]        # Number of timesteps in a demonstration
         self.DoF         = X.shape[1]-1      # Number of degrees of freedom for DMP

         self.K           = np.diag(K)                  # Spring constant
         self.D           = np.diag(2*np.sqrt(K))       # Damping constant, critically damped

         # Integrating the canonical system and mapping 's' to the time
         self.tau         = self.t[-1]  # Duration of the demonstration

         convergence_rate = 0.001
         self.alpha       = -np.log(convergence_rate)
         if self.task_number == 1:
             self.s       = np.exp(((-self.alpha/self.tau)*self.t))
         else:
             self.s       = np.exp(((-self.alpha/self.tau)*self.t[:self.dataLength//self.task_number]))
             self.s       = np.tile(self.s,(self.task_number,))

         # Creating basis functions and psiMatrix
         self.sumBFs, self.psiMatrix = self.getBasisFunctions()

         self.weights = self.learnWeights()

    def getBasisFunctions(self):
         # Centers logarithmically distributed between 0.001 and 1
         self.c    = np.logspace(-3,0,num=self.BF_number ) # centers of basis functions
         self.h    = self.BF_number/ (self.c**2)           # widths of basis functions

         psi  = np.zeros(self.dataLength)
         for i in range(0,self.BF_number):
                 psi = np.column_stack( [psi, np.exp(-self.h[i]*((self.s-self.c[i])**2))] )
         psiMatrix = psi[:,1:]

         sumBFs = np.sum(psiMatrix,axis = 1)

         return sumBFs, psiMatrix

    def learnWeights(self):
        f_target_holder = []
        for k in range(self.task_number):
             i = range(k*self.dataLength//self.task_number,
                       (k+1)*self.dataLength//self.task_number)
             x = self.x[i]
             t = self.t[i]
             dataLength = self.dataLength//self.task_number

             x0        = x[0,:]          # initial position
             g         = x[-1,:]          # goal position

             # Numerical Differentiations
             dt        = np.transpose(np.broadcast_to(np.diff(t),(self.DoF,dataLength-1)))
             dx        = np.diff(x,axis = 0)

             x_dot     = np.row_stack((np.zeros(self.DoF),dx/dt))
             dx_dot    = np.diff(x_dot, axis=0)
             x_dotdot  = np.row_stack((np.zeros(self.DoF),dx_dot/dt))

             v         = self.tau*x_dot
             v_dot     = self.tau*x_dotdot

             # Finding the target nonlinear function for each time step
             f_target = np.zeros(self.DoF)
             for i in range(dataLength):
                 f_temp = np.dot(np.linalg.inv(self.K), self.tau*v_dot[i,:] + \
                          np.dot(self.D,v[i,:]) )  + (x[i,:]-g) + self.s[i]*(g-x0)
                 
                 f_target = np.column_stack((f_target,f_temp))

             f_target = np.transpose(f_target)
             f_target = f_target[1:,:]
             f_target_holder.append(f_target)
                
        ft = np.array(f_target_holder)
        ft = np.reshape(ft, (self.dataLength,ft.shape[2]))
        # Linear Least Squares Regression
        X       = np.transpose(np.transpose(self.psiMatrix) * self.s/self.sumBFs)
        w_temp  = np.linalg.inv(np.dot(np.transpose(X),X))
        weights = np.dot(w_temp,np.dot(np.transpose(X),ft))
        
        return weights

    def getWeights(self):
        return self.weights

    def executeDMP(self, time, dt, des_tau, x0, g, x, xdot, zeta):

        # Integrate the canonical system
        s  = np.exp(((-self.alpha/des_tau)*time))

        # Basis functions for the new state
        psi        = np.exp(-self.h*((s-self.c)**2))
        sum_of_BFs = np.sum(psi)

        # Nonlinear function computation
        psi = np.reshape(psi,(self.BF_number,1))
        fs_nom =  np.sum( self.weights*psi,axis=0 )
        fs = (fs_nom/sum_of_BFs)*s

        # Scaled Velocity needed
        v = des_tau * xdot

        # Main equations
        v_dot = (1.0/des_tau) * (np.dot(self.K,g-x) -np.dot(self.D,v) - np.dot(self.K,(g-x0))*s + np.dot(self.K,fs) + zeta)
        v     =  v + v_dot * dt
        xdot  =  v/des_tau
        x     =  x + xdot * dt

        return x, xdot

