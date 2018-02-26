#!/usr/bin/env python
import numpy as np
from scipy.stats import multivariate_normal

def gmr(X, means, covariances):
    EPSILON = 1
    K       = means.shape[0] # number of hidden states
    OUT     = means.shape[1] # number of dimensions in input+output

    if type(X) == int or type(X) == float or type(X) == np.float64:
        DATA = 1
        INP  = 1 ;
    else :
        DATA = X.shape[0]
        INP  = X.shape[1]

    input_  = X

    #Finding sum of hi
    sum_all = []
    for datapoint in range(0,DATA):

        if type(input_) == int or type(input_) == float or type(X) == np.float64:
            x           = input_
        else:
            x           = input_[datapoint,:]

        sum_of_datapoint = 0
        for gaussian in range(0,K):

            mean_i = means[gaussian]
            mean_x = mean_i[0:INP]

            covars_i                = covariances[gaussian]
            covars_xx               = covars_i[0:INP , 0:INP]



            if len(mean_x) > 1:
                h_datapoint = multivariate_normal.pdf(x, mean_x, EPSILON * covars_xx, allow_singular=True)

            elif len(mean_x) == 1:
                covars_xx = covars_xx[0]
                h_datapoint = multivariate_normal.pdf(x, mean_x, EPSILON * covars_xx, allow_singular=True)

            sum_of_datapoint        = sum_of_datapoint + h_datapoint


        sum_all.append(sum_of_datapoint)

    #Gaussian Mixture Regression
    output = np.zeros(OUT-INP)
    for datapoint in range(0,DATA):

        if type(input_) == int or type(input_) == float or type(X) == np.float64:
            x           = input_
        else:
            x           = input_[datapoint,:]

        output_datapoint        = np.zeros(OUT-INP)
        for gaussian in range(0,K):

            mean_i         = means[gaussian]
            mean_x         = mean_i[0:INP]
            mean_y         = mean_i[INP:OUT]

            covars_i       = covariances[gaussian]
            covars_xx      = covars_i[0:INP , 0:INP]
            covars_yx      = covars_i[INP:OUT, 0:INP]


            if len(mean_x) > 1: # if MIMO and MISO

                cov      = np.dot( covars_yx, np.linalg.inv(covars_xx) )

                inside_i = mean_y + np.dot( cov, x-mean_x )
                h_i      = multivariate_normal.pdf(x, mean_x, EPSILON * covars_xx, allow_singular=True)

            elif len(mean_x) == 1: # if SIMO and SISO
                covars_xx = covars_xx[0]
                inside_i  = mean_y + np.dot( covars_yx ,(x - mean_x)/covars_xx )

                h_i = multivariate_normal.pdf(x, mean_x, EPSILON * covars_xx, allow_singular=True)

            h_i            = h_i / sum_all[datapoint]



            output_datapoint = output_datapoint + h_i * inside_i


        output = np.row_stack([output, output_datapoint])
        # print output
    return output[1:,:]
