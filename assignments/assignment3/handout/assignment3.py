import argparse
import numpy as np
from scipy.linalg import pinv, svd # Your only additional allowed imports!
# import matplotlib.pyplot as plt

class LDS:
    def __init__(self):
        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Assignment 3",
        epilog = "CSCI 4360/6360 Data Science II: Fall 2017",
        add_help = "How to use",
        prog = "python assignment3.py <arguments>")
    parser.add_argument("-f", "--infile", required = True,
        help = "Dynamic texture file, a NumPy array.")
    parser.add_argument("-q", "--dimensions", required = True, type = int,
        help = "Number of state-space dimensions to use.")
    parser.add_argument("-o", "--output", required = True,
        help = "Path where the 1-step prediction will be saved as a NumPy array.")

    args = vars(parser.parse_args())

    # Collect the arguments.
    input_file = args['infile']
    q = args['dimensions']
    output_file = args['output']

    # Read in the dynamic texture data.
    M = np.load(input_file)
    f, h, w = M.shape
    M = M.reshape((M.shape[0],-1))
    M = M.T
    # print(M.shape)
    U, s, V = svd(M)
    # print(U.shape, s.shape, V.shape)
    U = U[:,0:q]
    s = np.diag(s[0:q])
    V = V[0:q,:]
    # print(U.shape, s.shape, V.shape)


    A = np.matmul(np.matmul(s,V[:, 0:f-2]),pinv(np.matmul(s,V[:,1:f-1])))
    # print(A.shape)
    X_1_tp1 =np.matmul(A,np.matmul(s,V))
    X_tp1 = X_1_tp1[:,-1]

    y = np.matmul(U,X_tp1).reshape(h,w)
    np.save(output_file, y)
    # print(y.shape)
    # plt.imshow(y)
    # plt.show()
    ### FINISH ME
