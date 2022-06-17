'''
Code taken from Elias Vanderstuyft 
https://github.com/Eliasvan/Multiple-Quadrotor-SLAM/tree/master/Work/python_libs
'''

import numpy as np
import cv2

def linear_LS_triangulation(u1, P1, u2, P2):
    """
    Linear Least Squares based triangulation.
    Relative speed: 0.1
    
    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    
    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
    
    The status-vector will be True for all points.
    """
    
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))
    
    # Create array of triangulated points
    x = np.zeros((3, len(u1)))
    
    # Initialize C matrices
    # C1 = np.array(linear_LS_triangulation_C)
    # C2 = np.array(linear_LS_triangulation_C)
    C1 = np.array(-np.eye(2, 3))
    C2 = np.array(-np.eye(2, 3))
    
    for i in range(len(u1)):
        # Derivation of matrices A and b:
        # for each camera following equations hold in case of perfect point matches:
        #     u.x * (P[2,:] * x)     =     P[0,:] * x
        #     u.y * (P[2,:] * x)     =     P[1,:] * x
        # and imposing the constraint:
        #     x = [x.x, x.y, x.z, 1]^T
        # yields:
        #     (u.x * P[2, 0:3] - P[0, 0:3]) * [x.x, x.y, x.z]^T     +     (u.x * P[2, 3] - P[0, 3]) * 1     =     0
        #     (u.y * P[2, 0:3] - P[1, 0:3]) * [x.x, x.y, x.z]^T     +     (u.y * P[2, 3] - P[1, 3]) * 1     =     0
        # and since we have to do this for 2 cameras, and since we imposed the constraint,
        # we have to solve 4 equations in 3 unknowns (in LS sense).

        # Build C matrices, to construct A and b in a concise way
        C1[:, 2] = u1[i, :]
        C2[:, 2] = u2[i, :]
        
        # Build A matrix:
        # [
        #     [ u1.x * P1[2,0] - P1[0,0],    u1.x * P1[2,1] - P1[0,1],    u1.x * P1[2,2] - P1[0,2] ],
        #     [ u1.y * P1[2,0] - P1[1,0],    u1.y * P1[2,1] - P1[1,1],    u1.y * P1[2,2] - P1[1,2] ],
        #     [ u2.x * P2[2,0] - P2[0,0],    u2.x * P2[2,1] - P2[0,1],    u2.x * P2[2,2] - P2[0,2] ],
        #     [ u2.y * P2[2,0] - P2[1,0],    u2.y * P2[2,1] - P2[1,1],    u2.y * P2[2,2] - P2[1,2] ]
        # ]
        A[0:2, :] = C1.dot(P1[0:3, 0:3])    # C1 * R1
        A[2:4, :] = C2.dot(P2[0:3, 0:3])    # C2 * R2
        
        # Build b vector:
        # [
        #     [ -(u1.x * P1[2,3] - P1[0,3]) ],
        #     [ -(u1.y * P1[2,3] - P1[1,3]) ],
        #     [ -(u2.x * P2[2,3] - P2[0,3]) ],
        #     [ -(u2.y * P2[2,3] - P2[1,3]) ]
        # ]
        b[0:2, :] = C1.dot(P1[0:3, 3:4])    # C1 * t1
        b[2:4, :] = C2.dot(P2[0:3, 3:4])    # C2 * t2
        b *= -1
        
        # Solve for x vector
        cv2.solve(A, b, x[:, i:i+1], cv2.DECOMP_SVD)
    
    return x.T.astype(float), np.ones(len(u1), dtype=bool)


def iterative_LS_triangulation(u1, P1, u2, P2, tolerance=3.e-5):
    """
    Iterative (Linear) Least Squares based triangulation.
    From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997.
    Relative speed: 0.025
    
    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    "tolerance" is the depth convergence tolerance.
    
    Additionally returns a status-vector to indicate outliers:
        1: inlier, and in front of both cameras
        0: outlier, but in front of both cameras
        -1: only in front of second camera
        -2: only in front of first camera
        -3: not in front of any camera
    Outliers are selected based on non-convergence of depth, and on negativity of depths (=> behind camera(s)).
    
    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
    """
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))
    
    # Create array of triangulated points
    x = np.empty((4, len(u1))); x[3, :].fill(1)    # create empty array of homogenous 3D coordinates
    x_status = np.empty(len(u1), dtype=int)
    
    # Initialize C matrices
    C1 = np.array(-np.eye(2, 3))
    C2 = np.array(-np.eye(2, 3))
    
    for xi in range(len(u1)):
        # Build C matrices, to construct A and b in a concise way
        C1[:, 2] = u1[xi, :]
        C2[:, 2] = u2[xi, :]
        
        # Build A matrix
        A[0:2, :] = C1.dot(P1[0:3, 0:3])    # C1 * R1
        A[2:4, :] = C2.dot(P2[0:3, 0:3])    # C2 * R2
        
        # Build b vector
        b[0:2, :] = C1.dot(P1[0:3, 3:4])    # C1 * t1
        b[2:4, :] = C2.dot(P2[0:3, 3:4])    # C2 * t2
        b *= -1
        
        # Init depths
        d1 = d2 = 1.
        
        for i in range(10):    # Hartley suggests 10 iterations at most
            # Solve for x vector
            #x_old = np.array(x[0:3, xi])    # TODO: remove
            cv2.solve(A, b, x[0:3, xi:xi+1], cv2.DECOMP_SVD)
            
            # Calculate new depths
            d1_new = P1[2, :].dot(x[:, xi])
            d2_new = P2[2, :].dot(x[:, xi])
            
            # Convergence criterium
            #print i, d1_new - d1, d2_new - d2, (d1_new > 0 and d2_new > 0)    # TODO: remove
            #print i, (d1_new - d1) / d1, (d2_new - d2) / d2, (d1_new > 0 and d2_new > 0)    # TODO: remove
            #print i, np.sqrt(np.sum((x[0:3, xi] - x_old)**2)), (d1_new > 0 and d2_new > 0)    # TODO: remove
            ##print i, u1[xi, :] - P1[0:2, :].dot(x[:, xi]) / d1_new, u2[xi, :] - P2[0:2, :].dot(x[:, xi]) / d2_new    # TODO: remove
            #print bool(i) and ((d1_new - d1) / (d1 - d_old), (d2_new - d2) / (d2 - d1_old), (d1_new > 0 and d2_new > 0))    # TODO: remove
            ##if abs(d1_new - d1) <= tolerance and abs(d2_new - d2) <= tolerance: print "Orig cond met"    # TODO: remove
            if abs(d1_new - d1) <= tolerance and \
                    abs(d2_new - d2) <= tolerance:
            #if i and np.sum((x[0:3, xi] - x_old)**2) <= 0.0001**2:
            #if abs((d1_new - d1) / d1) <= 3.e-6 and \
                    #abs((d2_new - d2) / d2) <= 3.e-6: #and \
                    #abs(d1_new - d1) <= tolerance and \
                    #abs(d2_new - d2) <= tolerance:
            #if i and 1 - abs((d1_new - d1) / (d1 - d_old)) <= 1.e-2 and \    # TODO: remove
                    #1 - abs((d2_new - d2) / (d2 - d1_old)) <= 1.e-2 and \    # TODO: remove
                    #abs(d1_new - d1) <= tolerance and \    # TODO: remove
                    #abs(d2_new - d2) <= tolerance:    # TODO: remove
                break
            
            # Re-weight A matrix and b vector with the new depths
            A[0:2, :] *= 1 / d1_new
            A[2:4, :] *= 1 / d2_new
            b[0:2, :] *= 1 / d1_new
            b[2:4, :] *= 1 / d2_new
            
            # Update depths
            #d_old = d1    # TODO: remove
            #d1_old = d2    # TODO: remove
            d1 = d1_new
            d2 = d2_new
        
        # Set status
        x_status[xi] = ( i < 10 and                       # points should have converged by now
                         (d1_new > 0 and d2_new > 0) )    # points should be in front of both cameras
        if d1_new <= 0: x_status[xi] -= 1
        if d2_new <= 0: x_status[xi] -= 2
    
    return x[0:3, :].T.astype(float), x_status