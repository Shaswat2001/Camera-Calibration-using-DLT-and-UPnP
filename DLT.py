import numpy as np
import math
import sys
import matplotlib.pyplot as plt

def check_solution_exit(points):
    """
    Checks if Camera Matrix can be computed using the given points
    """
    # Minimum of 6 points are needed to calculate the camera matrix
    if points.shape[0]<6:
        return False
    
    # If all the 3D coordinates lie in a plane, No solution exits
    # Coefficients of plane are calculated using first 3 sample points
    a=(points[1][1]-points[0][1])*(points[2][2]-points[0][2])-(points[2][1]-points[0][1])*(points[1][2]-points[0][2])
    b=(points[1][2]-points[0][2])*(points[2][0]-points[0][0])-(points[2][2]-points[0][2])*(points[1][0]-points[0][0])
    c=(points[1][0]-points[0][0])*(points[2][1]-points[0][1])-(points[2][0]-points[0][0])*(points[1][1]-points[0][1])
    d=-(a*points[0][0]+b*points[0][1]+c*points[0][2])

    # Checks if rest of the points lie on the plane
    for i in range(3,points.shape[0]):
        if a*points[i][0]+b*points[i][1]+c*points[i][2]+d!=0:
            return False

    return True 

def normalize(pt):
    """
    Normalize the points
    """
    mat=None
    # Mean of the coordinates along the column
    mean=np.mean(pt,axis=0)
    s=math.sqrt(2)*pt.shape[0]/np.sqrt(np.sum(np.square(pt-mean)))

    # Normalization matrix is created
    if pt.shape[1]==3:
        mat=np.array([[s,0,0,-s*mean[0]],[0,s,0,-s*mean[1]],[0,0,s,-s*mean[2]],[0,0,0,1]])
    elif pt.shape[1]==2:
        mat=np.array([[s,0,-s*mean[0]],[0,s,-s*mean[1]],[0,0,1]])
    
    pt=np.concatenate((pt,np.ones((pt.shape[0],1))),axis=1)
    mat=np.linalg.inv(mat)
    #Normalized Coordinates
    pt_norm=np.dot(pt,mat)
    pt_norm=pt_norm[:,:-1]

    return pt_norm,mat


def compute_camera_matrix(pt_3d,pt_2d):
    """
    Calculates the Camera Matrix given sample of 2D and 3D coordinates
    """
    # The coordinates are normalized (Reduces the number of matrix conditions)
    nrm3d,mat3d=normalize(pt_3d)
    nrm2d,mat2d=normalize(pt_2d)
    
    A=[]
    for i in range(pt_3d.shape[0]):
        arr=[-nrm3d[i][0],-nrm3d[i][1],-nrm3d[i][2],-1,0,0,0,0,nrm2d[i][0]*nrm3d[i][0],nrm2d[i][0]*nrm3d[i][1],nrm2d[i][0]*nrm3d[i][2],nrm2d[i][0]]
        A.append(arr)
        arr=[0,0,0,0,-nrm3d[i][0],-nrm3d[i][1],-nrm3d[i][2],-1,nrm2d[i][1]*nrm3d[i][0],nrm2d[i][1]*nrm3d[i][1],nrm2d[i][1]*nrm3d[i][2],nrm2d[i][1]]
        A.append(arr)

    A=np.array(A)
    # SVD is performed
    U,sigma,V=np.linalg.svd(A)

    # Camera Matrix with the least error
    P_norm=V[-1,:].reshape(3,pt_3d.shape[1]+1)
    # Denormalizing the Camera Matrix
    P=np.dot(np.dot(np.linalg.inv(mat2d),P_norm),mat3d)

    pt_3d=np.concatenate((pt_3d,np.ones((pt_3d.shape[0],1))),axis=1)
    # Predicted 2D coordinates
    pred_2d=np.dot(pt_3d,P.T)
    pred_2d=pred_2d/pred_2d[:,-1].reshape(-1,1)
    pred_2d=pred_2d[:,:-1]
    # Error in the Camera Matrix 
    error=pt_2d-pred_2d

    return P,error

def decompose_matrix(matrix):
    """
    Decomposes the Camera Matrix to Rotation, Intrinsic and Translation Matrix
    """
    H_inf=matrix[:,:-1]
    H_inf_inv=np.linalg.inv(H_inf)
    h_vec=matrix[:,-1].reshape(-1,1)

    centre_proj=-np.dot(np.linalg.inv(H_inf),h_vec)

    #rot_inv,K_inv=np.linalg.qr(H_inf_inv)
    rot_inv,K_inv=QR_decomposition(H_inf_inv)

    rot_matrix=rot_inv.T
    intrinsic=np.linalg.inv(K_inv)
    intrinsic=intrinsic/intrinsic[-1,-1]

    return rot_matrix,intrinsic,centre_proj

def QR_decomposition(matrixA):
    """
    Performs QR Decomposition
    """

    U_mat=np.zeros_like(matrixA)
    U_mat[:,0]=matrixA[:,0]
    for i in range(1,matrixA.shape[1]):
        proj=np.dot(U_mat[:,:i].T,matrixA[:,i].reshape(-1,1))
        proj=np.multiply(proj.T,U_mat[:,:i])/np.sum(U_mat[:,:i]**2,axis=0)
        U_mat[:,i]=-matrixA[:,i]+np.sum(proj,axis=1)
    Q=U_mat/np.sqrt(np.sum(U_mat**2,axis=0))
    R=np.dot(Q.T,matrixA)

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if j<i:
                R[i,j]=0
    return Q,R
     

if __name__=="__main__":

    # Sample Points
    xyz = [[-875, 0, 9.755], [442, 0, 9.755], [1921, 0, 9.755], [2951, 0.5, 9.755], [-4132, 0.5, 23.618],[-876, 0, 23.618]]
    uv = [[76, 706], [702, 706], [1440, 706], [1867, 706], [264, 523], [625, 523]]
    
    cord_3d=np.array(xyz)
    cord_2d=np.array(uv)

    # Checks if solution can be found given the sample points
    result=check_solution_exit(cord_3d)

    if result:
        camera_mtx,error=compute_camera_matrix(cord_3d,cord_2d)
    else:
        sys.exit("Solution Doesn't exist")
    
    # Decompostion of Camera Matrix 
    Rot,Intrinsic,CP=decompose_matrix(camera_mtx)

    print(f"The mean error in Homography is : {np.mean(error)}")

