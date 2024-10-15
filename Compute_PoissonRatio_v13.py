
import numpy as np

def Get_Qij(E11,nu12,E22,G12,E33,nu13,nu23):

    S=np.zeros([4,4]);
    S[0,0]=1/E11;
    S[0,1]=-nu12/E11;
    S[0,2]=-nu13/E11;
    S[1,0]=S[0,1];
    S[1,1]=1/E22;
    S[1,2]=-nu23/E22;
    S[2,0]=S[0,2];
    S[2,1]=S[1,2];
    S[2,2]=1/E33;
    S[3,3]=1/G12;
    
    Q=np.linalg.inv(S)
    return Q


def rotation_matrix(theta):
    # change to radians
    theta = theta/180* np.pi

    M1 = np.zeros([4,4])
    M2 = np.zeros([4,4])
    c = np.cos(theta)
    s = np.sin(theta)

    M1[0,0] = c**2
    M1[0,1] = s**2
    M1[1,0] = s**2
    M1[1,1] = c**2
    M1[2,2] = 1
    M1[3,3] = s**2 - c**2

    M2[0,0] = c**2
    M2[0,1] = s**2
    M2[1,0] = s**2
    M2[1,1] = c**2
    M2[2,2] = 1
    M2[3,3] = s**2 - c**2


    M1[3,0] = c*s
    M1[3,1] = -c*s
    M1[0,3] = 2*c*s
    M1[1,3] = -2*c*s

    M2[3,0] = 2*c*s
    M2[3,1] = -2*c*s
    M2[0,3] = c*s
    M2[1,3] = -c*s

    return M1,M2


def Get_wQij(Q,angles):
    plynum=len(angles);
    wQij=[];
    
    for i in range(plynum):
        theta = angles[i]
        M1 ,M2 = rotation_matrix(theta)
        C_bar = M1@Q@M2
        wQij.append(C_bar)
    return wQij

def Get_zlist( ply_thickness,angles ):
    # total number of plys
    ply_num=len(angles);
    # total thickness
    t=ply_num*ply_thickness;  
    zlist=[];
    for i in range(ply_num):
        zlist.append(i*ply_thickness - 0.5*t)
    zlist.append(ply_num*ply_thickness - 0.5*t)
    return zlist

def get_ABD_mat(wQij, zlist, ply_num):
    A = np.zeros([4,4])
    B = np.zeros([4,4])
    D = np.zeros([4,4])
    for i in range(ply_num):
        A += wQij[i]*(zlist[i+1]-zlist[i])
        B += 0.5*wQij[i]*(zlist[i+1]**2-zlist[i]**2)
        D += (wQij[i]*(zlist[i+1]**3-zlist[i]**3))/3.0
    return A,B,D

def get_J(angles):

    ply_num=len(angles);

    ply_thickness=0.14;

    E11=159000 ;
    E22=9200 ;
    E33=9200 ;
    nu12=0.253 ;
    nu13=0.253 ;
    nu23=0.45 ;
    G12=4370 ;
    

    zlist=Get_zlist(ply_thickness,angles);

    Q=Get_Qij(E11,nu12,E22,G12,E33,nu13,nu23);
    wQij=Get_wQij(Q,angles);
    
    # get A,B,D
    A,B,D = get_ABD_mat(wQij, zlist, ply_num)
    
    # inv(A)*B
    inv_A =  np.linalg.inv(A)
    Tmpt = inv_A@B
    J=inv_A +Tmpt@np.linalg.inv(D-B@Tmpt)@Tmpt.T

    return J


def get_v(angles):
    J = get_J(angles)
    v_12 = -J[0,1]/J[0,0]
    v_13 = -J[2,0]/J[0,0]
    v_23 = -J[2,1]/J[1,1]
    return v_12, v_13, v_23

def get_v12(angles):
    J = get_J(angles)
    return  -J[1,0]/J[0,0]

def get_v13(angles):
    J = get_J(angles)
    return  -J[2,0]/J[0,0]

def get_v23(angles):
    J = get_J(angles)
    return  -J[2,1]/J[1,1]