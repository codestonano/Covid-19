import numpy as np
import glob
import time
import csv
import os
from scipy.optimize import fsolve
from numpy import inf
import datetime

def cross_entropy(I_ref,Iout_softmax):
    return np.sum(np.nan_to_num(-I_ref*np.log(Iout_softmax)-(1-I_ref)*np.log(1-Iout_softmax)))
def get_gra_CL_wij(Iout_softmax, Iout_exp, Iout_exp_sum, Iref, gra_Iout_wij):
    Iout_exp_sum_2d = np.kron(np.expand_dims(Iout_exp_sum, 1), np.ones([1, size_out]))
    # print(Isum.shape)
    Iout_exp_3d = np.kron(np.expand_dims(Iout_exp, 2), np.ones([1, 1, size_hidden_initial]))
    Iout_exp_sum_3d = np.kron(np.expand_dims(Iout_exp_sum_2d, 2), np.ones([1, 1, size_hidden_initial]))
    Iref_3d = np.kron(np.expand_dims(Iref, 2), np.ones([1, 1, size_hidden_initial]))
    # print(E.shape)
    Iout_exp_4d = np.kron(np.expand_dims(Iout_exp_3d, 3), np.ones([1, 1, 1, size_in]))
    Iout_exp_sum_4d = np.kron(np.expand_dims(Iout_exp_sum_3d, 3), np.ones([1, 1, 1, size_in]))
    # print(Isum.shape)
    Iref_4d = np.kron(np.expand_dims(Iref_3d, 3), np.ones([1, 1, 1, size_in]))
    # print(E.shape)
    # Iref_4d=Iref.swapaxes(1,3)
    # print(Iref_4d.shape)
    # Iout_exp_4d=Iout_exp_4d.swapaxes(0,3)
    # Isum=Isum.swapaxes(0,3)
    # E=E.swapaxes(0,1)
    # Isum=Isum.swapaxes(0,1)
    # print(Iout_exp_4d.shape,Iout_exp_sum_4d.shape)
    gra_Iout_exp_wij_4d = Iout_exp_4d * gra_Iout_wij
    gra_Isum_wij_3d = np.sum(gra_Iout_exp_wij_4d, 1)
    gra_Isum_wij_4d = np.kron(np.expand_dims(gra_Isum_wij_3d, 3), np.ones([1, 1, 1, size_out]))
    # print(gra_Isum_wij_4d.shape)
    gra_Isum_wij_4d = gra_Isum_wij_4d.swapaxes(2, 3)
    gra_Isum_wij_4d = gra_Isum_wij_4d.swapaxes(1, 2)
    # print(gra_Isum_wij_4d.shape)
    X = np.expand_dims((-1 / (Iout_softmax * Iref)), 2)
    X = np.kron(X, np.ones([1, 1, size_hidden_initial]))
    X = np.expand_dims(X, 3)
    X = np.kron(X, np.ones([1, 1, size_in]))
    # print(X.shape)
    X[X == inf] = 0
    X[X == -inf] = 0

    gra_CL_wij = X * ((gra_Iout_exp_wij_4d * Iout_exp_sum_4d - Iout_exp_4d * gra_Isum_wij_4d) / Iout_exp_sum_4d ** 2)

    # print(gra_CL_wij)
    gra_CL_wij = np.sum(gra_CL_wij, 1)
    gra_CL_wij = np.sum(gra_CL_wij, 0)
    # print(gra_CL_wij)
    # print(gra_CL_wij.shape)

    return gra_CL_wij



def get_gra_CL_wjk(Iout_softmax, Iout_exp, Iout_exp_sum, Iref, gra_Iout_wjk):
    Iout_exp_sum_2d = np.kron(np.expand_dims(Iout_exp_sum, 1), np.ones([1, size_out]))
    # print(Isum.shape)
    Iout_exp_3d = np.kron(np.expand_dims(Iout_exp, 2), np.ones([1, 1, size_hidden]))
    Iout_exp_sum_3d = np.kron(np.expand_dims(Iout_exp_sum_2d, 2), np.ones([1, 1, size_hidden]))
    Iref_3d = np.kron(np.expand_dims(Iref, 2), np.ones([1, 1, size_hidden]))
    # print(E.shape)
    Iout_exp_4d = np.kron(np.expand_dims(Iout_exp_3d, 3), np.ones([1, 1, 1, size_out]))
    Iout_exp_sum_4d = np.kron(np.expand_dims(Iout_exp_sum_3d, 3), np.ones([1, 1, 1, size_out]))
    # print(Isum.shape)
    Iref_4d = np.kron(np.expand_dims(Iref_3d, 3), np.ones([1, 1, 1, size_out]))
    # print(E.shape)
    # Iref_4d=Iref.swapaxes(1,3)
    # print(Iref_4d.shape)
    # Iout_exp_4d=Iout_exp_4d.swapaxes(0,3)
    # Isum=Isum.swapaxes(0,3)
    # E=E.swapaxes(0,1)
    # Isum=Isum.swapaxes(0,1)
    # print(Iout_exp_4d.shape,Iout_exp_sum_4d.shape)
    gra_Iout_exp_wjk_4d = Iout_exp_4d * gra_Iout_wjk.swapaxes(1, 2)
    gra_Isum_wjk_3d = np.sum(gra_Iout_exp_wjk_4d, 3)
    gra_Isum_wjk_4d = np.kron(np.expand_dims(gra_Isum_wjk_3d, 3), np.ones([1, 1, 1, size_out]))
    # print(gra_Isum_wjk_4d.shape)
    # gra_Isum_wjk_4d=gra_Isum_wjk_4d.swapaxes(2,3)
    # gra_Isum_wjk_4d=gra_Isum_wjk_4d.swapaxes(1,2)
    # print(gra_Isum_wjk_4d.shape)
    X = np.expand_dims((-1 / (Iout_softmax * Iref)), 2)
    X = np.kron(X, np.ones([1, 1, size_hidden]))
    X = np.expand_dims(X, 3)
    X = np.kron(X, np.ones([1, 1, 1, size_out]))
    # print(X.shape,gra_Iout_exp_wjk_4d.shape,Iout_exp_sum_4d.shape,Iout_exp_4d.shape,gra_Isum_wjk_4d.shape)

    X[X == inf] = 0
    X[X == -inf] = 0
    gra_CL_wjk = X * ((gra_Iout_exp_wjk_4d * Iout_exp_sum_4d - Iout_exp_4d * gra_Isum_wjk_4d) / Iout_exp_sum_4d ** 2)

    # print(gra_CL_wjk)
    gra_CL_wjk = np.sum(gra_CL_wjk, 1)
    gra_CL_wjk = np.sum(gra_CL_wjk, 0)
    # print(gra_CL_wjk)
    # print(gra_CL_wjk.shape)

    return gra_CL_wjk

def get_gra_Iout_wij(A, B, C, wij, wjk, V2h, Vin):
    # 计算V2h对wij的梯度

    # 增加一个维度，表示i 共784
    A = np.kron(np.expand_dims(A, 2), np.ones([1, 1, size_in]))
    B = np.kron(np.expand_dims(B, 2), np.ones([1, 1, size_in]))
    C = np.kron(np.expand_dims(C, 2), np.ones([1, 1, size_in]))
    V2h = np.kron(np.expand_dims(V2h[:, 0:size_hidden_initial], 2), np.ones([1, 1, size_in]))
    # 将Vin增加一个维度,并调整到与ABC一致
    Vin = np.kron(np.expand_dims(Vin, 2), np.ones([1, 1, size_hidden_initial]))
    Vin = Vin.swapaxes(1, 2)
    # print(A.shape, B.shape, C.shape, wij.shape, wjk.shape, V2h.shape, Vin.shape)
    # 计算梯度，此时为三维矩阵
    gra_V_wij = 10 * (np.ones(A.shape) - (np.tanh(10 * (A / B - (C + B) / B * V2h))) ** 2) * (
                Vin / B - A / (B) ** 2 + V2h * C / (B) ** 2) / (
                            C + 10 * (np.ones(A.shape) - (np.tanh(10 * (A / B - (C + B) / B * V2h))) ** 2) * (
                                C + B) / B)

    # 扩展到四维矩阵
    gra_Iout_wij_4d=np.zeros((size_batch, size_in, size_hidden_initial, size_out))
    for i in range(size_hidden_initial):
        # print (np.kron(np.expand_dims(gra_V_wij[:, i, :], 2), wjk[i, :]).shape)

        gra_Iout_wij_4d[:, :, i, :] = np.kron(np.expand_dims(gra_V_wij[:, i, :], 2), wjk[i, :])+ np.kron(np.expand_dims(-gra_V_wij[:, i, :], 2), wjk[i + size_hidden_initial, :])
    # print('shape2' ,   gra_Iout_wij_4d.shape)


    # 调整为 size_batch, size_in, size_hidden, size_out的顺序
    # gra_Iout_wij_4d = gra_Iout_wij_4d.swapaxes(1, 2)
    # gra_Iout_wij_4d = gra_Iout_wij_4d.swapaxes(1, 3)
    #print('gra_Iout_wij_4d',gra_Iout_wij_4d.shape)
    return gra_Iout_wij_4d.swapaxes(1,3)


def get_gra_L_wij(Iout, Iref, gra_Iout_wij):
    # print(gra_Iout_wij.shape)
    Iout = np.kron(np.expand_dims(Iout, 2), np.ones([1, 1, size_hidden_initial]))

    Iref = np.kron(np.expand_dims(Iref, 2), np.ones([1, 1, size_hidden_initial]))
    # print(E.shape)
    Iout = np.kron(np.expand_dims(Iout, 3), np.ones([1, 1, 1, size_in]))

    # print(Isum.shape)
    Iref = np.kron(np.expand_dims(Iref, 3), np.ones([1, 1, 1, size_in]))
    # print(E.shape)
    # Iref = Iref.swapaxes(1, 2)
    # # print(Iref.shape)
    # Iout = Iout.swapaxes(1, 2)

    gra_L_wij = (Iout - Iref) * gra_Iout_wij
    # print(gra_L_wij.shape)
    gra_L_wij = np.sum(gra_L_wij, 0)
    # print(gra_L_wij.shape)
    gra_L_wij = np.sum(gra_L_wij, 0)
    return gra_L_wij.T

def get_gra_Iout_wjk (A, B, C, wjk, V2h):


  #计算第二部分，得到[size_batch, size_hidden, size_out, size_out]的矩阵
  # gra_V_wjk= np.ones(A.shape)-((np.tanh(A/B-(C+B)/B*V2h))**2)*(-V2h/B)-V2h
  # gra_V_wjk2= ((np.ones(A.shape)-(np.tanh(A/B-(C+B)/B*V2h))**2)*((C+B)/B)+C)
  #print(gra_V_wjk.shape, gra_V_wjk2)
  grad_V_wjk= (10*(np.ones(A.shape)-(np.tanh(10*(A/B-(C+B)/B*V2h)))**2)*(V2h/B)+V2h)/(10*(np.ones(A.shape)-(np.tanh(10*(A/B-(C+B)/B*V2h)))**2)*((C+B)/B)+C)
  #print(grad_V_wjk.shape)
  # grad_V_wjk=grad_V_wjk.reshape(size_batch, size_hidden)
  #print(grad_V_wjk.shape,size_out)

  grad_V_wjk=np.expand_dims(grad_V_wjk, axis=2)
  #print(grad_V_wjk.shape)
  x=np.ones([1,1,size_out])
  #print((np.ones([1,1,size_out])).shape)
  grad_V_wjk=np.kron(grad_V_wjk,np.ones([1,1,size_out]))
  gradi_V_wjk=np.zeros([size_batch,size_hidden_initial,size_hidden_initial,size_out])
  #print(gradi_V_wjk.shape,grad_V_wjk.shape)
  for i in range(size_hidden_initial):
    gradi_V_wjk[:,i,i,:]=grad_V_wjk[:,i,:]
  #print(gradi_V_wjk.shape,grad_V_wjk.shape)
  gradi_V_wjk=gradi_V_wjk.swapaxes(2,3)
  #print(gradi_V_wjk.shape,grad_V_wjk.shape)
  part2=np.dot(gradi_V_wjk,wjk[0:size_hidden_initial,:])+np.dot(-gradi_V_wjk,wjk[size_hidden_initial:size_hidden_initial*2,:])
  # print(part2.shape)
  #第一个sizeout是权重，第二个sizeout是电流

    #计算第一部分，仅和V2h相关，最终是要得到一个[size_batch, size_hidden, size_out, size_out]的矩阵
  #V2h_neg=V2h.reshape(size_batch,size_hidden)
  part1_2d=np.expand_dims(V2h_neg,axis=2)
  #print(part1_2d.shape)
  part1_3d=np.kron(part1_2d,np.ones([1,1,size_out]))
  #print(part1_3d.shape)
  part1_4d=np.zeros([size_batch,size_hidden,size_out,size_out])
  #print(part1_4d.shape)
  for i in range(size_out):
    part1_4d[:,:,i,i]=part1_3d[:,:,i]
  # print(part1_4d.shape)
  #print(part1_4d[10,0,0,0])

  #final
  part1_4d[:,0:size_hidden_initial,:,:]=part1_4d[:,0:size_hidden_initial,:,:]+part2
  return (part1_4d)

def get_gra_L_wjk(Iout, Iref, gra_Iout_wjk):
    # print(gra_Iout_wjk.shape)
    Iout = np.kron(np.expand_dims(Iout, 2), np.ones([1, 1, size_hidden]))

    Iref = np.kron(np.expand_dims(Iref, 2), np.ones([1, 1, size_hidden]))
    # print(E.shape)
    Iout = np.kron(np.expand_dims(Iout, 3), np.ones([1, 1, 1, size_out]))

    # print(Isum.shape)
    Iref = np.kron(np.expand_dims(Iref, 3), np.ones([1, 1, 1, size_out]))
    # print(E.shape)
    Iref = Iref.swapaxes(1, 2)
    # # print(Iref.shape)
    Iout = Iout.swapaxes(1, 2)

    gra_L_wjk = (Iout - Iref) * gra_Iout_wjk
    # print(gra_L_wjk.shape)
    gra_L_wjk = np.sum(gra_L_wjk, 0)
    # print(gra_L_wij.shape)
    gra_L_wjk = np.sum(gra_L_wjk, 1)
    return gra_L_wjk


A= np.ones ( [1, 1] )
B=np.ones ( [1, 1] )
C=np.ones ( [1, 1] )
D=np.ones ( [1, 1] )
m=int(0)
n=int(0)

def f2(x):
    return np.tanh(10 * (A[m, n] / B[m, n] - (C[m, n] + B[m, n]) / B[m, n] * x)) - C[m, n] * x

A= np.ones ( [10] )
B=np.ones ( [10] )
C=np.ones ( [10] )
def f1(x):
    return np.tanh(10 * (A[m] / B[m] - (C[m] + B[m]) / B[m] * x)) - C[m] * x

Vin_txt=[]
Vout5_txt=[]
Vout10_txt=[]
Vout30_txt=[]
lables_txt=[]
for i in range (40):
    if i == 0:
        Vin=np.loadtxt(str(i)+'.csv', skiprows=1, delimiter=',')
        Vin=np.delete(Vin,[0],axis=1)
    else:
        Vin_txt=(np.loadtxt(str(i)+'.csv', skiprows=1, delimiter=','))
        Vin=np.vstack((Vin,np.delete(np.array(Vin_txt),[0],axis=1)))

    lables_txt.append(np.loadtxt(str(i)+'_output.csv', dtype=str,skiprows=1, delimiter=',',usecols=[0]))
    Vout5_txt.append(np.loadtxt(str(i) + '_output.csv', dtype=str,skiprows=1, delimiter=',', usecols=[1]))
    Vout10_txt.append(np.loadtxt(str(i) + '_output.csv', dtype=str, skiprows=1, delimiter=',', usecols=[2]))
    Vout30_txt.append(np.loadtxt(str(i) + '_output.csv', dtype=str, skiprows=1, delimiter=',', usecols=[3]))
lables=[]
ratio=1


for i in lables_txt:
    for j in i:
        if j == 'high-risk':
            lables.append(2*ratio)
        if j == 'medium-risk':
            lables.append(1*ratio)
        if j == 'medium-risk':
            lables.append(0*ratio)
Vout5=[]
for i in Vout5_txt:
    for j in i:
         Vout5.append(float(j.strip('%')))

Vout10=[]
for i in Vout10_txt:
    for j in i:
         Vout10.append(float(j.strip('%')))

Vout30=[]
for i in Vout30_txt:
    for j in i:
         Vout30.append(float(j.strip('%')))
# print(Vout5)
# print(Vout10)
# print(Vout30)
# print(lables)

#此处导入数据！！！！！
lables=Vout5
lables=np.array(lables)/10

#归一化Vin
# print(Vin.shape)
# print(Vin.max(0)-Vin.min(0))
Vin=10*(Vin - Vin.min(0)) / (Vin.max(0) - Vin.min(0))

epochmax=3
size_batch=3000
size_in_initial=Vin.shape[1]
size_in=size_in_initial*2+2

size_out=1

size_hidden_initial=20

size_hidden=size_hidden_initial*2+2

wij=np.random.random((size_in,size_hidden_initial))/size_in

wjk=np.random.random((size_hidden,size_out))/size_hidden



# wij=np.loadtxt('wij_tecent_10times')
# wjk=np.loadtxt('wjk_tencent_10times')
# wjk=np.expand_dims(wjk,1)
# print(wjk.shape)


Vin_train=Vin[0:size_batch,:]
Vin_all=Vin

# Vin_train=np.hstack([Vin_train,Vin_add])
# Vin_train=np.hstack([Vin_train,-Vin_add])

Iout=np.zeros(shape=(size_batch,size_out)) #定义一个batch的输出
Iref=lables[0:size_batch]
Iref=np.array(Iref)
Iref=Iref.T#Iref=np.zeros(shape=(size_batch,size_out)) # 定义一个batch的参考
Iref=np.expand_dims(Iref,1)
# print("Iref.shape",Iref.shape)
V2h=np.zeros(shape=(size_batch,size_hidden_initial))

A=np.zeros(shape=(size_batch,size_hidden_initial))
B=np.zeros(shape=(size_hidden_initial))
C=np.zeros(shape=(size_hidden_initial))
D = np.zeros(A.flatten().shape)
L=0

# L=np.loadtxt('L_tecent_10times')
CL=0

# wij=np.loadtxt('wij_tecent')
# wjk=np.loadtxt('wjk_tencent')
# wjk=np.expand_dims(wjk,1)
# L=np.loadtxt('L_tecent')

#定义步长
stepsize=0.00001



for epo in range(epochmax):
    stepsize = max(stepsize / 10, 0.0001)
    starttime = datetime.datetime.now()


    for x in range(int(Vin_train.shape[0] / size_batch)):
        # 准备
        Vin = Vin_train[x * size_batch:x * size_batch + size_batch, :]
        Iref =Iref[x * size_batch:x * size_batch + size_batch,:]

        # Vin = np.hstack((Vin, -Vin))

        Vin_add = np.ones([size_batch, 1])
        Vin=np.hstack([Vin,-Vin])
        Vin = np.hstack([Vin, Vin_add])
        Vin = np.hstack([Vin, -Vin_add])
        # 准备
        B = np.sum(wij, axis=0)
        C = np.sum(wjk[0:size_hidden_initial, :], axis=1)
        A = Vin.dot(wij)

        B = B - np.zeros(A.shape)  # 因为B，C维度要低于A，需要补充B，为矩阵求解做准备
        C = C - np.zeros(A.shape)  # 因为B，C维度要低于A，需要补充B，为矩阵求解做准备
        # A=np.reshape(A, (1,-1))
        # A = A.flatten()  # 转化为一维矩阵，为矩阵求解做准备
        # B = B.flatten()  # 转化为一维矩阵，为矩阵求解做准备
        # C = C.flatten()  # 转化为一维矩阵，为矩阵求解做准备


        # D=np.zeros(A.shape) #设置初始猜测值
        # print(A.shape,B.shape,C.shape,D.shape)
        # print(type(A),type(B),type(C),type(D))

        # 求解非线性方程
        print("time", datetime.datetime.now() - starttime)
        starttime = datetime.datetime.now()
        # V2h = fsolve(f2, D)
        for m in range (size_batch):
            for n in range (size_hidden_initial):
                V2h[m,n]=fsolve(f2,0)
        # D = V2h
        # endtime = datetime.datetime.now()

        # print(V2h.shape)
        # print(max(A),max(B),max(C))

        # 求出softmax前的输出电流
        # print(V2h.shape,wjk.shape)
        # V2h = np.reshape(V2h, (size_batch, size_hidden_initial))

        V2h_neg = np.hstack((V2h, -V2h))

        V2h_add = np.ones([size_batch, 1])
        # print(V2h_neg.shape)
        V2h_neg = np.hstack((V2h_neg, V2h_add))
        V2h_neg = np.hstack((V2h_neg, -V2h_add))

        # V2h=V2h.T
        # print(V2h.shape,wjk.shape)
        Iout = np.dot(V2h_neg, wjk)
        # print(Iout)
        # 求出softmax前的输出电流
        # print(Iout.shape)

        # Iout_exp = np.exp(Iout)
        # Iout_exp_sum = np.sum(Iout_exp, 1)
        # # print(Isum.shape)
        # Iout_softmax = Iout_exp / (np.kron(np.expand_dims(Iout_exp_sum, 1), np.ones([1, 10])))
        # Iref=Iout_softmax
        # print(Iref.shape)
        # print(Iout.T.shape)

        L = np.append(L, np.sum(0.5 * (Iout - Iref) ** 2))
        # L= np.sum(0.5*(Iout.T-Iref)**2)
        print(L)
        # CL = np.append(CL, cross_entropy(Iref, Iout_softmax))
        # print(CL)
        print("stepsize", stepsize)

        # 先将ABC调成size_batch,size_hiden的维度
        # A = A.reshape(size_batch, size_hidden_initial)
        # B = B.reshape(size_batch, size_hidden_initial)
        # C = C.reshape(size_batch, size_hidden_initial)
        # # V2h=V2h.T

        gra_Iout_wij = get_gra_Iout_wij(A, B, C, wij, wjk, V2h, Vin)
        # print(gra_Iout_wij.shape)
        # print(wij.shape)
        # print(Iout.shape)
        # print(Iref.shape)
        gra_L_wij=get_gra_L_wij(Iout,Iref,gra_Iout_wij)
        # print(gra_L_wij.shape)
        # print(gra_Iout_wij.shape)
        # print(gra_Iout_wij[0,0,0,:])

        # ∂Vjh2/∂wij ={1-tanh2[Aj/Bj-（Cj+Bj）/Bj Vjh2]} { Vi/Bj-Aj/(Bj)2j+Vjh2 Cj/(Bj)2}/{ Cj +{1-tanh2[Aj/Bj-（Cj+Bj）/Bj Vjh2]} [(Cj+Bj)/Bj ]  }

        # gra_CL_wij = get_gra_CL_wij(Iout_softmax, Iout_exp, Iout_exp_sum, Iref, gra_Iout_wij)
        # print(gra_L_wij.shape)
        wij = wij - stepsize * gra_L_wij
        # wij=np.maximum(wij, 0)
        # print(wij.shape)
        # print(gra_L_wij[0,:])

        gra_Iout_wjk = get_gra_Iout_wjk(A, B, C, wjk, V2h)
        # print(gra_Iout_wjk.shape)
        gra_L_wjk=get_gra_L_wjk(Iout,Iref,gra_Iout_wjk)
        # gra_CL_wjk = get_gra_CL_wjk(Iout_softmax, Iout_exp, Iout_exp_sum, Iref, gra_Iout_wjk)
        # print(gra_L_wjk.shape)
        wjk = wjk - stepsize * gra_L_wjk
        # wjk=np.maximum(wjk, 0)
        # print(wjk.shape)
        wij = np.maximum(wij, 0)
        wjk = np.maximum(wjk, 0)


    # np.savetxt('CL_128N',CL)

    Vin_test = Vin_all[size_batch:Vin_all.shape[0], :]
    lables_test=lables[size_batch:Vin_all.shape[0]]
    lables_test = np.array(lables_test)
    # print("lables_test.shape", lables_test.shape,lables_test)
    # lables_test = lables_test.T  # Iref=np.zeros(shape=(size_batch,size_out)) # 定义一个batch的参考
    # Iref = np.expand_dims(Iref, 1)
    # B = np.sum(wij, axis=0)
    # C = np.sum(wjk[0:size_hidden_initial, :], axis=1)
    # A = np.ones(B.shape)
    #
    # # A=np.reshape(A, (1,-1))
    # A = A.flatten()  # 转化为一维矩阵，为矩阵求解做准备
    # B = B.flatten()  # 转化为一维矩阵，为矩阵求解做准备
    # C = C.flatten()  # 转化为一维矩阵，为矩阵求解做准备
    B = np.sum(wij, axis=0)
    C = np.sum(wjk[0:size_hidden_initial, :], axis=1)
    V2h_test = np.zeros([Vin_test.shape[0], size_hidden_initial])
    right = 0
    wrong = 0
    accurency = 0
    for x in range(Vin_test.shape[0]):
        A = np.dot(Vin[x, :], wij)
        for m in range(size_hidden_initial):
            V2h_test[x,m] = fsolve(f1, 0)
        V2h_neg = np.hstack((V2h_test[x,:], -V2h_test[x,:]))
        # print(V2h.shape)
        V2h_add = np.ones([1])

        V2h_neg = np.hstack((V2h_neg, V2h_add))
        V2h_neg = np.hstack((V2h_neg, -V2h_add))
        Iout = np.dot(V2h_neg, wjk)
        # print("Iout.shape",Iout)
        Iout = np.sum(Iout)

        # if Iout< 0.5*ratio:
        #     Iout=0
        #
        # if 0.5*ratio<=Iout<= 1.5*ratio:
        #     Iout=1*ratio
        #
        # if Iout>1.5*ratio:
        #     Iout=2*ratio
        #
        # if Iout-lables_test[x]==0:
        #     right=right+1
        # if Iout-lables_test[x]!=0:
        #     wrong=wrong+1
        if int(x/100)-x/100==0:
            print (Iout, lables_test[x])
    if int(epo/1000)-epo/1000==0:
        np.savetxt('wij_tecent_100times', wij)
        np.savetxt('wjk_tencent_100times', wjk)
        np.savetxt('L_tecent_100times',L)
    # accurency = np.append(accurency, right / (right + wrong))
    # np.savetxt('accurency_128N', accurency)
    # print("all:", right + wrong, "right", right, "wrong", wrong)
    # print("accurrency", right / (right + wrong))
    # np.savetxt('accurency_tecent_class', accurency)