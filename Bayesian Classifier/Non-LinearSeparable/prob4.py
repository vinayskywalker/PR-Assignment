from array import *
from math import *
import numpy as np  
import matplotlib.pyplot as plt
import sys
from matplotlib import colors as mcolors
M=500
N=375
dimension=2


def read_file_data_set_train(filename):
  f=open(filename,'r')
  fl=f.readlines()
    # print(N)
  fl = fl[0:int(N)]
  data_set = [[0 for x in range(dimension)] for y in range(N)] # 2d vector which consist of train file 
  i=0                                                          # only 75% data has been taken
  for lines in fl:
    lines=lines.split()
    for j in range(dimension):
        data_set[i][j] = float(lines[j])
    i=i+1
  f.close()

  # for i in range(N):
  #   for j in range(dimension):
  #     print(data_set[j])
      
  return data_set


def read_file_data_set_test(filename):
  f=open(filename,'r')
  fl=f.readlines()[N:M]
  data_set = [[0 for x in range(dimension)] for y in range(N)]
  i=0
  for lines in fl:
    lines=lines.split()
    for j in range(dimension):
      data_set[i][j]=float(lines[j])
    i=i+1
  f.close()
  return data_set


def read_file_data_set(filename):
  f=open(filename,'r')
  fl=f.readlines()[N:M]
  data_set=[[0 for x in range(dimension)] for y in range(M)]
  i=0
  for lines in fl:
    lines=lines.split()
    for j in range(dimension):
      data_set[i][j]=float(lines[j])
    i=i+1
  f.close()
  return data_set


def calculate_mean(CLASS,length):
  x=0.0
  y=0.0
  for i in range(length):
    x=x+CLASS[i][0]
    y=y+CLASS[i][1]

  mean=[x/N,y/N]
  # print(x,y)
  return mean


def calculate_variance(CLASS,mean_vector,i,j):
  sum=0.0
  for k in range(len(CLASS)):
    sum+=(CLASS[k][i]-mean_vector[i])*(CLASS[k][j]-mean_vector[j])
  
  sum=(sum/(N-1))
  return sum


def calculate_co_variance_matrix_general_case(cov1,cov2,cov3):
    cov_mat_general=[[0.0 for x in range(2)] for y in range(2)]
    for i in range(dimension):
        for j in range(dimension):
            cov_mat_general[i][j]=float(cov1[i][j]+cov2[i][j]+cov3[i][j])
        cov_mat_general[i][j]=float(cov_mat_general[i][j]/3)
    
    return cov_mat_general



def calculate_co_variance_matrix(CLASS,mean_vector):
  cov_mat=[[0 for x in range(dimension)] for y in range(dimension)]
  for i in range(dimension):
    for j in range(dimension):
      cov_mat[i][j]=calculate_variance(CLASS,mean_vector,i,j)

  
  return cov_mat

def calculate_inverse_matrix(cov_matrix_b):
  det=(cov_matrix_b[0][0]*cov_matrix_b[1][1])-(cov_matrix_b[0][1]*cov_matrix_b[1][0])

  cov_matrix_b[0][1]=(-1*cov_matrix_b[0][1])
  cov_matrix_b[1][0]=(-1*cov_matrix_b[1][0])

  temp=cov_matrix_b[0][0]
  cov_matrix_b[0][0]=cov_matrix_b[1][1]
  cov_matrix_b[1][1]=temp;

  for i in range(dimension):
    for j in range(dimension):
      cov_matrix_b[i][j] /= det;
  # print("------------inverse---------------------")
  # print(cov_matrix_b[0][0]," ",cov_matrix_b[0][1])
  # print(cov_matrix_b[1][0]," ",cov_matrix_b[1][1])
  return cov_matrix_b



def calcualte_det_co_var_mat(matrix):
    det=(matrix[0][0]*matrix[1][1])-(matrix[1][0]*matrix[0][1])
    return det




def calculate_w1(mean,cov):
    x1=(mean[0]*cov[0][0])+(mean[1]+cov[1][0])
    y1=(mean[0]*cov[0][1])+(mean[1]*cov[1][1])
    w1=[x1,y1]
    return w1

def calculate_w2(cov):
    w2=[[0 for x in range(dimension)] for y in range(dimension)]
    for i in range(dimension):
        for j in range(dimension):
            w2[i][j]=(cov[i][j]/(-2))
    
    return w2



def calculate_w0(mean,prior,cov,det):
    x1=(mean[0]*cov[0][0])+(mean[1]+cov[1][0])
    y1=(mean[0]*cov[0][1])+(mean[1]*cov[1][1])
    t=[x1,y1]
    w0=(t[0]*mean[0])+(t[1]*mean[1])
    w0=(w0/(-2))
    w0=w0+log(prior)
    w0=w0-(log(det)/2)
    return w0

def calculate_g(w1,w2,w0,x):
  x1=(x[0]*w2[0][0])+(x[1]*w2[1][0])
  y1=(x[0]*w2[0][1])+(x[1]*w2[1][1])
  t=[x1,y1]
  g=(x[0]*t[0])+(x[1]*t[1])
  g=g+(x[0]*w1[0])+(x[1]*w1[1])
  g=g+w0
  return g

def calculate_matrix_range():
  X1=read_file_data_set("Class1.txt")
  X2=read_file_data_set("Class2.txt")
  X3=read_file_data_set("Class3.txt")
  xmin=sys.maxsize
  ymin=sys.maxsize
  xmax=(-xmin)
  ymax=(-ymin)
  for i in range(M):
    if(xmax<X1[i][0]):
      xmax=X1[i][0]
    if(xmax<X2[i][0]):
      xmax=X2[i][0]
    if(xmax<X3[i][0]):
      xmax=X3[i][0]
    if(ymax<X1[i][1]):
      ymax=X1[i][1]
    if(ymax<X2[i][1]):
      ymax=X2[i][1]
    if(ymax<X3[i][1]):
      ymax=X3[i][1]


    if(xmin>X1[i][0]):
      xmin=X1[i][0]
    if(xmin>X2[i][0]):
      xmin=X2[i][0]
    if(xmin>X3[i][0]):
      xmin=X3[i][0]
    if(ymin>X1[i][1]):
      ymin=X1[i][1]
    if(ymin>X2[i][1]):
      ymin=X2[i][1]
    if(ymin>X3[i][1]):
      ymin=X3[i][1]
  

  # print(xmin,xmax)
  # print(ymin,ymax)
  data_set=[[0 for x in range(2)] for y in range(2)]

  data_set=[[xmin,xmax],[ymin,ymax]]
  return data_set



def main():
    # print("read the file")

    ###################    Class1    ###################
    X_CLASS_1=read_file_data_set_train("Class1.txt")
    #read_file_data_set_train("Class1.txt")
    #print(len(X_CLASS_1))
    mean_X_CLASS_1=calculate_mean(X_CLASS_1,len(X_CLASS_1))
    # print(mean_X_CLASS_1[0],mean_X_CLASS_1[1])
    cov_X_CLASS_1=calculate_co_variance_matrix(X_CLASS_1,mean_X_CLASS_1)
    # print("cov matrix------------------")
    # print(cov_X_CLASS_1[0][0],"  ",cov_X_CLASS_1[0][1])
    # print(cov_X_CLASS_1[1][0],"  ",cov_X_CLASS_1[1][1])
    # print("--------------------------")
    det_CLASS_1=calcualte_det_co_var_mat(cov_X_CLASS_1)
    cov_X_CLASS_1=calculate_inverse_matrix(cov_X_CLASS_1)
    # print(cov_X_CLASS_1[0][0],"  ",cov_X_CLASS_1[0][1])
    # print(cov_X_CLASS_1[1][0],"  ",cov_X_CLASS_1[1][1])
    

    # print(cov_X_CLASS_1)



    ###################    Class2    ###################

    X_CLASS_2=read_file_data_set_train("Class2.txt")
    #read_file_data_set_train("Class1.txt")
    mean_X_CLASS_2=calculate_mean(X_CLASS_2,len(X_CLASS_2))
    print(mean_X_CLASS_2[0],mean_X_CLASS_2[1])
    cov_X_CLASS_2=calculate_co_variance_matrix(X_CLASS_2,mean_X_CLASS_2)
    det_CLASS_2=calcualte_det_co_var_mat(cov_X_CLASS_2)
    cov_X_CLASS_2=calculate_inverse_matrix(cov_X_CLASS_2)
    print(cov_X_CLASS_2[0][0],"  ",cov_X_CLASS_2[0][1])
    print(cov_X_CLASS_2[1][0],"  ",cov_X_CLASS_2[1][1])
    # print(cov_X_CLASS_1)



    ###################    Class3    ###################

    X_CLASS_3=read_file_data_set_train("Class3.txt")
    #read_file_data_set_train("Class1.txt")
    mean_X_CLASS_3=calculate_mean(X_CLASS_3,len(X_CLASS_3))
    print(mean_X_CLASS_3[0],mean_X_CLASS_3[1])
    cov_X_CLASS_3=calculate_co_variance_matrix(X_CLASS_3,mean_X_CLASS_3)
    det_CLASS_3=calcualte_det_co_var_mat(cov_X_CLASS_3)
    cov_X_CLASS_3=calculate_inverse_matrix(cov_X_CLASS_3)
    print(cov_X_CLASS_3[0][0],"  ",cov_X_CLASS_3[0][1])
    print(cov_X_CLASS_3[1][0],"  ",cov_X_CLASS_3[1][1])
    # print(cov_X_CLASS_1)
    # print(sigma)
    # print(sigma)
    


    #  CASE D             Covariance Matrix of all Classes are equal

    prior_CLASS1=(0.75*M)/((0.75*M)+(0.75*M)+(0.75*M))     #calculate prior probability
    # print(prior_CLASS1)
    prior_CLASS2=prior_CLASS1
    prior_CLASS3=prior_CLASS1


    # calculate w and w0


    # for class 1
    w1_CLASS1=calculate_w1(mean_X_CLASS_1,cov_X_CLASS_1)
    w2_CLASS1=calculate_w2(cov_X_CLASS_1)
    w0_CLASS1=calculate_w0(mean_X_CLASS_1,prior_CLASS1,cov_X_CLASS_1,det_CLASS_1)


    # for class 2
    w1_CLASS2=calculate_w1(mean_X_CLASS_2,cov_X_CLASS_2)
    w2_CLASS2=calculate_w2(cov_X_CLASS_2)
    w0_CLASS2=calculate_w0(mean_X_CLASS_2,prior_CLASS2,cov_X_CLASS_2,det_CLASS_2)




    # for class 3
    w1_CLASS3=calculate_w1(mean_X_CLASS_3,cov_X_CLASS_3)
    w2_CLASS3=calculate_w2(cov_X_CLASS_3)
    w0_CLASS3=calculate_w0(mean_X_CLASS_3,prior_CLASS3,cov_X_CLASS_3,det_CLASS_3)




    c1_1=0
    c1_2=0
    c1_3=0
    X1=read_file_data_set_test("Class1.txt")
    for i in range(M-N):
      g1=calculate_g(w1_CLASS1,w2_CLASS1,w0_CLASS1,X1[i])
      g2=calculate_g(w1_CLASS2,w2_CLASS2,w0_CLASS2,X1[i])
      g3=calculate_g(w1_CLASS3,w2_CLASS3,w0_CLASS3,X1[i])
      if(g1==max(g1,g2,g3)):
        c1_1=c1_1+1
      elif(g2==max(g1,g2,g3)):
        c1_2=c1_2+1
      elif(g3==max(g1,g2,g3)):
        c1_3=c1_3+1
    
    print("c1==",c1_1,"c2==",c1_2,"c3==",c1_3)


    c2_1=0
    c2_2=0
    c2_3=0
    X2=read_file_data_set_test("Class2.txt")
    for i in range(M-N):
      g1=calculate_g(w1_CLASS1,w2_CLASS1,w0_CLASS1,X2[i])
      g2=calculate_g(w1_CLASS2,w2_CLASS2,w0_CLASS2,X2[i])
      g3=calculate_g(w1_CLASS3,w2_CLASS3,w0_CLASS3,X2[i])
      if(g1==max(g1,g2,g3)):
        c2_1=c2_1+1
      elif(g2==max(g1,g2,g3)):
        c2_2=c2_2+1
      elif(g3==max(g1,g2,g3)):
        c2_3=c2_3+1
    
    print("c1==",c2_1,"c2==",c2_2,"c3==",c2_3)
      


    c3_1=0
    c3_2=0
    c3_3=0
    X3=read_file_data_set_test("Class3.txt")
    for i in range(M-N):
      g1=calculate_g(w1_CLASS1,w2_CLASS1,w0_CLASS1,X3[i])
      g2=calculate_g(w1_CLASS2,w2_CLASS2,w0_CLASS2,X3[i])
      g3=calculate_g(w1_CLASS3,w2_CLASS3,w0_CLASS3,X3[i])
      if(g1==max(g1,g2,g3)):
        c3_1=c3_1+1
      elif(g2==max(g1,g2,g3)):
        c3_2=c3_2+1
      elif(g3==max(g1,g2,g3)):
        c3_3=c3_3+1
    
    print("c1==",c3_1,"c2==",c3_2,"c3==",c3_3)


    print("-----------------Precision-----------------")
    print("precision for Class 1")
    precision_class_1=(c1_1)/(c1_1+c1_2+c1_3)
    print(precision_class_1)
    precision_class_2=(c2_2)/(c2_1+c2_2+c2_3)
    precision_class_3=(c3_3)/(c3_1+c3_2+c3_3)
    print("precision for Class 2")
    print(precision_class_2)
    print("precision for Class 3")
    print(precision_class_3)

    range_matrix=calculate_matrix_range()
    # print(range_matrix[0][0],range_matrix[0][1])
    # print(range_matrix[1][0],range_matrix[1][1])
    x_min=range_matrix[0][0]-5
    x_max=range_matrix[0][1]+5
    y_min=range_matrix[1][0]-5
    y_max=range_matrix[1][1]+5
    # print(x_min,x_max)
    # print(y_min,y_max)
    A=[[0 for x in range(2)] for y in range(2)]
    i=x_min
    while(i<x_max):
      j=y_min
      while(j<y_max):
        A[0]=i
        A[1]=j
        g1=calculate_g(w1_CLASS1,w2_CLASS1,w0_CLASS1,A)
        g2=calculate_g(w1_CLASS2,w2_CLASS2,w0_CLASS2,A)
        g3=calculate_g(w1_CLASS3,w2_CLASS3,w0_CLASS3,A)
        if g1==max(g1,g2,g3):
          plt.plot(i,j,color='#f6668f',marker='s')
        elif g2==max(g1,g2,g3):
          plt.plot(i,j,color='#33d7ff',marker='s')
        elif g3==max(g1,g2,g3):
          plt.plot(i,j,color='#75f740',marker='s')
        j=j+0.13
      i=i+0.13
    
    
    

    X1=read_file_data_set_train("Class1.txt")
    for i in range(N):
      plt.plot(X1[i][0],X1[i][1],'ro')

    X2=read_file_data_set_train("Class2.txt")
    for i in range(N):
      plt.plot(X2[i][0],X2[i][1],'bo')

    X3=read_file_data_set_train("Class3.txt")
    for i in range(N):
      plt.plot(X3[i][0],X3[i][1],'go')
    

    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    plt.show()
    


































if __name__== "__main__":
  main()