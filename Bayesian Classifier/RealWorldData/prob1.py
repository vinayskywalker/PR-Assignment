from array import *
from math import *
import numpy as np  
import matplotlib.pyplot as plt
import sys
from matplotlib import colors as mcolors
# M=500
# N=375
dimension=2


def read_file_data_set_train(filename):
  f=open(filename,'r')
  fl=f.readlines()
  N=int(ceil(0.75*len(fl)))
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
  fl=f.readlines()
  N=int(ceil(0.75*len(fl)))
  M=len(fl)
  # print(N,M)
  fl = fl[int(N):int(M)]
  # print(len(fl))
  data_set = [[0 for x in range(dimension)] for y in range(len(fl))]
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
  fl=f.readlines()
  N=int(len(fl))
  fl=fl[0:N]
  data_set=[[0 for x in range(dimension)] for y in range(N)]
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

  mean=[x/length,y/length]
  # print(x,y)
  return mean


def calculate_prior(CLASSNO):
  f=open("Class1.txt",'r')
  fl=f.readlines()
  c1=int(ceil(0.75*len(fl)))
  f=open("Class2.txt",'r')
  fl=f.readlines()
  c2=int(ceil(0.75*len(fl)))
  f=open("Class3.txt",'r')
  fl=f.readlines()
  c3=int(ceil(0.75*len(fl)))
  t=c1+c2+c3
  ans=0.0
  if(CLASSNO==1):
    ans=(c1/t)
  if(CLASSNO==2):
    ans=(c2/t)
  if(CLASSNO==3):
    ans=(c3/t)
  
  return ans
  





def calculate_variance(CLASS,mean_vector,i,j):
  sum=0.0
  for k in range(len(CLASS)):
    sum+=(CLASS[k][i]-mean_vector[i])*(CLASS[k][j]-mean_vector[j])
  
  sum=(sum/(len(CLASS)-1))
  return sum



def calculate_co_variance_matrix(CLASS,mean_vector):
  cov_mat=[[0 for x in range(dimension)] for y in range(dimension)]
  for i in range(dimension):
    for j in range(dimension):
      cov_mat[i][j]=calculate_variance(CLASS,mean_vector,i,j)

  
  return cov_mat


def calculate_w(sigma,mean_vector):
  sigma_inverse=(1/sigma)
  # print(sigma)
  # print(sigma_inverse)
  w=[sigma_inverse*mean_vector[0],sigma_inverse*mean_vector[1]]
  return w


def calculate_w0(prior_CLASS,mean_vector,sigma):
  w0=((mean_vector[0]*mean_vector[0])+(mean_vector[1]*mean_vector[1]))
  sigma_inverse=(1/sigma)
  w0=(w0*(sigma_inverse))
  w0=w0/2
  w0=(-w0)
  w0=w0+(log(prior_CLASS))
  return w0

def calculate_g(w,w0,x):
  g=(w[0]*x[0]+w[1]*x[1])+w0
  return g

def calculate_matrix_range():
  X1=read_file_data_set("Class1.txt")
  X2=read_file_data_set("Class2.txt")
  X3=read_file_data_set("Class3.txt")
  xmin=sys.maxsize
  ymin=sys.maxsize
  xmax=(-xmin)
  ymax=(-ymin)
  for i in range(len(X1)):
    if X1[i][0]>xmax:
      xmax=X1[i][0]
    if X1[i][0]<xmin:    
      xmin=X1[i][0]
        
    if X1[i][1]>ymax:
      ymax=X1[i][1]
    if X1[i][1]<ymin:    
      ymin=X1[i][1]

  for i in range(len(X2)):
    if X2[i][0]>xmax:
      xmax=X2[i][0]
    if X2[i][0]<xmin:    
      xmin=X2[i][0]
        
    if X2[i][1]>ymax:
      ymax=X2[i][1]
    if X2[i][1]<ymin:    
      ymin=X2[i][1]

  # print(xmin,xmax)
  # print(ymin,ymax)
  data_set=[[0 for x in range(2)] for y in range(2)]

  data_set=[[xmin,xmax],[ymin,ymax]]
  return data_set



def main():
    sigma=0.0
    # print("read the file")

    ###################    Class1    ###################
    X_CLASS_1=read_file_data_set_train("Class1.txt")
    #read_file_data_set_train("Class1.txt")
    #print(len(X_CLASS_1))
    mean_X_CLASS_1=calculate_mean(X_CLASS_1,len(X_CLASS_1))
    # print(mean_X_CLASS_1[0],mean_X_CLASS_1[1])
    cov_X_CLASS_1=calculate_co_variance_matrix(X_CLASS_1,mean_X_CLASS_1)
    for i in range(dimension):
      sigma+=cov_X_CLASS_1[i][i]
    

    # print(cov_X_CLASS_1)



    ###################    Class2    ###################

    X_CLASS_2=read_file_data_set_train("Class2.txt")
    #read_file_data_set_train("Class1.txt")
    mean_X_CLASS_2=calculate_mean(X_CLASS_2,len(X_CLASS_2))
    # print(mean_X_CLASS_1[0],mean_X_CLASS_1[1])
    cov_X_CLASS_2=calculate_co_variance_matrix(X_CLASS_2,mean_X_CLASS_2)
    for i in range(dimension):
      sigma+=cov_X_CLASS_2[i][i]
    # print(cov_X_CLASS_1)



    ###################    Class3    ###################

    X_CLASS_3=read_file_data_set_train("Class3.txt")
    #read_file_data_set_train("Class1.txt")
    mean_X_CLASS_3=calculate_mean(X_CLASS_3,len(X_CLASS_3))
    # print(mean_X_CLASS_1[0],mean_X_CLASS_1[1])
    cov_X_CLASS_3=calculate_co_variance_matrix(X_CLASS_3,mean_X_CLASS_3)
    for i in range(dimension):
      sigma+=cov_X_CLASS_3[i][i]
    # print(cov_X_CLASS_1)
    # print(sigma)


    sigma=sigma/6
    # print(sigma)
    


    #  CASE A             Covariance Matrix is equal to square of sigma

    cov_matrix_ALLCLASS=[[sigma,0.0],[0.0,sigma]]
    prior_CLASS1=calculate_prior(1)
    prior_CLASS2=calculate_prior(2)
    prior_CLASS3=calculate_prior(3)



    # calculate w and w0


    # for class 1
    w_CLASS1=calculate_w(sigma,mean_X_CLASS_1)
    w0_CLASS1=calculate_w0(prior_CLASS1,mean_X_CLASS_1,sigma)


    # for class 2
    w_CLASS2=calculate_w(sigma,mean_X_CLASS_2)
    w0_CLASS2=calculate_w0(prior_CLASS2,mean_X_CLASS_2,sigma)



    # for class 3
    w_CLASS3=calculate_w(sigma,mean_X_CLASS_3)
    w0_CLASS3=calculate_w0(prior_CLASS3,mean_X_CLASS_3,sigma)



    c1_1=0
    c1_2=0
    c1_3=0
    X1=read_file_data_set_test("Class1.txt")
    print(len(X1))
    for i in range(len(X1)):
      g1=calculate_g(w_CLASS1,w0_CLASS1,X1[i])
      g2=calculate_g(w_CLASS2,w0_CLASS2,X1[i])
      g3=calculate_g(w_CLASS3,w0_CLASS3,X1[i])
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
    for i in range(len(X2)):
      g1=calculate_g(w_CLASS1,w0_CLASS1,X2[i])
      g2=calculate_g(w_CLASS2,w0_CLASS2,X2[i])
      g3=calculate_g(w_CLASS3,w0_CLASS3,X2[i])
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
    for i in range(len(X3)):
      g1=calculate_g(w_CLASS1,w0_CLASS1,X3[i])
      g2=calculate_g(w_CLASS2,w0_CLASS2,X3[i])
      g3=calculate_g(w_CLASS3,w0_CLASS3,X3[i])
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
    x_min=range_matrix[0][0]-500
    x_max=range_matrix[0][1]+500
    y_min=range_matrix[1][0]-500
    y_max=range_matrix[1][1]+500
    # print(x_min,x_max)
    # print(y_min,y_max)
    A=[[0 for x in range(2)] for y in range(2)]
    i=x_min
    while(i<x_max):
      j=y_min
      while(j<y_max):
        A[0]=i
        A[1]=j
        g1=calculate_g(w_CLASS1,w0_CLASS1,A)
        g2=calculate_g(w_CLASS2,w0_CLASS2,A)
        g3=calculate_g(w_CLASS3,w0_CLASS3,A)
        if g1==max(g1,g2,g3):
          plt.plot(i,j,color='#f6668f',marker='s')
        elif g2==max(g1,g2,g3):
          plt.plot(i,j,color='#33d7ff',marker='s')
        elif g3==max(g1,g2,g3):
          plt.plot(i,j,color='#75f740',marker='s')
        j=j+30.0
      i=i+30.0
    
    
    

    X1=read_file_data_set_train("Class1.txt")
    for i in range(len(X1)):
      plt.plot(X1[i][0],X1[i][1],'ro')

    X2=read_file_data_set_train("Class2.txt")
    for i in range(len(X2)):
      plt.plot(X2[i][0],X2[i][1],'bo')

    X3=read_file_data_set_train("Class3.txt")
    for i in range(len(X3)):
      plt.plot(X3[i][0],X3[i][1],'go')
    

    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    plt.show()
    


































if __name__== "__main__":
  main()