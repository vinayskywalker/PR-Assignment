from array import *
from math import *
import numpy as np  
import matplotlib.pyplot as plt
import sys
from matplotlib import colors as mcolors


# M=500
# N=375
d=2

def graph(formula, x_range):  
    x = np.array(x_range)  
    y = formula(x)
    lines = plt.plot(y,x)
    # plt.axis([-20, 20, -20, 20])
    plt.setp(lines, color='r', linewidth=2.0)
    # plt.show()

def readDataSetWhole(fileName):
    f = open(fileName,"r")
    fl =f.readlines()
    N = int(ceil(len(fl)))
    # print(N)
    fl = fl[0:int(N)]
    dSet = [[0 for x in range(d)] for y in range(N)] # vector which consist of 2 features;
    i=0
    for lines in fl:
        lines=lines.split();
        for j in range(d):
            dSet[i][j] = float(lines[j])
        i=i+1
    f.close()
    return dSet 


def readDataSetTraining(fileName):
    f = open(fileName,"r")
    fl =f.readlines()
    N = int(ceil(0.75*len(fl)))
    # print(N)
    fl = fl[0:N]
    dSet = [[0 for x in range(d)] for y in range(N)] # vector which consist of 2 features;
    i=0
    for lines in fl:
        lines=lines.split();
        for j in range(d):
            dSet[i][j] = (lines[j])
        i=i+1
    f.close()
    return dSet


def readDataSetTesting(fileName):
    f = open(fileName,"r")
    fl =f.readlines()
    N = int(ceil(0.75*len(fl)))
    M = int(len(fl))
    # print(N, M)
    fl = fl[N:M]
    # print(len(fl))
    dSet = [[0 for x in range(d)] for y in range(len(fl))] # vector which consist of 2 features;
    i=0
    for lines in fl:
        lines=lines.split();
        for j in range(d):
            dSet[i][j] = (lines[j])
        i=i+1
    f.close()
    return dSet

def computeMean(X_class,N):
    x1=0
    x2=0
    for i in range(N):
        x1=x1+float(X_class[i][0])
        x2=x2+float(X_class[i][1])
    #Calculating Mean
    mean_x_class = [x1/N, x2/N];
    return mean_x_class

def computeCovariance(dSet,mean):
    cov_mat = [[0 for x in range(d)] for y in range(d)] # covariance matrix of size 2x2;
    for i in range(d):
        for j in range(d):
            cov_mat[i][j]=0
            for k in range(len(dSet)):
                cov_mat[i][j] += (float(dSet[k][i])-float(mean[i]))*(float(dSet[k][j])-float(mean[j]))
            cov_mat[i][j] /= len(dSet)-1
    return cov_mat

def calcW2(cov_matrix_inverse):
    w2 = [[0 for x in range(d)] for y in range(d)]
    for i in range(d):
        for j in range(d):
            w2[i][j] = cov_matrix_inverse[i][j]/(-2)
            
    return w2

def calcW1(mean,cov_matrix_inverse):
    w = [((mean[0]*cov_matrix_inverse[0][0])+(mean[1]*cov_matrix_inverse[1][0])), (mean[0]*cov_matrix_inverse[0][1])+(mean[1]*cov_matrix_inverse[1][1])]
    # print(w12)
    return w;

def calcW0(mean,cov_matrix_inverse,prior,det):
    t = [((mean[0]*cov_matrix_inverse[0][0])+(mean[1]*cov_matrix_inverse[1][0])), ((mean[0]*cov_matrix_inverse[0][1])+(mean[1]*cov_matrix_inverse[1][1]))]
    w0 = ((t[0]*mean[0])+(t[1]*mean[1]))/(-1*2)
    w0 += log(prior)
    w0 -= ((log(det))/2)
    return w0

def calcG(w2,w1,w0,x):
    x1 = float(x[0])
    x2 = float(x[1])
    # print(x1," ", x2);
    temp = [( (x1*w2[0][0]) + (x2*w2[1][0]) ), ( (x1*w2[0][1]) + (x2*w2[1][1]) )]
    g = ( (temp[0]*x1) + (temp[1]*x2) )
    
    g += ( (w1[0]*x1) + (w1[1]*x2) )

    g += w0
    # g=(w[0]*float(x[0]))+(w[1]*float(x[1]))+w0
    return g

def graph(formula, x_range):  
    x = np.array(x_range)  
    y = formula(x)
    lines = plt.plot(y,x)
    # plt.axis([-20, 20, -20, 20])
    plt.setp(lines, color='r', linewidth=2.0)
    # plt.show()

def computeCaseCCovariance(cov_mat_class):
    avg=0.0;
    for i in range(d):
        for j in range(d):
            avg += cov_mat_class[i][j]
    
    cov_mat_class[0][0] = cov_mat_class[1][1] = avg/4
    cov_mat_class[1][0] = cov_mat_class[0][1] = 0
    # print("--------------case c matrix")
    # print(cov_mat_class)
    return cov_mat_class

def calcDet(matrix):
    det = (matrix[0][0]*matrix[1][1]) - (matrix[0][1]*matrix[1][0])        
    return det

def inverseMatrix(cov_matrix_b):
    det = (cov_matrix_b[0][0]*cov_matrix_b[1][1]) - (cov_matrix_b[0][1]*cov_matrix_b[1][0])

    cov_matrix_b[0][1] = -1*cov_matrix_b[0][1]
    cov_matrix_b[1][0] = -1*cov_matrix_b[1][0]

    temp = cov_matrix_b[0][0]
    cov_matrix_b[0][0] = cov_matrix_b[1][1]
    cov_matrix_b[1][1] = temp;

    for i in range(d):
        for j in range(d):
            cov_matrix_b[i][j] /= det;
    
    print(cov_matrix_b)
    return cov_matrix_b

def computePrior(class_no):
    f = open("class1.txt","r")
    fl =f.readlines()
    c1 = int(ceil(0.75*len(fl)))
    f = open("class2.txt","r")
    fl =f.readlines()
    c2 = int(ceil(0.75*len(fl)))
    f = open("class3.txt","r")
    fl =f.readlines()
    c3 = int(ceil(0.75*len(fl)))
    t=c1+c2+c3
    ans=0.0
    if class_no==1:
        ans = (1.0*c1)/(1.0*t)
    elif class_no==2:
        ans = (1.0*c2)/(1.0*t)
    elif class_no==3:
        ans = (1.0*c3)/(1.0*t)
    return ans


def getRange():
    X1=readDataSetWhole("class1.txt")
    X2=readDataSetWhole("class2.txt")
    X3=readDataSetWhole("class3.txt")

    xmin=ymin=sys.maxsize
    xmax=ymax=-sys.maxsize

    for i in range(len(X1)):
        if X1[i][0] > xmax :
            xmax=X1[i][0]
        if X1[i][0] < xmin :    
            xmin=X1[i][0]
        
        if X1[i][1] > ymax :
            ymax=X1[i][1]
        if X1[i][1] < ymin :    
            ymin=X1[i][1]

    for i in range(len(X2)):
        if X2[i][0] > xmax :
            xmax=X2[i][0]
        if X2[i][0] < xmin :    
            xmin=X2[i][0]
        
        if X2[i][1] > ymax :
            ymax=X2[i][1]
        if X2[i][1] < ymin :    
            ymin=X2[i][1]

    for i in range(len(X3)):
        if X3[i][0] > xmax :
            xmax=X3[i][0]
        if X3[i][0] < xmin :    
            xmin=X3[i][0]
        
        if X3[i][1] > ymax :
            ymax=X3[i][1]
        if X3[i][1] < ymin :    
            ymin=X3[i][1]
        
    dSet = [[0 for x in range(2)] for y in range(2)]
    dSet[0][0]=xmin
    dSet[0][1]=xmax
    dSet[1][0]=ymin
    dSet[1][1]=ymax
    return dSet


def main():
    #########
    #Class 1#
    #########
    print ("Reading from file")
    X_class1=readDataSetTraining("Class1.txt")
    #Calculating Mean
    mean_x_class1 = computeMean(X_class1,len(X_class1))
    print(mean_x_class1[0]," --- ", mean_x_class1[1])
    #Calculating Covariance
    cov_mat_class1 = computeCovariance(X_class1,mean_x_class1)
    det_class1 = calcDet(cov_mat_class1)
    cov_mat_class1 = computeCaseCCovariance(cov_mat_class1)
    cov_mat_class1 = inverseMatrix(cov_mat_class1)
    print(cov_mat_class1[0][0], "  ", cov_mat_class1[0][1], "\n", cov_mat_class1[1][0], "  ", cov_mat_class1[1][1])

    print("\n--------------------------\n");

    #########
    #Class 2#
    #########
    #Reading from file
    X_class2=readDataSetTraining("Class2.txt")
    #Calculating Mean
    mean_x_class2 = computeMean(X_class2,len(X_class2))
    print(mean_x_class2[0]," --- ", mean_x_class2[1])
    #Calculating Covariance
    cov_mat_class2 = computeCovariance(X_class2,mean_x_class2)
    det_class2 = calcDet(cov_mat_class2)
    cov_mat_class2 = computeCaseCCovariance(cov_mat_class2)
    cov_mat_class2 = inverseMatrix(cov_mat_class2)
    print(cov_mat_class2[0][0], "  ", cov_mat_class2[0][1], "\n", cov_mat_class2[1][0], "  ", cov_mat_class2[1][1])
    
    print("\n--------------------------\n");

    #########
    #Class 3#
    #########
    #Reading from file
    X_class3=readDataSetTraining("Class3.txt")
    #Calculating Mean
    mean_x_class3 = computeMean(X_class3,len(X_class3))
    print(mean_x_class3[0]," --- ", mean_x_class3[1])
    #Calculating Covariance
    cov_mat_class3 = computeCovariance(X_class3,mean_x_class3)
    det_class3 = calcDet(cov_mat_class3)
    cov_mat_class3 = computeCaseCCovariance(cov_mat_class3)
    cov_mat_class3 = inverseMatrix(cov_mat_class3)
    print(cov_mat_class3[0][0], "  ", cov_mat_class3[0][1], "\n", cov_mat_class3[1][0], "  ", cov_mat_class3[1][1])

    print("\n--------------------------\n");
    
    #CASE - C (All covaraince matrix are not equal but diagonal)

    prior_class1 = computePrior(1)
    prior_class2 = computePrior(2)
    prior_class3 = computePrior(3)
    
    
    print(prior_class1, " ", prior_class2, " ", prior_class3)

    #Calc w2, w1 and w0
    #For Class 1
    w2_1 = calcW2(cov_mat_class1)
    w1_1 = calcW1(mean_x_class1,cov_mat_class1)
    w01 = calcW0(mean_x_class1,cov_mat_class1,prior_class1,det_class1)
    #For Class 2
    w2_2 = calcW2(cov_mat_class2)
    w1_2 = calcW1(mean_x_class2,cov_mat_class2)
    w02 = calcW0(mean_x_class2,cov_mat_class2,prior_class2,det_class2)
    #For Class 3
    w2_3 = calcW2(cov_mat_class3)
    w1_3 = calcW1(mean_x_class3,cov_mat_class3)
    w03 = calcW0(mean_x_class3,cov_mat_class3,prior_class3,det_class3)

    #Allocating Class to each point
    #For Class 1
    c1_1=c1_2=c1_3=0
    X1=readDataSetTesting("class1.txt")
    for i in range (len(X1)):
        g1=calcG(w2_1,w1_1,w01,X1[i])
        g2=calcG(w2_2,w1_2,w02,X1[i])
        g3=calcG(w2_3,w1_3,w03,X1[i])
        if g1==max(g1,g2,g3):
            c1_1+=1
            # print("1")
        elif g2==max(g1,g2,g3):
            c1_2+=1
            # print("2")
        elif g3==max(g1,g2,g3):
            c1_3+=1
            # print("3")
    print("c1_1 = ", c1_1, "c1_2 = ", c1_2, "c1_3 = ", c1_3)
    print ("End of Class 1")
    #For Class 2
    c2_1=c2_2=c2_3=0
    X2=readDataSetTesting("class2.txt")
    for i in range (len(X2)):
        g1=calcG(w2_1,w1_1,w01,X2[i])
        g2=calcG(w2_2,w1_2,w02,X2[i])
        g3=calcG(w2_3,w1_3,w03,X2[i])
        if g1==max(g1,g2,g3):
            c2_1+=1
            # print("1")
        elif g2==max(g1,g2,g3):
            c2_2+=1
            # print("2")
        elif g3==max(g1,g2,g3):
            c2_3+=1
            # print("3")
    print("c2_1 = ", c2_1, "c2_2 = ", c2_2, "c2_3 = ", c2_3)
    print ("End of Class 2")
    #For Class 3
    c3_1=c3_2=c3_3=0
    X3=readDataSetTesting("class3.txt")
    for i in range (len(X3)):
        g1=calcG(w2_1,w1_1,w01,X3[i])
        g2=calcG(w2_2,w1_2,w02,X3[i])
        g3=calcG(w2_3,w1_3,w03,X3[i])
        if g1==max(g1,g2,g3):
            c3_1+=1
            # print("1")
        elif g2==max(g1,g2,g3):
            c3_2+=1
            # print("2")
        elif g3==max(g1,g2,g3):
            c3_3+=1
            # print("3")
    print("c3_1 = ", c3_1, "c3_2 = ", c3_2, "c3_3 = ", c3_3)
    print ("End of Class 3")
    
    X=getRange()
    xmin=X[0][0]
    xmax=X[0][1]
    ymin=X[1][0]
    ymax=X[1][1]

    print ("xmin = ",xmin)
    print ("ymin = ",ymin)
    print ("xmax = ",xmax)
    print ("ymax = ",ymax)
    A = [[0 for x in range(2)] for y in range(2)]

    i=xmin
    while i<xmax :
        j=ymin
        while j<ymax:
            A[0]=i
            A[1]=j
            g1=calcG(w2_1,w1_1,w01,A)
            g2=calcG(w2_2,w1_2,w02,A)
            g3=calcG(w2_3,w1_3,w03,A)
            if g1==max(g1,g2,g3):
                plt.plot(i,j,color='#f6668f',marker='s')
            elif g2==max(g1,g2,g3):
                plt.plot(i,j,color='#33d7ff',marker='s')
            elif g3==max(g1,g2,g3):
                plt.plot(i,j,color='#75f740',marker='s')
            j+=30
        i+=30

    print("Plotting data points")

    X1=readDataSetTraining("Class1.txt")
    for i in range(len(X1)):
        plt.plot(X1[i][0],X1[i][1],'ro')

    X2=readDataSetTraining("Class2.txt")
    for i in range(len(X2)):
        plt.plot(X2[i][0],X2[i][1],'bo')

    X3=readDataSetTraining("Class3.txt")
    for i in range(len(X3)):
        plt.plot(X3[i][0],X3[i][1],'go')

    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.show()


if __name__== "__main__":
  main()

  # 1611275.5956010565