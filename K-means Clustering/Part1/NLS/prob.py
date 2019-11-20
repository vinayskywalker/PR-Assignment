from array import *
from math import *
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import sys
import random


def read_file_data_set():

    f1 = open('Class1.txt','r')
    num_lines_f1 = sum(1 for line in open('Class1.txt'))
    # print("file 1 ",num_lines_f1)
    f1_1=f1.readlines()[0:num_lines_f1]

    f2 = open('Class2.txt','r')
    num_lines_f2 = sum(1 for line in open('Class2.txt'))
    # print("file 2 ",num_lines_f2)
    f2_2=f2.readlines()[0:num_lines_f2]

    f3 = open('Class3.txt','r')
    num_lines_f3 = sum(1 for line in open('Class3.txt'))
    # print("file 3 ",num_lines_f3)
    f3_3=f3.readlines()[0:num_lines_f3]

    data_set = [[0 for x in range(3)] for y in range(num_lines_f1+num_lines_f2+num_lines_f3)]

    i=0
    for lines in f1_1:
        lines=lines.split()
        for j in range(2):
            data_set[i][j] = float(lines[j])
        i=i+1

    
    for lines in f2_2:
        lines=lines.split()
        for j in range(2):
            data_set[i][j] = float(lines[j])
        i=i+1

    
    for lines in f3_3:
        lines=lines.split()
        for j in range(2):
            data_set[i][j] = float(lines[j])
        i=i+1

    return data_set




def distance_cal(x1,y1,x2,y2):
    distance = ((x1-x2)*(x1-x2)) + ((y1-y2)*(y1-y2))
    return distance

def main():

    # k=input("Enter the size of k-means\n")
    # k=int(k)
    k=3
    data_set=read_file_data_set()
    # print(len(data_set))

    random_mean = [[0 for i in range(2)] for j in range(k)]
    random_mean_size = [0 for i in range(k)]
    for i in range(k):
        num = random.randint(0,1500)
        # print(num)
        random_mean[i][0]=data_set[int(num)][0]
        random_mean[i][1]=data_set[int(num)][1]

    i=0
    for i in range(20):
        for j in range(len(data_set)):
            cur_min = sys.maxsize
            index_k_mean = 9
            for l in range(k):
                distance = distance_cal(random_mean[l][0],random_mean[l][1],data_set[j][0],data_set[j][1])
                if(distance<cur_min):
                    cur_min=distance
                    index_k_mean=l
                    # data_set[j][2] = index_k_mean
            data_set[j][2] = index_k_mean
            # print(i,"--------------",data_set[j])
        random_mean_size = [0 for p in range(k)]
        random_mean = [[0 for s in range(2)] for t in range(k)]
        # print(random_mean)
        for j in range(len(data_set)):
            for l in range(k):
                if(data_set[j][2]==l):
                    random_mean[l][0]=random_mean[l][0] + data_set[j][0]
                    random_mean[l][1]=random_mean[l][1] + data_set[j][1]
                    random_mean_size[l]=random_mean_size[l]+1
        

        for l in range(k):
            # print(random_mean_size)
            if(random_mean_size[l]==0):
                pass
            else:
                random_mean[l][0] = random_mean[l][0]/random_mean_size[l]
                random_mean[l][1] = random_mean[l][1]/random_mean_size[l]
        # print(random_mean[l])

        print(random_mean_size)
            


    colours=['red','green','blue']
    # print(type(colors))
    for i in range(len(data_set)):
        # print(data_set)
        if(data_set[i][2]==0):
            plt.plot(data_set[i][0],data_set[i][1],marker='o',markersize=1,color='red')
        if(data_set[i][2]==1):
            plt.plot(data_set[i][0],data_set[i][1],marker='o',markersize=1,color='green')
        if(data_set[i][2]==2):
            plt.plot(data_set[i][0],data_set[i][1],marker='o',markersize=1,color='blue')

    plt.show()

            

    




                



    
    
    









if __name__ == "__main__":
  main()