from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import sys
import random


def distance_cal(x1,y1,z1,x2,y2,z2):
    distance = ((x1-x2)*(x1-x2)) + ((y1-y2)*(y1-y2)) + ((z1-z2)*(z1-z2))
    return distance

def main():


    filename = "input.jpg"
    image = Image.open(filename)
    print(np.shape(np.array(image)))
    image_data = list(image.getdata())
    data_set = [list(row) for row in image_data]
    # print(len(data_set))
    data_set = [x + [0] for x in data_set]

    


    
    k=7

    random_mean = [[0 for i in range(3)] for j in range(k)]
    random_mean_size = [0 for i in range(k)]
    for i in range(k):
        num = random.randint(10,15000)
        # print(num)
        random_mean[i][0]=data_set[int(num)][0]
        random_mean[i][1]=data_set[int(num)][1]
        random_mean[i][2]=data_set[int(num)][2]

    i=0
    for i in range(20):
        for j in range(len(data_set)):
            cur_min = sys.maxsize
            index_k_mean = 9
            for l in range(k):
                x1=random_mean[l][0]
                y1=random_mean[l][1]
                z1=random_mean[l][2]
                x2=data_set[j][0]
                y2=data_set[j][1]
                z2=data_set[j][1]
                distance = distance_cal(x1,y1,z1,x2,y2,z2)
                if(distance<cur_min):
                    cur_min=distance
                    index_k_mean=l
                    # data_set[j][2] = index_k_mean
            data_set[j][3] = index_k_mean
            # print(i,"--------------",data_set[j])
        random_mean_size = [0 for p in range(k)]
        random_mean = [[0 for s in range(3)] for t in range(k)]
        # print(random_mean)
        for j in range(len(data_set)):
            for l in range(k):
                if(data_set[j][3]==l):
                    random_mean[l][0]=random_mean[l][0] + data_set[j][0]
                    random_mean[l][1]=random_mean[l][1] + data_set[j][1]
                    random_mean[l][2]=random_mean[l][2] + data_set[j][2]
                    random_mean_size[l]=random_mean_size[l]+1
        

        for l in range(k):
            # print(random_mean_size)
            if(random_mean_size[l]==0):
                pass
            else:
                random_mean[l][0] = random_mean[l][0]/random_mean_size[l]
                random_mean[l][1] = random_mean[l][1]/random_mean_size[l]
                random_mean[l][2] = random_mean[l][2]/random_mean_size[l]
        # print(random_mean[l])

        # print(random_mean_size)
            


    colours=['red','green','blue','yellow']
    # # print(type(colors))
    data = np.zeros((1000,1000,3),dtype=np.uint8)
    for i in range(1000):
        for j in range(1000):
            data[i][j] = [0,0,0]

    
    # for i in range(392):
    #     for j in range(453):
    #         data[i][j] = random_mean[data_set[]]


    for i in range(len(data_set)):
        data[data_set[i][0]][data_set[i][1]]=random_mean[data_set[i][2]]
        print(data_set[i][0],data_set[i][1])

    img = Image.fromarray(data,'RGB')
    im.save("value.jpg")
    

    
    # for i in range(len(data_set)):
    #     for j in range(k):
    #         x=data_set[i][0]
    #         y=data_set[j][1]
    #         z=data_set[j][2]
    #         if(j==data_set[j][3]):
    #             plt.plot(x,y,z,color=)

    # plt.show()
    

            

    




                



    
    
    









if __name__ == "__main__":
  main()