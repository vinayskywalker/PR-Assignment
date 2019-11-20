import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from numpy import linalg as LA

file1 = 'data/train-images-idx3-ubyte'
file2 = 'data/train-labels-idx1-ubyte'

arr1 = idx2numpy.convert_from_file(file1)
arr2 = idx2numpy.convert_from_file(file2)

def principle_comp_analysis(data,k):
    cov_matrix = np.cov(data,rowvar=False)
    # print(cov_matrix.shape)
    eigen_values,eigen_vector = LA.eig(cov_matrix)
    eigen_values = np.real(eigen_values)
    eigen_vector = np.real(eigen_vector)
    print("Eigen Value and Eigen Vector")
    print(eigen_values.shape,eigen_vector.shape)
    # print(eigen_vector)
    eigen_vector_list = []

    for i in range(len(eigen_values)):
        eigen_vector_list.append([eigen_values[i],eigen_vector[:,i]])

    eigen_vector_list.sort(key = lambda x:x[0],reverse=True)
    # print(eigen_vector_list[0][1])

    K_top_eigen_vector = []

    for i in range(k):
        K_top_eigen_vector.append(eigen_vector_list[i][1])


    K_top_eigen_vector = np.asarray(K_top_eigen_vector)
    K_top_eigen_vector_transpose = np.transpose(K_top_eigen_vector)
    Z = np.dot(data,K_top_eigen_vector_transpose)
    Z = np.dot(Z,K_top_eigen_vector)
    img = np.reshape(Z[0,:],(28,28))
    plt.imshow(np.clip(img,0,255),'gray')
    plt.show()



if __name__ == '__main__':
    digit_list = [[] for i in range(10)]
    for i in range(len(arr1)):
        digit_list[arr2[i]].append(arr1[i].reshape(28*28))
    
    digit_list = np.asarray(digit_list)

    input_number = int(input("Enter the number 0-9\n"))
    k = int(input("Enter K:\n"))
    data = digit_list[input_number]
    data = np.asarray(data)
    # print(data.shape)
    mean_data = np.mean(data,axis=0,dtype=float)
    img = np.reshape(mean_data,(28,28))
    plt.imshow(img,'gray')
    plt.show()
    # print(mean_data[0])
    principle_comp_analysis(data,k)

