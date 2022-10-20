import numpy as np

if __name__ == '__main__':

    arr = np.array([[1, 2, 3],  [4, 5, 6]])

    arr1 = np.array([1, 2, 3])

    arr2 = np.array([4, 5, 6])

    arr = np.stack((arr1, arr2), axis=1)

    print(arr)