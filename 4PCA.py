# author:   Lena Luisa Feiler
# ID:       i6246119

import os
import cv2
import numpy as np


def do_pca(n_eigen_faces):
    # read in all images in directory
    images = read_images_from_folder("iivp/pictures/pca")

    # Size of images
    size = images[0].shape

    # Create data matrix for PCA.
    data = create_data_matrix(images) #np.array(create_data_matrix(images))

    # Compute the eigenvectors from the stack of images created
    print("Calculating PCA ", end="...")
    mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=n_eigen_faces)
    print("DONE")

    averageFace = mean.reshape(size)

######## ;?
    eigenFaces = [];

    for eigenVector in eigenVectors:
        eigenFace = eigenVector.reshape(size)
        eigenFaces.append(eigenFace)

    # Display result at 2x size
    output = cv2.resize(averageFace, (0, 0), fx=2, fy=2)
    cv2.imwrite('iivp/resultPictures/exercise3/BW_orangetree.jpg', output)


# https://learnopencv.com/eigenface-using-opencv-c-python/
# https://machinelearningmastery.com/face-recognition-using-principal-component-analysis/
def create_data_matrix(data):
    # create one data matrix for all the input images
    # size is ( w  * h  * 3, numImages ): width[0], height[1] of image in data set, [3] is for the 3 colour channels
    #
    print("Creating data matrix", end=" ... ")
    n_img = len(data)
    sz = data[0].shape
    data_matrix = np.zeros((n_img, sz[0] * sz[1] * sz[2]), dtype=np.float32)

    for i in range(0, n_img):
        image = data[i].flatten()
        data_matrix[i, :] = image

    print("DONE")
    print(data_matrix.shape)
    return data


# https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
def read_images_from_folder(folder_name):
    images = []
    for image in os.listdir(folder_name):
        print(image)
        img = cv2.imread(os.path.join(folder_name, image))
        if img is not None:
            if type(img) == np.ndarray:
                img = cv2.resize(img, (717, 717))
                print(img.shape)
                images.append(img)
    return images




do_pca(5)