# author:   Lena Luisa Feiler
# ID:       i6246119


# https://learnopencv.com/eigenface-using-opencv-c-python/
# https://machinelearningmastery.com/face-recognition-using-principal-component-analysis/
import math
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def do_pca(images):
    size = images[0].shape  # image size

    # Create data matrix for PCA
    data = create_data_matrix(images)  # np.array(create_data_matrix(images))

    # Compute the eigenvectors from the stack of images created
    print("Calculating PCA ", end="...")
    mean, eigenVectors = cv2.PCACompute(data, mean=None)
    print("DONE")

    averageFace = mean.reshape(size)

    ######## ;?
    eigenFaces = [];
    count = 1
    for eigenVector in eigenVectors:
        eigenFace = eigenVector.reshape(size)
        print("range: ", print(eigenFace))
        eigenFaces.append(eigenFace)
        cv2.imwrite("iivp/resultPictures/exercise4/" + str(count) + ".jpg", eigenFace)
        count += 1

    display_eigen_faces(eigenFaces, size)

    cv2.imwrite('iivp/resultPictures/exercise4/average face.jpg', averageFace)


### also include mirrored images!!!
# https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
def read_images_from_folder(folder_name):
    images = []
    for image in os.listdir(folder_name):
        img = cv2.imread(os.path.join(folder_name, image))
        if img is not None:
            if type(img) == np.ndarray:
                img = cv2.resize(img, (717, 717))
                images.append(img)
    return images


def create_data_matrix(data):
    # create one data matrix for all the input images
    # size is ( w  * h  * 3, numImages ): width[0], height[1] of image in data set, [3] is for the 3 colour channels
    num_img = len(data)
    sz = data[0].shape
    data_matrix = np.zeros((num_img, sz[0] * sz[1] * sz[2]),
                           dtype=np.float32)  # data_matrix = np.zeros((size[0] * size[1] * size[2]), dtype=np.float32)

    for i in range(0, num_img):
        image = data[i].flatten()
        data_matrix[i, :] = image  # ot this:   data_matrix[:, i] = image

    return data_matrix


'''eigen_faces is a list/ array og eigenfaces that is used to construct the face
weights is a list of weights corresponding to the list of faces, 
that measures how much each is going to be used in the reconstruction '''
def createNewFace(avg_face, eigen_faces, weights):
    # Start with the mean image
    output = avg_face
    num_eigen_faces = len(eigen_faces)

    # Add the eigen faces with the weights
    for i in range(0, num_eigen_faces):
        output = np.add(output, eigen_faces[i] * weights[i])

    # Display Result at 2x size
    output = cv2.resize(output, (0, 0), fx=2, fy=2)
    cv2.imwrite('iivp/resultPictures/exercise4/new face.jpg', output)


def display_eigen_faces(eigen_faces, size):
    # displaying all eigenfaces
    fig, axes = plt.subplots(4, 5, sharex=True, sharey=True, figsize=(8, 10))
    for i in range(len(eigen_faces)):
        axes[i % 4][i // 4].imshow(eigen_faces[i].reshape(size))  # , cmap="gray")
    plt.show()


# read in all images in directory
images = read_images_from_folder("iivp/pictures/pca")
do_pca(images)
