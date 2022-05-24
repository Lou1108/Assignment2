# author:   Lena Luisa Feiler
# ID:       i6246119


# https://learnopencv.com/eigenface-using-opencv-c-python/
# https://machinelearningmastery.com/face-recognition-using-principal-component-analysis/
import os
import cv2
import numpy as np


def do_pca(images):
    size = images[0].shape  # image size, all images have the smae size

    # step 1 and 2: vectorize samples and create data matrix
    # Create data matrix for PCA
    data = create_data_matrix(images)

    # using the data matrix, compute the mean and the eigenvectors for each image
    # already takes care of step 4: Covariance matrix and gives back the eigenvectors (step 5)
    # eigenvectors has shape (n_img, w*h*3)
    mx_mean, A_eigenVectors = cv2.PCACompute(data, mean=None)

    avg_faces = mx_mean.reshape(size)
    print("mean: ", avg_faces.shape)

    eigen_faces = [];
    count = 1  # counter to save all images
    for eigenVector in A_eigenVectors:
        eigenFace = eigenVector.reshape(size)  # reshaping to original format to retrieve eigen face
        eigen_faces.append(eigenFace)
        disp_img = cv2.normalize(np.abs(eigenFace), None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite("iivp/resultPictures/exercise4/" + str(count) + ".jpg", disp_img)
        count += 1

    cv2.imwrite('iivp/resultPictures/exercise4/average face.jpg', avg_faces)

    return mx_mean, A_eigenVectors


#################################### also include mirrored images!!!  #################################################
# reads in all images from a specified folder
def read_images_from_folder(folder_name):
    # reference: https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
    images = []
    for image in os.listdir(folder_name):  # find images in selected folder
        img = cv2.imread(os.path.join(folder_name, image))
        if img is not None:
            if type(img) == np.ndarray:  # only read if it is an image, prevents errors
                img = cv2.resize(img, (717, 717))  # let all images have the same size
                images.append(img)  # add to data array
    return images


# create one data matrix for all the input images
def create_data_matrix(data):
    num_img = len(data)
    size = data[0].shape  # size of one image, they all have the same size

    # data matrix will have dimension (size[0] * size[1] * size[2], num_images) = ( w  * h  * 3, numImages )
    data_matrix = np.zeros((num_img, size[0] * size[1] * size[2]),
                           dtype=np.float32)  # uint8

    for i in range(0, num_img):
        image = data[i].flatten()  # step 1: vectorize each image (will have dimension (3*w*h, 1)
        data_matrix[i, :] = image  # step 2: add vector to the data matrix

    return data_matrix


'''eigen_faces is a list/ array og eigenfaces that is used to construct the face
weights is a list of weights corresponding to the list of faces, 
that measures how much each is going to be used in the reconstruction '''


def createNewFace(avg_face, eigen_faces, y_weight, size):
    # Start with the mean image
    output_img = avg_face
    # len_eig_vector = eigen_faces.shape[1]
    len_eig_vector = len(eigen_faces)
    print("num: ", y_weight.shape)

    # Add the eigen faces with the weights
    for i in range(len_eig_vector):
        print("eigen_faces: ", eigen_faces[i].shape)
        print("weights: ", y_weight[i].shape)
        output_img = np.add(output_img, (eigen_faces[i] * y_weight[i]))

        print(output_img)

    print(output_img.shape)
    print(output_img-mx)
    # Display Result at 2x size
    cv2.imwrite('iivp/resultPictures/exercise4/newface.jpg', output_img.reshape(size))


# read in all images in directory
images_all = read_images_from_folder("iivp/pictures/pca")
data = create_data_matrix(images_all)

mx, A = do_pca(images_all)

weights_yi = np.dot(A, (data[0] - np.transpose(mx)))
createNewFace(mx, A[0:2], weights_yi[0:2], images_all[0].shape)
