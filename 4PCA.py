# author:   Lena Luisa Feiler
# ID:       i6246119

# references:
# https://learnopencv.com/eigenface-using-opencv-c-python/
# https://machinelearningmastery.com/face-recognition-using-principal-component-analysis/
# https://iq.opengenus.org/project-on-reconstructing-face/
import os
import cv2
import numpy as np


# calculates pca components
def do_pca(images):
    size = images[0].shape  # image size, all images have the smae size

    # step 1 and 2: vectorize samples and create data matrix
    data = create_data_matrix(images)  # Create data matrix for PCA

    # using the data matrix, compute the mean and the eigenvectors for each image
    # already takes care of step 4: Covariance matrix and gives back the eigenvectors (step 5)
    # eig_vec has shape (n_img, w*h*3)
    # mx_mean has shape (1, w*h*3)
    mx_mean, eig_vec = cv2.PCACompute(data, mean=None)

    average_face = mx_mean.reshape(size)  # average of all faces, shape (w, h, 3)
    cv2.imwrite('iivp/resultPictures/exercise4/average face.jpg', average_face)

    eigen_faces = [];
    count = 1  # counter to save all images
    for eigenVector in eig_vec:
        eigenFace = eigenVector.reshape(size)  # reshaping to original format to retrieve eigen face
        eigen_faces.append(eigenFace)

        display_img = cv2.normalize(np.abs(eigenFace), None, 0, 255, cv2.NORM_MINMAX)  # normalize to save it properly
        cv2.imwrite("iivp/resultPictures/exercise4/eigenfaces/eigenface_" + str(count) + ".jpg", display_img)
        count += 1

    return mx_mean, average_face, eig_vec, eigen_faces


# reads in all images from a specified folder
def read_images_from_folder(folder_name):
    # reference: https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
    images = []
    for image in os.listdir(folder_name):  # find images in selected folder
        img = cv2.imread(os.path.join(folder_name, image))
        if img is not None:
            if type(img) == np.ndarray:  # only read if it is an image, prevents errors
                img = cv2.resize(img, (500, 500))  # let all images have the same size
                images.append(img)  # add to data array

    return images


# create one data matrix for all the input images
def create_data_matrix(data):
    num_img = len(data)
    size = data[0].shape  # size of one image, they all have the same size

    # data matrix will have dimension (size[0] * size[1] * size[2], num_images) = ( w  * h  * 3, numImages )
    data_matrix = np.zeros((num_img, size[0] * size[1] * size[2]),
                           dtype=np.float32)

    for i in range(0, num_img):
        image = data[i].flatten()  # step 1: vectorize each image (will have dimension (3*w*h, 1)
        data_matrix[i, :] = image  # step 2: add vector to the data matrix

    return data_matrix


# face reconstruction
# reference: https://github.com/spmallick/learnopencv/blob/e355204b7e9657ce719208ab28879dd265a36a2e/ReconstructFaceUsingEigenFaces/reconstructFace.py
def rec_face(avg_face, mean, eig_vec, eig_faces, data, num_rec_vec):
    num_images = data.shape[0]  # number of faces
    reconstructedFaces = list()

    for i in range(0, num_images):
        final_output = avg_face
        for j in range(0, num_rec_vec):
            # calculate weights for the image: y = vi.(Ii - mx)
            weight = np.dot((data[i, :] - mean), eig_vec[j])
            # reconstruct face: Xr = Ii*y + mx
            final_output = final_output + eig_faces[j] * weight
        reconstructedFaces.append(final_output)  # save reconstructed face to list
        # saving images
        display_img = cv2.normalize(np.abs(final_output), None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite("iivp/resultPictures/exercise4/reconstructed/reconstructed_" + str(i+1) + ".jpg", display_img)


# face reconstruction
# reference: https://github.com/spmallick/learnopencv/blob/e355204b7e9657ce719208ab28879dd265a36a2e/ReconstructFaceUsingEigenFaces/reconstructFace.py
def rec_one_face(avg_face, mean, eig_vec, eig_faces, image, num_rec_vec, name):
    final_output = avg_face  # reconstruct face: Xr = Ii*y + mx
    for j in range(0, num_rec_vec):
        # calculate weights for the image: y = vi.(Ii - mx)
        weight = np.dot((image - mean), eig_vec[j])
        # reconstruct face: Xr = Ii*y + output
        final_output = final_output + eig_faces[j] * weight
    # saving images
    display_img = cv2.normalize(np.abs(final_output), None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("iivp/resultPictures/exercise4/reconstructed/reconstructed_" + name + ".jpg", display_img)


# read in all images in directory
images_all = read_images_from_folder("iivp/pictures/pca")
data_set = create_data_matrix(images_all)  # read pictures in from folder
mx, avgFace, eigenVectors, eigenFaces = do_pca(images_all)

################################################# exercise 2 #################################################
# reconstruct the 3 original faces using all eigen faces
rec_one_face(avgFace, mx, eigenVectors, eigenFaces, data_set[11], 18, "man_all")  # reconstruct the man
rec_one_face(avgFace, mx, eigenVectors, eigenFaces, data_set[9], 18, "old_all")  # reconstruct the old man
rec_one_face(avgFace, mx, eigenVectors, eigenFaces, data_set[14], 18, "woman_all")  # reconstruct the woman

# reconstruct the 3 original faces using all only 3 eigen faces
rec_one_face(avgFace, mx, eigenVectors, eigenFaces, data_set[11], 3, "man_3")  # reconstruct the man
rec_one_face(avgFace, mx, eigenVectors, eigenFaces, data_set[9], 3, "old_3")  # reconstruct the old man
rec_one_face(avgFace, mx, eigenVectors, eigenFaces, data_set[14], 3, "woman_3")  # reconstruct the woman

# reconstruct the 3 original faces using all only 3 eigen faces
rec_one_face(avgFace, mx, eigenVectors, eigenFaces, data_set[11], 1, "man_1")  # reconstruct the man
rec_one_face(avgFace, mx, eigenVectors, eigenFaces, data_set[9], 1, "old_1")  # reconstruct the old man
rec_one_face(avgFace, mx, eigenVectors, eigenFaces, data_set[14], 1, "woman_1")  # reconstruct the woman

