import os

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def read_images_from_folder(folder_name):
    # reference: https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
    images = []
    for image in os.listdir(folder_name):  # find images in selected folder
        img = cv2.imread(os.path.join(folder_name, image))
        if img is not None:
            if type(img) == np.ndarray:  # only read if it is an image, prevents errors
                img = cv2.resize(img, (50,50))  # let all images have the same size
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


def print_eigen():
    all_img = read_images_from_folder("iivp/pictures/pca")#[0:7]

    #data = create_data_matrix(all_img)
    #mean_face, faces_matrix = cv2.PCACompute(data, mean=None)

    neutral = []
    print('Enter path to 7 images below to produce mean face & eigen faces :\n')
    for i in range(7):
        img = all_img[i]

        # Vectorization of array
        img2 = np.array(img).flatten()
        neutral.append(img2)

    faces_matrix = np.vstack(neutral)
    mean_face = np.mean(faces_matrix, axis=0)

    print(faces_matrix.shape)

    # print the mean face
    #plt.imshow(mean_face.reshape(49, 58, 3), cmap='gray');
    cv2.imwrite('iivp/resultPictures/exercise4/average face.jpg', mean_face.reshape(50, 50, 3))

    print('Printed Mean Face. Check the output window for final results.\n ')
    #plt.title('Mean Face')
    #plt.show()

    # print 5 eigen faces
    # normalization of faces matrix
    faces_norm = faces_matrix - mean_face
    faces_norm = faces_norm.T
    face_cov = np.cov(faces_norm)
    eigen_vecs, eigen_vals, _ = np.linalg.svd(faces_norm)
    # 5 Eigen Faces Visualization
    average_face = mean_face.reshape(50, 50, 3)
    eigen_face = []

    # obtain eigenfaces
    for eigenVector in eigen_vecs:
        eigenFace = eigenVector.reshape(50, 50, 3)
        eigen_face.append(eigenFace)

    return average_face, eigen_vecs, eigen_face, mean_face


# recons() function helps to reconstruct the faces with different number of eigen faces used
def reconstruction(mean_face, eigen_vecs, eigen_faces, imVector):
    final_output = mean_face
    print("mean: ", mean_face.shape)
    print("eigen: ", len(eigen_faces))
    # We make percentage dict to store the weight of eigen face used
    # to reconstruct the current output
    percentage = {}
    for k in range(0, len(eigen_vecs)):
        weight = np.dot(imVector, eigen_vecs[k])
        final_output = final_output + eigen_faces[k] * weight
        # store weight in percentage dict

        percentage[k] = abs(weight)
    # display output

    disp_img = cv2.normalize(np.abs(final_output), None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("iivp/resultPictures/exercise4/final.jpg", disp_img)

    # Display the Percentage of Eigen Faces used for reconstruction dynamically
    total = 0
    if (len(percentage) > 0):
        print("\nPercentage of Eigen Faces that make current output :")
    for i in percentage:
        total = total + abs(percentage[i])
    for i in percentage:
        val = float(abs((percentage[i] / total) * 100))
        if (val > 0):
            print(str("{:.2f}".format(val)) + "% of Face " + str(i + 1))



mean_face, eigen_vecs, eigen_faces, mean = print_eigen()
eigen_vec=[]
im = cv2.imread("iivp/pictures/pca/old.jpg")
im = cv2.resize(im, (50,50))
im = np.float32(im)/255.0
imVector = im.flatten() - mean;
eigen_vec.append(imVector)
disp_img = cv2.normalize(np.abs(np.array(eigen_vec).reshape(50,50,3)), None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite("iivp/resultPictures/exercise4/test.jpg", disp_img)
# im_vec = eigen_vecs[1]
reconstruction(mean_face, eigen_vecs, eigen_faces, eigen_vec)