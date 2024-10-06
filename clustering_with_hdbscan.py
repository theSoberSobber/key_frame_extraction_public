from __future__ import print_function

import os
from glob import glob
import tempfile
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import img_as_float
import hdbscan
# from multiprocessing import Pool, Process, cpu_count
import matplotlib.pyplot as plt
import seaborn as sns

import time

from extracting_candidate_frames import Configs as config


class ImageSelector(object):
    """Class for selection of best top N images from input list of images. Currently following selection methods are implemented:
    brightness filtering, contrast/entropy filtering, clustering of frames, and variance of Laplacian for non-blurred images.
    """

    def __get_brightness_score__(self, image):
        """Internal function to compute the brightness of input image, returns brightness score between 0 to 100.0.

        :param image: Input image in OpenCV Numpy format.
        :return: Brightness score as float value between 0.0 to 100.0.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        total_brightness = np.sum(v, dtype=np.float32)
        num_of_pixels = v.shape[0] * v.shape[1]
        brightness_score = (total_brightness * 100.0) / (num_of_pixels * 255.0)
        return brightness_score

    def __get_entropy_score__(self, image):
        """Internal function to compute the entropy/contrast of input image, returns entropy score between 0 to 10.

        :param image: Input image in OpenCV Numpy format.
        :return: Entropy score as float value between 0.0 to 10.0.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        entr_img = entropy(gray, disk(5))
        total_entropy = np.sum(entr_img)
        num_of_pixels = entr_img.shape[0] * entr_img.shape[1]
        entropy_score = total_entropy / num_of_pixels

        return entropy_score

    def __variance_of_laplacian__(self, image):
        """Internal function to compute the Laplacian of the image and return the focus
        measure, which is simply the variance of the Laplacian.

        :param image: Input image in OpenCV Numpy format.
        :return: Variance of Laplacian as float.
        """
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def __filter_optimum_brightness_and_contrast_images__(self, input_img_files):
        """Internal function to filter input images with optimum brightness and contrast/entropy.

        :param input_img_files: List of input image files.
        :return: List of filtered image files with optimum brightness and contrast/entropy.
        """
        n_files = len(input_img_files)

        # Calculate brightness and entropy score
        brightness_score = np.asarray(list(map(self.__get_brightness_score__, input_img_files)))
        entropy_score = np.asarray(list(map(self.__get_entropy_score__, input_img_files)))

        # Check if brightness and contrast scores are in the min and max defined range
        brightness_ok = np.logical_and(
            brightness_score > config.min_brightness_value,
            brightness_score < config.max_brightness_value,
        )
        contrast_ok = np.logical_and(
            entropy_score > config.min_entropy_value,
            entropy_score < config.max_entropy_value,
        )

        # Return only images that have good brightness and contrast
        return [
            input_img_files[i]
            for i in range(n_files)
            if brightness_ok[i] and contrast_ok[i]
        ]

    def __prepare_cluster_sets_hdbscan(self, files):
        """Internal function for clustering input image files using HDBSCAN.

        :param files: List of input image files.
        :return: Array of indices representing which cluster a file belongs to.
        """
        all_dst = []

        # Calculate DCT for each image and store in all_dst
        for img_file in files:
            img = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (256, 256))
            imf = np.float32(img) / 255.0  # Float conversion/scale
            dst = cv2.dct(imf)  # DCT computation
            dst = dst[:16, :16]
            dst = dst.reshape((256))
            all_dst.append(dst)

        # Apply HDBSCAN clustering
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=2, metric='manhattan')
        hdbscan_model.fit(all_dst)

        labels = np.add(hdbscan_model.labels_, 1)
        nb_clusters = len(np.unique(hdbscan_model.labels_))

        # Cluster index array
        files_clusters_index_array = []
        files_clusters_index_array_of_only_one_image = []

        for i in range(nb_clusters):
            if i == 0:
                index_array = np.where(labels == i)
                files_clusters_index_array_of_only_one_image.append(index_array)
            else:
                index_array = np.where(labels == i)
                files_clusters_index_array.append(index_array)

        files_clusters_index_array = np.array(files_clusters_index_array)
        return files_clusters_index_array, files_clusters_index_array_of_only_one_image

    def __get_laplacian_scores(self, files, cluster_indices):
        """Function to iterate over images in the cluster and calculate Laplacian (blurriness) score.

        :param files: List of input image filenames.
        :param cluster_indices: Cluster indices to process.
        :return: List of Laplacian scores for each image in the cluster.
        """
        variance_laplacians = []

        for image_i in cluster_indices:
            img_file = files[image_i]
            img = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)

            # Calculate blurriness using Laplacian
            variance_laplacian = self.__variance_of_laplacian__(img)
            variance_laplacians.append(variance_laplacian)

        return variance_laplacians

    def __get_best_images_index_from_each_cluster__(self, files, cluster_indices):
        """Internal function to return the index of the best image from each cluster.

        :param files: List of input image files.
        :param cluster_indices: Cluster indices to process.
        :return: Index of the best image for each cluster.
        """
        best_images_index = []

        for cluster in cluster_indices:
            if len(cluster[0]) == 1:
                best_images_index.append(cluster[0][0])
            else:
                laplacian_scores = self.__get_laplacian_scores(files, cluster[0])
                best_image = cluster[0][np.argmax(laplacian_scores)]
                best_images_index.append(best_image)

        return best_images_index
