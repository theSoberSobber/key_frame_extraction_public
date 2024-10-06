import os
import cv2
import numpy as np
import hdbscan
from skimage.filters.rank import entropy
from skimage.morphology import disk

class ImageSelector:
    def __init__(self):
        # Initialize configuration parameters
        self.min_brightness_value = 20  # Example value, adjust as needed
        self.max_brightness_value = 80  # Example value, adjust as needed
        self.min_entropy_value = 3  # Example value, adjust as needed
        self.max_entropy_value = 7  # Example value, adjust as needed
        self.nb_clusters = 5  # Example value, adjust as needed

    def __get_brightness_score__(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        sum = np.sum(v, dtype=np.float32)
        num_of_pixels = v.shape[0] * v.shape[1]
        brightness_score = (sum * 100.0) / (num_of_pixels * 255.0)
        return brightness_score

    def __get_entropy_score__(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        entr_img = entropy(gray, disk(5))
        all_sum = np.sum(entr_img)
        num_of_pixels = entr_img.shape[0] * entr_img.shape[1]
        entropy_score = all_sum / num_of_pixels
        return entropy_score

    def __variance_of_laplacian__(self, image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def __filter_optimum_brightness_and_contrast_images__(self, input_img_files):
        n_files = len(input_img_files)
        brightness_score = np.asarray(list(map(self.__get_brightness_score__, input_img_files)))
        entropy_score = np.asarray(list(map(self.__get_entropy_score__, input_img_files)))

        brightness_ok = np.logical_and(
            brightness_score > self.min_brightness_value,
            brightness_score < self.max_brightness_value
        )
        contrast_ok = np.logical_and(
            entropy_score > self.min_entropy_value,
            entropy_score < self.max_entropy_value
        )

        return [
            input_img_files[i]
            for i in range(n_files)
            if brightness_ok[i] and contrast_ok[i]
        ]

    def __prepare_cluster_sets__hdbscan(self, files):
        all_dst = []
        for img_file in files:
            img = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (256, 256))
            imf = np.float32(img) / 255.0
            dst = cv2.dct(imf)
            dst = dst[:16, :16]
            dst = dst.reshape((256))
            all_dst.append(dst)

        Hdbascan = hdbscan.HDBSCAN(min_cluster_size=2, metric='manhattan').fit(all_dst)
        labels = np.add(Hdbascan.labels_, 1)
        nb_clusters = len(np.unique(Hdbascan.labels_))

        files_clusters_index_array = []
        files_clusters_index_array_of_only_one_image = []
        for i in np.arange(nb_clusters):
            if i == 0:
                index_array = np.where(labels == i)
                files_clusters_index_array_of_only_one_image.append(index_array)
            else:
                index_array = np.where(labels == i)
                files_clusters_index_array.append(index_array)

        files_clusters_index_array = np.array(files_clusters_index_array)
        return files_clusters_index_array, files_clusters_index_array_of_only_one_image

    def __get_laplacian_scores(self, files, n_images):
        variance_laplacians = []
        for image_i in n_images:
            img_file = files[image_i]
            img = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)
            variance_laplacian = self.__variance_of_laplacian__(img)
            variance_laplacians.append(variance_laplacian)
        return variance_laplacians

    def __get_best_images_index_from_each_cluster__(self, files, files_clusters_index_array):
        filtered_items = []
        clusters = np.arange(len(files_clusters_index_array))
        for cluster_i in clusters:
            curr_row = files_clusters_index_array[cluster_i][0]
            n_images = np.arange(len(curr_row))
            variance_laplacians = self.__get_laplacian_scores(files, curr_row)
            try:
                selected_frame_of_current_cluster = curr_row[np.argmax(variance_laplacians)]
                filtered_items.append(selected_frame_of_current_cluster)
            except ValueError:
                continue
        return filtered_items

    def select_best_frames(self, input_key_frames, output_folder):
        filtered_images_list = []

        if len(input_key_frames) >= 1:
            files_clusters_index_array, files_clusters_index_array_of_only_one_image = self.__prepare_cluster_sets__hdbscan(input_key_frames)
            selected_images_index = self.__get_best_images_index_from_each_cluster__(
                input_key_frames, files_clusters_index_array
            )
            files_clusters_index_array_of_only_one_image = [item for t in files_clusters_index_array_of_only_one_image for item in t]
            files_clusters_index_array_of_only_one_image = files_clusters_index_array_of_only_one_image[0].tolist()
            selected_images_index.extend(files_clusters_index_array_of_only_one_image)
            
            for index in selected_images_index:
                img = input_key_frames[index]
                filtered_images_list.append(img)
            
            # Saving images of same clusters
            for i, images in enumerate(files_clusters_index_array):
                path = os.path.join(output_folder, str(i))
                os.makedirs(path, exist_ok=True)
                for image in images[0]:
                    cv2.imwrite(os.path.join(path, f"{image}.jpeg"), input_key_frames[image])
        else:
            filtered_images_list = input_key_frames

        # Saving clusters of single image cluster
        for i, image in enumerate(files_clusters_index_array_of_only_one_image[0], start=len(files_clusters_index_array)):
            path = os.path.join(output_folder, str(i))
            os.makedirs(path, exist_ok=True)
            cv2.imwrite(os.path.join(path, f"{image}.jpeg"), input_key_frames[image])

        return filtered_images_list
