# Project:
This repository is for key frame extraction process.

# Requirements:

All the required libraries are mentioned in requirement.txt file. Use pip install -r requirement.txt to install all the requirements.

# How to run the code:

run "python candidate_frames_folder.py --input_videos "sample_video.mp4" --output_folder_video_image candidate_frames_and_their_cluster_folder --output_folder_video_final_image final_images"

#This command will create a new folder with the same name as input video name and inside that folder, candidate frames and their clusters based on similarity will be created in "candidate_frames_and_their_cluster_folder" and final key frames in "final_images" folder respectively
