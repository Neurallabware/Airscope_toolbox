import os
import deeplabcut
from pathlib import Path


ProjectFolderName = "D:/Users/86177/DeepLabCut_Mice/PICO_three_mice-yuanlong-2024-06-26/" # note this has to be a full path
VideoType = 'mp4' #, mp4, MOV, or avi, whatever you uploaded!

videofile_path = [ProjectFolderName+"/videoss/"] #Enter the list of videos or folder to analyze.

path_config_file = ProjectFolderName + "/config.yaml"

shuffle = 0

tracktype= 'ellipse' #box, skeleton, ellipse -- ellipse is recommended.


deeplabcut.analyze_videos(config = path_config_file,
                          videos = videofile_path, # this can be different
                          videotype=VideoType,
                          shuffle = shuffle,
                          save_as_csv = True,
                          in_random_order = False),

numAnimals = 5 # how many animals do you expect to find?

deeplabcut.convert_detections2tracklets(config=path_config_file,
                                        videos=videofile_path,
                                        videotype=VideoType,
                                        shuffle=shuffle,  # used for loading training weights
                                        track_method=tracktype, overwrite=True)

deeplabcut.stitch_tracklets(config_path=path_config_file,
                            videos=videofile_path,
                            track_method=tracktype,
                             shuffle=shuffle,
                              n_tracks=numAnimals)


deeplabcut.filterpredictions(config=path_config_file,
                                 video = videofile_path,
                                 shuffle=shuffle,
                                 videotype=VideoType,
                                 track_method = tracktype)


deeplabcut.plot_trajectories(config=path_config_file,
                             videos=videofile_path,
                             shuffle=shuffle,
                             videotype=VideoType,
                             track_method=tracktype)

deeplabcut.create_labeled_video(config=path_config_file,
                                videos=videofile_path,
                                shuffle=shuffle,
                                color_by="individual",
                                videotype=VideoType,
                                save_frames=False,
                                filtered=True,
                                track_method = tracktype)