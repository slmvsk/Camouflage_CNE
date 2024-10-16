

####### TRIIMMING THE VIDEOS ########
import subprocess
import numpy as np
import cv2
import datetime
import os


def convert_seconds(seconds):
    # Convert total seconds into hours, minutes, and seconds format (HH:MM:SS)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    # Return the time in HH:MM:SS format
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def find_boundaries(video, step, fps=23, logpath=None):
    # This function calculates time boundaries in a video based on step size and frame rate (fps)

    time_points = list()  # To store time points
    step_frame = step * fps  # Calculate number of frames for each step (step in seconds * fps)

    frame = 1  # Initialize starting frame
    cap = cv2.VideoCapture(video)  # Capture video from file
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    if number_of_frames == 0:
        # If no frames are detected, try reading the video file with .avi extension
        cap.release()
        new_video = video.split('.')[0] + '.avi'
        cap = cv2.VideoCapture(new_video)
        number_of_frames = int(cv2.CAP_PROP_FRAME_COUNT)  # Get total number of frames from the new video

    print('!!!!!!!!', number_of_frames)  # Debugging message for frame count
    # Log number of frames
    joblog = open(logpath, 'a')
    joblog.write(str(datetime.datetime.now()) + 'video contains {} frames\n'.format(number_of_frames))
    joblog.close()

    succ, img = cap.read()  # Read first frame
    reds = list()  # Store average pixel intensity for each frame
    while frame < number_of_frames:
        if logpath is not None:
            # Log progress of frame extraction
            joblog = open(logpath, 'a')
            joblog.write(str(datetime.datetime.now()) + 'getting frame {} \n'.format(frame + step_frame))
            joblog.close()

        # Set video to the current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        time_points.append(frame / fps)  # Convert frame to time and store

        # Read frame and calculate the average color intensity
        succ, img = cap.read()
        img = np.array(img)
        reds.append(img[:,:,:].mean())  # Append mean of pixel values to reds
        frame += step_frame  # Increment frame by the step size in frames
            
    cap.release()  # Release video capture
    time_points = np.array(time_points)  # Convert list to numpy array
    reds = np.array(reds)  # Convert list to numpy array

    mean_I = np.mean(reds)  # Calculate mean intensity of frames

    # Find boundaries where intensity is below the mean
    b1 = int(time_points[np.where(reds < mean_I)[0][0]])
    b2 = int(time_points[np.where(reds < mean_I)[0][-1]])
    t1 = convert_seconds(b1)  # Convert boundary 1 to time format
    t2 = convert_seconds(b2)  # Convert boundary 2 to time format
    t3 = convert_seconds(time_points[-2])  # Convert second last time point to time format
    
    TIMESTAMPS = [("00:00:00", t1), (t2,t3)]  # Define time boundaries

    # Log the timestamps
    joblog = open(logpath, 'a')
    joblog.write(str(datetime.datetime.now()) + 'timestamps are {}, {}, {}\n'.format(t1, t2, t3))
    joblog.close()

    return TIMESTAMPS  # Return calculated timestamps

# Define log path for the job
logpath = '/work/ReiterU/temp_videos/joblog_enriched.txt'
joblog = open(logpath, 'w')
joblog.write(str(datetime.datetime.now()) + ' STARTING FOR ENRICHED CONDITION WOOHOO \n')
joblog.close()

# Define the folder where videos are stored
FOLDER = '/bucket/ReiterU/Development_project/movies/enriched2/'
#video = '/bucket/ReiterU/Development_project/movies/enriched/cam3_2023-03-11-08-16-36.avi'

# Get list of all video files in the folder with .mp4 extension
ALL_VIDEOS = os.listdir(FOLDER)
ALL_VIDEOS = [FOLDER + v for v in ALL_VIDEOS if v.endswith('mp4')]

# Loop over each video in the folder
for video in ALL_VIDEOS:

    try:
        # Log the copying of video
        joblog = open(logpath, 'a')
        joblog.write(str(datetime.datetime.now()) + ' copying video {} to temp \n'.format(video))
        joblog.close()

        # Uncomment the following lines if necessary to copy the video to temp
        # new_path = '/work/ReiterU/temp_videos/cam3_2023-04-15-11-41-11.avi'
        # subprocess.run(['scp', video, new_path])

        # Log the boundary search process
        joblog = open(logpath, 'a')
        joblog.write(str(datetime.datetime.now()) + ' searching for time boundaries... \n')
        joblog.close()

        # Find boundaries in the video
        TIMESTAMPS = find_boundaries(video, step=600, fps=23, logpath=logpath)
        print(TIMESTAMPS)

        # Log that boundaries were found
        joblog = open(logpath, 'a')
        joblog.write(str(datetime.datetime.now()) + ' boundaries found !\n')
        joblog.close()

        # Define output file names for the trimmed videos
        video_output_names = ['/work/ReiterU/temp_videos/' + video.split('/')[-1].split('.')[0] + '_trim1.MP4', 
                              '/work/ReiterU/temp_videos/' + video.split('/')[-1].split('.')[0] + '_trim2.MP4']
        # Log trimming process
        joblog = open(logpath, 'a')
        joblog.write(str(datetime.datetime.now()) + ' trimming video...\n')
        joblog.write(str(datetime.datetime.now()) + ' videos paths are {} \n and {}\n'.format(video_output_names[0], video_output_names[1]))
        joblog.close()

        print(video)

        # Define ffmpeg paths and commands for trimming the video based on found timestamps
        ffmpeg = '/apps/unit/ReiterU/olivier/ffmpeg/ffmpeg/ffmpeg'
        ffmpeg_command1 = [ffmpeg, '-y', '-i', video, '-ss', TIMESTAMPS[0][0], '-to', TIMESTAMPS[0][1],
                           '-c', 'copy', '-avoid_negative_ts', 'make_zero', video_output_names[0]]

        ffmpeg_command2 = [ffmpeg, '-y', '-i', video, '-ss', TIMESTAMPS[1][0], '-to', TIMESTAMPS[1][1],
                           '-c', 'copy', '-avoid_negative_ts', 'make_zero', video_output_names[1]]

        # Execute ffmpeg commands to trim the video
        subprocess.run(ffmpeg_command1, check=True)
        subprocess.run(ffmpeg_command2, check=True)

        # Log that video trimming is complete
        joblog = open(logpath, 'a')
        joblog.write(str(datetime.datetime.now()) + ' video trimmed!\n')
        joblog.close()

        # Define working folder for video processing
        work_folder = '/work/ReiterU/temp_videos/' + video.split('/')[-1].split('.')[0] + '/'
        os.makedirs(work_folder, exist_ok=True)


                ######## DETECTION AND WARPING #######

        # Opening a job log file to track the process
        joblog = open(logpath, 'a')
        joblog.write(str(datetime.datetime.now()) + ' setting up detectron model...\n')
        joblog.close()

        # Importing necessary libraries and modules for object detection and transformation
        import os
        import cv2
        import numpy as np
        import detectron2
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2.utils.visualizer import Visualizer, ColorMode
        from detectron2.data import MetadataCatalog, DatasetCatalog
        import h5py
        import argparse
        from tqdm import tqdm
        from uuid import uuid1
        from scipy import ndimage
        import random
        import string
        import torch
        import pickle

        # Logging that all imports have been successfully done
        joblog = open(logpath, 'a')
        joblog.write(str(datetime.datetime.now()) + ' imports done!\n')
        joblog.close()

        # Dictionary mapping camera numbers to specific models for detection
        DETECTRON_DICT = {
            'cam1':'development1',
            'cam0':'development2',
            'cam5':'development3',
            'cam4':'development4',
            'cam2':'development5',
            'cam3':'development6'
        }

        # Extract the camera number from the video file name
        if '/' in video:
            camera_number = video.split('/')[-1][:4]
        else:
            camera_number = video[:4]

        # Assign the detectron model based on the camera number
        detectron_model = DETECTRON_DICT[camera_number]

        # Logging that the model has been detected
        joblog = open(logpath, 'a')
        joblog.write(str(datetime.datetime.now()) + ' model detected!\n')
        joblog.close()  

        # Configuration for the detectron model
        cfg = get_cfg()

        # Merging with the base configuration file
        cfg.merge_from_file("/apps/unit/ReiterU/olivier/detectron2Configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        # Setting output directory to the model-specific folder
        cfg.OUTPUT_DIR = '/apps/unit/ReiterU/olivier/trained_models/' + detectron_model + '/output'

        # Setting the number of workers for the data loader
        cfg.DATALOADER.NUM_WORKERS = 4

        # Number of object classes in the model (e.g., 2 for binary classification)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

        # Load the model weights (pre-trained) from the specified path
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

        # Set the threshold for object detection; higher threshold means more confident detections
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9

        # Initialize the predictor (a wrapper for the model) with the given configuration
        predictor = DefaultPredictor(cfg)

        # Create a dummy dataset name using UUID and set up metadata for the segmentation
        currDataset_name = str(uuid1()) # Dummy dataset name
        segmentation_titles = ['cuttlefish'] # List of class names
        meta = MetadataCatalog.get(currDataset_name + '_train').set(thing_classes=segmentation_titles)

        # Log that the detectron model setup is complete
        joblog = open(logpath, 'a')
        joblog.write(str(datetime.datetime.now()) + ' detectron model set-up!\n')
        joblog.close()

        # Function to estimate the affine transformation between two masks
        def estimateAffine(src_mask, trg_mask, mode='similarity'):
            # Find contours (outlines) of the source and target masks
            cnts, _ = cv2.findContours(src_mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            src_ellipse = cv2.fitEllipse(cnts[0])  # Fit an ellipse to the source mask
            cnts, _ = cv2.findContours(trg_mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            trg_ellipse = cv2.fitEllipse(cnts[0])  # Fit an ellipse to the target mask

            # Calculate the rotation between the ellipses (in radians)
            rotation = (src_ellipse[2] - trg_ellipse[2]) / 180. * np.pi

            # Determine the scaling based on the specified mode (rotation, similarity, or full affine)
            if mode == 'rotation':
                scale_x = scale_y = 1  # No scaling, only rotation
            elif mode == 'similarity':
                # Scale uniformly (average scaling factor between both axes)
                scale_x = scale_y = (trg_ellipse[1][0] / src_ellipse[1][0] + trg_ellipse[1][1] / src_ellipse[1][1]) / 2
            elif mode == 'full':
                # Independent scaling for x and y axes
                scale_x = trg_ellipse[1][0] / src_ellipse[1][0]
                scale_y = trg_ellipse[1][1] / src_ellipse[1][1]
            else:
                raise RuntimeError('mode %s not in [\'rotation\', \'similarity\', \'full\']' % mode)

            # Calculate the shift between the centroids of the two ellipses
            shift_src = src_ellipse[0]
            shift_trg = trg_ellipse[0]

            # Compute the transformation matrix t0 for the affine transformation
            alpha = scale_x * np.cos(rotation)
            beta = scale_y * np.sin(rotation)
            t0 = np.array([
                [+alpha, +beta, (1. - alpha) * shift_src[0] - beta * shift_src[1] + shift_trg[0] - shift_src[0]],
                [-beta, +alpha, beta * shift_src[0] + (1. - alpha) * shift_src[1] + shift_trg[1] - shift_src[1]]
            ], 'float32')

            # Compute a second transformation matrix t1 (with a 180-degree phase shift)
            alpha = scale_x * np.cos(np.pi + rotation)
            beta = scale_y * np.sin(np.pi + rotation)
            t1 = np.array([
                [+alpha, +beta, (1. - alpha) * shift_src[0] - beta * shift_src[1] + shift_trg[0] - shift_src[0]],
                [-beta, +alpha, beta * shift_src[0] + (1. - alpha) * shift_src[1] + shift_trg[1] - shift_src[1]]
            ], 'float32')

            return t0, t1  # Return the two possible affine transformation matrices


        # Function to create a log file for a given dataset
        def create_log_file(dataset_name, work_folder):
            import datetime
            logfilename = work_folder + dataset_name + '.log'  # Log file path
            logfile = open(logfilename, 'w')  # Open the log file in write mode
            logfile.write('--------LOG OF DATASET {} CREATED AT {}-------- \n\n\n'.format(dataset_name, datetime.datetime.now()))  # Write the log header
            logfile.close()  # Close the file
            return logfilename  # Return the log file name

        # Function to divide a video into smaller slices
        def divide_video(source_video, start_frame, stop_frame, step_frame, num_workers, dataset_name, work_folder):
            
            # Import the FFMPEG tool from moviepy to handle video slicing
            from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
            
            # If stop_frame is -1, find the total number of frames in the video
            if stop_frame == -1:
                cap = cv2.VideoCapture(source_video)  # Open the video
                stop_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total frame count
                cap.release()  # Release the video capture object

            # Define slices: calculate the number of frames to divide among workers
            num_frames = stop_frame - start_frame - (num_workers * step_frame)  # Number of frames excluding safety seconds
            len_slice = int(np.floor(num_frames / num_workers))  # Length of each slice

            # Adjust length to ensure contiguous dataset
            leftover = len_slice % step_frame
            len_slice = len_slice - leftover

            # Create a list to store start and stop frames for each slice
            slices = list()
            start = start_frame
            stop = start + len_slice
            slices.append((start, stop))  # Append the first slice

            # Create additional slices for the remaining workers
            for k in range(num_workers - 1):
                start = stop + step_frame
                stop = start + len_slice
                slices.append((start, stop))

            # Slice the video and create logs for each slice
            VIDEOS = list()  # List to store the output video slice names
            LOGS = list()  # List to store the log file names
            for start, stop in slices:
                # Define the output file name based on the dataset name and frame range
                slice_name = "{}_{}-{}.mp4".format(dataset_name, start, stop)
                file_name = work_folder + slice_name  # Complete path for the sliced video
                # Extract the subclip (ffmpeg requires time in seconds; FPS assumed to be 23)
                ffmpeg_extract_subclip(source_video, start / 23, stop / 23, file_name)
                log_filename = create_log_file(slice_name, work_folder)  # Create log for the slice

                # Append the video file name and log file to respective lists
                VIDEOS.append(file_name)
                LOGS.append(log_filename)

            # Return the slices, video file paths, and log file paths
            return slices, VIDEOS, LOGS

        # Function to process the video and apply warp transformations
        def run_through_video2(video, work_folder, logfilename, warp_params, start_frame, stop_frame, step_frame):
            
            # Extract warp parameters from the provided dictionary
            refmask = warp_params['refMask']
            src_mask = warp_params['refMask']
            resizedMask = warp_params['resizedMask']
            scale_percent = warp_params['scale_percent']
            dim = warp_params['dim']
            rotation = warp_params['rotation']
            maskRotated = warp_params['maskRotated']
            invMaskRotated = warp_params['invMaskRotated']
            maskCrop = warp_params['maskCrop']
            (h, w) = warp_params['heightWidth']

            # Setup output file name
            video_suffix = video.split('/')[-1].split('.')[0]  # Get the video suffix (e.g., 'warp_trim1_123-456')
            video_suffix2 = video.split('/')[-2]  # Get the parent folder name
            # Create the HDF5 file path for storing the processed data
            output_filename = work_folder + '{}_data_{}-{}.h5'.format(video_suffix2, video_suffix, start_frame, stop_frame)
            num_frames = len(np.arange(start_frame, stop_frame, step_frame))  # Calculate the number of frames to process

            # Create the HDF5 file and datasets for storing patterns, masks, and positions
            pattern_file = h5py.File(output_filename, 'w')
            pattern_dset1 = pattern_file.create_dataset('patterns1', shape=[num_frames * 8, h, w, 3], dtype='uint8')
            pattern_dset2 = pattern_file.create_dataset('patterns2', shape=[num_frames * 8, h, w, 3], dtype='uint8')
            pattern_file.create_dataset('mask', data=resizedMask, dtype='bool')
            pos_dset = pattern_file.create_dataset('positions', shape=[num_frames * 8, 2], dtype='float32')
            instance_idx = pattern_file.create_dataset('instance', shape=[num_frames * 8], dtype='uint8')
            # Store some additional attributes like scaling percentage, start frame, and video name
            pattern_file.attrs.create('scale_percent', scale_percent, dtype='uint32')
            pattern_file.attrs.create('startFrame', start_frame, dtype='uint32')
            pattern_file.attrs.create('video', video, dtype=h5py.special_dtype(vlen=str))

            instance_idx[:] = 0  # Initialize the instance index to zero

            # Process the video frame by frame
            goodRun = 1  # Flag to track if the video processing goes well
            cap = cv2.VideoCapture(video)  # Open the video
            logfile = open(logfilename, 'a')  # Open the log file for writing
            frames_to_process = np.arange(0, stop_frame - start_frame, step_frame)  # Define frames to process
            N = 0  # Initialize frame counter
            for frame in tqdm(frames_to_process, total=len(frames_to_process)):  # Process each frame
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)  # Set video to the current frame
                succ, img = cap.read()  # Read the frame

                # If frame was read successfully
                if succ:
                    outputs = predictor(img)  # Perform detection on the frame
                    N_MASKS = len(outputs["instances"].pred_masks)  # Get the number of detected masks
                    if N_MASKS == 0:  # If no masks were detected, log the issue
                        logfile = open(logfilename, 'a')
                        logfile.write('no masks detected on frame ' + str(frame + start_frame) + '!\n')
                        logfile.close()
                    else:
                        # For each detected instance, apply the warp transformations
                        for INST in range(N_MASKS):
                            try:
                                mask = np.array(outputs["instances"].pred_masks[INST].to("cpu"))  # Extract the mask

                                mask_coos = np.where(mask)  # Get coordinates of mask pixels
                                xpos = mask_coos[1].mean()  # Calculate the mean x position
                                ypos = mask_coos[0].mean()  # Calculate the mean y position

                                # Estimate the affine transformation matrices
                                t0, t1 = estimateAffine(src_mask, mask)

                                # Apply the first affine transformation
                                t_inv0 = cv2.invertAffineTransform(t0)
                                img1 = cv2.warpAffine(img, t_inv0, src_mask.shape[1::-1])
                                resized = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
                                resized = ndimage.rotate(resized, rotation) * maskRotated + 1
                                resized = resized * invMaskRotated - 1
                                embeddingData = np.array(resized[maskCrop[0]:maskCrop[1], maskCrop[2]:maskCrop[3]])
                                pattern_dset1[N] = embeddingData[:, :, ::-1]  # Store the RGB pattern

                                # Apply the second affine transformation
                                t_inv1 = cv2.invertAffineTransform(t1)
                                img2 = cv2.warpAffine(img, t_inv1, src_mask.shape[1::-1])
                                resized = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)
                                resized = ndimage.rotate(resized, rotation) * maskRotated + 1
                                resized = resized * invMaskRotated - 1
                                embeddingData = np.array(resized[maskCrop[0]:maskCrop[1], maskCrop[2]:maskCrop[3]])
                                pattern_dset2[N] = embeddingData[:, :, ::-1]  # Store the RGB pattern

                                pos_dset[N, :] = np.array([xpos, ypos])  # Store the mask position
                                instance_idx[N] = INST + 1  # Update the instance index

                                N += 1  # Increment frame counter
                                # Log successful warping of the mask
                                logfile = open(logfilename, 'a')
                                logfile.write('warping of mask ' + str(INST) + ' succeeded on frame ' + str(frame) + '\n')
                                logfile.close()

                            except Exception as e:  # If an error occurs during mask warping
                                logfile = open(logfilename, 'a')  # Open the log file to append error details
                                logfile.write('warping of mask ' + str(INST) +  ' failed on frame ' + str(frame) + '\n')  # Log the failure
                                logfile.write(str(e) + '\n')  # Log the exception message
                                logfile.close()  # Close the log file

                # If frame could not be read (succ is False)
                elif not succ:
                    logfile = open(logfilename, 'a')  # Open the log file to append error
                    logfile.write('something wrong on frame ' + str(frame) + '\n')  # Log the frame failure
                    logfile.close()  # Close the log file
                    goodRun = 0  # Mark the run as unsuccessful

                else:  # This branch seems unreachable but is handled as a fallback
                    if not succ:  # If succ is False (again)
                        logfile = open(logfilename, 'a')  # Open the log file to log the failure
                        logfile.write('frame #{} was not included because cap not succ \n'.format(frame))  # Log the issue
                        logfile.close()  # Close the log file
                    else:  # If succ is True but the frame wasn't processed as expected
                        logfile = open(logfilename, 'a')  # Open the log file
                        logfile.write('frame #{} was not included because as expected \n'.format(frame))  # Log the frame exclusion
                        logfile.close()  # Close the log file
                    pass  # Continue with the next frame

            logfile.close()  # Close the log file at the end of processing
            cap.release()  # Release the video capture object
            pattern_file.close()  # Close the HDF5 file
            return True  # Indicate successful processing

        # Variables for dataset processing
        dataset_name = "warped_dark"  # Name of the dataset being processed
        num_workers = 4  # Number of parallel workers for processing
        start_frame = 0  # Starting frame for the video
        stop_frame = -1  # Use all frames in the video if stop_frame is -1
        step_frame = 690  # Step frame interval for frame sampling

        # Importing necessary modules for parallel processing and handling video metadata
        from joblib import Parallel, delayed
        import pickle

        # Load the warp parameters from a pickle file
        with open('/apps/unit/ReiterU/olivier/temp/warp_params.pickle', 'rb') as f:
            warp_params = pickle.load(f)

        # Define the list of videos to be processed
        # work_folder = '/work/ReiterU/temp_videos/'
        videolist = video_output_names  # List of videos to process

        # Loop through each video in the videolist and process it
        for source_video in videolist:
            
            # Log the start of processing for the video
            joblog = open(logpath, 'a')
            joblog.write(str(datetime.datetime.now()) + ' processing video... {} \n'.format(source_video))
            joblog.close()

            # Define the suffix for the video based on its name
            video_suffix = "_" + source_video.split("_")[-1].split('.')[0]  # Extract suffix (e.g., _trim1)
            # Divide the video into slices and get the corresponding logs
            slices, videos, logs = divide_video(source_video, start_frame, stop_frame, step_frame, num_workers, dataset_name + video_suffix, work_folder)
            
            # Parallel execution of video processing on multiple workers
            validations = Parallel(n_jobs=num_workers)(delayed(run_through_video2)(vid, work_folder, log, warp_params, Slice[0], Slice[1], step_frame)
                                             for (Slice, vid, log) in zip(slices, videos, logs))

            # Log the completion of video processing
            joblog = open(logpath, 'a')                               
            joblog.write(str(datetime.datetime.now()) + ' {} processing done!\n'.format(source_video))
            joblog.write(str(datetime.datetime.now()) + ' cleaning up subvideos...\n')  # Log cleanup start
            joblog.close()

            # Remove the temporary subvideos after processing
            for v in videos:
                subprocess.run(['rm', v], check=True)  # Delete each video file

            # Log the completion of cleanup
            joblog = open(logpath, 'a')
            joblog.write(str(datetime.datetime.now()) + ' cleaning up subvideos finished!\n')
            joblog.close()

        # After processing all videos, remove the original videos from the videolist
        for v in videolist:
            subprocess.run(['rm', v], check=True)  # Delete the original video files

    except:  # General exception handling for the entire block
        joblog = open(logpath, 'a')  # Open the log file
        joblog.write('processing failed on video {}, check joblog or slurm output'.format(video))  # Log the failure
        joblog.close()  # Close the log file