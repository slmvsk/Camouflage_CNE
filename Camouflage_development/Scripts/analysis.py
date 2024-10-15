

####### TRIIMMING THE VIDEOS ########
import subprocess
import numpy as np
import cv2
import datetime
import os


def convert_seconds(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)


    return f"{hours:02}:{minutes:02}:{seconds:02}"
            
            
def find_boundaries(video, step, fps=23, logpath=None):
    
    time_points = list()
    step_frame = step * fps
    
    frame = 1
    cap = cv2.VideoCapture(video)
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if number_of_frames == 0:
        cap.release()
        new_video = video.split('.')[0] + '.avi'
        cap = cv2.VideoCapture(new_video)
        number_of_frames = int(cv2.CAP_PROP_FRAME_COUNT)

    print('!!!!!!!!', number_of_frames)
    joblog = open(logpath, 'a')
    joblog.write(str(datetime.datetime.now()) + 'video contains {} frames\n'.format(number_of_frames))
    joblog.close()
    succ, img = cap.read()
    reds = list()
    while frame < number_of_frames:
        if logpath is not None:
            joblog = open(logpath, 'a')
            joblog.write(str(datetime.datetime.now()) + 'getting frame {} \n'.format(frame + step_frame))
            joblog.close()
        
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        time_points.append(frame / fps)
        

        succ, img = cap.read()
        img = np.array(img)
        reds.append(img[:,:,:].mean())
        frame += step_frame
            
    cap.release()    
    time_points = np.array(time_points)
    reds = np.array(reds)
    
    mean_I = np.mean(reds)
    
    b1 = int(time_points[np.where(reds < mean_I)[0][0]])
    b2 = int(time_points[np.where(reds < mean_I)[0][-1]])
    t1 = convert_seconds(b1)
    t2 = convert_seconds(b2)
    t3 = convert_seconds(time_points[-2])
    
    TIMESTAMPS = [("00:00:00", t1), (t2,t3)]
    
    joblog = open(logpath, 'a')
    joblog.write(str(datetime.datetime.now()) + 'timestamps are {}, {}, {}\n'.format(t1, t2, t3))
    joblog.close()

    return TIMESTAMPS

logpath = '/work/ReiterU/temp_videos/joblog_enriched.txt'
joblog = open(logpath, 'w')
joblog.write(str(datetime.datetime.now()) + ' STARTING FOR ENRICHED CONDITION WOOHOO \n')
joblog.close()


FOLDER = '/bucket/ReiterU/Development_project/movies/enriched2/'
#video = '/bucket/ReiterU/Development_project/movies/enriched/cam3_2023-03-11-08-16-36.avi'


ALL_VIDEOS = os.listdir(FOLDER)
ALL_VIDEOS = [FOLDER + v for v in ALL_VIDEOS if v.endswith('mp4')]


for video in ALL_VIDEOS:

    try:
        joblog = open(logpath, 'a')
        joblog.write(str(datetime.datetime.now()) + ' copying video {} to temp \n'.format(video))
        joblog.close()

        #new_path = '/work/ReiterU/temp_videos/cam3_2023-04-15-11-41-11.avi'
        #subprocess.run(['scp', video, new_path])



        joblog = open(logpath, 'a')
        joblog.write(str(datetime.datetime.now()) + ' searching for time boundaries... \n')
        joblog.close()
        TIMESTAMPS = find_boundaries(video, step=600, fps=23, logpath=logpath)
        print(TIMESTAMPS)
        joblog = open(logpath, 'a')
        joblog.write(str(datetime.datetime.now()) + ' boundaries found !\n')
        joblog.close()




        video_output_names = ['/work/ReiterU/temp_videos/' + video.split('/')[-1].split('.')[0] + '_trim1.MP4', '/work/ReiterU/temp_videos/' + video.split('/')[-1].split('.')[0] + '_trim2.MP4']
        joblog = open(logpath, 'a')
        joblog.write(str(datetime.datetime.now()) + ' trimming video...\n')
        joblog.write(str(datetime.datetime.now()) + ' videos paths are {} \n and {}\n'.format(video_output_names[0], video_output_names[1]))
        joblog.close()
        print(video)
        ffmpeg = '/apps/unit/ReiterU/olivier/ffmpeg/ffmpeg/ffmpeg'
        ffmpeg_command1 = [ffmpeg, '-y', '-i', video, '-ss', TIMESTAMPS[0][0], '-to', TIMESTAMPS[0][1],\
                           '-c', 'copy', '-avoid_negative_ts', 'make_zero', video_output_names[0]]

        ffmpeg_command2 = [ffmpeg, '-y', '-i', video, '-ss', TIMESTAMPS[1][0], '-to', TIMESTAMPS[1][1],\
                           '-c', 'copy', '-avoid_negative_ts', 'make_zero', video_output_names[1]]

        subprocess.run(ffmpeg_command1, check=True)
        subprocess.run(ffmpeg_command2, check=True)

        joblog = open(logpath, 'a')
        joblog.write(str(datetime.datetime.now()) + ' video trimmed!\n')
        joblog.close()

        work_folder = '/work/ReiterU/temp_videos/' + video.split('/')[-1].split('.')[0] + '/'
        os.makedirs(work_folder, exist_ok=True)


        ######## DETECTION AND WARPING #######

        joblog = open(logpath, 'a')
        joblog.write(str(datetime.datetime.now()) + ' setting up detectron model...\n')
        joblog.close()
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


        joblog = open(logpath, 'a')
        joblog.write(str(datetime.datetime.now()) + ' imports done!\n')
        joblog.close()


        # determining which detectron model to use

        DETECTRON_DICT = {'cam1':'development1',
                          'cam0':'development2',
                          'cam5':'development3',
                          'cam4':'development4',
                          'cam2':'development5',
                          'cam3':'development6'}




        if '/' in video:
            camera_number = video.split('/')[-1][:4]
        else:
            camera_number = video[:4]

        detectron_model = DETECTRON_DICT[camera_number]
        joblog = open(logpath, 'a')
        joblog.write(str(datetime.datetime.now()) + ' model detected!\n')
        joblog.close()  

        cfg = get_cfg()
        cfg.merge_from_file("/apps/unit/ReiterU/olivier/detectron2Configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.OUTPUT_DIR = '/apps/unit/ReiterU/olivier/trained_models/' + detectron_model + '/output'
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9

        predictor = DefaultPredictor(cfg)
        currDataset_name= str(uuid1()) #dummy
        segmentation_titles=['cuttlefish']
        meta=MetadataCatalog.get(currDataset_name + '_train').set(thing_classes=segmentation_titles)


        joblog = open(logpath, 'a')
        joblog.write(str(datetime.datetime.now()) + ' detectron model set-up !\n')
        joblog.close()



        def estimateAffine(src_mask,trg_mask,mode='similarity'):
            cnts, _ = cv2.findContours(src_mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            src_ellipse = cv2.fitEllipse(cnts[0])
            cnts, _ = cv2.findContours(trg_mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            trg_ellipse = cv2.fitEllipse(cnts[0])
            rotation = (src_ellipse[2] - trg_ellipse[2]) / 180. * np.pi
            if mode == 'rotation':
                scale_x = scale_y = 1
            elif mode == 'similarity':
                scale_x = scale_y = (trg_ellipse[1][0] / src_ellipse[1][0] \
                        + trg_ellipse[1][1] / src_ellipse[1][1]) / 2
            elif mode == 'full':
                scale_x = trg_ellipse[1][0] / src_ellipse[1][0]
                scale_y = trg_ellipse[1][1] / src_ellipse[1][1]
            else:
                raise RuntimeError('mode %s not in ' \
                        '[\'rotation\', \'similarity\', \'full\']' % mode)
            shift_src = src_ellipse[0]
            shift_trg = trg_ellipse[0]

            # Compute transformation matrices
            alpha = scale_x * np.cos(rotation)
            beta = scale_y * np.sin(rotation)
            t0 = np.array([[+alpha, +beta,   (1. - alpha) * shift_src[0] \
                                                   - beta * shift_src[1] \
                                           + shift_trg[0] - shift_src[0]], \
                           [-beta, +alpha,           beta * shift_src[0] \
                                           + (1. - alpha) * shift_src[1] \
                                           + shift_trg[1] - shift_src[1]]], 'float32')

            alpha = scale_x * np.cos(np.pi + rotation)
            beta = scale_y * np.sin(np.pi + rotation)
            t1 = np.array([[+alpha, +beta,   (1. - alpha) * shift_src[0] \
                                                   - beta * shift_src[1] \
                                           + shift_trg[0] - shift_src[0]], \
                           [-beta, +alpha,           beta * shift_src[0] \
                                           + (1. - alpha) * shift_src[1] \
                                           + shift_trg[1] - shift_src[1]]], 'float32')

            return t0, t1



        def create_log_file(dataset_name, work_folder):
            import datetime
            logfilename = work_folder + dataset_name + '.log'
            logfile = open(logfilename, 'w')
            logfile.write('--------LOG OF DATASET {} CREATED AT {}-------- \n\n\n'.format(dataset_name, datetime.datetime.now()))
            logfile.close()
            return logfilename

        def divide_video(source_video, start_frame, stop_frame, step_frame, num_workers, dataset_name, work_folder):

            from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip



            if stop_frame == -1:
                cap = cv2.VideoCapture(source_video)
                stop_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()


            # define slices
            num_frames = stop_frame - start_frame - (num_workers * step_frame) # safety seconds
            len_slice = int(np.floor(num_frames / num_workers))

            leftover = len_slice % step_frame
            len_slice = len_slice - leftover # this is to ensure to end up with contiguous dataset

            slices = list()
            start = start_frame
            stop = start + len_slice
            slices.append((start, stop))
            for k in range(num_workers - 1):
                start = stop + step_frame
                stop = start + len_slice
                slices.append((start,stop))


            # slice the video
            VIDEOS = list()
            LOGS = list()
            for start, stop in slices:
                slice_name = "{}_{}-{}.mp4".format(dataset_name, start, stop) #warped_trim1_123-456.mp4
                file_name = work_folder + slice_name # /work/ReiterU/temp_videos/camXXXXXXXXXXXX/warp_trim1-123-456.mp4
                ffmpeg_extract_subclip(source_video, start/23, stop/23, file_name) # function take time as sec, FPS is usually 23
                log_filename = create_log_file(slice_name, work_folder)

                VIDEOS.append(file_name)
                LOGS.append(log_filename)


            return slices, VIDEOS, LOGS


        def run_through_video2(video, work_folder, logfilename, warp_params, start_frame, stop_frame, step_frame):

            refmask = warp_params['refMask']
            src_mask = warp_params['refMask']
            resizedMask = warp_params['resizedMask']
            scale_percent = warp_params['scale_percent']
            dim = warp_params['dim']
            rotation = warp_params['rotation']
            maskRotated = warp_params['maskRotated']
            invMaskRotated = warp_params['invMaskRotated']
            maskCrop = warp_params['maskCrop']
            (h,w) = warp_params['heightWidth']


            #start_frame = int(video.split('-')[0].split('_')[-1]) # /work/ReiterU/temp_videos/camXXXXXXXXXXXX/warp_trim1_123-456.mp4
            #stop_frame = int(video.split('-')[-1].split('.')[0])
            print(start_frame, stop_frame, step_frame)




            #setup output file
            video_suffix = video.split('/')[-1].split('.')[0]  #/work/ReiterU/temp_videos/camXXXXXXXXXXXX/warp_trim1_123-456.mp4 -> warp_trim1-123-456
            video_suffix2 = video.split('/')[-2]
            output_filename = work_folder + '{}_data_{}-{}.h5'.format(video_suffix2, video_suffix, start_frame, stop_frame)
            num_frames = len(np.arange(start_frame, stop_frame, step_frame))

            pattern_file = h5py.File(output_filename, 'w')
            pattern_dset1 = pattern_file.create_dataset('patterns1', \
                        shape=[num_frames*8,h,w,3], \
                        dtype='uint8')
            pattern_dset2 = pattern_file.create_dataset('patterns2', \
                        shape=[num_frames*8,h,w,3], \
                        dtype='uint8')
            pattern_file.create_dataset('mask', \
                        data=resizedMask, \
                        dtype='bool')
            pos_dset = pattern_file.create_dataset('positions', \
                    shape = [num_frames*8, 2], \
                    dtype = 'float32')
            instance_idx = pattern_file.create_dataset('instance', \
                                                      shape = [num_frames*8],
                                                      dtype='uint8')
            pattern_file.attrs.create('scale_percent',scale_percent, \
                                dtype='uint32')
            pattern_file.attrs.create('startFrame',start_frame, \
                                dtype='uint32')
            pattern_file.attrs.create('video', \
                    video, \
                    dtype=h5py.special_dtype(vlen=str))

            instance_idx[:] = 0

            #run through video
            goodRun=1
            cap = cv2.VideoCapture(video)
            print(video, logfilename)
            logfile = open(logfilename, 'a')
            frames_to_process = np.arange(0, stop_frame - start_frame, step_frame)
            print(frames_to_process)
            N = 0 
            for frame in tqdm(frames_to_process, total=len(frames_to_process)): #length

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                succ, img = cap.read() 

                if (succ) and (True):
                    outputs = predictor(img)
                    N_MASKS = len(outputs["instances"].pred_masks)
                    if N_MASKS == 0:   #it will use the previous mask in this case

                        logfile = open(logfilename, 'a')
                        logfile.write('no masks detected on frame ' + str(frame + start_frame) + '!\n')
                        logfile.close()

                    else:
                        for INST in range(N_MASKS):

                            try:
                                mask=np.array(outputs["instances"].pred_masks[INST].to("cpu"))

                                mask_coos = np.where(mask)
                                xpos = mask_coos[1].mean()
                                ypos = mask_coos[0].mean()


                                t0, t1 = estimateAffine(src_mask, mask)

                                t_inv0 = cv2.invertAffineTransform(t0)
                                img1 = cv2.warpAffine(img, t_inv0, src_mask.shape[1::-1])
                                resized = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
                                resized=ndimage.rotate(resized, rotation)*maskRotated+1
                                resized=resized*invMaskRotated-1
                                embeddingData=np.array(resized[maskCrop[0]:maskCrop[1],maskCrop[2]:maskCrop[3]])
                                pattern_dset1[N]=embeddingData[:,:,::-1] #back to rgb


                                t_inv1 = cv2.invertAffineTransform(t1)
                                img2 = cv2.warpAffine(img, t_inv1, src_mask.shape[1::-1])
                                resized = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
                                resized=ndimage.rotate(resized, rotation)*maskRotated+1
                                resized=resized*invMaskRotated-1
                                embeddingData=np.array(resized[maskCrop[0]:maskCrop[1],maskCrop[2]:maskCrop[3]])
                                pattern_dset2[N]=embeddingData[:,:,::-1] #back to rgb


                                pos_dset[N,:] = np.array([xpos, ypos])
                                instance_idx[N] = INST + 1

                                N += 1
                                logfile = open(logfilename, 'a')
                                logfile.write('warping of mask ' + str(INST) +  ' succedeedon frame' + str(frame) + '\n')
                                logfile.close()

                            except Exception as e:
                                logfile = open(logfilename, 'a')
                                logfile.write('warping of mask ' + str(INST) +  ' failed on frame' + str(frame) + '\n')
                                logfile.write(str(e) + '\n')
                                logfile.close()



                elif not succ:
                    logfile = open(logfilename, 'a')
                    logfile.write('something wrong on frame ' + str(frame) + '\n')
                    logfile.close()
                    goodRun=0

                else:
                    if not succ:
                        logfile = open(logfilename, 'a')
                        logfile.write('frame #{} was not included because cap not succ \n'.format(frame))
                        logfile.close()
                    else:
                        logfile = open(logfilename, 'a')
                        logfile.write('frame #{} was not included because as expected \n'.format(frame))
                        logfile.close()
                    pass



            logfile.close()
            cap.release()    
            pattern_file.close()
            return True





        dataset_name = "warped_dark"
        num_workers = 4
        start_frame = 0
        stop_frame = -1
        step_frame = 690






        from joblib import Parallel, delayed
        import pickle

        with open('/apps/unit/ReiterU/olivier/temp/warp_params.pickle', 'rb') as f:
            warp_params = pickle.load(f)


        #work_folder = '/work/ReiterU/temp_videos/'
        videolist = video_output_names


        for source_video in videolist:

            joblog = open(logpath, 'a')
            joblog.write(str(datetime.datetime.now()) + ' processing video... {} \n'.format(source_video))
            joblog.close()
            video_suffix = "_" + source_video.split("_")[-1].split('.')[0] # _trim1
            slices, videos, logs = divide_video(source_video, start_frame, stop_frame,\
                                                step_frame, num_workers, dataset_name+video_suffix, work_folder)
                                           #warped_trim1
            validations = Parallel(n_jobs=num_workers)(delayed(run_through_video2)(vid, work_folder, log, warp_params, Slice[0], Slice[1], step_frame)\
                                             for (Slice, vid, log) in zip(slices, videos, logs))

            joblog = open(logpath, 'a')                               
            joblog.write(str(datetime.datetime.now()) + ' {} processing done!\n'.format(source_video))
            joblog.write(str(datetime.datetime.now()) + ' cleaning up subvideos...\n')
            joblog.close()

            for v in videos:
                subprocess.run(['rm', v], check=True)

            joblog = open(logpath, 'a')
            joblog.write(str(datetime.datetime.now()) + ' cleaning up subvideos finished!\n')
            joblog.close()


        for v in videolist:
            subprocess.run(['rm', v], check=True)
    except:
        joblog = open(logpath, 'a')
        joblog.write('processing failed on video {}, check joblog or slurm output'.format(video))



