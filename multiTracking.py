import cv2
import sys
import os
from random import randint

tracker_types = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(tracker_type):
    ## Create a tracker based on tracker name
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == 'MIL': 
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    elif tracker_type == 'CSRT':
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in tracker_types:
            print(t)
        
    return tracker

def multiTracking(video_path, toimage=0, tracker_type='CSRT'):
     ## Create a video capture object to Read video
    try:
        video = cv2.VideoCapture("modify_code\data\passageway0.mp4")
        if not video.isOpened():
            raise NameError('Could not open video')
            sys.exit()
    except cv2.error as e:
        print("cv2.error:", e)
    except Exception as e:
        print("Exception:", e)
    else:
        print("read video no problem reported")

    video.set(cv2.CAP_PROP_POS_MSEC, 4000)
    original_video = video

    ## Read first frame.
    video_ret, video_frame = video.read()
    if not video_ret:
        print("Cannot read video file")
        sys.exit()

    ## Select boxes
    bboxes = []
    colors = [] 

    while True:
        ## draw bounding boxes over objects
        ## selectROI's default behaviour is to draw box starting from the center
        ## when fromCenter is set to false, you can draw box starting from top left corner
        bbox = cv2.selectROI('MultiTracker', video_frame)
        bboxes.append(bbox)
        colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
        print("Press q to quit selecting boxes and start tracking")
        print("Press any other key to select next object")
        k = cv2.waitKey(0) & 0xFF
        if (k == 113):  ## q is pressed
            break
  
    print('Selected bounding boxes {}'.format(bboxes))

    ## Create MultiTracker object
    multiTracker = cv2.MultiTracker_create()

    ## Initialize MultiTracker 
    for bbox in bboxes:
        multiTracker.add(createTrackerByName(tracker_type), video_frame, bbox)

    filewriter =  open("modify_code/data/tracker.txt", "w+")
    filewriter.write("Arr:   Lea:  \n")

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('track_point.mp4', fourcc, 10, (640, 360))

    ## Video frame index
    i = -1

    while video.isOpened():
        i += 1

        ## Read a new frame
        video_ret, video_frame = video.read()
        original_ret, original_frame = original_video.read()
        
        if (not video_ret) or (not original_ret):
            break

        ## Start timer
        timer = cv2.getTickCount()

        ## Update tracker
        ret, boxes = multiTracker.update(video_frame)
        print('bounding boxes position {}'.format(boxes))
        ## Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        ## draw tracked objects
        if video_ret:
            for index, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(video_frame, p1, p2, colors[index], 2, 1)
                filewriter.write("object" + str(i) + "frame" + str(index) +  ": " + str(newbox[0]) + ", " + str(newbox[1]) + "\n")
                if toimage == 1:
                    if i%35 == 0:
                        cv2.imwrite(video_path + "object%dframe%d.jpg" % (index, i), video_frame)
                        region = original_frame[int(newbox[1]):int(newbox[1])+int(newbox[3]), int(newbox[0]):int(newbox[0])+int(newbox[2])]
                        cv2.imwrite(video_path + "object%dregion%d.jpg" %  (index, i), region)
        else:
            cv2.putText(video_frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
        
        ## Display tracker type on frame
        cv2.putText(video_frame, tracker_type + " Tracker", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2)
        ## Display FPS on frame
        cv2.putText(video_frame, "FPS : " + str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2)
        ## Display frame index  on frame
        cv2.putText(video_frame, "Frame : " + str(i), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2)

        # out.write(frame)
        ## show frame
        cv2.imshow("Tracking", video_frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
    
    filewriter.close()
    # out.release()

if __name__ == '__main__' :
    # major_ver, minor_ver, subminor_ver = (cv2.__version__).split('.')

    ## Set up data path & Whether to record the tracking result & Set up tracker type
    video_path = "modify_code/data/"
    toimage = 1
    tracker_type = 'CSRT'

    multiTracking(video_path, toimage, tracker_type)

   