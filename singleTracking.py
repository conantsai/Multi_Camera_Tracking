import cv2
import sys
import os

def singleTracking(video_path, toimage=0, tracker_type='CSRT'):
    ## Set up tracker.
    # tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

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

    ## Create a video capture object to read video
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

    video.set(cv2.CAP_PROP_POS_MSEC, 2000)
    original_video = video

    ## Read first frame.
    video_ret, video_frame = video.read()
    if not video_ret:
        print("Cannot read video file")
        sys.exit()

    ## Define an initial bounding box
    # initbox = (67, 19, 25, 60)
    ## Uncomment the line below to select a different bounding box
    initbox = cv2.selectROI(video_frame, False)
    print("initial bounding box size:", initbox[2], initbox[3])

    bbox = initbox

    filewriter =  open("modify_code/data/tracker.txt", "w+")
    filewriter.write("Arr:   Lea:  \n")

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('track_point.mp4', fourcc, 10, (640, 360))

    ## Initialize tracker with first frame and bounding box
    tracker.init(video_frame, bbox)

    ## Video frame index
    i = -1

    while video.isOpened():
        i += 1

        ## Read a new frame
        video_ret, video_frame = video.read( )
        original_ret, original_frame = original_video.read( )
        
        if (not video_ret) or (not original_ret):
            break

        ## Start timer
        timer = cv2.getTickCount()

        ## Update tracker
        ret, bbox = tracker.update(video_frame)
        print("bounding box's position:", bbox)
        ## Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        ## Draw bounding box
        if video_ret:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(video_frame, p1, p2, (255,0,0), 2, 1)

            filewriter.write("frame" + str(i) + ": " + str(bbox[0]) + ", " + str(bbox[1]) + "\n")
            if toimage == 1:
                if i%35 == 0:
                    cv2.imwrite(path + "frame2%d.jpg" % i, video_frame)

                    region = original_frame[int(bbox[1]):int(bbox[1])+int(bbox[3]), int(bbox[0]):int(bbox[0])+int(bbox[2])]
                    cv2.imwrite(path + "region%d.jpg" % i, region)

        else:
            cv2.putText(video_frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

        ## Display tracker type on frame
        cv2.putText(video_frame, tracker_type + " Tracker", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2)
        ## Display FPS on frame
        cv2.putText(video_frame, "FPS : " + str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2)
        ## Display frame index  on frame
        cv2.putText(video_frame, "Frame : " + str(i), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2)

        # out.write(frame)
        ## Display result
        cv2.imshow("Tracking", video_frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

    filewriter.close()
    # out.release()


if __name__ == '__main__' :
    # major_ver, minor_ver, subminor_ver = (cv2.__version__).split('.')

    ## Set up data path & Whether to record the tracking result
    path = "modify_code/data/"
    toimage = 1

    singleTracking(path, toimage=1, tracker_type='CSRT')

    