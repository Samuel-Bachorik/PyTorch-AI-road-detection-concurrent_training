import cv2
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from segmentation_inference import *

#Load desired video
cap = cv2.VideoCapture("C:/Users/Samuel/PycharmProjects/Condapytorch/Mesto8.mp4")

show_video = False
save_video = True

# Set same height and width as trained model have
height = 480
width = 640

#Load segmentation inference
si = SegmentationInference(2)

if save_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('output.avi', fourcc, 25.0, (width, height))

fps_smooth = 0.0
frame_skip = 20
next_frame = 0
cnt = 0

def print_video(image, text):
    x = cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

while (True):
    ret, frame = cap.read()

    if ret == False:
        break

    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    if cnt > next_frame:
        time_start = time.time()

        # Sent frame to segmentation inferece and get result
        prediction_np, mask, result = si.process(frame)
        #Count FPS

        time_stop = time.time()
        fps = 1.0 / (time_stop - time_start)
        result = (result * 255).astype(numpy.uint8)


        # Print FPS
        text = "fps= " + str(round(fps, 1))
        
        im_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        print_video(im_bgr, text)

        if show_video:
            cv2.imshow('frame', im_bgr)

        if save_video:
            writer.write(im_bgr)

        frame_skip = 25 / fps
        frame_skip = int(numpy.clip(frame_skip, 1, 500))

        next_frame = cnt + frame_skip

        #print(fps, frame_skip, fps * frame_skip)

    cnt += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
