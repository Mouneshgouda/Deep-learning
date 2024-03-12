import argparse
import cv2
import imutils
import datetime
import time

# TODO: add erosion before dilation
#       optimum sensitivity

# Constructing a parser
ap = argparse.ArgumentParser()

# Adding arguments
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

# Initialize the video stream from the webcam
vs = cv2.VideoCapture(0)  # 0 represents the default webcam

# First frame of the Video
firstFrame = None

while True:
    # grab the current frame and initialize the occupied/unoccupied text
    ret, frame = vs.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    text = "Unoccupied"

    # resize the frame, convert it to grayscale, and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue

    # compute the absolute difference between the current frame and first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    # assign a threshold
    thresh = cv2.threshold(frameDelta, 100, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes,
    # then find contours on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    firstFrame = gray
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"

    # draw the text and timestamp on the frame
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # show the frame and record if the user presses a key
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    
    # Check for user input to quit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the video stream and close all windows
vs.release()
cv2.destroyAllWindows()
