# USAGE
# python pi_face_recognition.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import os
import collections
import sys
import logging
import threading
from datetime import datetime

def video():
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    # vs = VideoStream(src=0).start()
    vs = VideoStream(usePiCamera=True).start()

    # the raspi says something at startup (to check if the speakers are connected)
    def robot(text):
        os.system("espeak ' " + text + " ' ")

    robot("Hi this is flume")
    time.sleep(2.0)

    # start the FPS counter
    fps = FPS().start()

    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        frame = cv2.flip(frame,-1)

        # convert the input frame from (1) BGR to grayscale (for face
        # detection) and (2) from BGR to RGB (for face recognition)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect faces in the grayscale frame
        rects = detector.detectMultiScale(gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        # OpenCV returns bounding box coordinates in (x, y, w, h) order
        # but we need them in (top, right, bottom, left) order, so we
        # need to do a bit of reordering
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

        # compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(rgb, boxes)
        

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                encoding)
            name = "Unknown"
            probability = 0

            # check to see if we have found a match
            if True in matches:
                print("Found matches for the face in the video!")
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # Iterate over key/value pairs in dict and print them
                for name, count in counts.items():
                    print(name, ' : ', count)

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)
                print("The most matched name = " + name)

                print("Dataset occurences = " + str(name_occurrences.get(name)))

                probability = counts[name]/name_occurrences.get(name)
                print("Probability = " + str(probability))
                print("")


                # take a picture from the matched face
           #     now = datetime.now()
           #     dt_string = now.strftime("%d%m%Y_%H:%M:%S")
           #     name_now = dt_string + name
           #     os.system("raspistill -vf -hf -o  " + name_now + ".jpg -t 500")

            if(probability > 0.8):
                if(name not in names):
                    # update the list of names
                    names.append(name)
                    print(names)
                    if name != 'Sandra':
                        robot("Hi " + name)

                    if name == 'Sandra':
                        robot("Hi " + name + " kissies from bae")
                #else:
                 #   pass
                    # robot("I already recognized you " + name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
                (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            text = "Name: " + name + " Probability: " + str(round(probability, 2)) + "%"
            cv2.putText(frame, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)

        # display the image to our screen
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

def empty_names():
    names.clear()


def empty_names_repeat():
    empty_names()
    threading.Timer(WAIT_SECONDS, empty_names_repeat).start()



if __name__== "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--cascade", required=True,
        help = "path to where the face cascade resides")
    ap.add_argument("-e", "--encodings", required=True,
        help="path to serialized db of facial encodings")
    args = vars(ap.parse_args())

    # load the known faces and embeddings along with OpenCV's Haar
    # cascade for face detection
    print("[INFO] loading encodings + face detector...")
    data = pickle.loads(open(args["encodings"], "rb").read())
    detector = cv2.CascadeClassifier(args["cascade"])

    # explore the existing person images dataset (necesary to improve the model later on)
    name_occurrences = collections.Counter(data.get("names"))
    print("Dataset pictures used:")
    print(name_occurrences)
    print("")
    names = []
    
    # reset the recognised names list periodically to be recognised again
    WAIT_SECONDS = 60
    empty_names_repeat()

    # start the video-stream
    video()
    

    
    

    
    
    
