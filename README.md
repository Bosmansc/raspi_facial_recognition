# raspi_facial_recognition

This project tries to recognize faces using a raspberry pi. 

Based on this blogpost:

https://www.pyimagesearch.com/2018/06/25/raspberry-pi-face-recognition/

## Preparing the data

Use the correct python virtual environment (workon cv)

Make a folder for each person with a collection of pictures.

Create facial embeddings based on the pictures: 

```
python encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method hog
```

This command creates the pickle file. (--detection-method cnn argument can be used on another device than the raspi)

## Run the model

Start the facial recognition: (make sure you are in the python virtualenv)

```
python pi_face_recognition.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle
```

