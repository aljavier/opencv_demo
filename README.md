# Face Recognition with OpenCV 

This is an example of face recognition with OpenCV, this was for an assignament in
college, but here it is for reference and help for anyone who needed. 

## Requirements

* Numpy
	```
	sudo pip install numpy
	```
* OpenCV
	```
	sudo pip install opencv-contrib-python
	```
* Pillow (Python Image Library, PIL)
	```
	sudo pip install pillow
	```

## How to use

First, you need to run `trainer.py` to create a *trainer.yml* file, which is the one
with the info from your database of images. Therefore, you also need a database of images.
Which is simply, a folder with a bunch of images of people with faces.

```
trainer.py images_folder/
```

This creates the file *trainer.yml*, which is used by `face_detector.py`.

Then, you can just ran the `face_detector.py` script passing as an argument the image of the 
person you want to check if exists in your *database of images* (the one used for the `trainer.py`).

```face_detector.py ~/downloads/sammy_sosa.png```

![Demo](opencv_demo.gif)
