# FocusFrame
Reformate 16:9 live video to 9:16 aspect reation by cropping most focued object. 

# for starting of project

1.create new env by execute in terminal "python -m venv .venv"

2.then activate it by execute in terminal ".venv\Scripts\activate"

3. pip install -r requirements.txt

3.adjust video input path,output path and number of frame to process

4.then just run app1.py and app2.py


# apporach 1 (app1.py)

Overview
used Yolo11x to first identify object then storing all detection coordinate of boxes in variable.
priotizing accourdingly(read apporach 1) and then calculate mid value of most focused object and cropped the main frame
if no object found take middle point of video as mid value.
more info in approach 1 file.


# apporach 2 (app2.py)
Overview
used Yolo11x to first track object then storing all detection coordinate of boxes in a variable.
here i created a point system in whcih default value is 102 for person 101 for living things and 100 for others.
if any objects moves most in video add +3% to that object's score.
if any ojects is fixed to one place then -3% to score.
and more rules in approach 2 file
