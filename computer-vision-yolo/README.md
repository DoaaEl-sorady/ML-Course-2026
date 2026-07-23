# YOLO Object Detection App

A simple web app built with **Streamlit** and **YOLOv8** (Ultralytics) that detects objects in an uploaded image.

## What it does

1. The user uploads an image (JPG, PNG, or WEBP).
2. The YOLOv8 model detects objects in the image (people, cars, motorcycles, etc.).
3. The app draws bounding boxes around each detected object.
4. The app counts how many objects of each type were found and displays them as metric cards.

## How to run it locally

1. Clone this repository and navigate to this folder.
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```
4. The app will open in your browser. Upload an image to see the detection results.

## Tech stack

- [Streamlit](https://streamlit.io/) — web app framework
- [Ultralytics YOLOv8](https://docs.ultralytics.com/) — object detection model
- [Pillow](https://python-pillow.org/) — image handling
- [NumPy](https://numpy.org/) — array/image data conversion

## Example output

The app displays the annotated image with bounding boxes, followed by a count of each detected object type (e.g. `car: 4`, `person: 7`, `motorcycle: 1`).
