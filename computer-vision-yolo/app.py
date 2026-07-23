"""
YOLO Object Detection App
--------------------------
A simple Streamlit app that lets the user upload an image, runs YOLOv8
object detection on it, draws bounding boxes on the detected objects,
and displays a count of each object type found (e.g. how many cars,
people, motorcycles, etc.).
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import Counter

# تحميل موديل YOLO المدرب مسبقاً على اكتشاف 80 نوع جسم مختلف
model = YOLO("yolov8n.pt")

st.title("YOLO Detection")

# زرار لرفع صورة من المستخدم بامتدادات محددة فقط
file = st.file_uploader(
    "Upload Image", type=["jpg", "jpeg", "png", "webp"]
)

if file:
    # فتح الصورة اللي رفعها المستخدم
    image = Image.open(file)

    # تشغيل الموديل على الصورة بعد تحويلها لمصفوفة أرقام (numpy array)
    results = model(
        np.array(image)
    )

    # رسم المربعات والأسماء على الصورة الأصلية
    result = results[0].plot()

    st.image(result)

    # استخراج رقم (index) كل جسم اتكتشف في الصورة
    class_indices = results[0].boxes.cls.tolist()

    # تحويل كل رقم لاسم الجسم الحقيقي باستخدام قاموس الموديل model.names
    class_names = [model.names[int(i)] for i in class_indices]

    # عد كام مرة كل اسم جسم اتكرر في الصورة
    counts = Counter(class_names)

    # عرض عدد كل نوع جسم اتكتشف في شكل كروت جنب بعض
    st.subheader("Detected Objects")

    # إنشاء عدد أعمدة (columns) يساوي عدد الأنواع المختلفة اللي اتكتشفت
    cols = st.columns(len(counts))

    # المتغير i بيمثل رقم الكارت (0, 1, 2...) عشان نحدد هيتحط في أي عمود
    for i, (name, count) in enumerate(counts.items()):
        with cols[i]:
            st.metric(label=name, value=count)
