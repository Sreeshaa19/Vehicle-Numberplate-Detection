import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import pytesseract
from PIL import Image


model = YOLO("/content/YOLO_Results/vehicle_numberplate_v8_aug/weights/best.pt")  


reader = easyocr.Reader(['en'])
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.title("🚘 Vehicle & Number Plate Detection with OCR")
st.write("Upload an image or use webcam to detect vehicles and extract number plates.")


uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])


use_webcam = st.checkbox("Use Webcam")

def detect_and_ocr(image):
    results = model.predict(image, conf=0.4)  
    annotated = results[0].plot()  
    detected_texts = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        if label == "Number plate":  
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = image[y1:y2, x1:x2]

            if plate_img.size == 0:
                continue

            
            plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            _, plate_bin = cv2.threshold(plate_gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            
            easy_text = reader.readtext(plate_bin, detail=0)
            easy_text = easy_text[0] if easy_text else None

            
            tess_text = pytesseract.image_to_string(plate_bin, config="--psm 7").strip()

            detected_texts.append({
                "bbox": (x1, y1, x2, y2),
                "easyocr": easy_text,
                "tesseract": tess_text
            })

    return annotated, detected_texts

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    annotated, texts = detect_and_ocr(img)

    st.image(annotated, caption="Detection Results", use_column_width=True)
    st.subheader("📑 OCR Results")
    for t in texts:
        st.write(f"**EasyOCR:** {t['easyocr']} | **Tesseract:** {t['tesseract']}")

elif use_webcam:
    st.write("Click 'Start' below to capture webcam feed.")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        annotated, texts = detect_and_ocr(frame)
        st.image(annotated, caption="Webcam Detection", use_column_width=True)
        st.subheader("📑 OCR Results")
        for t in texts:
            st.write(f"**EasyOCR:** {t['easyocr']} | **Tesseract:** {t['tesseract']}")
    cap.release()
