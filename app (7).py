
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import easyocr
import pytesseract
import tempfile
import time


model = YOLO("/content/best (5).pt")
reader = easyocr.Reader(['en'])

st.title("🚘 Vehicle & Number Plate Detection with OCR")


classes = ["All", "Car", "Truck", "Bus", "Bike", "Person", "Number plate"]
selected_class = st.selectbox("Select Class to Display", classes)


uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    np_img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_display = np_img.copy()

    
    start_time = time.time()

    
    results = model.predict(np_img)

    
    inference_time = time.time() - start_time

    output_data = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for i, cls_id in enumerate(class_ids):
            class_name = result.names[int(cls_id)]
            conf_score = confidences[i]

            if selected_class != "All" and class_name != selected_class:
                continue

            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_display, f"{class_name} {conf_score:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            easyocr_text = ""
            tesseract_text = ""

            
            if class_name == "Number plate":
                plate_img = np_img[y1:y2, x1:x2]

                
                ocr_result = reader.readtext(plate_img)
                easyocr_text = " ".join([res[1] for res in ocr_result])

                
                gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                _, thresh_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                tesseract_text = pytesseract.image_to_string(thresh_plate, config='--psm 7').strip()

                cv2.putText(img_display, easyocr_text, (x1, y2+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                cv2.putText(img_display, tesseract_text, (x1, y2+40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            output_data.append({
                "Class": class_name,
                "Confidence": conf_score,
                "EasyOCR": easyocr_text,
                "Tesseract": tesseract_text,
                "Box": f"[{x1},{y1},{x2},{y2}]"
            })

    st.image(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    st.write(f"**Inference Time:** {inference_time:.3f} seconds")

    
    df = pd.DataFrame(output_data)
    st.dataframe(df)

   
    if not df.empty:
        tmp_download_link = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(tmp_download_link.name, index=False)
        st.download_button("Download Detection CSV", tmp_download_link.name, file_name="detection_results.csv")
