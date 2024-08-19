import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
import os
import cv2

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
all_categories = weights.meta["categories"]  # Full category list
animal_categories = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"] # animal categories
categories_not = [i for i in all_categories if i not in ['N/A', '__background__']]  # Filter out 'N/A' and '__background__'
categories = [i for i in categories_not if i in animal_categories]

img_preprocess = weights.transforms()  # Scales values from 0-255 range to 0-1 range.

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()  # Setting Model for Evaluation/Prediction   
    return model

model = load_model()

def make_prediction(img):
    img_processed = img_preprocess(img)  # (3,500,500) 
    prediction = model(img_processed.unsqueeze(0))  # (1,3,500,500)
    prediction = prediction[0]  # Dictionary with keys "boxes", "labels", "scores".
    prediction["labels"] = [all_categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction, selected_category):  # Adds Bounding Boxes around original Image.
    img_tensor = torch.tensor(img)  # Transpose
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"], labels=prediction["labels"],
                                          colors=["red" if label == selected_category else "green" for label in prediction["labels"]], width=2)
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)  # (3,W,H) -> (W,H,3), Channel first to channel last.
    return img_with_bboxes_np

def send_email(subject, message, from_addr, to_addr, password, img_path):
    msg = MIMEMultipart()
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg['Subject'] = subject

    body = message
    msg.attach(MIMEText(body, 'plain'))

    img_data = open(img_path, 'rb').read()
    image = MIMEImage(img_data, name=os.path.basename(img_path))
    msg.attach(image)

    try:
        server = smtplib.SMTP('smtp.naver.com', 587)
        server.starttls()
        server.login(from_addr, password)
        text = msg.as_string()
        server.sendmail(from_addr, to_addr, text)
        server.quit()
        print("Email sent successfully!")
    except smtplib.SMTPAuthenticationError:
        print("Error: Unable to authenticate with the SMTP server.")
    except smtplib.SMTPException as e:
        print(f"Error: {e}")
    except TimeoutError:
        print("Error: Connection timed out. Please try again.")
    except Exception as e:
        print(f"Error: {e}")

def detect_object_in_frame(frame, selected_category):
    img = Image.fromarray(frame)
    prediction = make_prediction(img)
    img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2, 0, 1), prediction, selected_category)  # (W,H,3) -> (3,W,H)
    object_detected = any(label == selected_category for label in prediction["labels"])
    return img_with_bbox, object_detected


st.title("Object Detector:")
selected_category = st.selectbox("Select a category to detect:", categories)
use_webcam = st.checkbox("Use Webcam")


if use_webcam:
    st.write("Webcam feed will be displayed below. Please wait for the camera to start.")
    camera = cv2.VideoCapture(0)
    st_frame = st.empty()
    while camera.isOpened():
        ret, frame = camera.read()
        if ret:
            img_with_bbox, object_detected = detect_object_in_frame(frame, selected_category)
            st_frame.image(frame, channels="BGR")

        else:
            break

        if object_detected:
            st.image(img_with_bbox, channels="BGR")
            st.header(f"{selected_category} Detected!")
            st.write(f"A {selected_category} has been detected in the image.")

            cv2.imwrite('image_with_bboxes.png', cv2.cvtColor(img_with_bbox, cv2.COLOR_RGB2BGR))
            image_rgb = cv2.imread('image_with_bboxes.png', cv2.IMREAD_UNCHANGED)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite('image_with_bboxes.png', image_bgr)

            break

    
    send_email(f"{selected_category} Detection Alert", f"A {selected_category} has been detected in the image. Please find the image attached.", "sendEmail", "receiveEmail", "sendEmailPW", 'image_with_bboxes.png')

    camera.release()
    cv2.destroyAllWindows()

else:
    upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])
    
    if upload:
        img = Image.open(upload)

        prediction = make_prediction(img)  # Dictionary
        img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2, 0, 1), prediction, selected_category)  # (W,H,3) -> (3,W,H)

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        plt.imshow(img_with_bbox)
        plt.xticks([], [])
        plt.yticks([], [])
        ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

        st.image(img_with_bbox)

        object_detected = any(label == selected_category for label in prediction["labels"])

        if object_detected:
            st.header(f"{selected_category} Detected!")
            st.write(f"A {selected_category} has been detected in the image.")
        else:
            st.header(f"No {selected_category} Detected")
            st.write(f"No {selected_category} has been detected in the image.")

        # Save the image with bounding boxes to a file
        plt.savefig('image_with_bboxes.png')
        
        # Send the email
        if object_detected:
            send_email(f"{selected_category} Detection Alert", f"A {selected_category} has been detected in the image. Please find the image attached.", "sendEmail", "receiveEmail", "sendEmailPW", 'image_with_bboxes.png')
