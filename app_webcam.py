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
categories = weights.meta["categories"] ## ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',]
img_preprocess = weights.transforms() ## Scales values from 0-255 range to 0-1 range.

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval() ## Setting Model for Evaluation/Prediction   
    return model

model = load_model()

def make_prediction(img): 
    img_processed = img_preprocess(img) ## (3,500,500) 
    prediction = model(img_processed.unsqueeze(0)) # (1,3,500,500)
    prediction = prediction[0]                       ## Dictionary with keys "boxes", "labels", "scores".
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction): ## Adds Bounding Boxes around original Image.
    img_tensor = torch.tensor(img) ## Transpose
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"], labels=prediction["labels"],
                                          colors=["red" if label=="dog" else "green" for label in prediction["labels"]], width=2)
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1,2,0) ### (3,W,H) -> (W,H,3), Channel first to channel last.
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

def detect_dog_in_frame(frame):
    img = Image.fromarray(frame)
    prediction = make_prediction(img)
    img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2,0,1), prediction) ## (W,H,3) -> (3,W,H)
    dog_detected = any(label == "dog" for label in prediction["labels"])
    return img_with_bbox, dog_detected

def play_video(video_source):
    camera = cv2.VideoCapture(video_source)

    st_frame = st.empty()
    while(camera.isOpened()):
        ret, frame = camera.read()

        if ret:
            st_frame.image(frame, channels = "BGR")
        else:
            camera.release()
            break

st.title("Dog Detector :")
use_webcam = st.checkbox("Use Webcam")
upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])

if use_webcam:
    st.write("Webcam feed will be displayed below. Please wait for the camera to start.")
    camera = cv2.VideoCapture(0)
    st_frame = st.empty()
    while camera.isOpened():
        ret, frame = camera.read()
        if ret:
            img_with_bbox, dog_detected = detect_dog_in_frame(frame)
            st_frame.image(frame, channels = "BGR")

        else:
            break
        
        if dog_detected:
            st.image(img_with_bbox, channels="BGR")
            st.header("Dog Detected!")
            st.write("A dog has been detected in the image.")
            
            image_rgb = cv2.imread('./image_with_bboxes.png', cv2.IMREAD_UNCHANGED)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite('./image_with_bboxes.png', image_bgr)
            
            send_email("Dog Detection Alert", "A dog has been detected in the image. Please find the image attached.", "sendEmail", "receiveEmail", "sendEmailPW", './image_with_bboxes.png')
            break

    camera.release()
    cv2.destroyAllWindows()

elif upload:
    img = Image.open(upload)

    prediction = make_prediction(img) ## Dictionary
    img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2,0,1), prediction) ## (W,H,3) -> (3,W,H)


    dog_detected = False
    for label in prediction["labels"]:
        if label == "dog":
            dog_detected = True
            break

    st.image(img_with_bbox)
    
    if dog_detected:
        st.header("Dog Detected!")
        st.write("A dog has been detected in the image.")
    else:
        st.header("No Dog Detected")
        st.write("No dog has been detected in the image.")

    # Send the email
    if dog_detected:
        send_email("Dog Detection Alert", "A dog has been detected in the image. Please find the image attached.", "sendEmail", "receiveEmail", "sendEmailPW", './image_with_bboxes.png')
