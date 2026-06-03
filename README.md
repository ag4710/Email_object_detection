# NOpineapple
https://sites.google.com/view/nopineapple

***

# 🐾 Wildlife Guardian: Email Object Detection Cam Alert

This project was developed by **ITE + BRAIN AI CO. LTD**.

## 📌 About the Project (Social Impact)
**"Wildlife Guardian" - AI-powered Conservation in Korea**

**Background & Problem Statement:**
Korea's endangered turtle species and other wildlife face severe challenges from habitat loss, illegal hunting, and the rapid growth of invasive species. Despite the efforts of conservation organizations and local communities, they struggle with inadequate funding and poor monitoring systems, which puts the marine ecosystems and species survival at high risk. 

**Our Objective & 4W's:**
*   **Who:** We aim to assist Korea marine researchers and conservation centers, including the National Institute of Fisheries Science (NIFS).
*   **What:** We are providing an AI-powered real-time detection and classification system to overcome inadequate monitoring.
*   **Where:** The primary focus is on Korea's wetlands and specifically Jeju Island, where the majority of endangered turtle species reside.
*   **Why:** Protecting these animals ensures marine ecosystem sustainability and boosts the local economy through eco-tourism.

**Solution & Implementation:**
To safeguard these habitats, this project utilizes AI and computer vision to execute real-time inference on edge devices (such as cameras). Whenever a target species is discovered in the frame, the system acts as an **Alert System** that instantly captures the image and notifies conservationists and authorities via email so they can take immediate action. 

---

## 🚀 Development Phases (Version History)
Our application has been developed through continuous improvements:

1.  **`dog detection v1`**: The initial phase supported basic image uploads. It detected whether the uploaded image contained a dog and notified the user via email.
2.  **`dog detection v2`**: Upgraded the system by adding a real-time detection feature via webcam.
3.  **`animal detection v1` (Final Version)**: Expanded the capability from just dogs to a user-selectable list of multiple animals. Currently, the system can detect birds, cats, dogs, horses, sheep, cows, elephants, bears, zebras, and giraffes.

---

## ✨ Key Features
Based on the final **`animal detection v1`**, the application includes the following functionalities:

*   **Custom Target Selection**: Users can utilize a dropdown menu to select the exact animal category they want the camera to detect.
*   **Real-Time Webcam Inference**: Actively monitors the live video feed using the computer's webcam.
*   **Bounding Box Visualization**: Draws distinct bounding boxes around detected objects. The user-selected target animal is highlighted with a **red box**, while any other recognized objects are marked with a **green box** for clear distinction.
*   **Automated Email Alerts**: The moment the target animal is spotted, the system saves the captured frame (`image_with_bboxes.png`) and automatically dispatches an email alert with the attached image to the designated administrator.
*   **Image File Uploading**: Provides a secondary feature to upload static images (`png`, `jpg`, `jpeg`) for detection if a webcam is unavailable.

---

## 🛠 Technology Stack
*   **UI Framework**: Streamlit (For web deployment and dashboard interface)
*   **AI Model**: PyTorch, `fasterrcnn_resnet50_fpn_v2` (A two-stage object detection model with ResNet-50 backbone and Feature Pyramid Network for strong multi-scale feature extraction)
*   **Image Processing**: OpenCV (`cv2`) for webcam handling, and PIL
*   **Email Automation**: Python's `smtplib`, `email.mime`

---

## ⚙️ How it Works
1.  **Data Inference**: The model takes the live frame, resizes it, and normalizes pixel values to make predictions using pre-trained weights (COCO dataset).
2.  **Detection & Alert**: If the `selected_category` matches the model's prediction, the system applies the bounding box, saves the BGR image, logs into the SMTP server, and immediately sends the alert email.
