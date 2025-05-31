import cv2
import numpy as np
import os
import sys

# --- Face Detection Function (remains the same as in your previous script) ---
def highlightFace(net, frame, conf_threshold=0.5):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], swapRB=True, crop=False) # Assuming swapRB=True is correct for your face detector

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes

# --- Main Emotion Prediction Function ---
def predict_emotions_from_webcam():
    project_root = os.path.dirname(os.path.abspath(__file__))

    faceProto = os.path.join(project_root, "Model", "deploy.prototxt.txt")
    faceModel = os.path.join(project_root, "Model", "res10_300x300_ssd_iter_140000.caffemodel")
    emotionModelPath = os.path.join(project_root, "Model", "emotion-ferplus-8.onnx")

    EMOTION_LIST = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

    print("[INFO] Loading models...")
    try:
        faceNet = cv2.dnn.readNet(faceModel, faceProto)
        emotionNet = cv2.dnn.readNet(emotionModelPath)
    except cv2.error as e:
        print(f"[ERROR] Could not load one or more models: {e}")
        sys.exit(1)

    if faceNet.empty() or emotionNet.empty():
        print("[ERROR] One or more models are empty after loading.")
        sys.exit(1)
    else:
        print("[INFO] Models loaded successfully.")

    print("[INFO] Starting video capture...")
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("[ERROR] Could not open video capture device (webcam).")
        sys.exit(1)
    
    print("[INFO] Webcam opened successfully. Press 'q' to quit.")
    padding = 20
    frame_count = 0
    skip_frames = 3 
    
    last_known_face_boxes = []
    last_known_emotions_text = []

    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            print("[INFO] End of video stream or cannot read frame.")
            break

        frame_count += 1
        display_frame = frame.copy() 

        if frame_count % skip_frames == 0:
            frame_with_new_boxes_drawn, current_face_boxes = highlightFace(faceNet, frame, conf_threshold=0.5)
            last_known_face_boxes = current_face_boxes
            last_known_emotions_text = [] 

            if current_face_boxes:
                display_frame = frame_with_new_boxes_drawn
                for faceBox in current_face_boxes:
                    face_roi = frame[max(0, faceBox[1]-padding):
                                 min(faceBox[3]+padding, frame.shape[0]-1), 
                                 max(0, faceBox[0]-padding):
                                 min(faceBox[2]+padding, frame.shape[1]-1)]

                    if face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
                        last_known_emotions_text.append(None)
                        continue

                    gray_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    resized_face = cv2.resize(gray_face_roi, (64, 64)) 

                    cv2.imshow("Emotion Input Debug", resized_face)

                    # ##################################################################
                    # ## THIS IS THE LINE THAT WAS MODIFIED FOR THIS TEST ##
                    # ##################################################################
                    # Test with scalefactor=1.0 (0-255 range) AND subtract mean 127.5
                    emotion_blob = cv2.dnn.blobFromImage(resized_face, 1.0, (64, 64), (127.5), swapRB=False, crop=False)
                    # ##################################################################
                    
                    print(f"[DEBUG] Emotion Blob Shape: {emotion_blob.shape}")
                    
                    emotionNet.setInput(emotion_blob)
                    emotionPreds = emotionNet.forward() 
                    
                    print(f"[DEBUG] Emotion Preds Raw Scores: {emotionPreds[0]}")
                    
                    emotion_index = emotionPreds[0].argmax()
                    emotion = EMOTION_LIST[emotion_index] 
                    print(f"[DEBUG] Predicted Index: {emotion_index}, Emotion: {emotion}")
                    
                    prediction_text = emotion
                    last_known_emotions_text.append(prediction_text)

                    cv2.putText(display_frame, prediction_text, 
                                (faceBox[0], faceBox[1]-10 if faceBox[1]-10 > 10 else faceBox[1]+10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
        else: # Skipped frame
            if last_known_face_boxes:
                frameHeight = display_frame.shape[0] 
                for i, faceBox in enumerate(last_known_face_boxes):
                    cv2.rectangle(display_frame, (faceBox[0], faceBox[1]), (faceBox[2], faceBox[3]), (0, 255, 0), int(round(frameHeight/150)), 8)
                    if i < len(last_known_emotions_text) and last_known_emotions_text[i]:
                        cv2.putText(display_frame, last_known_emotions_text[i], 
                                    (faceBox[0], faceBox[1]-10 if faceBox[1]-10 > 10 else faceBox[1]+10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Detecting Emotion", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    print("[INFO] Video stream closed and windows destroyed.")

if __name__ == "__main__":
    print("[INFO] Starting emotion detector...")
    predict_emotions_from_webcam()