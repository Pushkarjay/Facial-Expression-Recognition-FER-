# realtime_detection.py
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from collections import deque
from PIL import Image
import numpy as np
from utils import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- CNN MODEL --------------------- #
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512*3*3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

_, _, num_classes = load_data("data/raw")
model = EmotionCNN(num_classes).to(device)
model.load_state_dict(torch.load("models/emotion_cnn_model.pth", map_location=device))
model.eval()

emotion_labels = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
risk_map = {
    "angry": 0.4,
    "disgusted": 0.5,
    "fearful": 0.6,
    "happy": 0.0,
    "neutral": 0.2,
    "sad": 0.8,
    "surprised": 0.3
}

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

recent_emotions = deque(maxlen=30) 


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("🎥 Press 'C' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_emotion = "neutral"

    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi_pil = Image.fromarray(roi) 
        roi_tensor = transform(roi_pil).unsqueeze(0).to(device) 

        with torch.no_grad():
            outputs = model(roi_tensor)
            pred = torch.argmax(outputs, dim=1).item()
            current_emotion = emotion_labels[pred]

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, current_emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    recent_emotions.append(current_emotion)

    risk_score = np.mean([risk_map[e] for e in recent_emotions]) if recent_emotions else 0

    if risk_score < 0.3:
        risk_level = "LOW"
        color = (0,255,0)
    elif risk_score < 0.6:
        risk_level = "MODERATE"
        color = (0,255,255)
    else:
        risk_level = "HIGH"
        color = (0,0,255)

    bar_x, bar_y = 50, frame.shape[0]-60
    bar_width = int(risk_score*300)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+300, bar_y+30), (50,50,50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_width, bar_y+30), color, -1)
    cv2.putText(frame, f"Risk: {risk_level}", (bar_x, bar_y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if risk_level=="HIGH":
        cv2.putText(frame, "⚠ Emotional distress detected!",
                    (bar_x, bar_y-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)

    cv2.imshow("CNN Emotion Detection + Risk Level", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Stream closed.")
