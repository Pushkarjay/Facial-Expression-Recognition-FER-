import cv2
import torch
import torch.nn.functional as F
from utils import CNN_SNN, ensemble_predict
from torchvision import transforms
from PIL import Image
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7
emotion_labels = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

model_paths = [
    "models/snn_model_1.pth",
    "models/snn_model_2.pth",
    "models/snn_model_3.pth"
]

models = []
for path in model_paths:
    model = CNN_SNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    model.eval()
    models.append(model)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])


risk_mapping = {
    "sad": ("High Risk", (0, 0, 255)),         # Red
    "fearful": ("Medium Risk", (0, 165, 255)), # Orange
    "angry": ("Medium Risk", (0, 165, 255)),   # Orange
    "neutral": ("Low Risk", (0, 255, 0)),      # Green
    "happy": ("Very Low Risk", (0, 255, 0)),   # Green
    "surprised": ("Low Risk", (0, 255, 0)),    # Green
    "disgusted": ("Low Risk", (0, 255, 0))     # Green
}


cap = cv2.VideoCapture(0)
print("🎥 Press 'Q' to quit the real-time detection window...")

prev_time = 0  # For FPS calculation

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        face_img_resized = cv2.resize(face_img, (48, 48))
        face_pil = Image.fromarray(face_img_resized)           # Convert to PIL Image
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        # Ensemble prediction
        with torch.no_grad():
            outputs = ensemble_predict(models, face_tensor, device)
            probs = F.softmax(outputs, dim=1)
            emotion_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][emotion_idx].item()

        emotion = emotion_labels[emotion_idx]
        label = f"{emotion} ({confidence * 100:.1f}%)"

        # Suicide-risk label & color
        risk_level, color = risk_mapping.get(emotion, ("Unknown", (255, 255, 255)))

        # Draw bounding boxes and labels
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Risk: {risk_level}", (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Real-Time Emotion & Suicide Risk Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
