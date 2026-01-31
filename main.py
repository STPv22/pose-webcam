import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
# Holistic combines face, hands, and pose into one efficient model
holistic = mp_holistic.Holistic(
    model_complexity=0, # 0 is critical for CPU latency
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
skip_frames = 3  # Increase this if you still feel lag

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    # Performance Hack: Increment counter
    frame_count += 1
    
    # Only process MediaPipe every 'skip_frames'
    if frame_count % skip_frames == 0:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = holistic.process(img_rgb)
        img_rgb.flags.writeable = True

        # Draw the results (Face, Hands, etc)
        if results.face_landmarks:
            mp_draw.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        if results.left_hand_landmarks:
            mp_draw.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_draw.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        if results.face_landmarks:
            landmarks = results.face_landmarks.landmark
            
            # 1. Mouth Open Detection (Surprise)
            # Point 13 (Upper Lip) and Point 14 (Lower Lip)
            mouth_open = abs(landmarks[13].y - landmarks[14].y)
            
            # 2. Smile Detection
            # Point 61 (Left corner) and Point 291 (Right corner)
            mouth_width = abs(landmarks[61].x - landmarks[291].x)
            
            # Simple Logic Thresholds (You may need to tweak these numbers)
            expression = "Neutral"
            if mouth_open > 0.05:
                expression = "Surprise / Mouth Open"
            elif mouth_width > 0.15:
                expression = "Smiling"

            # Display it on your Bazzite GUI
            cv2.putText(img, expression, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Zero Latency Mode", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
