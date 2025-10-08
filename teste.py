import cv2
import mediapipe as mp
import numpy as np # Adicione esta linha


def calculate_angle(a, b, c):
    """Calcula o ângulo entre três pontos (em graus)."""
    a = np.array(a)  # Primeiro ponto
    b = np.array(b)  # Ponto do meio (vértice do ângulo)
    c = np.array(c)  # Terceiro ponto
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Initialize MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully. Press 'q' to quit.")

while cap.isOpened():
    # Read a frame from the webcam
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for a later selfie-view display
    # This makes the output feel more like a mirror.
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB.
    # OpenCV reads images in BGR format, but MediaPipe requires RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB image to find the pose landmarks
    results = pose.process(rgb_frame)

    # Draw the pose annotation on the BGR image.
    # We draw on the original BGR frame.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        try:
            # Extrai as coordenadas dos marcos
            landmarks = results.pose_landmarks.landmark
            
            # Coordenadas para o agachamento (lado esquerdo do corpo)
            # O MediaPipe retorna uma lista, então usamos o enum para pegar o índice correto
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # O código de cálculo e lógica virá aqui...

            # Calcula o ângulo do joelho
            knee_angle = calculate_angle(hip, knee, ankle)
            print(f"Ângulo do Joelho: {knee_angle}") # Linha para debug, pode remover depois

            # Calcula o ângulo do quadril (para verificar a postura das costas)
            hip_angle = calculate_angle(shoulder, hip, knee)
            print(f"Ângulo do Quadril: {hip_angle}") # Linha para debug

        except Exception as e:
            print(f"Erro ao processar os marcos: {e}")


    # Display the resulting frame
    cv2.imshow('MediaPipe Pose Estimation', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
pose.close()

print("Script finished.")
