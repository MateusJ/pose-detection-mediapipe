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

# Variáveis para controlar o estado do agachamento
squat_stage = "em pé"  # Pode ser "em pé" ou "agachado"
squat_count = 0

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
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Pontos adicionais para análise mais completa
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            
            # ========== CÁLCULO DOS ÂNGULOS PRINCIPAIS ==========
            
            # 1. Ângulo do JOELHO (hip-knee-ankle)
            # Ideal: ~90° no agachamento, ~170-180° em pé
            knee_angle = calculate_angle(hip, knee, ankle)
            
            # 2. Ângulo do QUADRIL (shoulder-hip-knee)
            # Ideal: ~80-100° no agachamento (mantém tronco relativamente reto)
            hip_angle = calculate_angle(shoulder, hip, knee)
            
            # 3. Ângulo do TRONCO (nose-hip-knee) 
            # Verifica se o tronco está muito inclinado para frente
            trunk_angle = calculate_angle(nose, hip, knee)
            
            # 4. Ângulo do TORNOZELO (knee-ankle-ponta do pé)
            # Para simplificar, verificamos se o joelho não ultrapassa muito o tornozelo
            # Calculamos a distância horizontal entre joelho e tornozelo
            knee_ankle_horizontal_distance = abs(knee[0] - ankle[0])
            
            # ========== ANÁLISE DA FORMA DO AGACHAMENTO ==========
            
            form_feedback = []
            is_correct_form = True
            
            # Verifica se está na posição de agachamento (joelho flexionado)
            if knee_angle < 120:  # Considera agachamento quando joelho < 120°
                # Análise 1: Profundidade do agachamento
                if knee_angle > 100:
                    form_feedback.append("Agachamento raso - Desça mais!")
                    is_correct_form = False
                elif knee_angle < 70:
                    form_feedback.append("Muito profundo - Cuidado com os joelhos!")
                    is_correct_form = False
                else:
                    form_feedback.append("Profundidade CORRETA!")
                
                # Análise 2: Postura do quadril e tronco
                if hip_angle < 70:
                    form_feedback.append("Tronco muito inclinado - Mantenha o peito para cima!")
                    is_correct_form = False
                elif hip_angle > 120:
                    form_feedback.append("Fique mais sentado - Empine o quadril para trás!")
                    is_correct_form = False
                else:
                    form_feedback.append("Postura do tronco CORRETA!")
                
                # Análise 3: Posição do joelho em relação ao tornozelo
                if knee_ankle_horizontal_distance > 0.15:  # Joelho muito à frente
                    form_feedback.append("Joelho muito à frente - Empurre o quadril para trás!")
                    is_correct_form = False
                else:
                    form_feedback.append("Posição do joelho CORRETA!")
            
            # ========== CONTADOR DE REPETIÇÕES ==========
            
            # Lógica para contar repetições
            if knee_angle > 160:
                squat_stage = "em pé"
            if knee_angle < 90 and squat_stage == "em pé":
                squat_stage = "agachado"
                squat_count += 1
            
            # ========== INTERFACE VISUAL ==========
            
            # Converter coordenadas normalizadas para pixels
            h, w, _ = frame.shape
            
            # Exibir ângulos na tela
            cv2.putText(frame, f'Joelho: {int(knee_angle)}', 
                       (int(knee[0]*w), int(knee[1]*h)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.putText(frame, f'Quadril: {int(hip_angle)}', 
                       (int(hip[0]*w), int(hip[1]*h)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Contador de repetições
            cv2.rectangle(frame, (0, 0), (150, 100), (245, 117, 16), -1)
            cv2.putText(frame, 'REPETICOES', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, str(squat_count), (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # Status da forma
            status_y = 120
            status_color = (0, 255, 0) if is_correct_form and knee_angle < 120 else (0, 165, 255)
            
            cv2.putText(frame, 'FORMA DO EXERCICIO:', (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Exibir feedback linha por linha
            for i, feedback in enumerate(form_feedback):
                color = (0, 255, 0) if "CORRETA" in feedback or "CORRETO" in feedback else (0, 0, 255)
                cv2.putText(frame, feedback, (10, status_y + 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Indicador geral
            if is_correct_form and knee_angle < 120:
                cv2.putText(frame, 'FORMA CORRETA!', (10, status_y + 30 + len(form_feedback)*25 + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif knee_angle < 120:
                cv2.putText(frame, 'CORRIJA A FORMA!', (10, status_y + 30 + len(form_feedback)*25 + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
