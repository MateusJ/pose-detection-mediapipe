import cv2
import mediapipe as mp
import numpy as np


def calculate_angle(a, b, c):
    """Calcula o ângulo entre três pontos (em graus)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

squat_stage = "em pé"
squat_count = 0
development_stage = "baixo"
development_count = 0
exercise_name = ""
lista_exercicios = ["Agachamento", "Desenvolvimento", "Elevacao Lateral"]

print("Webcam opened successfully. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        if exercise_name == "":
            cv2.putText(frame, 'Pressione "e" para selecionar um exercicio.', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for i, ex in enumerate(lista_exercicios):
                cv2.putText(frame, f'{i+1}. {ex}', (10, 60 + i*30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        elif exercise_name == "Agachamento":

            try:
                landmarks = results.pose_landmarks.landmark
                
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                
                # ========== CÁLCULO DOS ÂNGULOS PRINCIPAIS ==========
                
                knee_angle = calculate_angle(hip, knee, ankle)
                hip_angle = calculate_angle(shoulder, hip, knee)
                trunk_angle = calculate_angle(nose, hip, knee)
                knee_ankle_horizontal_distance = abs(knee[0] - ankle[0])
                
                # ========== ANÁLISE DA FORMA DO AGACHAMENTO ==========
                
                form_feedback = []
                is_correct_form = True
                
                if knee_angle < 120:
                    if knee_angle > 100:
                        form_feedback.append("Agachamento raso - Desça mais!")
                        is_correct_form = False
                    elif knee_angle < 70:
                        form_feedback.append("Muito profundo - Cuidado com os joelhos!")
                        is_correct_form = False
                    else:
                        form_feedback.append("Profundidade CORRETA!")
                    
                    if hip_angle < 70:
                        form_feedback.append("Tronco muito inclinado - Mantenha o peito para cima!")
                        is_correct_form = False
                    elif hip_angle > 120:
                        form_feedback.append("Fique mais sentado - Empine o quadril para tras!")
                        is_correct_form = False
                    else:
                        form_feedback.append("Postura do tronco CORRETA!")
                    
                    if knee_ankle_horizontal_distance > 0.15:
                        form_feedback.append("Joelho muito a frente - Empurre o quadril para tras!")
                        is_correct_form = False
                    else:
                        form_feedback.append("Posicao do joelho CORRETA!")
                
                # ========== CONTADOR DE REPETIÇÕES ==========
                
                if knee_angle > 160:
                    squat_stage = "em pé"
                if knee_angle < 90 and squat_stage == "em pé":
                    squat_stage = "agachado"
                    squat_count += 1
                
                # ========== INTERFACE VISUAL ==========
                
                h, w, _ = frame.shape
                
                cv2.putText(frame, f'Joelho: {int(knee_angle)}', 
                        (int(knee[0]*w), int(knee[1]*h)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                cv2.putText(frame, f'Quadril: {int(hip_angle)}', 
                        (int(hip[0]*w), int(hip[1]*h)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                cv2.rectangle(frame, (0, 0), (150, 100), (245, 117, 16), -1)
                cv2.putText(frame, 'REPETICOES', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(frame, str(squat_count), (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                status_y = 120
                status_color = (0, 255, 0) if is_correct_form and knee_angle < 120 else (0, 165, 255)
                
                cv2.putText(frame, 'FORMA DO EXERCICIO:', (10, status_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                for i, feedback in enumerate(form_feedback):
                    color = (0, 255, 0) if "CORRETA" in feedback or "CORRETO" in feedback else (0, 0, 255)
                    cv2.putText(frame, feedback, (10, status_y + 30 + i*25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if is_correct_form and knee_angle < 120:
                    cv2.putText(frame, 'FORMA CORRETA!', (10, status_y + 30 + len(form_feedback)*25 + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif knee_angle < 120:
                    cv2.putText(frame, 'CORRIJA A FORMA!', (10, status_y + 30 + len(form_feedback)*25 + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            except Exception as e:
                print(f"Erro ao processar os marcos: {e}")
            
        elif exercise_name == "Desenvolvimento":

            try:
                landmarks = results.pose_landmarks.landmark
                
                shoulder_L = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                shoulder_R = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_L = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_L = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                wrist_R = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                hip_L = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                hip_R = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                
                # ========== CÁLCULO DOS ÂNGULOS PRINCIPAIS ==========
                
                elbow_angle_L = calculate_angle(shoulder_L, elbow_L, wrist_L)
                elbow_angle_R = calculate_angle(shoulder_R, elbow_R, wrist_R)
                elbow_angle = (elbow_angle_L + elbow_angle_R) / 2
                shoulder_angle_L = calculate_angle(hip_L, shoulder_L, elbow_L)
                shoulder_angle_R = calculate_angle(hip_R, shoulder_R, elbow_R)
                shoulder_angle = (shoulder_angle_L + shoulder_angle_R) / 2
                elbow_symmetry_diff = abs(elbow_angle_L - elbow_angle_R)
                shoulder_alignment = abs(shoulder_L[1] - shoulder_R[1])
                wrist_above_elbow_L = wrist_L[1] < elbow_L[1]
                wrist_above_elbow_R = wrist_R[1] < elbow_R[1]


                # ========== ANÁLISE DA FORMA DO DESENVOLVIMENTO ==========
                
                form_feedback = []
                is_correct_form = True
                
                if elbow_angle < 160:
                    if elbow_angle > 110:
                        form_feedback.append("Desca mais os bracos - Cotovelo em 90°!")
                        is_correct_form = False
                    elif elbow_angle < 70:
                        form_feedback.append("Bracos muito baixos - Risco de lesao!")
                        is_correct_form = False
                    else:
                        form_feedback.append("Amplitude do movimento CORRETA!")
                    
                    if elbow_symmetry_diff > 15:
                        form_feedback.append("Bracos ASSIMETRICOS - Iguale os dois lados!")
                        is_correct_form = False
                    else:
                        form_feedback.append("Simetria dos braços CORRETA!")
                    
                    if shoulder_angle < 60:
                        form_feedback.append("Cotovelos muito para tras - Traga para frente!")
                        is_correct_form = False
                    elif shoulder_angle > 110:
                        form_feedback.append("Cotovelos muito para frente - Alinhe com ombros!")
                        is_correct_form = False
                    else:
                        form_feedback.append("Posicao dos ombros CORRETA!")
                    
                    if shoulder_alignment > 0.05:
                        form_feedback.append("Tronco INCLINADO - Mantenha-se reto!")
                        is_correct_form = False
                    else:
                        form_feedback.append("Postura do tronco CORRETA!")
                    
                    if not wrist_above_elbow_L or not wrist_above_elbow_R:
                        form_feedback.append("Pulsos abaixo dos cotovelos - Corrija a pegada!")
                        is_correct_form = False
                    else:
                        form_feedback.append("Posicao dos pulsos CORRETA!")
                
                elif elbow_angle >= 160:
                    if elbow_angle < 170:
                        form_feedback.append("Estenda completamente os bracos no topo!")
                        is_correct_form = False
                    else:
                        form_feedback.append("Extensao completa CORRETA!")

                
                # ========== CONTADOR DE REPETIÇÕES ==========
                
                if elbow_angle > 160:
                    development_stage = "alto"
                if elbow_angle < 100 and development_stage == "alto":
                    development_stage = "baixo"
                    development_count += 1 
                    

                # ========== INTERFACE VISUAL ==========
                
                h, w, _ = frame.shape
                
                cv2.putText(frame, f'Cotovelo E: {int(elbow_angle_L)}', 
                        (int(elbow_L[0]*w), int(elbow_L[1]*h) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                cv2.putText(frame, f'Cotovelo D: {int(elbow_angle_R)}', 
                        (int(elbow_R[0]*w), int(elbow_R[1]*h) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                cv2.putText(frame, f'Ombro: {int(shoulder_angle)}', 
                        (int(shoulder_L[0]*w), int(shoulder_L[1]*h)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                cv2.rectangle(frame, (0, 0), (150, 100), (245, 117, 16), -1)
                cv2.putText(frame, 'REPETICOES', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(frame, str(development_count), (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                status_y = 120
                status_color = (0, 255, 0) if is_correct_form and elbow_angle < 160 else (0, 165, 255)
                
                cv2.putText(frame, 'FORMA DO EXERCICIO:', (10, status_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                for i, feedback in enumerate(form_feedback):
                    color = (0, 255, 0) if "CORRETA" in feedback or "CORRETO" in feedback else (0, 0, 255)
                    cv2.putText(frame, feedback, (10, status_y + 30 + i*25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if is_correct_form and elbow_angle < 160:
                    cv2.putText(frame, 'FORMA CORRETA!', (10, status_y + 30 + len(form_feedback)*25 + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif elbow_angle < 160:
                    cv2.putText(frame, 'CORRIJA A FORMA!', (10, status_y + 30 + len(form_feedback)*25 + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            except Exception as e:
                print(f"Erro ao processar os marcos: {e}")
            


    cv2.imshow('MediaPipe Pose Estimation', frame)

    k = cv2.waitKey(5)
    if k != -1:
        if k == ord('q'):
            break
        elif k == ord('e'):
            exercise_name = ""
        elif int(chr(k)) in range(1, len(lista_exercicios) + 1) and exercise_name == "":
            print(f"Exercício selecionado: {lista_exercicios[int(chr(k)) - 1]}")
            exercise_name = lista_exercicios[int(chr(k)) - 1]

cap.release()
cv2.destroyAllWindows()
pose.close()

print("Script finished.")
