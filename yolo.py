import numpy as np
import cv2
from ultralytics import YOLO
import random
import csv
from datetime import datetime
import os
import threading
import queue
import time
from collections import deque
import math

# Ouvrir le fichier contenant les classes
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Modèle pour la détection de genre
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
gender_model = cv2.dnn.readNetFromCaffe(genderProto, genderModel)

# Modèle pour la détection d'âge
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
age_model = cv2.dnn.readNetFromCaffe(ageProto, ageModel)

# Détecteur de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

print("Système d'alignement basique activé (OpenCV uniquement)")

# Classes d'âge et genre
age_classes = ['(0-3)', '(4-7)', '(8-14)', '(15-20)', '(21-24)', '(25-31)', '(32-37)', '(38-45)', '(46-53)', '(60-100)']
gender_classes = ["Homme", "Femme"]

# Générer des couleurs aléatoires pour chaque classe
detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(class_list))]

# Charger le modèle YOLOv8
model = YOLO("weights/yolov8n.pt")

# Dimensions de la fenêtre vidéo
frame_wid = 640
frame_hyt = 480

# Configuration CSV
csv_filename = f"detection_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_headers = ['timestamp', 'frame_number', 'total_persons', 'men_count', 'women_count'] + \
              [f'age_{age}' for age in age_classes] + \
              ['detected_objects', 'processing_fps']

# Créer le fichier CSV avec les en-têtes
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_headers)

# Variables globales pour le multithreading
frame_queue = queue.Queue(maxsize=10)
detection_queue = queue.Queue(maxsize=10)
result_queue = queue.Queue(maxsize=10)
stop_threads = threading.Event()

# Variables partagées avec verrous
stats_lock = threading.Lock()
shared_stats = {
    'person_count': 0,
    'men_count': 0,
    'women_count': 0,
    'age_count': {},
    'object_counter': {},
    'frame_number': 0,
    'processing_fps': 0
}

# Video recording configuration
video_filename = f"session_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
video_writer = None
is_recording = False

def align_face_basic(face_img, left_eye, right_eye):
    """
    Alignement basique du visage basé sur les yeux
    """
    # Calculer l'angle entre les yeux
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = math.degrees(math.atan2(dy, dx))
    
    # Centre entre les yeux
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    
    # Matrice de rotation
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    
    # Appliquer la rotation
    aligned_face = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]))
    
    return aligned_face

def align_face_advanced_opencv(face_roi):
    """
    Alignement avancé du visage avec OpenCV uniquement
    """
    try:
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 5, minSize=(10, 10))
        
        if len(eyes) >= 2:
            # Trier les yeux par position x (gauche à droite)
            eyes = sorted(eyes, key=lambda x: x[0])
            left_eye = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
            right_eye = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)
            
            return align_face_basic(face_roi, left_eye, right_eye)
        
        # Méthode 2: Estimation basée sur la symétrie du visage
        return align_face_symmetry_based(face_roi)
        
    except Exception:
        return face_roi

def align_face_symmetry_based(face_roi):
    """
    Alignement basé sur la symétrie du visage
    """
    try:
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Appliquer un filtre pour améliorer les contours
        edges = cv2.Canny(gray_face, 50, 150)
        
        # Calculer la ligne de symétrie verticale
        height, width = gray_face.shape
        center_x = width // 2
        
        # Calculer l'asymétrie pour différents angles
        best_angle = 0
        min_asymmetry = float('inf')
        
        for angle in range(-15, 16, 3):  # Tester les angles de -15 à +15 degrés
            # Rotation de test
            M = cv2.getRotationMatrix2D((center_x, height//2), angle, 1.0)
            rotated = cv2.warpAffine(edges, M, (width, height))
            
            # Calculer l'asymétrie
            left_half = rotated[:, :center_x]
            right_half = cv2.flip(rotated[:, center_x:], 1)
            
            # Redimensionner si nécessaire
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            asymmetry = np.sum(np.abs(left_half.astype(int) - right_half.astype(int)))
            
            if asymmetry < min_asymmetry:
                min_asymmetry = asymmetry
                best_angle = angle
        
        # Appliquer la meilleure rotation
        if abs(best_angle) > 1:  # Seulement si l'angle est significatif
            M = cv2.getRotationMatrix2D((center_x, height//2), best_angle, 1.0)
            aligned_face = cv2.warpAffine(face_roi, M, (width, height))
            return aligned_face
        
    except Exception:
        pass
    
    return face_roi

def detect_and_align_face(person_roi):
    """
    Détecte et aligne un visage dans la ROI d'une personne
    """
    gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_roi, 1.1, 4, minSize=(30, 30))
    
    if len(faces) == 0:
        return None, None
    
    # Prendre le plus grand visage
    largest_face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = largest_face
    face_roi = person_roi[y:y+h, x:x+w]
    
    # Alignement avec OpenCV uniquement
    aligned_face = face_roi
    
    if face_roi.shape[0] > 50 and face_roi.shape[1] > 50:
        try:
            aligned_face = align_face_advanced_opencv(face_roi)
        except Exception:
            # Si l'alignement avancé échoue, utiliser l'alignement basique
            aligned_face = align_face_with_eyes(face_roi)
    else:
        # Pour les petits visages, utiliser l'alignement basique
        aligned_face = align_face_with_eyes(face_roi)
    
    return aligned_face, (x, y, w, h)

def align_face_with_eyes(face_roi):
    """
    Alignement basique en détectant les yeux avec amélioration
    """
    try:
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Améliorer le contraste pour la détection des yeux
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        enhanced_gray = clahe.apply(gray_face)
        
        eyes = eye_cascade.detectMultiScale(enhanced_gray, 1.1, 5, minSize=(10, 10))
        
        if len(eyes) >= 2:
            # Trier les yeux par position x
            eyes = sorted(eyes, key=lambda x: x[0])
            left_eye = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
            right_eye = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)
            
            return align_face_basic(face_roi, left_eye, right_eye)
        
        # Si pas assez d'yeux détectés, essayer avec des paramètres plus permissifs
        eyes = eye_cascade.detectMultiScale(enhanced_gray, 1.05, 3, minSize=(8, 8))
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda x: x[0])
            left_eye = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
            right_eye = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)
            return align_face_basic(face_roi, left_eye, right_eye)
        
    except Exception:
        pass
    
    return face_roi

def video_capture_thread():
    """
    Thread pour la capture vidéo
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    frame_count = 0
    
    while not stop_threads.is_set():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Validate frame dimensions and type
        if frame is None or frame.size == 0:
            print("Invalid frame received")
            continue
            
        try:
            frame = cv2.resize(frame, (frame_wid, frame_hyt))
        except cv2.error as e:
            print(f"Error resizing frame: {e}")
            continue
            
        frame_count += 1
        
        try:
            frame_queue.put((frame_count, frame), timeout=0.1)
        except queue.Full:
            # Supprimer le frame le plus ancien si la queue est pleine
            try:
                frame_queue.get_nowait()
                frame_queue.put((frame_count, frame), timeout=0.1)
            except queue.Empty:
                pass
    
    cap.release()

def object_detection_thread():
    """
    Thread pour la détection d'objets YOLO
    """
    while not stop_threads.is_set():
        try:
            frame_num, frame = frame_queue.get(timeout=1.0)
            
            # Prédictions YOLO
            detect_params = model.predict(source=[frame], conf=0.45, save=False, verbose=False)
            
            detection_queue.put((frame_num, frame, detect_params), timeout=0.1)
            
        except queue.Empty:
            continue
        except queue.Full:
            # Supprimer l'élément le plus ancien
            try:
                detection_queue.get_nowait()
                detection_queue.put((frame_num, frame, detect_params), timeout=0.1)
            except queue.Empty:
                pass

def age_gender_prediction_thread():
    """
    Thread pour les prédictions d'âge et de genre
    """
    while not stop_threads.is_set():
        try:
            frame_num, frame, detect_params = detection_queue.get(timeout=1.0)
            
            # Traitement des détections
            processed_frame, stats = process_detections(frame, detect_params, frame_num)
            
            result_queue.put((frame_num, processed_frame, stats), timeout=0.1)
            
        except queue.Empty:
            continue
        except queue.Full:
            # Supprimer l'élément le plus ancien
            try:
                result_queue.get_nowait()
                result_queue.put((frame_num, processed_frame, stats), timeout=0.1)
            except queue.Empty:
                pass

def process_detections(frame, detect_params, frame_num):
    """
    Traite les détections et effectue les prédictions d'âge/genre
    """
    start_time = time.time()
    
    # Compteurs locaux
    person_count = 0
    men_count = 0
    women_count = 0
    age_count = {}
    object_counter = {}
    
    DP = detect_params[0].numpy()
    font = cv2.FONT_HERSHEY_COMPLEX
    
    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]
            class_name = class_list[int(clsID)]
            
            # Compter les objets
            object_counter[class_name] = object_counter.get(class_name, 0) + 1
            
            # Rectangle autour de l'objet
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )
            
            # Nom de la classe et confiance
            cv2.putText(
                frame,
                class_name + " " + str(round(conf, 2)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                0.6,
                (255, 255, 255),
                2,
            )
            
            if class_name == "person" and bb[2] - bb[0] > 50 and bb[3] - bb[1] > 50:
                person_count += 1
                person_roi = frame[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
                
                if person_roi.shape[0] > 0 and person_roi.shape[1] > 0:
                    # Détecter et aligner le visage
                    aligned_face, face_coords = detect_and_align_face(person_roi)
                    
                    if aligned_face is not None and aligned_face.shape[0] > 30 and aligned_face.shape[1] > 30:
                        # Dessiner un rectangle autour du visage détecté
                        if face_coords:
                            x, y, w, h = face_coords
                            face_x1 = int(bb[0]) + x
                            face_y1 = int(bb[1]) + y
                            face_x2 = face_x1 + w
                            face_y2 = face_y1 + h
                            cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (0, 255, 0), 2)
                        
                        # Prédire l'âge et le genre avec le visage aligné
                        gender_pred, age_pred = predict_age_gender_optimized(aligned_face)
                        
                        if gender_pred and age_pred:
                            gender, gender_conf = gender_pred
                            age, age_conf = age_pred
                            
                            # Mise à jour des compteurs avec seuils de confiance
                            if gender_conf > 0.5:
                                if gender == "Homme":
                                    men_count += 1
                                else:
                                    women_count += 1
                            
                            if age_conf > 0.3:
                                age_count[age] = age_count.get(age, 0) + 1
                            
                            # Affichage avec indicateur d'alignement
                            cv2.putText(frame, f"{gender} ({gender_conf:.2f}) *", 
                                      (int(bb[0]), int(bb[3]) + 20), font, 0.5, (255, 255, 255), 1)
                            cv2.putText(frame, f"{age} ({age_conf:.2f}) *", 
                                      (int(bb[0]), int(bb[3]) + 40), font, 0.5, (255, 255, 255), 1)
                        else:
                            # Fallback sans alignement
                            gender_pred, age_pred = predict_age_gender_fallback(person_roi)
                            if gender_pred and age_pred:
                                gender, age = gender_pred, age_pred
                                if gender == "Homme":
                                    men_count += 1
                                else:
                                    women_count += 1
                                age_count[age] = age_count.get(age, 0) + 1
                                
                                cv2.putText(frame, f"{gender} (fallback)", 
                                          (int(bb[0]), int(bb[3]) + 20), font, 0.5, (128, 128, 128), 1)
                                cv2.putText(frame, f"{age} (fallback)", 
                                          (int(bb[0]), int(bb[3]) + 40), font, 0.5, (128, 128, 128), 1)
    
    processing_time = time.time() - start_time
    processing_fps = 1.0 / processing_time if processing_time > 0 else 0
    
    stats = {
        'person_count': person_count,
        'men_count': men_count,
        'women_count': women_count,
        'age_count': age_count,
        'object_counter': object_counter,
        'frame_number': frame_num,
        'processing_fps': processing_fps
    }
    
    return frame, stats

def predict_age_gender_optimized(face_roi):
    """
    Prédiction optimisée d'âge et genre avec visage aligné
    """
    if face_roi is None or face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
        return None, None
    
    try:
        # Redimensionner pour améliorer la qualité
        if face_roi.shape[0] < 100 or face_roi.shape[1] < 100:
            face_roi = cv2.resize(face_roi, (128, 128), interpolation=cv2.INTER_CUBIC)
        
        # Appliquer un léger lissage pour réduire le bruit
        face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
        
        # Normalisation d'histogramme pour améliorer le contraste
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4)).apply(lab[:,:,0])
        face_roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Créer un blob
        blob = cv2.dnn.blobFromImage(face_roi, 1, (227, 227), (104, 117, 123), swapRB=False)
        
        # Prédiction genre
        gender_model.setInput(blob)
        gender_preds = gender_model.forward()
        gender = gender_classes[np.argmax(gender_preds)]
        gender_confidence = np.max(gender_preds)
        
        # Prédiction âge
        age_model.setInput(blob)
        age_preds = age_model.forward()
        age = age_classes[np.argmax(age_preds)]
        age_confidence = np.max(age_preds)
        
        return (gender, gender_confidence), (age, age_confidence)
        
    except Exception as e:
        return None, None

def predict_age_gender_fallback(person_roi):
    """
    Méthode de fallback pour la prédiction sans alignement
    """
    try:
        blob = cv2.dnn.blobFromImage(person_roi, 1, (227, 227), (104, 117, 123), swapRB=False)
        
        gender_model.setInput(blob)
        gender_preds = gender_model.forward()
        gender = gender_classes[np.argmax(gender_preds)]
        
        age_model.setInput(blob)
        age_preds = age_model.forward()
        age = age_classes[np.argmax(age_preds)]
        
        return gender, age
    except:
        return None, None

def log_to_csv(timestamp, frame_num, person_count, men_count, women_count, age_count, object_counter, fps):
    """
    Enregistre les statistiques dans le fichier CSV
    """
    row = [timestamp, frame_num, person_count, men_count, women_count]
    
    # Ajouter les compteurs d'âge
    for age_class in age_classes:
        row.append(age_count.get(age_class, 0))
    
    # Ajouter les objets détectés
    objects_str = ','.join([f"{obj}:{count}" for obj, count in object_counter.items() if obj != "person"])
    row.append(objects_str)
    row.append(round(fps, 2))
    
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)

def start_recording():
    """Start recording the session"""
    global video_writer, is_recording
    if not is_recording:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame_wid + 380, frame_hyt))
        is_recording = True
        print(f"Recording started: {video_filename}")

def stop_recording():
    """Stop recording the session"""
    global video_writer, is_recording
    if is_recording:
        video_writer.release()
        is_recording = False
        print(f"Recording saved: {video_filename}")

def main():
    """
    Fonction principale avec interface utilisateur
    """
    print("=== SYSTÈME DE DÉTECTION AVANCÉ ===")
    print("Fonctionnalités:")
    print("- Détection multi-objets avec YOLO")
    print("- Alignement de visages (OpenCV) pour précision améliorée")
    print("- Traitement multithreadé pour performances optimales")
    print("- Logging CSV automatique")
    print("- Alignement basique avec détection d'yeux et symétrie")
    print("- Enregistrement vidéo des sessions")
    print("\nCommandes:")
    print("- 'q': Quitter")
    print("- 's': Sauvegarder capture d'écran")
    print("- 'r': Démarrer/Arrêter l'enregistrement")
    print(f"\nStatistiques sauvegardées dans: {csv_filename}")
    print("=" * 50)
    
    # Démarrage des threads
    threads = []
    
    # Thread de capture vidéo
    capture_thread = threading.Thread(target=video_capture_thread, daemon=True)
    capture_thread.start()
    threads.append(capture_thread)
    
    # Thread de détection d'objets
    detection_thread = threading.Thread(target=object_detection_thread, daemon=True)
    detection_thread.start()
    threads.append(detection_thread)
    
    # Thread de prédiction âge/genre
    prediction_thread = threading.Thread(target=age_gender_prediction_thread, daemon=True)
    prediction_thread.start()
    threads.append(prediction_thread)
    
    # Variables pour le FPS et les statistiques
    fps_deque = deque(maxlen=30)
    last_log_time = time.time()
    
    try:
        while True:
            try:
                frame_num, processed_frame, stats = result_queue.get(timeout=1.0)
                
                # Calcul du FPS d'affichage
                current_time = time.time()
                fps_deque.append(current_time)
                if len(fps_deque) > 1:
                    display_fps = len(fps_deque) / (fps_deque[-1] - fps_deque[0])
                else:
                    display_fps = 0
                
                # Mettre à jour les statistiques partagées
                with stats_lock:
                    shared_stats.update(stats)
                
                # Créer le panneau de statistiques
                panel = create_stats_panel(stats, display_fps)
                
                # Combiner l'image et le panneau
                combined = np.hstack((processed_frame, panel))
                
                # Save frame if recording
                if is_recording and video_writer is not None:
                    video_writer.write(combined)
                
                cv2.imshow('Advanced Detection System', combined)
                
                # Logging périodique (toutes les 2 secondes)
                if current_time - last_log_time > 2.0:
                    log_to_csv(
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        stats['frame_number'],
                        stats['person_count'],
                        stats['men_count'],
                        stats['women_count'],
                        stats['age_count'],
                        stats['object_counter'],
                        stats['processing_fps']
                    )
                    last_log_time = current_time
                
                # Gestion des touches
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_name = f"advanced_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(screenshot_name, combined)
                    print(f"Capture sauvegardée: {screenshot_name}")
                elif key == ord('r'):
                    if is_recording:
                        stop_recording()
                    else:
                        start_recording()
                
            except queue.Empty:
                continue
                
    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur...")
    
    finally:
        # Stop recording if active
        if is_recording:
            stop_recording()
            
        # Arrêter tous les threads
        print("Arrêt des threads...")
        stop_threads.set()
        
        # Attendre que tous les threads se terminent
        for thread in threads:
            thread.join(timeout=2.0)
        
        cv2.destroyAllWindows()
        
        print(f"Système arrêté. Statistiques finales dans: {csv_filename}")

def create_stats_panel(stats, display_fps):
    """
    Crée le panneau de statistiques
    """
    panel = np.zeros((frame_hyt, 380, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_COMPLEX
    
    cv2.putText(panel, "=== STATISTIQUES AVANCEES ===", (10, 30), font, 0.6, (0, 255, 255), 2)
    cv2.putText(panel, f"Frame: {stats['frame_number']}", (10, 60), font, 0.45, (200, 200, 200), 1)
    cv2.putText(panel, f"FPS Affichage: {display_fps:.1f}", (10, 85), font, 0.45, (200, 200, 200), 1)
    cv2.putText(panel, f"FPS Traitement: {stats['processing_fps']:.1f}", (10, 110), font, 0.45, (200, 200, 200), 1)
    
    cv2.putText(panel, f"Personnes : {stats['person_count']}", (10, 145), font, 0.55, (255, 255, 255), 1)
    cv2.putText(panel, f"Hommes : {stats['men_count']}", (10, 175), font, 0.55, (200, 200, 255), 1)
    cv2.putText(panel, f"Femmes : {stats['women_count']}", (10, 205), font, 0.55, (255, 200, 200), 1)
    
    y_offset = 235
    cv2.putText(panel, "Ages (* = aligne):", (10, y_offset), font, 0.5, (255, 255, 0), 1)
    y_offset += 20
    for age_label, count in sorted(stats['age_count'].items(), key=lambda x: x[0]):
        cv2.putText(panel, f"{age_label} : {count}", (10, y_offset), font, 0.4, (180, 255, 180), 1)
        y_offset += 18
    
    y_offset += 10
    cv2.putText(panel, "Objets:", (10, y_offset), font, 0.5, (255, 255, 0), 1)
    y_offset += 20
    for obj, count in stats['object_counter'].items():
        if obj != "person" and y_offset < frame_hyt - 20:
            cv2.putText(panel, f"{obj}: {count}", (10, y_offset), font, 0.4, (220, 220, 220), 1)
            y_offset += 18
    
    return panel

if __name__ == "__main__":
    main()