"""
Run:  python base_prototype.py
Quit: ESC        Toggle fullscreen: F

Dependencies: opencv-python, numpy, pygame, ultralytics, deepface
(+ tensorflow, retina-face — pulled in by deepface / the retinaface backend).
First run needs internet once, to download yolo26n.pt and DeepFace's weights.
"""

import os
import sys
import math
import time
import json
import threading
import urllib.request

import cv2
import numpy as np
import pygame
from pygame import mixer
from ultralytics import YOLO
from deepface import DeepFace

CAMERA_INDEX = 0

MODEL_NAME     = "yolo26n.pt"
OBJ_CONFIDENCE = 0.45
OBJ_INTERVAL   = 0.12     
OBJ_COLOR      = (0, 200, 255)

OBJECT_LABEL_ALIASES = {
    "cellphone": "cell phone",
    "cell-phone": "cell phone",
    "mobile phone": "cell phone",
    "smartphone": "cell phone",
    "phone": "cell phone",
    "notebook": "book",
    "textbook": "book",
    "books": "book",
    "book": "book",
}


INTERVAL = 1.5            
MIN_SIZE = 10               
MIN_CONF = 0.40             
DETECTOR = "retinaface"
FACE_COLOR = (0, 255, 100)

FS    = True                
RINGS = 6                   


TS_KEY    = os.environ.get("THINGSPEAK_API_KEY", "P9HTEV3BYHZ2MKP9")
TS_FIELDS = ["happy", "sad", "angry", "surprise", "neutral", "fear"]

ALERT_COOLDOWN = 3.0       
ALERT_REPEAT = 0.8          
DISTRACTION_EMOTIONS = ["angry", "sad", "surprise"]  
ALERT_VOLUME = 0.3

COLORS = {
    "happy":    (255, 140,   0),
    "sad":      ( 30, 100, 255),
    "angry":    (220,  30,  30),
    "surprise": (255, 220,   0),
    "neutral":  (160, 160, 160),
    "fear":     (150,  50, 200),
    "disgust":  ( 50, 180,  50),
}

def get_folder():
    return os.path.dirname(os.path.abspath(__file__))


def generate_alert_sound(frequency=1000, duration=0.2, sample_rate=22050):
    """Generate a simple beep sound at given frequency."""
    num_samples = int(sample_rate * duration)
    frames = np.sin(2 * np.pi * frequency * np.linspace(0, duration, num_samples))
    frames = np.repeat(frames.reshape(-1, 1), 2, axis=1)
    frames = (frames * 32767).astype(np.int16)
    sound = pygame.sndarray.make_sound(frames)
    sound.set_volume(ALERT_VOLUME)
    return sound


def play_alert(alert_type="normal"):
    try:
        if alert_type == "distraction":
            sound = generate_alert_sound(frequency=800, duration=0.15)
            sound.play()
            time.sleep(0.2)
            sound.play()
        elif alert_type == "phone":
            sound = generate_alert_sound(frequency=1200, duration=0.1)
            sound.play()
            time.sleep(0.15)
            sound.play()
            time.sleep(0.15)
            sound.play()
        else:
            sound = generate_alert_sound(frequency=1000, duration=0.2)
            sound.play()
    except Exception as e:
        print(f"[alert] sound error: {e}")



def normalize_object_label(label: str) -> str:
    
    text = str(label or "").strip().lower()
    if not text:
        return ""
    text = text.replace("_", " ")
    text = " ".join(text.split())
    return OBJECT_LABEL_ALIASES.get(text, text)


def label_matches_any(obj_counts: dict, *patterns: str) -> bool:
    if not obj_counts:
        return False
    for label, count in obj_counts.items():
        if count <= 0:
            continue
        l = str(label).lower()
        for pattern in patterns:
            if pattern.lower() in l:
                return True
    return False


def load_yolo_model():
    folder = get_folder()
    local_model = os.path.join(folder, MODEL_NAME)
    try:
        if os.path.exists(local_model):
            print(f"[yolo] loading local model: {local_model}")
            return YOLO(local_model)
        print(f"[yolo] {MODEL_NAME} not found locally — Ultralytics will download it once.")
        return YOLO(MODEL_NAME)
    except Exception as e:
        print(f"\n[yolo] could not load the object-detection model: {e}")
        print("Make sure this computer has internet access for the first-run download.")
        sys.exit(1)


state = {
    "emotion": "neutral",
    "color":   COLORS["neutral"],
    "faces":   0,
    "bd":      {},           
    "regions": [],           
    "objects": [],          
    "obj_counts": {},       
    "frame":   None,
    "frame_version": 0,      
    "alert_message": "",     
    "alert_time": 0,         
    "last_alert_type": "",   
}
lock  = threading.Lock()     
flock = threading.Lock()     


def ok_face(region: dict) -> bool:
    w = region.get("w", 0)
    h = region.get("h", 0)
    c = region.get("face_confidence", 1.0)
    return w >= MIN_SIZE and h >= MIN_SIZE and c >= MIN_CONF


def check_alerts(emotion: str, obj_counts: dict, last_alert_time: float, last_alert_type: str) -> tuple:

    current_time = time.time()
    has_phone = label_matches_any(obj_counts, "cell phone", "phone", "smartphone")
    has_book = label_matches_any(obj_counts, "book", "notebook", "textbook")
    has_person = label_matches_any(obj_counts, "person", "man", "woman", "girl", "boy")
    
    
    current_alert_type = ""
    alert_msg = ""
    
   
    if has_phone and has_book and has_person:
        current_alert_type = "phone"
        alert_msg = "⚠️  PHONE WHILE STUDYING!"
    )
    elif has_phone and not has_book and has_person:
        current_alert_type = "phone"
        alert_msg = "⚠️  PHONE DETECTED!"
    
    elif emotion in DISTRACTION_EMOTIONS and has_person:
        current_alert_type = "distraction"
        alert_msg = f"⚠️  DISTRACTED ({emotion.upper()})!"
    
    elif has_person and has_book == 0:
        if emotion == "sad" or emotion == "angry":
            current_alert_type = "distraction"
            alert_msg = "⚠️  CHECK FOCUS!"
    
    
    if not current_alert_type:
        return "", False, ""
    
    
    if current_alert_type == last_alert_type:
        if current_time - last_alert_time >= ALERT_REPEAT:
            return alert_msg, True, current_alert_type
        else:
            return alert_msg, False, current_alert_type
    
   
    if current_time - last_alert_time < ALERT_COOLDOWN:
        return alert_msg, False, current_alert_type
    
    return alert_msg, True, current_alert_type



def post_dashboard(payload: dict):
    
    if not DASHBOARD_URL:
        return
    try:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            DASHBOARD_URL, data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception as e:
        print(f"[dashboard] failed: {e}")


def post_thingspeak(bd: dict, face_count: int, object_count: int):
    try:
        params = "&".join(f"field{i+1}={bd.get(k, 0)}" for i, k in enumerate(TS_FIELDS))
        url = (
            f"https://api.thingspeak.com/update?api_key={TS_KEY}&{params}"
            f"&field7={face_count}&field8={object_count}"
        )
        urllib.request.urlopen(url, timeout=5)
        top = max(bd, key=bd.get) if bd else "neutral"
        print(f"[thingspeak] sent  top={top}  faces={face_count}  objects={object_count}")
    except Exception as e:
        print(f"[thingspeak] failed: {e}")



def worker(yolo_model):
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_obj_analysis = 0.0
    last_emo_analysis = 0.0
    last_upload        = 0.0
    last_alert_time    = 0.0
    print("[worker] started — first emotion detection takes ~15s while DeepFace loads model weights.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        with flock:
            state["frame"] = frame.copy()
            state["frame_version"] += 1

        now = time.time()

        
        if now - last_obj_analysis >= OBJ_INTERVAL:
            last_obj_analysis = now
            try:
                results = yolo_model.predict(source=frame, conf=OBJ_CONFIDENCE, verbose=False)
                boxes, counts = [], {}
                for b in results[0].boxes:
                    raw_label = yolo_model.names.get(int(b.cls[0]), "")
                    label = normalize_object_label(raw_label)
                    if not label:
                        continue
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": label})
                    counts[label] = counts.get(label, 0) + 1
                with lock:
                    state["objects"]    = boxes
                    state["obj_counts"] = counts
            except Exception as e:
                print(f"[yolo] {e}")

       
        if now - last_emo_analysis < INTERVAL:
            continue
        last_emo_analysis = now

        emotions, regions = [], []
        try:
            results = DeepFace.analyze(
                frame, actions=["emotion"],
                enforce_detection=False, detector_backend=DETECTOR,
            )
            for face in (results if isinstance(results, list) else [results]):
                region = face.get("region", {})
                if not ok_face(region):
                    continue
                emotions.append(face["dominant_emotion"])
                regions.append(region)
        except Exception as e:
            print(f"[deepface] {e}")

        if emotions:
            counts = {}
            for e in emotions:
                counts[e] = counts.get(e, 0) + 1
            top = max(counts, key=counts.get)
            breakdown = {
                e: round(c / len(emotions) * 100)
                for e, c in sorted(counts.items(), key=lambda x: -x[1])
            }
        else:
            top, breakdown = "neutral", {}

       
        with lock:
            obj_counts_snapshot = dict(state["obj_counts"])
            last_alert_type_snapshot = state.get("last_alert_type", "")
        
        alert_msg, should_play, alert_type = check_alerts(top, obj_counts_snapshot, last_alert_time, last_alert_type_snapshot)
        
        if should_play:
            last_alert_time = now
            threading.Thread(target=play_alert, args=(alert_type,), daemon=True).start()

        with lock:
            state.update({
                "emotion": top,
                "color":   COLORS.get(top, (160, 160, 160)),
                "faces":   len(emotions),
                "bd":      breakdown,
                "regions": regions,
                "alert_message": alert_msg,
                "alert_time": now if should_play else state.get("alert_time", 0),
                "last_alert_type": alert_type,
            })
            obj_total = sum(state["obj_counts"].values())

        post_dashboard({
            "emotion": top, "faces": len(emotions), "bd": breakdown,
            "objects": dict(state["obj_counts"]),
        })

        if now - last_upload >= 15:
            post_thingspeak(breakdown, len(emotions), obj_total)
            last_upload = now

    cap.release()  



def lerp(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def build_glow_cache(color, rad, rings=RINGS):
    """Pre-render glow ring surfaces. Rebuilt only when radius/colour drift enough to matter."""
    surfaces = []
    for i in range(rings, 0, -1):
        a  = int(22 * (1 - i / rings))
        gr = rad + i * 32
        gs = pygame.Surface((gr * 2, gr * 2), pygame.SRCALPHA)
        r, g, b = color
        pygame.draw.circle(gs, (r, g, b, a), (gr, gr), gr)
        surfaces.append((gs, gr))
    return surfaces


def draw_glow(screen, glow_cache, cx, cy):
    for gs, gr in glow_cache:
        screen.blit(gs, (cx - gr, cy - gr))


def draw_faces(screen, regions, sx, sy, ox, oy):
    for r in regions:
        x = int(r.get("x", 0) * sx) + ox
        y = int(r.get("y", 0) * sy) + oy
        w = int(r.get("w", 0) * sx)
        h = int(r.get("h", 0) * sy)
        pygame.draw.rect(screen, FACE_COLOR, (x, y, w, h), 2)


def draw_objects(screen, objects, sx, sy, ox, oy, font):
    for obj in objects:
        x = int(obj["x1"] * sx) + ox
        y = int(obj["y1"] * sy) + oy
        w = int((obj["x2"] - obj["x1"]) * sx)
        h = int((obj["y2"] - obj["y1"]) * sy)
        pygame.draw.rect(screen, OBJ_COLOR, (x, y, w, h), 2)
        lbl = font.render(obj["label"], True, OBJ_COLOR)
        screen.blit(lbl, (x, max(0, y - 14)))


def frame_to_surface(frame, pw, ph):
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
    return pygame.transform.scale(surf, (pw, ph))



def main():
    yolo_model = load_yolo_model()   

    pygame.init()
    mixer.init(frequency=22050, size=-16, channels=2, buffer=512)  
    screen = (
        pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        if FS else pygame.display.set_mode((1280, 720))
    )
    W, H = screen.get_size()
    pygame.display.set_caption("CROWD PULSE — base prototype")

    f_emo = pygame.font.SysFont("Arial", max(60, H // 9),  bold=True)
    f_cnt = pygame.font.SysFont("Arial", max(32, H // 18))
    f_det = pygame.font.SysFont("Arial", max(20, H // 30))
    f_tit = pygame.font.SysFont("Arial", max(18, H // 40))
    f_obj = pygame.font.SysFont("Arial", max(16, H // 55))
    f_lbl = pygame.font.SysFont("Arial", 13)

    
    PW, PH = 320, 240
    PX, PY = W - PW - 20, H - PH - 20
    SX, SY = PW / 640, PH / 480

    clock  = pygame.time.Clock()
    cur    = COLORS["neutral"]
    phase  = 0.0
    base_r = min(W, H) * 0.26

    prev_rad, prev_color, glow_cache = -1, (-1, -1, -1), []
    preview_surf, last_seen_version = None, -1

    threading.Thread(target=worker, args=(yolo_model,), daemon=True).start()
    print("Crowd Pulse (base prototype) running.  ESC = quit   F = toggle fullscreen")

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                if ev.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()

        with lock:
            target     = state["color"]
            emo        = state["emotion"]
            faces      = state["faces"]
            bd         = dict(state["bd"])
            regions    = list(state["regions"])
            objects    = list(state["objects"])
            obj_counts = dict(state["obj_counts"])
            alert_msg  = state.get("alert_message", "")
            alert_time = state.get("alert_time", 0)

        
        with flock:
            fv = state["frame_version"]
            frame = state["frame"] if fv != last_seen_version else None

        if frame is not None:
            try:
                preview_surf = frame_to_surface(frame, PW, PH)
                last_seen_version = fv
            except Exception:
                pass

        cur = lerp(cur, target, 0.035)
        screen.fill((8, 8, 10))

        phase += 0.018
        rad = int(base_r * (0.74 + 0.26 * math.sin(phase)))
        cx, cy = W // 2, H // 2

        color_drift = max(abs(cur[i] - prev_color[i]) for i in range(3))
        if rad != prev_rad or color_drift > 3:
            glow_cache = build_glow_cache(cur, rad)
            prev_rad, prev_color = rad, cur

        draw_glow(screen, glow_cache, cx, cy)
        pygame.draw.circle(screen, cur, (cx, cy), rad)

        et = f_emo.render(emo.upper(), True, (8, 8, 10))
        screen.blit(et, et.get_rect(center=(cx, cy)))

        ft = f_cnt.render(f"{faces} {'face' if faces == 1 else 'faces'}", True, (180, 180, 180))
        screen.blit(ft, ft.get_rect(center=(cx, cy + rad + 48)))

        if bd:
            s  = "   ·   ".join(f"{e.upper()} {p}%" for e, p in list(bd.items())[:4])
            bt = f_det.render(s, True, (90, 90, 90))
            screen.blit(bt, bt.get_rect(center=(cx, cy + rad + 90)))

        r, g, b = cur
        tt = f_tit.render("CROWD PULSE", True, (r // 3, g // 3, b // 3))
        screen.blit(tt, (22, 20))

        if obj_counts:
            top_objs = sorted(obj_counts.items(), key=lambda x: -x[1])[:5]
            os_txt = "sees: " + ", ".join(f"{k} x{v}" if v > 1 else k for k, v in top_objs)
            ot = f_obj.render(os_txt, True, (120, 120, 120))
            screen.blit(ot, (22, 20 + tt.get_height() + 6))

        
        if alert_msg and (time.time() - alert_time < 1.5):  
            alert_surf = f_det.render(alert_msg, True, (255, 50, 50))
            alert_rect = alert_surf.get_rect(center=(W // 2, 60))
           
            bg_rect = alert_rect.inflate(20, 10)
            pygame.draw.rect(screen, (80, 20, 20), bg_rect)
            pygame.draw.rect(screen, (255, 50, 50), bg_rect, 2)
            screen.blit(alert_surf, alert_rect)

        if preview_surf is not None:
            screen.blit(preview_surf, (PX, PY))
            pygame.draw.rect(screen, (50, 50, 50), (PX, PY, PW, PH), 1)
            draw_objects(screen, objects, SX, SY, PX, PY, f_lbl)
            draw_faces(screen, regions, SX, SY, PX, PY)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
