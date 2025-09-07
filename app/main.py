# app/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import re, json
import time
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# .env 로드 (DATABASE_URL, FRAMES_DIR, CAMERA_SOURCE 등)
from dotenv import load_dotenv
load_dotenv()

# ====== DB (SQLAlchemy sync + psycopg3) ======
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

# -----------------------------------------------------------------------------
# 환경 설정
# -----------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://obuser:password@localhost:5432/db")
FRAMES_DIR = Path(os.getenv("FRAMES_DIR", "./frames"))
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")  # "0","1"... or file/rtsp url
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "1280"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "720"))
INFER_SIZE = int(os.getenv("INFER_SIZE", "640"))
TARGET_FPS = float(os.getenv("TARGET_FPS", "15"))

# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(title="A:EYE YOLO Live + DB Save", version="1.2.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 도메인 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# YOLO 및 카메라 상태
# -----------------------------------------------------------------------------
model: Optional[YOLO] = None
device: str = "cpu"

_cam_thread: Optional[threading.Thread] = None
_cam_running: bool = False
_cam_lock = threading.Lock()
_last_frame_bgr: Optional[np.ndarray] = None
_last_draw_bgr: Optional[np.ndarray] = None

# 최근 탐지 결과(저장을 위해 유지)
# 예: {"object_type": "person", "confidence": 0.91, "location": {"x":0.32,"y":0.7,"z":0.15}}
_last_detections: List[Dict[str, Any]] = []

# -----------------------------------------------------------------------------
# DB 세션팩토리 & 의존성 (동기)
# -----------------------------------------------------------------------------
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)

def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------------------------------------------------------
# 응답 모델
# -----------------------------------------------------------------------------
class Box(BaseModel):
    object_type: str
    confidence: float
    location: Dict[str, float]  # {"x":0~1,"y":0~1,"z":0~1}

class SaveResult(BaseModel):
    frame_id: int
    file_path: str
    saved_objects: int

# -----------------------------------------------------------------------------
# 카메라 오픈
# -----------------------------------------------------------------------------
def _open_capture():
    src = CAMERA_SOURCE
    if src.isdigit():
        cam_index = int(src)
        cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)  # macOS 우선
        if not cap or not cap.isOpened():
            cap = cv2.VideoCapture(cam_index)  # 백업
    else:
        cap = cv2.VideoCapture(src)

    if not cap or not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    return cap

# -----------------------------------------------------------------------------
# 카메라 루프
# -----------------------------------------------------------------------------
def _camera_loop():
    global _cam_running, _last_frame_bgr, _last_draw_bgr, _last_detections, model

    cap = _open_capture()
    if cap is None:
        _cam_running = False
        return

    min_interval = 1.0 / max(TARGET_FPS, 1.0)
    last_ts = 0.0

    while _cam_running:
        ok, frame_bgr = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        now = time.time()
        if now - last_ts < min_interval:
            with _cam_lock:
                _last_frame_bgr = frame_bgr
            time.sleep(0.001)
            continue
        last_ts = now

        H, W = frame_bgr.shape[:2]
        # 추론 입력 전처리
        inp = cv2.resize(frame_bgr, (INFER_SIZE, INFER_SIZE), interpolation=cv2.INTER_AREA)
        inp_rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(inp_rgb)

        detections: List[Dict[str, Any]] = []
        try:
            results = model.predict(source=pil_img, conf=0.25, iou=0.45, device=device, verbose=False)
            r = results[0]
            draw = r.plot()  # BGR, INFER_SIZE 기준
            draw_resized = cv2.resize(draw, (W, H), interpolation=cv2.INTER_LINEAR)

            # bbox → (x,y,z) 변환 (정규화 중심 + 면적비)
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                names = r.names if hasattr(r, "names") else model.names

                for (x1, y1, x2, y2), conf, k in zip(xyxy, confs, clss):
                    cx = float(((x1 + x2) / 2.0) / W)
                    cy = float(((y1 + y2) / 2.0) / H)
                    area_ratio = float(((x2 - x1) * (y2 - y1)) / (W * H))
                    label = names.get(k, str(k)) if isinstance(names, dict) else str(k)
                    detections.append({
                        "object_type": label,
                        "confidence": float(conf),
                        "location": {
                            "x": max(0.0, min(1.0, cx)),
                            "y": max(0.0, min(1.0, cy)),
                            "z": max(0.0, min(1.0, area_ratio)),
                        }
                    })
        except Exception:
            draw_resized = frame_bgr
            detections = []

        with _cam_lock:
            _last_frame_bgr = frame_bgr
            _last_draw_bgr = draw_resized
            _last_detections = detections

    cap.release()

def _start_camera_if_needed():
    global _cam_thread, _cam_running
    if _cam_running:
        return
    _cam_running = True
    _cam_thread = threading.Thread(target=_camera_loop, daemon=True)
    _cam_thread.start()

def _stop_camera():
    global _cam_running, _cam_thread
    _cam_running = False
    if _cam_thread and _cam_thread.is_alive():
        _cam_thread.join(timeout=2.0)
    _cam_thread = None

# ==== 안전한 마스킹 함수 (에러 방지) ====
def _mask_db_url(url: str | None) -> str:
    if not url:
        return ""
    try:
        return re.sub(r'(?<=://[^:]+:)[^@]+', '****', url)
    except Exception:
        return url  # 마스킹 실패해도 원문 반환(진단용)

# -----------------------------------------------------------------------------
# 앱 라이프사이클
# -----------------------------------------------------------------------------
@app.on_event("startup")
def load_model():
    global model, device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = YOLO("yolov8n.pt")
    try:
        model.predict(source=Image.new("RGB", (320, 320)), device=device, verbose=False)
    except Exception:
        pass

@app.on_event("shutdown")
def shutdown():
    _stop_camera()

# -----------------------------------------------------------------------------
# 헬스/카메라/스트림/HTML
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "device": device, "model": "yolov8n.pt", "camera_running": _cam_running}

# DB URL만 보여주기
@app.get("/db/url")
def db_url():
    try:
        raw = os.getenv("DATABASE_URL") or DATABASE_URL or ""
        return {"url": _mask_db_url(raw), "empty": raw == ""}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# DB 핑 (동기)
@app.get("/db/ping")
def db_ping(session: Session = Depends(get_session)):
    try:
        r = session.execute(text("SELECT 1"))
        return {"db": "ok", "result": r.scalar_one(), "url": _mask_db_url(os.getenv("DATABASE_URL") or DATABASE_URL)}
    except Exception as e:
        return JSONResponse(
            {"db": "error", "url": _mask_db_url(os.getenv("DATABASE_URL") or DATABASE_URL),
             "error_type": e.__class__.__name__, "detail": str(e)},
            status_code=500,
        )

@app.post("/camera/start")
def camera_start():
    _start_camera_if_needed()
    return {"ok": True, "camera_running": _cam_running, "source": CAMERA_SOURCE}

@app.get("/camera/status")
def camera_status():
    return {"camera_running": _cam_running, "source": CAMERA_SOURCE}

@app.post("/camera/stop")
def camera_stop():
    _stop_camera()
    return {"ok": True, "camera_running": _cam_running}

def _mjpeg_generator(use_draw: bool = True):
    boundary = "frame"
    while True:
        if not _cam_running:
            time.sleep(0.05)
            continue
        with _cam_lock:
            frame = (_last_draw_bgr if use_draw else _last_frame_bgr)
        if frame is None:
            time.sleep(0.01)
            continue
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            time.sleep(0.01)
            continue
        payload = jpg.tobytes()
        yield (
            b"--" + boundary.encode() + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" +
            payload + b"\r\n"
        )

@app.get("/video")
def video_stream():
    _start_camera_if_needed()
    return StreamingResponse(_mjpeg_generator(use_draw=True),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video/raw")
def video_stream_raw():
    _start_camera_if_needed()
    return StreamingResponse(_mjpeg_generator(use_draw=False),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/", response_class=HTMLResponse)
def index_page():
    return """
    <html>
      <body style="margin:0;background:#111;color:#eee;font-family:sans-serif;">
        <div style="padding:12px">
          <h3>A:EYE Live</h3>
          <button onclick="fetch('/camera/start',{method:'POST'}).then(()=>location.reload())">Start</button>
          <button onclick="fetch('/camera/stop',{method:'POST'}).then(()=>location.reload())">Stop</button>
          <button onclick="fetch('/save',{method:'POST'}).then(r=>r.json()).then(x=>alert(JSON.stringify(x)))">Save(Current)</button>
          <div style="display:flex;gap:16px;margin-top:12px;align-items:flex-start">
            <div><div>Annotated</div><img src="/video" style="max-width:640px;border:1px solid #444"/></div>
            <div><div>Raw</div><img src="/video/raw" style="max-width:640px;border:1px solid #444"/></div>
          </div>
        </div>
      </body>
    </html>
    """

# ===== 진단용 조회 API =====

@app.get("/debug/detections")
def debug_detections():
    with _cam_lock:
        n = len(_last_detections)
        sample = _last_detections[:5]
    return {"count": n, "sample": sample}

@app.get("/frames/recent")
def frames_recent(limit: int = 10, session: Session = Depends(get_session)):
    rows = session.execute(text("""
        SELECT frame_id, object_count, captured_at, file_path
        FROM frame
        ORDER BY frame_id DESC
        LIMIT :limit
    """), {"limit": limit}).mappings().all()
    return {"rows": [dict(r) for r in rows]}

@app.get("/objects/recent")
def objects_recent(limit: int = 20, session: Session = Depends(get_session)):
    rows = session.execute(text("""
        SELECT object_id, frame_id, object_type, confidence, location, vector_id, detected_at
        FROM object
        ORDER BY object_id DESC
        LIMIT :limit
    """), {"limit": limit}).mappings().all()
    return {"rows": [dict(r) for r in rows]}

@app.get("/objects/by_frame/{frame_id}")
def objects_by_frame(frame_id: int, session: Session = Depends(get_session)):
    rows = session.execute(text("""
        SELECT object_id, object_type, confidence, location, vector_id, detected_at
        FROM object
        WHERE frame_id = :fid
        ORDER BY object_id ASC
    """), {"fid": frame_id}).mappings().all()
    return {"rows": [dict(r) for r in rows]}


# -----------------------------------------------------------------------------
# 저장 API: 현재 프레임 + 탐지결과를 DB에 기록
#   - frame: file_path 저장(annotated JPG를 디스크에 저장)
#   - object: _last_detections 벌크 insert (location JSONB)
# -----------------------------------------------------------------------------
@app.post("/save", response_model=SaveResult)
def save_current(session: Session = Depends(get_session)):
    # 스냅 확보
    with _cam_lock:
        draw = _last_draw_bgr.copy() if _last_draw_bgr is not None else None
        detections = list(_last_detections)

    if draw is None:
        raise HTTPException(status_code=503, detail="No frame available yet")

    # 파일로 저장
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"frame_{ts}.jpg"
    file_path = str(FRAMES_DIR / filename)
    ok, jpg = cv2.imencode(".jpg", draw, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="JPEG encode failed")
    with open(file_path, "wb") as f:
        f.write(jpg.tobytes())

    try:
        # 트랜잭션
        with session.begin():
            # 1) frame insert
            result = session.execute(
                text("""
                    INSERT INTO frame (file_path)
                    VALUES (:file_path)
                    RETURNING frame_id
                """),
                {"file_path": file_path}
            )
            frame_id = result.scalar_one()

            # 2) object bulk insert
            if detections:
                rows = [
                    {
                        "frame_id": frame_id,
                        "object_type": d["object_type"],
                        "confidence": float(d["confidence"]),
                        "location": json.dumps(d["location"]),  # ✅ 꼭 넣기 (JSON 직렬화)
                        "vector_id": None,
                    }
                    for d in detections
                ]
                session.execute(
                    text("""
                        INSERT INTO object (frame_id, object_type, confidence, location, vector_id)
                        VALUES (:frame_id, :object_type, :confidence, :location, :vector_id)
                    """),
                    rows
                )

        return SaveResult(frame_id=frame_id, file_path=file_path, saved_objects=len(detections))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB save failed: {e}")
