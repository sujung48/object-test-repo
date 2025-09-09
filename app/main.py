# app/main.py
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import os
import re, json
import time
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO

from dotenv import load_dotenv
load_dotenv()

# ====== DB (SQLAlchemy sync + psycopg3) ======
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

# ====== Milvus (선택) ======
_MILVUS_AVAILABLE = False
try:
    from pymilvus import connections, Collection, db, utility
    _MILVUS_AVAILABLE = True
except Exception:
    _MILVUS_AVAILABLE = False

# -----------------------------------------------------------------------------
# 환경 설정
# -----------------------------------------------------------------------------
DATABASE_URL   = os.getenv("DATABASE_URL", "postgresql+psycopg://obuser:password@localhost:5432/db")
FRAMES_DIR     = Path(os.getenv("FRAMES_DIR", "./frames")); FRAMES_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_SOURCE  = os.getenv("CAMERA_SOURCE", "0")  # "0","1"... or file/rtsp url
FRAME_WIDTH    = int(os.getenv("FRAME_WIDTH", "1280"))
FRAME_HEIGHT   = int(os.getenv("FRAME_HEIGHT", "720"))
INFER_SIZE     = int(os.getenv("INFER_SIZE", "640"))
TARGET_FPS     = float(os.getenv("TARGET_FPS", "15"))
CAMERA_ID      = int(os.getenv("CAMERA_ID", "1"))

# Tracker
TRACKER_TYPE   = os.getenv("TRACKER_TYPE", "simple").lower()  # "simple" | "deepsort"
_DEEPSORT_AVAILABLE = False
try:
    if TRACKER_TYPE == "deepsort":
        from deep_sort_realtime.deepsort_tracker import DeepSort  # pip install deep-sort-realtime
        _DEEPSORT_AVAILABLE = True
except Exception:
    _DEEPSORT_AVAILABLE = False
    TRACKER_TYPE = "simple"

# Milvus
MILVUS_HOST        = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT        = os.getenv("MILVUS_PORT", "19530")
MILVUS_DB          = os.getenv("MILVUS_DB", "cctv")
MILVUS_COLLECTION  = os.getenv("MILVUS_COLLECTION", "objects_vec")
MILVUS_METRIC      = os.getenv("MILVUS_METRIC", "IP")  # IP | COSINE | L2
MILVUS_TOPK        = int(os.getenv("MILVUS_TOPK", "20"))
MILVUS_USE_HNSW    = os.getenv("MILVUS_USE_HNSW", "false").lower() == "true"
MILVUS_PARAM       = int(os.getenv("MILVUS_PARAM", "16"))  # ✅ IVF:nprobe / HNSW:ef

# === env 토글 추가 (파일 상단 다른 env들 옆에) ===
AEYE_OFFLINE = os.getenv("AEYE_OFFLINE", "1") == "1"      # 기본 오프라인 안전 모드
AEYE_USE_OPENCLIP = os.getenv("AEYE_USE_OPENCLIP", "0") == "1"

# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(title="A:EYE YOLO Live + Tracking + Milvus", version="1.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# YOLO 및 카메라/트래커 상태
# -----------------------------------------------------------------------------
model: Optional[YOLO] = None
device: str = "cpu"

_cam_thread: Optional[threading.Thread] = None
_cam_running: bool = False
_cam_lock = threading.Lock()
_last_frame_bgr: Optional[np.ndarray] = None
_last_draw_bgr: Optional[np.ndarray] = None

# 최근 탐지 결과
# 예: {
#   "object_type":"person","confidence":0.91,"track_id":3,
#   "bbox":[x1,y1,x2,y2], "location":{"x":..,"y":..,"z":..}
# }
_last_detections: List[Dict[str, Any]] = []

# -----------------------------------------------------------------------------
# 간단 IOU 트래커
# -----------------------------------------------------------------------------
def _iou_xyxy(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (ay2 - ay1)
    union = area_a + area_b - inter
    return inter / max(1e-6, union)

class _SimpleTrack:
    __slots__ = ("track_id", "bbox", "miss", "cls_id", "score")
    def __init__(self, tid, bbox, cls_id, score):
        self.track_id = tid
        self.bbox = bbox  # xyxy
        self.miss = 0
        self.cls_id = cls_id
        self.score = score

class SimpleTracker:
    def __init__(self, iou_thresh: float = 0.3, max_miss: int = 10):
        self.tracks: List[_SimpleTrack] = []
        self.next_id = 1
        self.iou_thresh = iou_thresh
        self.max_miss = max_miss

    def update(self, dets: List[Tuple[Tuple[float,float,float,float], float, int]]) -> List[_SimpleTrack]:
        assigned_det = set()
        assigned_trk = set()
        pairs = []
        for ti, trk in enumerate(self.tracks):
            for di, (bb, conf, cid) in enumerate(dets):
                iou = _iou_xyxy(trk.bbox, bb)
                if iou >= self.iou_thresh:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)
        for iou, ti, di in pairs:
            if ti in assigned_trk or di in assigned_det:
                continue
            trk = self.tracks[ti]
            bb, conf, cid = dets[di]
            trk.bbox = bb
            trk.cls_id = cid
            trk.score = conf
            trk.miss = 0
            assigned_trk.add(ti)
            assigned_det.add(di)

        for ti, trk in enumerate(self.tracks):
            if ti not in assigned_trk:
                trk.miss += 1

        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]

        for di, (bb, conf, cid) in enumerate(dets):
            if di in assigned_det:
                continue
            self.tracks.append(_SimpleTrack(self.next_id, bb, cid, conf))
            self.next_id += 1

        return self.tracks

# 전역 트래커
_tracker = None
def _init_tracker():
    global _tracker
    if TRACKER_TYPE == "deepsort" and _DEEPSORT_AVAILABLE:
        _tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.2, nms_max_overlap=1.0, bgr=True)
    else:
        _tracker = SimpleTracker(iou_thresh=0.3, max_miss=10)

# -----------------------------------------------------------------------------
# 임베딩 추출기 (open_clip -> torchvision(resnet18) 폴백)
# -----------------------------------------------------------------------------
class EmbeddingExtractor:
    def __init__(self, device: str, out_dim: int = 512):
        self.device = device
        self.dim = out_dim
        self.use_openclip = False
        self._init_models()

    def _init_models(self):
        # 1) OpenCLIP (선택)
        if AEYE_USE_OPENCLIP:
            try:
                import open_clip
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', pretrained=None  # ✅ 오프라인 기본: pretrained=None
                )
                # 온라인 사용 원하면 AEYE_OFFLINE=0 일 때만 pretrained='openai' 시도
                if not AEYE_OFFLINE:
                    try:
                        model, _, preprocess = open_clip.create_model_and_transforms(
                            'ViT-B-32', pretrained='openai'
                        )
                    except Exception:
                        pass
                self.oc_model = model.eval().to(self.device)
                self.oc_preprocess = preprocess
                self.use_openclip = True
                return
            except Exception:
                self.use_openclip = False  # 계속 진행(폴백)

        # 2) torchvision resnet18
        from torchvision.models import resnet18
        weights = None
        if not AEYE_OFFLINE:
            try:
                from torchvision.models import ResNet18_Weights
                weights = ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        try:
            self.tv_model = resnet18(weights=weights)
        except Exception:
            # 가중치 다운로드 실패 등 → 완전 오프라인
            self.tv_model = resnet18(weights=None)
        # 임베딩 차원 512로 뽑기
        import torch.nn as nn
        self.tv_model.fc = nn.Identity()
        self.tv_model.eval().to(self.device)

        # 기본 전처리 (ImageNet)
        import torch
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(self.device)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(self.device)

    @torch.no_grad()
    def _preprocess_tv(self, imgs: List[np.ndarray]) -> torch.Tensor:
        import torch, cv2
        ts = []
        for bgr in imgs:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            im = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(im).float().permute(2,0,1).unsqueeze(0) / 255.0
            ts.append(t)
        x = torch.cat(ts, dim=0).to(self.device)
        x = (x - self.mean) / self.std
        return x

    @torch.no_grad()
    def extract(self, crops_bgr: List[np.ndarray]) -> List[List[float]]:
        import torch.nn.functional as F
        if len(crops_bgr) == 0:
            return []
        if self.use_openclip:
            from PIL import Image
            import cv2, torch
            pil_list = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
            x = torch.cat([self.oc_preprocess(p).unsqueeze(0) for p in pil_list], dim=0).to(self.device)
            feats = self.oc_model.encode_image(x)
        else:
            x = self._preprocess_tv(crops_bgr)
            feats = self.tv_model(x)
        feats = F.normalize(feats, dim=1)  # L2 정규화
        return feats.cpu().numpy().tolist()

# 전역 임베딩 추출기 & Milvus 핸들
_embedder: Optional[EmbeddingExtractor] = None
_milvus_coll: Optional["Collection"] = None
_milvus_ok: bool = False

def _init_milvus():
    global _milvus_coll, _milvus_ok
    if not _MILVUS_AVAILABLE:
        _milvus_ok = False
        return
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        try:
            db.create_database(MILVUS_DB)
        except Exception:
            pass
        db.using_database(MILVUS_DB)
        _milvus_coll = Collection(MILVUS_COLLECTION)
        try:
            _milvus_coll.load()
        except Exception:
            pass
        _milvus_ok = True
    except Exception as e:
        print(f"[Milvus] connect/load failed: {e}")
        _milvus_ok = False

def milvus_insert_embeddings(embeds: List[List[float]], metas: List[Dict[str, Any]]) -> List[int]:
    """
    embeds: [[512], ...]
    metas:  [{"camera_id":int,"ts":int,"cls":str,"track_id":int}, ...]
    return: inserted primary keys
    """
    if not _milvus_ok or _milvus_coll is None or len(embeds) == 0:
        return []
    try:
        entities = [
            [m["camera_id"] for m in metas],
            [m["ts"] for m in metas],
            [m["cls"] for m in metas],
            [m.get("track_id", -1) for m in metas],
            embeds
        ]
        # 컬렉션 스키마 순서에 맞춰 이름 지정 insert
        mr = _milvus_coll.insert(
            data=entities,
            insert_fields=["camera_id", "ts", "cls", "track_id", "embedding"]
        )
        pks = [int(x) for x in mr.primary_keys]
        return pks
    except Exception as e:
        print(f"[Milvus] insert failed: {e}")
        return []

def milvus_search(embeds: List[List[float]], topk: int, expr: Optional[str] = None) -> List[List[Dict[str, Any]]]:
    """
    return: [[{"id":..., "score":..., "camera_id":..., "ts":..., "cls":..., "track_id":...}], ...]
    """
    if not _milvus_ok or _milvus_coll is None or len(embeds) == 0:
        return [[]]
    params = {"metric_type": MILVUS_METRIC}
    # IVF vs HNSW 파라미터
    if MILVUS_USE_HNSW:
        params["params"] = {"ef": MILVUS_PARAM}
    else:
        params["params"] = {"nprobe": MILVUS_PARAM}
    try:
        res = _milvus_coll.search(
            data=embeds,
            anns_field="embedding",
            param=params,
            limit=topk,
            expr=expr,
            output_fields=["camera_id", "ts", "cls", "track_id"]
        )
        out = []
        for hits in res:
            cur = []
            for h in hits:
                ent = h.entity
                cur.append({
                    "id": int(h.id),
                    "score": float(h.score),
                    "camera_id": int(ent.get("camera_id")),
                    "ts": int(ent.get("ts")),
                    "cls": str(ent.get("cls")),
                    "track_id": int(ent.get("track_id")),
                })
            out.append(cur)
        return out
    except Exception as e:
        print(f"[Milvus] search failed: {e}")
        return [[]]

# -----------------------------------------------------------------------------
# DB 세션팩토리 & 의존성 (동기)
# -----------------------------------------------------------------------------
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)

def get_session():
    dbs = SessionLocal()
    try:
        yield dbs
    finally:
        dbs.close()

# -----------------------------------------------------------------------------
# 응답 모델
# -----------------------------------------------------------------------------
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
            cap = cv2.VideoCapture(cam_index)
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
        inp = cv2.resize(frame_bgr, (INFER_SIZE, INFER_SIZE), interpolation=cv2.INTER_AREA)
        inp_rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(inp_rgb)

        detections: List[Dict[str, Any]] = []
        draw_resized = frame_bgr

        try:
            results = model.predict(source=pil_img, conf=0.25, iou=0.45, device=device, verbose=False)
            r = results[0]

            if r.boxes is not None and len(r.boxes) > 0:
                xyxy_infer = r.boxes.xyxy.cpu().numpy()
                confs      = r.boxes.conf.cpu().numpy()
                clss       = r.boxes.cls.cpu().numpy().astype(int)
            else:
                xyxy_infer = np.zeros((0, 4))
                confs      = np.zeros((0,))
                clss       = np.zeros((0,), dtype=int)

            scale_x = W / float(INFER_SIZE)
            scale_y = H / float(INFER_SIZE)
            xyxy_full = xyxy_infer.copy()
            if xyxy_full.size:
                xyxy_full[:, [0, 2]] *= scale_x
                xyxy_full[:, [1, 3]] *= scale_y

            dets_for_tracker = [ (tuple(bb.tolist()), float(cf), int(ci)) for bb, cf, ci in zip(xyxy_full, confs, clss) ]

            # === 트래커 ===
            tracks_info = []
            if isinstance(_tracker, SimpleTracker):
                tracks = _tracker.update(dets_for_tracker)
                for t in tracks:
                    tracks_info.append({
                        "track_id": t.track_id,
                        "bbox": t.bbox,     # xyxy
                        "cls_id": t.cls_id,
                        "score": t.score
                    })
            else:
                # DeepSORT
                ds_tracks = _tracker.update_tracks(dets_for_tracker, frame=frame_bgr)
                for tr in ds_tracks:
                    if not tr.is_confirmed():
                        continue
                    x1, y1, x2, y2 = tr.to_tlbr()  # xyxy
                    tracks_info.append({
                        "track_id": int(tr.track_id),
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        "cls_id": int(getattr(tr, "det_class", -1)),
                        "score": float(getattr(tr, "det_confidence", 0.0))
                    })

            # 가시화
            draw = r.plot()  # INFER_SIZE 기준
            draw_resized = cv2.resize(draw, (W, H), interpolation=cv2.INTER_LINEAR)
            for ti in tracks_info:
                x1, y1, x2, y2 = ti["bbox"]
                tid = ti["track_id"]
                cv2.rectangle(draw_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                cv2.putText(draw_resized, f"id:{tid}", (int(x1), max(0, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # detections 기록 (bbox 포함)
            names = r.names if hasattr(r, "names") else (model.names if hasattr(model, "names") else {})
            for ti in tracks_info:
                x1, y1, x2, y2 = ti["bbox"]
                w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)
                cx = (x1 + x2) / 2.0 / W
                cy = (y1 + y2) / 2.0 / H
                area_ratio = (w * h) / (W * H)
                k = ti["cls_id"]
                label = names.get(k, str(k)) if isinstance(names, dict) else str(k)
                detections.append({
                    "object_type": label,
                    "confidence": float(ti["score"]),
                    "track_id": int(ti["track_id"]),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "location": {"x": float(cx), "y": float(cy), "z": float(area_ratio)}
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

# ==== 안전한 마스킹 함수 ====
def _mask_db_url(url: str | None) -> str:
    if not url:
        return ""
    try:
        return re.sub(r'(?<=://[^:]+:)[^@]+', '****', url)
    except Exception:
        return url

# -----------------------------------------------------------------------------
# 앱 라이프사이클
# -----------------------------------------------------------------------------
@app.on_event("startup")
def load_model():
    global model, device, _embedder
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

    _init_tracker()
    _embedder = EmbeddingExtractor(device=device, out_dim=512)
    _init_milvus()

@app.on_event("shutdown")
def shutdown():
    _stop_camera()

# -----------------------------------------------------------------------------
# 헬스/카메라/스트림/HTML
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "model": "yolov8n.pt",
        "camera_running": _cam_running,
        "tracker": TRACKER_TYPE,
        "deepsort_available": _DEEPSORT_AVAILABLE,
        "milvus_available": _MILVUS_AVAILABLE,
        "milvus_connected": _milvus_ok,
        "milvus": {
            "db": MILVUS_DB, "collection": MILVUS_COLLECTION,
            "metric": MILVUS_METRIC, "topk": MILVUS_TOPK,
            "param": MILVUS_PARAM, "use_hnsw": MILVUS_USE_HNSW
        }
    }

@app.get("/db/url")
def db_url():
    raw = os.getenv("DATABASE_URL") or DATABASE_URL or ""
    return {"url": _mask_db_url(raw), "empty": raw == ""}

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
          <h3>A:EYE Live (YOLO + Tracking + Milvus)</h3>
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
# 저장 API: 현재 프레임 + 탐지결과 -> 파일 저장 + Postgres + (선택)Milvus 임베딩 삽입
#   * bbox를 이용해 원본 프레임에서 crop → 임베딩 추출 → Milvus insert
#   * 반환된 PK를 Postgres object.vector_id로 매핑하여 저장
# -----------------------------------------------------------------------------
@app.post("/save", response_model=SaveResult)
def save_current(session: Session = Depends(get_session)):
    with _cam_lock:
        draw = _last_draw_bgr.copy() if _last_draw_bgr is not None else None
        raw  = _last_frame_bgr.copy() if _last_frame_bgr is not None else None
        detections = list(_last_detections)

    if draw is None or raw is None:
        raise HTTPException(status_code=503, detail="No frame available yet")

    # 1) 파일 저장(annotated)
    ts_str = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"frame_{ts_str}.jpg"
    file_path = str(FRAMES_DIR / filename)
    ok, jpg = cv2.imencode(".jpg", draw, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="JPEG encode failed")
    with open(file_path, "wb") as f:
        f.write(jpg.tobytes())

    # 2) 임베딩 준비(필요 시)
    milvus_ids: List[Optional[int]] = [None] * len(detections)
    try:
        if _milvus_ok and _embedder is not None and len(detections) > 0:
            H, W = raw.shape[:2]
            crops, metas = [], []
            epoch_sec = int(time.time())
            for d in detections:
                if "bbox" not in d:
                    continue
                x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                # 클리핑
                x1 = max(0, min(W-1, x1)); x2 = max(0, min(W, x2))
                y1 = max(0, min(H-1, y1)); y2 = max(0, min(H, y2))
                if x2 <= x1 or y2 <= y1:
                    crops.append(None); metas.append(None); continue
                crop = raw[y1:y2, x1:x2].copy()
                crops.append(crop)
                metas.append({
                    "camera_id": CAMERA_ID,
                    "ts": epoch_sec,
                    "cls": d.get("object_type", "unknown"),
                    "track_id": int(d.get("track_id", -1))
                })
            # None 제거
            valid = [(i, c, m) for i,(c,m) in enumerate(zip(crops, metas)) if c is not None and m is not None]
            if valid:
                idxs, vcrops, vmetas = zip(*valid)
                embeds = _embedder.extract(list(vcrops))
                pks = milvus_insert_embeddings(embeds, list(vmetas))
                for i, pk in zip(idxs, pks):
                    milvus_ids[i] = pk
    except Exception as e:
        print(f"[Milvus] embedding/insert skipped due to error: {e}")

    # 3) DB 트랜잭션
    try:
        with session.begin():
            # frame insert
            result = session.execute(
                text("""
                    INSERT INTO frame (file_path)
                    VALUES (:file_path)
                    RETURNING frame_id
                """),
                {"file_path": file_path}
            )
            frame_id = result.scalar_one()

            # object bulk insert
            if detections:
                rows = []
                for idx, d in enumerate(detections):
                    loc = dict(d.get("location", {}))
                    if "track_id" in d:
                        loc["tid"] = int(d["track_id"])
                    rows.append({
                        "frame_id": frame_id,
                        "object_type": d["object_type"],
                        "confidence": float(d["confidence"]),
                        "location": json.dumps(loc),
                        "vector_id": (str(milvus_ids[idx]) if milvus_ids[idx] is not None else None),
                    })
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

# -----------------------------------------------------------------------------
# Milvus 유사도 검색 API
#   - object_id를 받아 vector_id로 임베딩을 가져와 Top-K 검색 수행
#   - (벡터를 Milvus에서 다시 가져오거나, 동일 object_id의 bbox를 현재 프레임/파일에서 재추출하는 방식)
# -----------------------------------------------------------------------------
class SimilarReq(BaseModel):
    object_id: int
    topk: Optional[int] = None
    cls: Optional[str] = None
    camera_ids: Optional[List[int]] = None
    ts_from: Optional[int] = None
    ts_to: Optional[int] = None

@app.post("/search/similar")
def search_similar(req: SimilarReq, session: Session = Depends(get_session)):
    if not _milvus_ok or _milvus_coll is None:
        raise HTTPException(status_code=503, detail="Milvus not available")

    # 1) vector_id 조회
    row = session.execute(
        text("SELECT vector_id FROM object WHERE object_id = :oid"),
        {"oid": req.object_id}
    ).mappings().first()
    if not row or not row["vector_id"]:
        raise HTTPException(status_code=404, detail="vector_id not found for object_id")

    try:
        pk = int(row["vector_id"])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid vector_id format")

    # 2) 해당 PK의 벡터를 Milvus에서 조회 (output_fields에 embedding 포함)
    try:
        q = _milvus_coll.query(expr=f"id in [{pk}]", output_fields=["embedding"])
        if not q:
            raise HTTPException(status_code=404, detail="Embedding not found in Milvus")
        emb = q[0]["embedding"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus query failed: {e}")

    # 3) 필터 구성
    filters = []
    if req.cls: filters.append(f"cls == '{req.cls}'")
    if req.camera_ids: filters.append(f"camera_id in [{','.join(str(i) for i in req.camera_ids)}]")
    if req.ts_from is not None: filters.append(f"ts >= {int(req.ts_from)}")
    if req.ts_to is not None: filters.append(f"ts < {int(req.ts_to)}")
    expr = " && ".join(filters) if filters else None

    # 4) 검색
    topk = int(req.topk or MILVUS_TOPK)
    res = milvus_search([emb], topk=topk, expr=expr)
    return {"hits": res[0], "expr": expr}

# -----------------------------------------------------------------------------
# 업로드된 이미지로 유사도 검색 (선택)
#   - bbox를 [x1,y1,x2,y2]로 주면 해당 크롭 임베딩으로 검색
# -----------------------------------------------------------------------------
class SearchByCropReq(BaseModel):
    bbox: List[int]  # [x1,y1,x2,y2]
    cls: Optional[str] = None
    camera_ids: Optional[List[int]] = None
    ts_from: Optional[int] = None
    ts_to: Optional[int] = None
    topk: Optional[int] = None

@app.post("/search/by_image")
async def search_by_image(file: UploadFile = File(...), meta: str = "{}"):
    if not _milvus_ok or _milvus_coll is None or _embedder is None:
        raise HTTPException(status_code=503, detail="Milvus or embedder not available")
    try:
        meta_obj = json.loads(meta or "{}")
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        bbox = meta_obj.get("bbox")
        if not bbox or len(bbox) != 4:
            raise HTTPException(status_code=400, detail="bbox required")
        x1,y1,x2,y2 = [int(v) for v in bbox]
        H,W = img.shape[:2]
        x1 = max(0, min(W-1, x1)); x2 = max(0, min(W, x2))
        y1 = max(0, min(H-1, y1)); y2 = max(0, min(H, y2))
        if x2 <= x1 or y2 <= y1:
            raise HTTPException(status_code=400, detail="invalid bbox")
        crop = img[y1:y2, x1:x2].copy()
        emb = _embedder.extract([crop])[0]

        filters = []
        if meta_obj.get("cls"): filters.append(f"cls == '{meta_obj['cls']}'")
        if meta_obj.get("camera_ids"): filters.append(f"camera_id in [{','.join(str(i) for i in meta_obj['camera_ids'])}]")
        if meta_obj.get("ts_from") is not None: filters.append(f"ts >= {int(meta_obj['ts_from'])}")
        if meta_obj.get("ts_to")   is not None: filters.append(f"ts < {int(meta_obj['ts_to'])}")
        expr = " && ".join(filters) if filters else None

        topk = int(meta_obj.get("topk") or MILVUS_TOPK)
        res = milvus_search([emb], topk=topk, expr=expr)
        return {"hits": res[0], "expr": expr}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
