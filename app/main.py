# app/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .analyze import router as analyze_router

import io, json, os, threading, time
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import numpy as np
import faiss
import cv2
from PIL import Image, ImageOps

import torch
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO
import joblib

# ==============================
# Config
# ==============================
MODEL_ID = "openai/clip-vit-base-patch32"
USE_FAST = True

DET_MODEL = YOLO("yolov8n.pt")
PET_CLASSES = [15, 16]  # 15=cat, 16=dog

TOPN_PRERANK = 600            # pré-filtro FAISS
KRECIP_SAFE_MAXN = 2000
TOPK_RERANK_SOURCE = 32       # pool para fusão (HSV + head)

QUALITY_MIN_EDGE = 256
QUALITY_MIN_FOCUS = 60.0
QUALITY_MIN_BRIGHT = 20
QUALITY_MAX_BRIGHT = 235

COCO2SPECIES = {15: "cat", 16: "dog"}
TEXT_PROMPTS = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a bird",
    "a photo of an animal",
]

INDEX_DIR = Path("index")
INDEX_PATH = INDEX_DIR / "pets.faiss"
META_PATH = INDEX_DIR / "pets_meta.json"

# ==============================
# Persistência
# ==============================
def save_index_atomic(index: faiss.Index):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_PATH.with_suffix(".faiss.tmp")
    faiss.write_index(index, str(tmp))
    os.replace(tmp, INDEX_PATH)

def save_meta_atomic(rows: List[dict]):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    tmp = META_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    os.replace(tmp, META_PATH)

def _normpath(p: str) -> str:
    """Normaliza caminho para comparar/deduplicar de forma robusta."""
    try:
        return str(Path(p).resolve()).replace("\\", "/").lower()
    except Exception:
        return str(Path(p)).replace("\\", "/").lower()

# ==============================
# k-reciprocal (estável e consistente com L2 do FAISS)
# ==============================
def _soft_weights_from_dist(d: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """
    Converte distâncias em pesos estáveis:
      - desloca para começar em 0
      - normaliza pela mediana (escala robusta)
      - limita expoente a [-50, 50] (evita overflow)
    Retorna vetor que soma 1 (ou uniforme se underflow).
    """
    d = d.astype(np.float64)
    d = d - d.min()
    scale = np.median(d) + 1e-12
    z = -d / (tau * scale)
    z = np.clip(z, -50.0, 50.0)
    w = np.exp(z)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(w, dtype=np.float64) / max(1, len(w))
    return w / s

def _l2_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Retorna ||a - b||^2 para todos os pares (linhas)
    # a: [Na, D], b: [Nb, D]
    a2 = (a * a).sum(axis=1, keepdims=True)        # [Na, 1]
    b2 = (b * b).sum(axis=1, keepdims=True).T      # [1, Nb]
    return a2 + b2 - 2.0 * (a @ b.T)

def kreciprocal_rerank_single(q: np.ndarray, g: np.ndarray, k1=20, k2=6, lambda_value=0.3) -> np.ndarray:
    """
    q: [1,D], g: [N,D]. Custo ~ O(N^2). Use somente com N<=~2000.
    Retorna distâncias re-rankeadas (menor = melhor), shape [N].
    """
    # Distâncias euclidianas quadráticas reais (consistentes com o IndexFlatL2)
    original_dist = _l2_sq_dists(q, g)[0]          # shape [N]
    gallery_dist  = _l2_sq_dists(g, g)             # shape [N, N]

    # vizinhos
    nn_g = np.argsort(original_dist)[:k1]
    nn_g_g = np.argsort(gallery_dist[nn_g], axis=1)[:, :k1]
    k_reciprocal = [gi for i, gi in enumerate(nn_g) if gi in nn_g_g[i]]
    k_reciprocal = np.array(k_reciprocal, dtype=int)
    if k_reciprocal.size == 0:
        return original_dist

    # pesos estáveis
    V = np.zeros(g.shape[0], dtype=np.float64)
    V[k_reciprocal] = _soft_weights_from_dist(original_dist[k_reciprocal], tau=1.0)

    if k2 > 1:
        nn_k2 = np.argsort(original_dist)[:k2]
        V2 = np.zeros_like(V)
        V2[nn_k2] = _soft_weights_from_dist(original_dist[nn_k2], tau=1.0)
        V = 0.5 * (V + V2)

    invIndex = []
    for j in range(g.shape[0]):
        d = gallery_dist[j]
        idx = np.argsort(d)[:k1]
        vj = np.zeros(g.shape[0], dtype=np.float64)
        vj[idx] = _soft_weights_from_dist(d[idx], tau=1.0)
        invIndex.append(vj)

    jaccard = np.zeros(g.shape[0], dtype=np.float64)
    for j in range(g.shape[0]):
        jaccard[j] = 1.0 - float(np.minimum(V, invIndex[j]).sum())

    # combinação
    orig_norm = original_dist / (original_dist.max() + 1e-12)
    final_dist = (1 - lambda_value) * jaccard + lambda_value * orig_norm
    final_dist = np.nan_to_num(final_dist, nan=1.0, posinf=1.0, neginf=0.0).astype(np.float32)
    return final_dist

# ==============================
# Head/face (heurística)
# ==============================
_haar_cat = None
_haar_path = cv2.data.haarcascades + "haarcascade_frontalcatface.xml"
if os.path.exists(_haar_path):
    _haar_cat = cv2.CascadeClassifier(_haar_path)

def extract_head_crop_from_bgr(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    if _haar_cat is not None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = _haar_cat.detectMultiScale(gray, 1.1, 3)
        if len(faces):
            x, y, ww, hh = max(faces, key=lambda r: r[2]*r[3])
            pad = int(0.15 * max(ww, hh))
            x1 = max(0, x - pad); y1 = max(0, y - pad)
            x2 = min(w, x + ww + pad); y2 = min(h, y + hh + pad)
            crop = bgr[y1:y2, x1:x2]
            if crop.size > 0:
                return crop
    # fallback: região superior central
    y2 = int(0.55 * h); x1 = int(0.15 * w); x2 = int(0.85 * w)
    crop = bgr[:y2, x1:x2]
    return crop if crop.size else bgr

def cosine_sim_vec(qv: np.ndarray, gv: np.ndarray) -> float:
    return float((qv @ gv.T).ravel()[0])

# ==============================
# App + modelos
# ==============================
app = FastAPI(title="PataNet Vision API", version="0.7.3")
app.include_router(analyze_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
proc  = CLIPProcessor.from_pretrained(MODEL_ID, use_fast=USE_FAST)

# --------- Índice principal: tenta carregar PCA+índice novo, senão o original ---------
GALLERY_PCA = None
try:
    GALLERY_PCA = joblib.load("models/gallery_pca.joblib")
    index = faiss.read_index("index/pets_pca.faiss")
    print("[gallery] PCA carregada + index_pca.faiss")
except Exception as e:
    print("[gallery] sem PCA de galeria; usando índice original:", e)
    index = faiss.read_index(str(INDEX_PATH))

# meta e matriz da galeria (no espaço DO ÍNDICE carregado)
meta  = json.load(open(META_PATH, "r", encoding="utf-8"))
XB    = index.reconstruct_n(0, index.ntotal)

# Classificador de raça (opcional) + PCA (para o classificador)
BREED_CLF = None; BREED_LABELS: List[str] = []; BREED_PCA = None
try:
    BREED_CLF = joblib.load("models/breed_clf.joblib")
    BREED_LABELS = json.load(open("models/breed_labels.json", "r", encoding="utf-8"))
    try:
        BREED_PCA = joblib.load("models/breed_pca.joblib")  # espera input 512D
        print("[breed] PCA carregada.")
    except Exception:
        BREED_PCA = None
    print(f"[breed] loaded {len(BREED_LABELS)} classes.")
except Exception as e:
    print("[breed] classificador não carregado:", e)

# Head index (FAISS) + lookup: crop_norm -> posição no head_meta
HEAD_INDEX: Optional[faiss.Index] = None
HEAD_META: Optional[List[dict]] = None
HEAD_LOOKUP: Dict[str, int] = {}

try:
    HEAD_INDEX = faiss.read_index("index/head.faiss")
    with open("index/head_meta.json", "r", encoding="utf-8") as f:
        HEAD_META = json.load(f)
    for i, m in enumerate(HEAD_META):
        c = m.get("crop")
        if isinstance(c, str):
            HEAD_LOOKUP[_normpath(c)] = i
    print(f"[head] index com {HEAD_INDEX.ntotal} itens. lookup={len(HEAD_LOOKUP)}")
except Exception:
    HEAD_INDEX = None; HEAD_META = None; HEAD_LOOKUP = {}
    print("[head] sem head-index.")

LOCK = threading.Lock()

# ==============================
# Utils
# ==============================
def letterbox_pil(pil: Image.Image, target_min_edge=QUALITY_MIN_EDGE, border=16) -> Image.Image:
    w, h = pil.size
    scale = target_min_edge / min(w, h)
    if scale < 1.0:
        return ImageOps.expand(pil, border=border, fill=(0, 0, 0))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = pil.resize((nw, nh), Image.BICUBIC)
    return ImageOps.expand(resized, border=border, fill=(0, 0, 0))

def quality_report(img_bgr: np.ndarray) -> Tuple[bool, List[str]]:
    h, w = img_bgr.shape[:2]
    notes = []
    if min(h, w) < QUALITY_MIN_EDGE:
        notes.append(f"min_edge<{QUALITY_MIN_EDGE}px (está {min(h,w)}px)")
    lap = cv2.Laplacian(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
    focus = float(lap.var())
    if focus < QUALITY_MIN_FOCUS:
        notes.append(f"foco baixo (varLap={focus:.1f} < {QUALITY_MIN_FOCUS})")
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_mean = float(hsv[..., 2].mean())
    if v_mean < QUALITY_MIN_BRIGHT:
        notes.append(f"muito escura (V_mean={v_mean:.1f})")
    elif v_mean > QUALITY_MAX_BRIGHT:
        notes.append(f"muito clara (V_mean={v_mean:.1f})")
    return (len(notes) == 0), notes

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def embed_pil(pil_img: Image.Image) -> np.ndarray:
    inputs = proc(images=pil_img.convert("RGB"), return_tensors="pt").to(device)
    with torch.no_grad():
        e = model.get_image_features(**inputs)
    e = torch.nn.functional.normalize(e, dim=-1)
    return e.squeeze(0).cpu().numpy().astype("float32")

def clip_zero_shot_species(pil_img: Image.Image) -> Tuple[str, float]:
    inputs = proc(text=TEXT_PROMPTS, images=pil_img.convert("RGB"),
                  return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(**inputs)
        image_emb = torch.nn.functional.normalize(out.image_embeds, dim=-1)
        text_emb  = torch.nn.functional.normalize(out.text_embeds,  dim=-1)
        sims = (image_emb @ text_emb.T).squeeze(0)
        probs = torch.softmax(sims, dim=0)
        idx = int(torch.argmax(probs).item())
    return TEXT_PROMPTS[idx].split()[-1], float(probs[idx].item())

def color_hist_hsv(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.normalize(cv2.calcHist([hsv], [0], None, [32], [0, 180]), None).flatten()
    s = cv2.normalize(cv2.calcHist([hsv], [1], None, [32], [0, 256]), None).flatten()
    return np.concatenate([h, s]).astype("float32")

def hist_intersection(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.minimum(a, b).sum())  # ~[0..2]

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(); ex = np.exp(x); return ex / (ex.sum() + 1e-12)

def breed_topk_from_vec(vec512: np.ndarray, k=3):
    """Recebe SEMPRE embedding 512D (CLIP)."""
    if BREED_CLF is None:
        return []
    z = vec512
    if BREED_PCA is not None:
        z = BREED_PCA.transform(vec512)
    if hasattr(BREED_CLF, "predict_proba"):
        p = BREED_CLF.predict_proba(z)
        order = np.argsort(-p[0])[:k]
        return [{"label": BREED_LABELS[i], "prob": float(p[0][i])} for i in order]
    if hasattr(BREED_CLF, "decision_function"):
        d = BREED_CLF.decision_function(z)
        if d.ndim == 1: d = d.reshape(1, -1)
        pr = softmax(d[0]); order = np.argsort(-pr)[:k]
        return [{"label": BREED_LABELS[i], "prob": float(pr[i])} for i in order]
    return []

# ---- avaliação de aparência (neutra) ----
def describe_coat_neutral(hsv_img: np.ndarray) -> str:
    """Descrição neutra do tom predominante da pelagem (apenas explicação)."""
    h_mean = float(hsv_img[..., 0].mean())
    s_mean = float(hsv_img[..., 1].mean())
    v_mean = float(hsv_img[..., 2].mean())
    brilho = "claro" if v_mean > 180 else ("escuro" if v_mean < 70 else "médio")
    if s_mean < 35:
        tom = "neutro (branco/cinza/creme)"
    else:
        if 15 <= h_mean <= 35:
            tom = "amarelado/dourado"
        elif h_mean < 15 or h_mean > 165:
            tom = "avermelhado/marrom"
        elif 35 < h_mean <= 85:
            tom = "esverdeado/oliva"
        elif 85 < h_mean <= 135:
            tom = "azulado/preto"
        else:
            tom = "indefinido"
    return f"tom predominante {tom}, brilho {brilho}"

def has_white_chest_mask(hsv_img: np.ndarray) -> bool:
    h, w = hsv_img.shape[:2]
    roi = hsv_img[int(0.45*h):int(0.75*h), int(0.25*w):int(0.75*w)]
    mask = (roi[...,2] > 180) & (roi[...,1] < 60)
    return bool(mask.mean() > 0.12)

def pct(x: float) -> float:
    """Converte [0..1] em porcentagem (0.0..100.0) com clamp."""
    return round(100.0 * max(0.0, min(1.0, float(x))), 1)

# ==============================
# Endpoints
# ==============================
@app.get("/")
def root():
    return {"ok": True, "message": "PataNet Vision API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"ok": True, "ntotal": index.ntotal}

@app.get("/version")
def version():
    return {
        "model": MODEL_ID,
        "use_fast": USE_FAST,
        "device": device,
        "ntotal": index.ntotal,
        "dim": index.d,
        "gallery_pca": bool(GALLERY_PCA is not None),
        "head_index": bool(HEAD_INDEX is not None),
        "head_lookup": len(HEAD_LOOKUP),
    }

@app.post("/search")
async def search(
    file: UploadFile = File(...),
    k: int = 5,
    use_head: int = 1,   # 1=liga head similarity; 0=desliga
    use_color: int = 1,  # 1=liga cor HSV; 0=desliga
    # ---- controles ----
    w_krec: float = 0.60,
    w_head: float = 0.25,
    w_color: float = 0.15,
    krec_k1: int = 20,
    krec_k2: int = 6,
    krec_lambda: float = 0.30,
    return_diagnostics: int = 0,
):
    t0 = time.perf_counter()
    t_stage = {}

    # 1) carregar imagem
    pil_raw = Image.open(io.BytesIO(await file.read())).convert("RGB")
    pil = letterbox_pil(pil_raw, QUALITY_MIN_EDGE)
    t_stage["load"] = time.perf_counter() - t0

    # 2) quality
    t1 = time.perf_counter()
    bgr = pil_to_bgr(pil)
    quality_ok, notes = quality_report(bgr)
    t_stage["quality"] = time.perf_counter() - t1

    # 3) espécie zero-shot
    t2 = time.perf_counter()
    species, s_conf = clip_zero_shot_species(pil)
    apply_filter = species in ("dog", "cat", "bird") and s_conf >= 0.40
    t_stage["zero_shot"] = time.perf_counter() - t2

    # 4) embeddings da query
    #    - q512: espaço CLIP 512D (para classificador de raça)
    #    - q   : espaço DA GALERIA (PCA 256D se ativa; senão 512D)
    t3 = time.perf_counter()
    q512 = embed_pil(pil).reshape(1, -1)             # 512D
    q = GALLERY_PCA.transform(q512).astype("float32") if GALLERY_PCA is not None else q512
    t_stage["embed"] = time.perf_counter() - t3

    # 5) FAISS Top-N
    t4 = time.perf_counter()
    _, I_all = index.search(q, min(TOPN_PRERANK, index.ntotal))
    I_all = I_all[0]
    t_stage["faiss"] = time.perf_counter() - t4

    # 6) filtro opcional por espécie
    if apply_filter:
        mask = np.array([COCO2SPECIES.get(int(meta[i].get("cls", -1))) == species if i < len(meta) else False for i in I_all])
        g_idx = I_all[mask]
        if g_idx.size == 0:
            g_idx = I_all
    else:
        g_idx = I_all
    g = XB[g_idx]
    N_sub = g.shape[0]

    # 7) k-reciprocal (ou cosseno normalizado)
    t5 = time.perf_counter()
    if N_sub <= KRECIP_SAFE_MAXN and N_sub >= 2:
        dist_rr = kreciprocal_rerank_single(q, g, k1=krec_k1, k2=krec_k2, lambda_value=krec_lambda)
        sim_base = 1.0 - (dist_rr / (dist_rr.max() + 1e-12))
    else:
        g_norm = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        sim_base = (q_norm @ g_norm.T).ravel()  # ∈ [-1, 1]
        sim_base = (sim_base - sim_base.min()) / (sim_base.max() - sim_base.min() + 1e-12)
    t_stage["krecip"] = time.perf_counter() - t5

    # 8) prepara atributos da query
    q_hist = color_hist_hsv(bgr)
    q_head_bgr = extract_head_crop_from_bgr(bgr)
    q_head_pil = Image.fromarray(cv2.cvtColor(q_head_bgr, cv2.COLOR_BGR2RGB))
    q_head_vec = embed_pil(q_head_pil).reshape(1, -1)  # (sem PCA — espaço CLIP original)

    # 9) fusão (pool)
    M = min(TOPK_RERANK_SOURCE, N_sub)
    order0 = np.argsort(-sim_base)[:M]

    fused = []
    seen_ids = set()
    seen_crops = set()
    head_time = 0.0
    color_time = 0.0

    for r in order0:
        gi = int(g_idx[int(r)])
        if gi in seen_ids:
            continue
        seen_ids.add(gi)

        m  = meta[gi]
        crop_path = m["crop"]
        crop_key = _normpath(crop_path)
        if crop_key in seen_crops:
            continue
        seen_crops.add(crop_key)

        # ---- Similaridade de cor (HSV) ----
        s_hist_norm = 0.0
        if use_color:
            tC = time.perf_counter()
            if "hsv_hist" in m and isinstance(m["hsv_hist"], list) and len(m["hsv_hist"]) == 64:
                cand_hist = np.array(m["hsv_hist"], dtype="float32")
                s_hist = hist_intersection(q_hist, cand_hist)
                s_hist_norm = s_hist / 2.0
            else:
                img_bgr = cv2.imread(crop_path, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    s_hist = hist_intersection(q_hist, color_hist_hsv(img_bgr))
                    s_hist_norm = s_hist / 2.0
            color_time += (time.perf_counter() - tC)

        # ---- Similaridade de cabeça ----
        s_head = 0.0
        if use_head:
            tH = time.perf_counter()
            head_pos = None
            if HEAD_INDEX is not None and HEAD_LOOKUP:
                head_pos = HEAD_LOOKUP.get(crop_key, None)

            if head_pos is not None and 0 <= head_pos < (HEAD_INDEX.ntotal if HEAD_INDEX is not None else 0):
                try:
                    g_head_vec = HEAD_INDEX.reconstruct(head_pos)  # vetor já normalizado
                    s_head = cosine_sim_vec(q_head_vec, g_head_vec)
                except Exception:
                    s_head = 0.0
            else:
                img_bgr = cv2.imread(crop_path, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    head_bgr = extract_head_crop_from_bgr(img_bgr)
                    if head_bgr is not None and head_bgr.size > 0:
                        head_pil = Image.fromarray(cv2.cvtColor(head_bgr, cv2.COLOR_BGR2RGB))
                        g_head_vec = embed_pil(head_pil).reshape(1, -1)  # (sem PCA)
                        s_head = cosine_sim_vec(q_head_vec, g_head_vec)
            head_time += (time.perf_counter() - tH)

        s_krec = float(sim_base[int(r)])
        s_krec = max(0.0, min(1.0, s_krec))
        s_head = max(0.0, min(1.0, float(s_head)))
        s_hist_norm = max(0.0, min(1.0, float(s_hist_norm)))

        # ---- pesos (controláveis por query) ----
        w_k = float(max(0.0, w_krec))
        w_h = float(max(0.0, w_head)) if use_head else 0.0
        w_c = float(max(0.0, w_color)) if use_color else 0.0
        total_w = w_k + w_h + w_c + 1e-12
        w_k, w_h, w_c = w_k/total_w, w_h/total_w, w_c/total_w

        S = w_k * s_krec + w_h * s_head + w_c * s_hist_norm

        fused.append((S, gi, s_krec, s_head, s_hist_norm, w_k, w_h, w_c))
        if len(fused) >= k:
            break

    t_stage["head"] = head_time
    t_stage["color"] = color_time

    fused.sort(key=lambda x: x[0], reverse=True)
    top = fused[:k]

    results = []
    for S, gi, s_krec, s_head, s_hist, wk, wh, wc in top:
        m = meta[gi]
        results.append({
            "score": round(float(S), 4),
            "score_pct": pct(S),
            "score_krecip": round(float(s_krec), 4),
            "score_krecip_pct": pct(s_krec),
            "score_head": round(float(s_head), 4),
            "score_head_pct": pct(s_head),
            "score_color": round(float(s_hist), 4),
            "score_color_pct": pct(s_hist),
            "weights": {"krec": round(wk,3), "head": round(wh,3), "color": round(wc,3)},
            "species_pred": species,
            "species_conf": round(float(s_conf), 3),
            "crop": Path(m["crop"]).as_posix(),
            "src": (Path(m["src"]).as_posix() if isinstance(m.get("src"), str) else m.get("src")),
            "cls": int(m.get("cls", -1)),
        })

    # raça (usa 512D SEMPRE)
    breed_top3 = breed_topk_from_vec(q512, k=3)

    # aparência
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    attrs = {
        "coat": describe_coat_neutral(hsv),
        "chest_white": bool(has_white_chest_mask(hsv)),
        "ear": None,
    }

    parts = []
    if breed_top3:
        parts.append(f"Raça provável: {breed_top3[0]['label'].replace('_',' ')} ({breed_top3[0]['prob']:.2f})")
        if len(breed_top3) > 1:
            parts.append(f"Alternativa: {breed_top3[1]['label'].replace('_',' ')} ({breed_top3[1]['prob']:.2f})")
    else:
        parts.append("Raça provável: n/d")
    parts.append(f"Aparência: {attrs['coat']}")
    parts.append("marca branca no peito: sim" if attrs["chest_white"] else "marca branca no peito: não")
    explanation = "; ".join([p for p in parts if p])

    latency_ms = (time.perf_counter() - t0) * 1000.0

    payload = {
        "ok": True,
        "latency_ms": round(latency_ms, 1),
        "quality_ok": bool(quality_ok),
        "quality_notes": notes,
        "filtered_by_species": bool(apply_filter),
        "k": int(k),
        "breed_top3": breed_top3,
        "attributes": attrs,
        "explanation": explanation,
        "topk": results,
        "flags": {"use_head": bool(use_head), "use_color": bool(use_color)},
    }
    if int(return_diagnostics) == 1:
        payload["diagnostics"] = {
            "timings_sec": {k: round(float(v), 4) for k, v in t_stage.items()},
            "weights": {"w_krec": float(w_krec), "w_head": float(w_head), "w_color": float(w_color)},
            "krecip": {"k1": int(krec_k1), "k2": int(krec_k2), "lambda": float(krec_lambda)},
        }

    return JSONResponse(content=payload)

@app.post("/index/add")
async def index_add(
    files: List[UploadFile] = File(...),
    pet_id: str | None = Form(default=None),
    src_note: str | None = Form(default=None),
    det_conf: float = Form(default=0.35),
    min_edge: int = Form(default=40),
):
    """
    Adiciona 1..N imagens:
      YOLO -> crop -> embedding CLIP -> (PCA se ativa) -> add FAISS + meta (persistência atômica).
    """
    global XB, meta, index
    added: List[dict] = []

    out_dir = Path("outputs/crops")
    out_dir.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with LOCK:
        for up in files:
            raw = await up.read()
            try:
                pil = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                continue
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

            try:
                res = DET_MODEL.predict(bgr, classes=PET_CLASSES, conf=det_conf, verbose=False)[0]
            except Exception:
                continue

            for j, box in enumerate(res.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop = bgr[y1:y2, x1:x2]
                if crop.size == 0 or min(crop.shape[:2]) < min_edge:
                    continue

                base = Path(up.filename or "upload").stem
                out_path = out_dir / f"{base}_{j}.jpg"
                k_ = 1
                while out_path.exists():
                    out_path = out_dir / f"{base}_{j}_{k_}.jpg"; k_ += 1
                cv2.imwrite(str(out_path), crop)

                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                vec512 = embed_pil(crop_pil).astype("float32").reshape(1, -1)
                vec_add = GALLERY_PCA.transform(vec512).astype("float32") if GALLERY_PCA is not None else vec512

                index.add(vec_add)
                XB = np.vstack([XB, vec_add]) if XB.size else vec_add

                hsv_hist = color_hist_hsv(crop)
                row = {
                    "src": src_note or f"upload:{up.filename}",
                    "crop": str(out_path.as_posix()),
                    "cls": int(box.cls.item()),
                    "conf": float(box.conf.item()),
                    "hsv_hist": [float(x) for x in hsv_hist.tolist()],
                }
                if pet_id is not None:
                    row["pet_id"] = pet_id
                meta.append(row); added.append(row)

        # salva com segurança (para PCA, persiste em pets_pca.faiss)
        if GALLERY_PCA is None:
            save_index_atomic(index)
        else:
            tmp = INDEX_DIR / "pets_pca.faiss.tmp"
            faiss.write_index(index, str(tmp))
            os.replace(tmp, INDEX_DIR / "pets_pca.faiss")

        save_meta_atomic(meta)

    return {"ok": True, "count": len(added), "added": added}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
