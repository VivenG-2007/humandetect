"""
Human Detection + Filter + Face Effects Webcam App
====================================================
Uses YOLOv8 for person segmentation + OpenCV Haar for face detection.
Clickable LEFT-SIDE PANEL — no keyboard needed!

Install:
    pip install ultralytics opencv-python numpy

Controls (keyboard still works too):
  ─── BODY FILTERS ───────────────────────── ─── FACE / CHARACTER EFFECTS ───
  1  Raw              6  Cyberpunk            E  Elf Ears
  2  Neon Outline     7  Pixel Art            V  Vampire Teeth
  3  Cartoon          8  Oil Painting         A  Angel Wings + Halo
  4  Anime            9  Heat Vision          D  Demon Horns
  5  Pencil Sketch    0  Glitch               M  Mermaid Look
  R  Skeleton         X  Stick Figure         F  Fairy Sparkles
  U  Bubble Replace                           Y  Baby Face
                                              C  Child Look
                                              T  Teen Look
                                              O  Old Age
                                              P  Age Progression

  B  Cycle background    K  Toggle boxes    S  Save    Q  Quit
"""

import cv2
import numpy as np
import time
import os
import math
import warnings
warnings.filterwarnings("ignore")
os.environ["YOLO_VERBOSE"] = "False"

from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR LAYOUT CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PANEL_W        = 230
BTN_H          = 32
SECTION_H      = 24
PAD            = 6

# Colour palette (BGR)
C_BG           = (14,  14,  22)
C_BG2          = (20,  20,  32)
C_SECTION_BG   = (26,  26,  44)
C_BTN          = (32,  32,  52)
C_BTN_HOVER    = (48,  48,  75)
C_BTN_ACTIVE   = (0,  210, 130)
C_BTN_ACTIVE2  = (0,  170, 100)
C_BORDER       = (50,  50,  80)
C_BORDER_ACT   = (0,  240, 150)
C_ACTIVE_TEXT  = (8,    8,   8)
C_TEXT         = (185, 185, 208)
C_SECTION_TEXT = (0,  220, 180)
C_ACCENT       = (0,  180, 255)
C_SAVE_BTN     = (30, 120, 200)
C_QUIT_BTN     = (160,  40,  40)
C_BOX_BTN_ON   = (20, 130,  20)
C_BOX_BTN_OFF  = (55,  55,  80)
C_NEW_BADGE    = (0,  140, 255)

# Filter category colours (dot indicators)
CAT_BODY   = (0,  200, 140)
CAT_NEW    = (0,  160, 255)
CAT_FACE   = (200, 80, 220)
CAT_BG     = (220, 140,  20)

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FS         = 0.37
FS_SEC     = 0.40
FS_BADGE   = 0.28

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA FINDER
# ─────────────────────────────────────────────────────────────────────────────

def find_camera(max_index=5):
    print("Scanning for cameras...")
    found = []
    for i in range(max_index + 1):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    found.append(i)
                    print(f"  [OK] Camera at index {i}")
            cap.release()
        except Exception:
            pass
    return found

def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap

# ─────────────────────────────────────────────────────────────────────────────
# YOLO SEGMENTOR
# ─────────────────────────────────────────────────────────────────────────────

class YOLOSegmentor:
    def __init__(self):
        print("Loading YOLOv8 segmentation model...")
        self.model  = YOLO("yolov8n-seg.pt")
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        print("Model loaded!")

    def get_mask_and_boxes(self, frame):
        h, w  = frame.shape[:2]
        mask  = np.zeros((h, w), dtype=np.uint8)
        boxes = []
        results = self.model(frame, classes=[0], verbose=False)
        r = results[0]
        if r.masks is not None:
            for seg in r.masks.data:
                m = seg.cpu().numpy()
                m = cv2.resize(m, (w, h))
                m = (m > 0.5).astype(np.uint8) * 255
                mask = cv2.bitwise_or(mask, m)
        if r.boxes is not None:
            for box in r.boxes.xyxy.cpu().numpy():
                boxes.append(box.astype(int))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        return mask, boxes

# ─────────────────────────────────────────────────────────────────────────────
# FACE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        eye_path     = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.eye_cascade  = cv2.CascadeClassifier(eye_path)

    def detect(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        return faces if len(faces) > 0 else []

# ─────────────────────────────────────────────────────────────────────────────
# DRAWING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def draw_glow_line(img, p1, p2, col, glow_col, thickness=2, glow_t=8):
    """Draw a line with a soft glow halo."""
    cv2.line(img, p1, p2, glow_col, glow_t, cv2.LINE_AA)
    cv2.line(img, p1, p2, col,      thickness, cv2.LINE_AA)

def draw_glow_circle(img, center, r, col, glow_col, thickness=-1, glow_t=10):
    cv2.circle(img, center, r + glow_t//2, glow_col, -1)
    cv2.circle(img, center, r, col, thickness, cv2.LINE_AA)

def blend_overlay(canvas, overlay_bgr, overlay_alpha, x, y):
    oh, ow = overlay_bgr.shape[:2]
    ch, cw = canvas.shape[:2]
    x1c = max(x, 0);        y1c = max(y, 0)
    x2c = min(x + ow, cw);  y2c = min(y + oh, ch)
    x1o = x1c - x;          y1o = y1c - y
    x2o = x1o + (x2c - x1c); y2o = y1o + (y2c - y1c)
    if x2c <= x1c or y2c <= y1c: return canvas
    roi = canvas[y1c:y2c, x1c:x2c].astype(np.float32)
    src = overlay_bgr[y1o:y2o, x1o:x2o].astype(np.float32)
    a   = overlay_alpha[y1o:y2o, x1o:x2o].astype(np.float32) / 255.0
    a3  = np.stack([a, a, a], axis=2)
    canvas[y1c:y2c, x1c:x2c] = np.clip(src*a3 + roi*(1-a3), 0, 255).astype(np.uint8)
    return canvas

# ─────────────────────────────────────────────────────────────────────────────
# ★  FACE / CHARACTER EFFECT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def effect_elf_ears(frame, faces, t):
    out = frame.copy()
    for (fx, fy, fw, fh) in faces:
        ear_w = fw // 5;  ear_h = fw // 3
        lx    = fx - ear_w // 2
        ly    = fy - ear_h + ear_h // 6
        pts_l = np.array([[lx, ly+ear_h],[lx+ear_w//2,ly],[lx+ear_w,ly+ear_h]], np.int32)
        cv2.fillPoly(out,[pts_l],(130,190,235)); cv2.polylines(out,[pts_l],True,(60,110,160),2)
        inner_l = pts_l.copy(); inner_l[:,0]+=ear_w//5; inner_l[:,1]+=ear_h//5
        cv2.polylines(out,[inner_l],False,(100,150,200),1)
        rx    = fx + fw - ear_w//2
        pts_r = np.array([[rx,ly+ear_h],[rx+ear_w//2,ly],[rx+ear_w,ly+ear_h]],np.int32)
        cv2.fillPoly(out,[pts_r],(130,190,235)); cv2.polylines(out,[pts_r],True,(60,110,160),2)
        inner_r = pts_r.copy(); inner_r[:,0]-=ear_w//5; inner_r[:,1]+=ear_h//5
        cv2.polylines(out,[inner_r],False,(100,150,200),1)
    return out

def effect_vampire_teeth(frame, faces, t):
    out = frame.copy()
    for (fx, fy, fw, fh) in faces:
        mx=fx+fw//2; my=fy+int(fh*0.75); tw=fw//10; th=fw//8; gap=fw//16
        for side in [-1,1]:
            tx=mx+side*gap-tw//2; ty=my
            pts=np.array([[tx,ty],[tx+tw,ty],[tx+tw//2+2,ty+th],[tx+tw//2-2,ty+th]],np.int32)
            cv2.fillPoly(out,[pts],(245,245,255)); cv2.polylines(out,[pts],True,(180,180,200),1)
            drip=int(4+4*abs(math.sin(t*1.5)))
            cv2.line(out,(tx+tw//2,ty+th),(tx+tw//2,ty+th+drip),(0,0,200),2)
            cv2.circle(out,(tx+tw//2,ty+th+drip),3,(0,0,180),-1)
    return out

def effect_angel(frame, faces, t):
    out = frame.copy()
    for (fx, fy, fw, fh) in faces:
        cx=fx+fw//2
        halo_y=fy-fw//6; halo_rx=fw//3; halo_ry=fw//10
        pulse=1.0+0.04*math.sin(t*3); halo_rx_p=int(halo_rx*pulse)
        for thickness,alpha in [(20,40),(12,80),(6,160),(3,255)]:
            ov=out.copy(); cv2.ellipse(ov,(cx,halo_y),(halo_rx_p,halo_ry),0,0,360,(30,200,255),thickness)
            out=cv2.addWeighted(out,1-alpha/255,ov,alpha/255,0)
        wing_w=fw*2; wing_h=int(fh*1.6); wing_y=fy+fh//3
        for side in [-1,1]:
            pts_outer=[]
            for ang in range(0,181,8):
                rad=math.radians(ang)
                ex=cx+side*int(wing_w*math.sin(rad)); ey=wing_y-int(wing_h*0.5*math.sin(rad*0.5))
                pts_outer.append([ex,ey])
            pts_outer=np.array(pts_outer,np.int32)
            ov=out.copy(); cv2.fillPoly(ov,[pts_outer],(240,240,255))
            out=cv2.addWeighted(out,0.45,ov,0.55,0)
            for i in range(10):
                r=i/10; fx2=int(cx+side*wing_w*r)
                fy2=int(wing_y-wing_h*0.4*math.sin(math.pi*r)); fy3=int(wing_y+fh*0.1)
                cv2.line(out,(fx2,fy3),(fx2,fy2),(200,200,230),1)
    return out

def effect_demon_horns(frame, faces, t):
    out = frame.copy()
    for (fx, fy, fw, fh) in faces:
        cx=fx+fw//2; horn_h=int(fw*0.5); horn_w=fw//7
        for side in [-1,1]:
            hx=cx+side*fw//4; hy=fy; sway=int(4*math.sin(t*2))
            pts=np.array([[hx-horn_w//2,hy],[hx+side*sway-2,hy-horn_h],[hx+horn_w//2,hy]],np.int32)
            shadow=pts.copy(); shadow[:,0]+=3; shadow[:,1]+=3
            cv2.fillPoly(out,[shadow],(0,0,30)); cv2.fillPoly(out,[pts],(20,20,160))
            mid=np.array([[hx+side*2,hy-5],[hx+side*sway+side*2,hy-horn_h+8],[hx+side*2+horn_w//4,hy-5]],np.int32)
            cv2.fillPoly(out,[mid],(60,60,200)); cv2.polylines(out,[pts],True,(0,0,80),2)
        ley=fy+int(fh*0.38); lex=fx+int(fw*0.28); rex=fx+int(fw*0.72); glow_r=int(fw*0.07)
        for eye_cx,eye_cy in [(lex,ley),(rex,ley)]:
            ov=out.copy(); cv2.circle(ov,(eye_cx,eye_cy),glow_r+6,(0,0,255),-1)
            out=cv2.addWeighted(out,0.6,ov,0.4,0)
            cv2.circle(out,(eye_cx,eye_cy),glow_r,(0,80,255),-1)
            cv2.circle(out,(eye_cx,eye_cy),glow_r//2,(100,200,255),-1)
    return out

def effect_mermaid(frame, faces, t):
    out=frame.copy(); h_f,w_f=frame.shape[:2]
    tint=np.zeros_like(out,dtype=np.float32); tint[:,:,0]=60; tint[:,:,1]=40
    out=np.clip(out.astype(np.float32)*0.80+tint,0,255).astype(np.uint8)
    for i in range(8):
        lx=int(w_f*(0.1+0.1*i+0.04*math.sin(t*1.2+i))); ly=int(h_f*(0.2+0.06*math.sin(t*0.8+i*1.3)))
        cv2.ellipse(out,(lx,ly),(int(30+10*math.sin(t+i)),8),30+i*10,0,360,(200,230,220),-1)
    for (fx,fy,fw,fh) in faces:
        cx=fx+fw//2; scale_r=fw//14
        for row in range(8):
            for col in range(12):
                sx=fx+col*scale_r*2-scale_r; sy=fy+row*scale_r+(scale_r if col%2 else 0)
                if fx<=sx<=fx+fw and fy<=sy<=fy+fh:
                    hue=int((row*30+col*15+t*40)%180)
                    col_hsv=np.uint8([[[hue,200,200]]]); col_bgr=cv2.cvtColor(col_hsv,cv2.COLOR_HSV2BGR)[0][0]
                    ov3=out.copy()
                    cv2.ellipse(ov3,(sx,sy),(scale_r,scale_r-1),0,0,360,(int(col_bgr[0]),int(col_bgr[1]),int(col_bgr[2])),-1)
                    out=cv2.addWeighted(out,0.75,ov3,0.25,0)
        fin_pts=np.array([[cx-fw//3,fy],[cx-fw//5,fy-fw//5],[cx,fy-fw//4],[cx+fw//5,fy-fw//5],[cx+fw//3,fy]],np.int32)
        cv2.fillPoly(out,[fin_pts],(0,180,130)); cv2.polylines(out,[fin_pts],False,(0,220,180),2)
    return out

def effect_fairy(frame, faces, t):
    out=frame.copy(); h_f,w_f=frame.shape[:2]; rng=np.random.default_rng(7)
    positions=[]
    for i in range(40):
        positions.append((int(rng.integers(0,w_f)),int(rng.integers(0,h_f)),rng.uniform(0.5,3.0),rng.uniform(0,2*math.pi)))
    for sx,sy,sp,ph in positions:
        bri=0.5+0.5*math.sin(t*sp+ph); sz=int(1+bri*5); alpha=int(bri*255)
        col=(int(180+bri*75),int(180+bri*75),255); ov=out.copy()
        cv2.line(ov,(sx-sz,sy),(sx+sz,sy),col,1); cv2.line(ov,(sx,sy-sz),(sx,sy+sz),col,1)
        cv2.circle(ov,(sx,sy),max(1,sz//2),(255,255,255),-1)
        out=cv2.addWeighted(out,1-alpha/400,ov,alpha/400,0)
    for (fx,fy,fw,fh) in faces:
        for i in range(20):
            ang=t*90+i*18; rad=math.radians(ang); r=fw*0.55
            px=int(fx+fw//2+r*math.cos(rad)); py=int(fy+fh//2+r*math.sin(rad)*0.6)
            bri2=0.5+0.5*math.sin(t*3+i); sz2=int(2+bri2*4)
            cv2.circle(out,(px,py),sz2,(int(200*bri2),int(160*bri2),255),-1)
    return out

def effect_baby_face(frame, faces, t):
    out=frame.copy()
    for (fx,fy,fw,fh) in faces:
        cx=fx+fw//2; cy=fy+fh//2; roi=out[fy:fy+fh,fx:fx+fw]
        if roi.size==0: continue
        big_w=int(fw*1.25); big_h=int(fh*1.15); big=cv2.resize(roi,(big_w,big_h))
        bx=cx-big_w//2; by=cy-big_h//2; bx2,by2=bx+big_w,by+big_h
        bx=max(0,bx); by=max(0,by); bx2=min(out.shape[1],bx2); by2=min(out.shape[0],by2)
        cw=bx2-bx; ch=by2-by
        if cw<=0 or ch<=0: continue
        resized_crop=cv2.resize(big,(cw,ch))
        paste_mask=np.zeros((ch,cw),dtype=np.uint8)
        cv2.ellipse(paste_mask,(cw//2,ch//2),(cw//2-2,ch//2-2),0,0,360,255,-1)
        paste_mask=cv2.GaussianBlur(paste_mask,(21,21),0); a3=np.stack([paste_mask/255.]*3,axis=2)
        out[by:by2,bx:bx2]=np.clip(resized_crop*a3+out[by:by2,bx:bx2]*(1-a3),0,255).astype(np.uint8)
        cheek_r=fw//6
        for side in [-1,1]:
            chx=cx+side*int(fw*0.32); chy=fy+int(fh*0.62); ov=out.copy()
            cv2.circle(ov,(chx,chy),cheek_r,(100,130,220),-1); out=cv2.addWeighted(out,0.75,ov,0.25,0)
    return out

def effect_child(frame, faces, t):
    out=frame.copy()
    for (fx,fy,fw,fh) in faces:
        cx=fx+fw//2; roi=out[fy:fy+fh,fx:fx+fw]
        if roi.size==0: continue
        smooth=cv2.bilateralFilter(roi,9,60,60)
        hsv=cv2.cvtColor(smooth,cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1]=np.clip(hsv[:,:,1]*0.80,0,255); hsv[:,:,2]=np.clip(hsv[:,:,2]*1.15,0,255)
        out[fy:fy+fh,fx:fx+fw]=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
        rng=np.random.default_rng(3)
        for _ in range(18):
            frx=int(fx+rng.integers(fw//4,3*fw//4)); fry=int(fy+rng.integers(fh//3,2*fh//3))
            cv2.circle(out,(frx,fry),2,(80,100,160),-1)
    return out

def effect_teen(frame, faces, t):
    out=frame.copy()
    for (fx,fy,fw,fh) in faces:
        roi=out[fy:fy+fh,fx:fx+fw]
        if roi.size==0: continue
        kernel=np.array([[0,-0.3,0],[-0.3,2.2,-0.3],[0,-0.3,0]]); sharpened=cv2.filter2D(roi,-1,kernel)
        sharpened=sharpened.astype(np.float32)
        sharpened[:,:,0]=np.clip(sharpened[:,:,0]*1.1,0,255)
        sharpened[:,:,1]=np.clip(sharpened[:,:,1]*1.05,0,255)
        sharpened[:,:,2]=np.clip(sharpened[:,:,2]*0.93,0,255)
        out[fy:fy+fh,fx:fx+fw]=sharpened.astype(np.uint8)
    return out

def effect_old_age(frame, faces, t):
    out=frame.copy()
    for (fx,fy,fw,fh) in faces:
        roi=out[fy:fy+fh,fx:fx+fw].copy()
        if roi.size==0: continue
        cx=fw//2
        hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1]=np.clip(hsv[:,:,1]*0.40,0,255); hsv[:,:,2]=np.clip(hsv[:,:,2]*0.90,0,255)
        roi=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
        noise=np.random.normal(0,6,roi.shape).astype(np.int16)
        roi=np.clip(roi.astype(np.int16)+noise,0,255).astype(np.uint8)
        for i in range(4):
            wy=int(fh*(0.15+i*0.05)); wave=[int(4*math.sin(x*0.3+i)) for x in range(fw)]
            pts=[(x,wy+wave[x]) for x in range(fw)]
            for j in range(len(pts)-1): cv2.line(roi,pts[j],pts[j+1],(100,100,110),1)
        out[fy:fy+fh,fx:fx+fw]=roi
    return out

AGE_PROGRESSION_STAGE=[0]

def effect_age_progression(frame, faces, t):
    stage=AGE_PROGRESSION_STAGE[0]
    funcs=[effect_baby_face,effect_child,effect_teen,lambda f,fa,t_: f.copy(),effect_old_age]
    names=["Baby","Child","Teen","Adult","Elderly"]
    out=funcs[stage](frame,faces,t)
    h_f,w_f=frame.shape[:2]
    label=f"Age Stage: {names[stage]}"
    cv2.rectangle(out,(0,h_f-55),(len(label)*11+20,h_f-32),(0,0,0),-1)
    cv2.putText(out,label,(10,h_f-38),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,230,180),1)
    return out

FACE_EFFECTS={
    ord('e'):("Elf Ears",       effect_elf_ears),
    ord('v'):("Vampire Teeth",  effect_vampire_teeth),
    ord('a'):("Angel Wings",    effect_angel),
    ord('d'):("Demon Horns",    effect_demon_horns),
    ord('m'):("Mermaid",        effect_mermaid),
    ord('f'):("Fairy Sparkles", effect_fairy),
    ord('y'):("Baby Face",      effect_baby_face),
    ord('c'):("Child Look",     effect_child),
    ord('t'):("Teen Look",      effect_teen),
    ord('o'):("Old Age",        effect_old_age),
    ord('p'):("Age Progress",   effect_age_progression),
}

# ─────────────────────────────────────────────────────────────────────────────
# BODY FILTERS (original)
# ─────────────────────────────────────────────────────────────────────────────

def m3f(mask):
    return cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR).astype(np.float32)/255.0

def f_raw(frame,mask): return frame.copy()

def f_neon_outline(frame,mask):
    dark=(frame*0.12).astype(np.uint8)
    hard=(mask>127).astype(np.uint8)*255
    outline=cv2.Canny(hard,30,100)
    outline=cv2.dilate(outline,np.ones((3,3),np.uint8),iterations=3)
    gw=cv2.GaussianBlur(outline,(25,25),0); gn=cv2.GaussianBlur(outline,(7,7),0)
    neon=np.zeros_like(frame,dtype=np.float32)
    neon[:,:,0]+=gw.astype(np.float32)*1.2; neon[:,:,1]+=gw.astype(np.float32)*1.4; neon[:,:,2]+=gw.astype(np.float32)*0.3
    neon[:,:,0]+=gn.astype(np.float32)*1.5; neon[:,:,1]+=gn.astype(np.float32)*1.5; neon[:,:,2]+=gn.astype(np.float32)*1.5
    neon=np.clip(neon,0,255).astype(np.uint8); alpha=m3f(mask)
    body=(frame*alpha*0.35+dark*(1-alpha)).astype(np.uint8)
    return cv2.add(body,neon)

def f_cartoon(frame,mask):
    data=frame.reshape((-1,3)).astype(np.float32); crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,8,1.0)
    _,labels,centers=cv2.kmeans(data,8,None,crit,3,cv2.KMEANS_RANDOM_CENTERS)
    quant=np.uint8(centers)[labels.flatten()].reshape(frame.shape)
    gray=cv2.medianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),7)
    edges=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,6)
    cartoon=cv2.bitwise_and(quant,cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR))
    hsv=cv2.cvtColor(cartoon,cv2.COLOR_BGR2HSV).astype(np.float32); hsv[:,:,1]=np.clip(hsv[:,:,1]*1.8,0,255)
    return cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)

def f_anime(frame,mask):
    s=frame.copy()
    for _ in range(5): s=cv2.bilateralFilter(s,9,75,75)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges=cv2.adaptiveThreshold(cv2.medianBlur(gray,5),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,3)
    anime=cv2.bitwise_and(s,cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR))
    hsv=cv2.cvtColor(anime,cv2.COLOR_BGR2HSV).astype(np.float32); hsv[:,:,1]=np.clip(hsv[:,:,1]*2.0,0,255)
    result=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
    bright=cv2.cvtColor(result,cv2.COLOR_BGR2GRAY); _,hi=cv2.threshold(bright,200,255,cv2.THRESH_BINARY)
    glow=cv2.GaussianBlur(cv2.cvtColor(hi,cv2.COLOR_GRAY2BGR),(15,15),0)
    return cv2.addWeighted(result,1.0,glow,0.3,0)

def f_pencil_sketch(frame,mask):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); blur=cv2.GaussianBlur(cv2.bitwise_not(gray),(25,25),0)
    sketch=cv2.divide(gray,cv2.bitwise_not(blur),scale=256.0)
    noise=np.random.normal(0,4,sketch.shape).astype(np.int16)
    sketch=np.clip(sketch.astype(np.int16)+noise,0,255).astype(np.uint8)
    return cv2.cvtColor(sketch,cv2.COLOR_GRAY2BGR)

def f_cyberpunk(frame,mask):
    dark=(frame*0.08).astype(np.uint8); gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges=cv2.dilate(cv2.Canny(gray,30,100),np.ones((2,2),np.uint8)); eb=cv2.GaussianBlur(edges,(5,5),0)
    neon=np.zeros_like(frame,dtype=np.float32)
    neon[:,:,0]=np.roll(eb,5,axis=1).astype(np.float32)*2.5
    neon[:,:,1]=np.roll(eb,-5,axis=0).astype(np.float32)*1.5
    neon[:,:,2]=eb.astype(np.float32)*2.0
    neon=np.clip(neon,0,255).astype(np.uint8)
    tinted=frame.copy().astype(np.float32)
    tinted[:,:,0]=np.clip(tinted[:,:,0]*1.4,0,255); tinted[:,:,2]=np.clip(tinted[:,:,2]*0.6,0,255)
    tinted=tinted.astype(np.uint8); alpha=m3f(mask)
    body=(tinted*alpha*0.6+dark*(1-alpha)).astype(np.uint8)
    return cv2.add(body,neon)

def f_pixel_art(frame,mask):
    h,w=frame.shape[:2]; px=10
    sm=cv2.resize(frame,(w//px,h//px),interpolation=cv2.INTER_LINEAR)
    pix=cv2.resize(sm,(w,h),interpolation=cv2.INTER_NEAREST)
    return (pix//(256//6))*(256//6)

def f_oil_paint(frame,mask):
    oil=frame.copy()
    for _ in range(7): oil=cv2.bilateralFilter(oil,9,150,150)
    gray=cv2.cvtColor(oil,cv2.COLOR_BGR2GRAY)
    emboss=cv2.cvtColor(cv2.filter2D(gray,-1,np.array([[-2,-1,0],[-1,1,1],[0,1,2]],dtype=np.float32)),cv2.COLOR_GRAY2BGR)
    hsv=cv2.cvtColor(oil,cv2.COLOR_BGR2HSV).astype(np.float32); hsv[:,:,1]=np.clip(hsv[:,:,1]*1.3,0,255)
    oil=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
    return cv2.addWeighted(oil,0.88,emboss,0.12,0)

def f_heat_vision(frame,mask):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); alpha=mask.astype(np.float32)/255.0
    boosted=np.clip(gray.astype(np.float32)+alpha*60,0,255).astype(np.uint8)
    return cv2.GaussianBlur(cv2.applyColorMap(boosted,cv2.COLORMAP_INFERNO),(3,3),0)

def f_glitch(frame,mask):
    h,w=frame.shape[:2]; out=frame.copy(); shift=8
    b,g,r=cv2.split(out); out=cv2.merge([np.roll(b,-shift,axis=1),g,np.roll(r,shift,axis=1)])
    rng=np.random.default_rng(int(time.time()*10)%10000)
    for _ in range(6):
        y1=int(rng.integers(0,h)); y2=min(y1+int(rng.integers(2,12)),h)
        out[y1:y2]=np.roll(out[y1:y2],int(rng.integers(-30,30)),axis=1)
    out[::4,:,1]=np.clip(out[::4,:,1].astype(np.int16)+40,0,255).astype(np.uint8)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# ★  NEW FILTERS
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_skeleton_joints(x, y, w, h):
    """Estimate 16 body joint positions from a bounding box."""
    cx = x + w // 2
    # Head & torso
    head_c  = (cx,               y + int(h * 0.07))
    neck    = (cx,               y + int(h * 0.17))
    l_sho   = (x + int(w*0.23), y + int(h * 0.24))
    r_sho   = (x + int(w*0.77), y + int(h * 0.24))
    spine_m = (cx,               y + int(h * 0.38))
    pelvis  = (cx,               y + int(h * 0.54))
    # Arms
    l_elb   = (x + int(w*0.10), y + int(h * 0.42))
    r_elb   = (x + int(w*0.90), y + int(h * 0.42))
    l_wri   = (x + int(w*0.06), y + int(h * 0.60))
    r_wri   = (x + int(w*0.94), y + int(h * 0.60))
    # Legs
    l_hip   = (x + int(w*0.37), y + int(h * 0.54))
    r_hip   = (x + int(w*0.63), y + int(h * 0.54))
    l_kne   = (x + int(w*0.34), y + int(h * 0.73))
    r_kne   = (x + int(w*0.66), y + int(h * 0.73))
    l_ank   = (x + int(w*0.32), y + int(h * 0.93))
    r_ank   = (x + int(w*0.68), y + int(h * 0.93))

    joints = [head_c, neck, l_sho, r_sho, spine_m, pelvis,
              l_elb, r_elb, l_wri, r_wri,
              l_hip, r_hip, l_kne, r_kne, l_ank, r_ank]

    bones = [
        (neck,  head_c),
        (neck,  spine_m),
        (spine_m, pelvis),
        (neck,  l_sho), (l_sho, l_elb), (l_elb, l_wri),
        (neck,  r_sho), (r_sho, r_elb), (r_elb, r_wri),
        (pelvis, l_hip), (l_hip, l_kne), (l_kne, l_ank),
        (pelvis, r_hip), (r_hip, r_kne), (r_kne, r_ank),
    ]
    head_r = max(10, int(w * 0.12))
    return joints, bones, head_c, head_r


def f_skeleton(frame, mask):
    """
    X-ray / neon skeleton: hides the body, draws an estimated
    glowing green skeleton on a dark background.
    """
    h, w = frame.shape[:2]
    t    = time.time()

    # Dark background that still lets the un-masked scene bleed through faintly
    bg_dark = (frame * 0.08).astype(np.uint8)
    out     = bg_dark.copy()

    hard = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pulse timing for joint glow
    pulse = 0.85 + 0.15 * math.sin(t * 4)

    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw < 30 or bh < 50:
            continue

        joints, bones, head_c, head_r = _estimate_skeleton_joints(bx, by, bw, bh)

        # ── Bones ─────────────────────────────────────────────────
        for p1, p2 in bones:
            # outer glow (wide, dark green)
            cv2.line(out, p1, p2, (0, 60, 20), 10, cv2.LINE_AA)
            # mid glow
            cv2.line(out, p1, p2, (0, 160, 60), 4, cv2.LINE_AA)
            # bright core
            cv2.line(out, p1, p2, (0, 255, 100), 2, cv2.LINE_AA)

        # ── Ribcage hint ──────────────────────────────────────────
        r_cx, r_cy = joints[0][0], joints[4][1]     # spine mid x, spine mid y
        rib_w = int(bw * 0.30); rib_h = int(bh * 0.12)
        for i in range(3):
            ry = r_cy - int(rib_h * 0.5 * i)
            cv2.ellipse(out, (r_cx, ry), (rib_w, max(4, rib_h-i*2)),
                        0, 0, 360, (0, 100, 40), 1, cv2.LINE_AA)

        # ── Skull ─────────────────────────────────────────────────
        cv2.circle(out, head_c, head_r + 6, (0, 40, 15), -1)     # glow fill
        cv2.circle(out, head_c, head_r,     (0, 60, 25), -1)     # skull fill
        cv2.circle(out, head_c, head_r,     (0, 220, 80), 2, cv2.LINE_AA)

        # Eye sockets
        eye_off = int(head_r * 0.35)
        eye_r   = int(head_r * 0.22)
        for ex_off in [-eye_off, eye_off]:
            ex = head_c[0] + ex_off
            ey = head_c[1] + int(head_r * 0.1)
            glow_a = int(pulse * 255)
            ov = out.copy()
            cv2.circle(ov, (ex, ey), eye_r + 4, (0, 200, 60), -1)
            out = cv2.addWeighted(out, 1 - 0.4 * pulse, ov, 0.4 * pulse, 0)
            cv2.circle(out, (ex, ey), eye_r, (0, 255, 120), -1)

        # Teeth hint
        teeth_y = head_c[1] + int(head_r * 0.55)
        t_w     = int(head_r * 0.55)
        cv2.line(out, (head_c[0]-t_w, teeth_y), (head_c[0]+t_w, teeth_y), (0, 200, 70), 1)
        for ti in range(-2, 3):
            tx = head_c[0] + ti * (t_w // 2)
            cv2.line(out, (tx, teeth_y), (tx, teeth_y + int(head_r*0.25)), (0, 180, 60), 1)

        # ── Joints (spheres) ──────────────────────────────────────
        for j in joints[1:]:          # skip head_c, drawn separately
            jr = max(4, int(bw * 0.025))
            cv2.circle(out, j, jr + 4, (0, 50, 20), -1)
            cv2.circle(out, j, jr,     (0, 255, 100), -1)
            cv2.circle(out, j, jr,     (100, 255, 180), 1, cv2.LINE_AA)

    # Faint green scanline overlay for X-ray feel
    scanline = np.zeros_like(out, dtype=np.uint8)
    scanline[::3] = [0, 12, 4]
    out = np.clip(out.astype(np.int16) + scanline.astype(np.int16), 0, 255).astype(np.uint8)

    return out


def f_stick_figure(frame, mask):
    """
    Minimalist animated stick figure: erases the person and draws a
    colourful neon stick figure with a round head.
    """
    h, w = frame.shape[:2]
    t    = time.time()

    # Dark-tinted background (background scene, person erased)
    bg = (frame * 0.10).astype(np.uint8)
    # Keep background behind non-person pixels
    bg_alpha = (1.0 - mask.astype(np.float32)/255.0)
    scene    = (frame.astype(np.float32) * np.stack([bg_alpha]*3, axis=2)
                + bg.astype(np.float32) * np.stack([1-bg_alpha]*3, axis=2))
    out = np.clip(scene, 0, 255).astype(np.uint8)

    hard = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Colour cycles over time
    hue   = int((t * 40) % 180)
    c_hsv = np.uint8([[[hue, 255, 255]]])
    stick_col = tuple(int(v) for v in cv2.cvtColor(c_hsv, cv2.COLOR_HSV2BGR)[0][0])
    glow_col  = tuple(max(0, v//3) for v in stick_col)

    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw < 30 or bh < 50:
            continue

        joints, bones, head_c, head_r = _estimate_skeleton_joints(bx, by, bw, bh)

        # Slight wiggle animation
        wiggle_scale = 0.015
        def w_pt(p):
            wx = int(p[0] + bw * wiggle_scale * math.sin(t * 3 + p[1] * 0.05))
            wy = int(p[1] + bh * wiggle_scale * math.cos(t * 2.5 + p[0] * 0.05))
            return (wx, wy)

        # Draw bones
        for p1, p2 in bones:
            wp1, wp2 = w_pt(p1), w_pt(p2)
            cv2.line(out, wp1, wp2, glow_col, 10, cv2.LINE_AA)
            cv2.line(out, wp1, wp2, stick_col, 4, cv2.LINE_AA)
            cv2.line(out, wp1, wp2, (255,255,255), 1, cv2.LINE_AA)

        # Joints
        for j in joints[1:]:
            wj = w_pt(j)
            cv2.circle(out, wj, 7, glow_col, -1)
            cv2.circle(out, wj, 4, stick_col, -1)
            cv2.circle(out, wj, 4, (255,255,255), 1, cv2.LINE_AA)

        # Head circle
        wh = w_pt(head_c)
        cv2.circle(out, wh, head_r + 6, glow_col, -1)
        cv2.circle(out, wh, head_r,     stick_col, 3, cv2.LINE_AA)
        cv2.circle(out, wh, head_r,     (255,255,255), 1, cv2.LINE_AA)

        # Smiley dots (eyes + smile)
        eye_off = int(head_r * 0.35)
        eye_r   = max(2, int(head_r * 0.12))
        for ex_off in [-eye_off, eye_off]:
            cv2.circle(out, (wh[0]+ex_off, wh[1]-int(head_r*0.15)), eye_r,
                       (255,255,255), -1)
        # Smile arc
        cv2.ellipse(out, (wh[0], wh[1]+int(head_r*0.1)),
                    (int(head_r*0.35), int(head_r*0.22)), 0, 0, 180,
                    (255,255,255), 1, cv2.LINE_AA)

    return out


# ── Bubble state (persists between frames for smooth physics) ──────────────
_BUBBLE_STATE: list = []

def _init_bubbles(n=60):
    rng = np.random.default_rng(int(time.time()*100) % 100000)
    bubbles = []
    for _ in range(n):
        bubbles.append({
            'rx': rng.uniform(0.05, 0.95),    # relative x in mask bbox
            'ry': rng.uniform(0.05, 0.95),    # relative y
            'r':  int(rng.integers(10, 40)),  # radius px
            'vx': rng.uniform(-0.002, 0.002),
            'vy': rng.uniform(-0.005, -0.001),
            'hue': int(rng.integers(0, 180)),
            'phase': rng.uniform(0, math.pi*2),
            'born': time.time(),
        })
    return bubbles

def f_bubbles(frame, mask):
    """
    Replaces the person's silhouette with a swarm of colourful
    animated bubbles; background is kept visible.
    """
    global _BUBBLE_STATE
    h, w = frame.shape[:2]
    t    = time.time()

    if len(_BUBBLE_STATE) == 0:
        _BUBBLE_STATE = _init_bubbles(70)

    hard  = (mask > 127).astype(np.uint8) * 255

    # Background: keep non-person pixels, darken + blue-tint person area
    bg_alpha = 1.0 - mask.astype(np.float32) / 255.0
    tint     = np.zeros_like(frame, dtype=np.float32)
    tint[:,:,0] = 30; tint[:,:,1] = 10
    bg = np.clip(
        frame.astype(np.float32) * np.stack([bg_alpha]*3, axis=2)
        + tint * np.stack([1-bg_alpha]*3, axis=2),
        0, 255
    ).astype(np.uint8)

    bubble_layer = np.zeros((h, w, 4), dtype=np.uint8)  # BGRA for alpha compositing

    # Find overall person bounding box for coordinate mapping
    cnts, _ = cv2.findContours(hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_pts = []
    for c in cnts:
        all_pts.extend(c.reshape(-1,2).tolist())

    if not all_pts:
        return bg  # nobody detected

    all_pts = np.array(all_pts)
    gx1 = int(all_pts[:,0].min()); gx2 = int(all_pts[:,0].max())
    gy1 = int(all_pts[:,1].min()); gy2 = int(all_pts[:,1].max())
    gw  = max(1, gx2 - gx1);  gh = max(1, gy2 - gy1)

    dead = []
    for i, b in enumerate(_BUBBLE_STATE):
        # Update position
        b['ry'] += b['vy']
        b['rx'] += b['vx'] + 0.001 * math.sin(t * 1.5 + b['phase'])

        # Wrap / recycle bubble that floats out
        if b['ry'] < -b['r']/gh or b['rx'] < -0.1 or b['rx'] > 1.1:
            dead.append(i)
            continue

        # World coords
        bx_c = int(gx1 + b['rx'] * gw)
        by_c = int(gy1 + b['ry'] * gh)
        br   = b['r']

        # Only draw bubble if its centre is roughly inside the mask
        if 0 <= bx_c < w and 0 <= by_c < h and hard[by_c, bx_c] > 64:
            # Colour: slowly shift hue
            hue = int((b['hue'] + t * 20) % 180)
            sat = 200; val = 230
            c_hsv = np.uint8([[[hue, sat, val]]])
            bgr   = tuple(int(v) for v in cv2.cvtColor(c_hsv, cv2.COLOR_HSV2BGR)[0][0])

            # Wobble radius
            wobble = 1.0 + 0.06 * math.sin(t * 3 + b['phase'])
            wr = max(4, int(br * wobble))

            # Draw on bubble_layer (BGRA)
            pulse_a = int(160 + 40 * math.sin(t * 2 + b['phase']))

            # Outer glow
            cv2.circle(bubble_layer, (bx_c, by_c), wr + 4,
                       (bgr[0]//2, bgr[1]//2, bgr[2]//2, pulse_a//4), -1)
            # Fill (semi-transparent)
            cv2.circle(bubble_layer, (bx_c, by_c), wr,
                       (bgr[0], bgr[1], bgr[2], pulse_a), -1)
            # Bright rim
            cv2.circle(bubble_layer, (bx_c, by_c), wr,
                       (255, 255, 255, 200), 1, cv2.LINE_AA)
            # Specular highlight (top-left)
            hx = bx_c - wr // 3;  hy = by_c - wr // 3
            hl_r = max(2, wr // 4)
            cv2.circle(bubble_layer, (hx, hy), hl_r, (255, 255, 255, 220), -1)
            # Inner shine arc
            cv2.ellipse(bubble_layer, (bx_c - wr//4, by_c - wr//4),
                        (wr//3, wr//4), -30, 200, 320,
                        (255, 255, 255, 100), 1, cv2.LINE_AA)

    # Recycle dead bubbles
    rng2 = np.random.default_rng(int(t * 1000) % 100000)
    for i in sorted(dead, reverse=True):
        _BUBBLE_STATE[i] = {
            'rx': rng2.uniform(0.05, 0.95),
            'ry': rng2.uniform(0.6, 1.1),
            'r':  int(rng2.integers(10, 40)),
            'vx': rng2.uniform(-0.002, 0.002),
            'vy': rng2.uniform(-0.005, -0.001),
            'hue': int(rng2.integers(0, 180)),
            'phase': rng2.uniform(0, math.pi*2),
            'born': t,
        }

    # Composite bubble_layer onto bg
    b_bgr  = bubble_layer[:,:,:3].astype(np.float32)
    b_a    = bubble_layer[:,:,3].astype(np.float32) / 255.0
    b_a3   = np.stack([b_a, b_a, b_a], axis=2)

    # Only paste bubbles inside person mask area
    mask_f = (mask.astype(np.float32) / 255.0)
    mask3  = np.stack([mask_f, mask_f, mask_f], axis=2)
    combined_a = b_a3 * mask3

    out = np.clip(
        bg.astype(np.float32) * (1 - combined_a) + b_bgr * combined_a,
        0, 255
    ).astype(np.uint8)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# FILTER REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

BODY_FILTERS = {
    ord('1'): ("Raw",           f_raw),
    ord('2'): ("Neon Outline",  f_neon_outline),
    ord('3'): ("Cartoon",       f_cartoon),
    ord('4'): ("Anime",         f_anime),
    ord('5'): ("Pencil Sketch", f_pencil_sketch),
    ord('6'): ("Cyberpunk",     f_cyberpunk),
    ord('7'): ("Pixel Art",     f_pixel_art),
    ord('8'): ("Oil Painting",  f_oil_paint),
    ord('9'): ("Heat Vision",   f_heat_vision),
    ord('0'): ("Glitch",        f_glitch),
    # ── NEW ──
    ord('r'): ("Skeleton",      f_skeleton),
    ord('x'): ("Stick Figure",  f_stick_figure),
    ord('u'): ("Bubbles",       f_bubbles),
}

# Ordered lists for sidebar (body filters split into two columns visually)
BODY_FILTER_LIST = [
    (ord('1'), "1", "Raw",         False),
    (ord('2'), "2", "Neon Outline",False),
    (ord('3'), "3", "Cartoon",     False),
    (ord('4'), "4", "Anime",       False),
    (ord('5'), "5", "Pencil Sketch",False),
    (ord('6'), "6", "Cyberpunk",   False),
    (ord('7'), "7", "Pixel Art",   False),
    (ord('8'), "8", "Oil Painting",False),
    (ord('9'), "9", "Heat Vision", False),
    (ord('0'), "0", "Glitch",      False),
    (ord('r'), "R", "Skeleton",    True),   # True = NEW badge
    (ord('x'), "X", "Stick Figure",True),
    (ord('u'), "U", "Bubbles",     True),
]

FACE_EFFECT_LIST = [
    (ord('e'), "E", "Elf Ears"),
    (ord('v'), "V", "Vampire"),
    (ord('a'), "A", "Angel Wings"),
    (ord('d'), "D", "Demon Horns"),
    (ord('m'), "M", "Mermaid"),
    (ord('f'), "F", "Fairy"),
    (ord('y'), "Y", "Baby Face"),
    (ord('c'), "C", "Child"),
    (ord('t'), "T", "Teen"),
    (ord('o'), "O", "Old Age"),
    (ord('p'), "P", "Age Prog"),
]

BG_LIST = [
    ("Space",  (80,  0, 100)),
    ("Forest", (20,130,  20)),
    ("Ocean",  (160, 80,   0)),
    ("Sunset", (20, 100, 200)),
    ("Matrix", (0,  160,   0)),
    ("None",   (60,  60,  60)),
]

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUNDS
# ─────────────────────────────────────────────────────────────────────────────

def make_space_bg(h,w):
    bg=np.zeros((h,w,3),dtype=np.uint8)
    for y in range(h):
        r=y/h; bg[y]=[int(30*r),0,int(60+60*r)]
    rng=np.random.default_rng(42)
    xs,ys=rng.integers(0,w,400),rng.integers(0,h,400); br=rng.integers(150,255,400)
    for x,y,b in zip(xs,ys,br): cv2.circle(bg,(int(x),int(y)),1,(int(b),int(b),int(b)),-1)
    for cx,cy,radius,col in [(w//3,h//3,90,(80,0,130)),(2*w//3,2*h//3,70,(0,50,140))]:
        nm=np.zeros((h,w),dtype=np.float32); cv2.circle(nm,(cx,cy),radius,1.0,-1)
        nm=cv2.GaussianBlur(nm,(101,101),0)
        for c,v in enumerate(col): bg[:,:,c]=np.clip(bg[:,:,c].astype(np.float32)+nm*v,0,255).astype(np.uint8)
    return bg

def make_forest_bg(h,w):
    bg=np.zeros((h,w,3),dtype=np.uint8); bg[:h//2]=[180,220,120]; bg[h//2:]=[25,90,15]
    rng=np.random.default_rng(5)
    for tx in range(20,w,80):
        th=int(rng.integers(70,130)); ty=h//2
        cv2.rectangle(bg,(tx+28,ty-th),(tx+42,ty+5),(35,70,15),-1)
        cv2.circle(bg,(tx+35,ty-th),55,(15,130,25),-1)
    return bg

def make_ocean_bg(h,w):
    bg=np.zeros((h,w,3),dtype=np.uint8)
    for y in range(h):
        r=y/h; bg[y]=[int(160*(1-r)),int(90+60*r),int(190+65*r)]
    return bg

def make_sunset_bg(h,w):
    bg=np.zeros((h,w,3),dtype=np.uint8)
    for y in range(h):
        r=y/h; bg[y]=[int(20+40*r),int(60*(1-r)+30*r),int(180*(1-r)+10*r)]
    cv2.circle(bg,(w//2,h//2+50),75,(20,170,255),-1)
    return bg

def make_matrix_bg(h,w):
    bg=np.zeros((h,w,3),dtype=np.uint8); rng=np.random.default_rng(99); chars="01ABXYZ"
    for col_x in range(0,w,14):
        ln=int(rng.integers(5,max(6,h//14))); sy=int(rng.integers(0,h))
        for i in range(ln):
            y2=(sy+i*14)%h; b=max(40,255-i*20)
            cv2.putText(bg,chars[int(rng.integers(0,len(chars)))],(col_x,y2),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,b,0),1)
    return bg

_BG_CACHE={}
BG_NAMES=["Space","Forest","Ocean","Sunset","Matrix"]
BG_FUNCS=[make_space_bg,make_forest_bg,make_ocean_bg,make_sunset_bg,make_matrix_bg]

def get_bg(name,h,w):
    key=(name,h,w)
    if key not in _BG_CACHE: _BG_CACHE[key]=BG_FUNCS[BG_NAMES.index(name)](h,w)
    return _BG_CACHE[key].copy()

def composite(person,bg,mask):
    a=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR).astype(np.float32)/255.0
    return np.clip(person*a+bg*(1-a),0,255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# ★  IMPROVED SIDEBAR UI
# ─────────────────────────────────────────────────────────────────────────────

class SidebarUI:
    def __init__(self):
        self.hit_boxes: list = []
        self._frame_count = 0

    # ── drawing helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _rounded_rect(img, x1, y1, x2, y2, col, r=5):
        cv2.rectangle(img, (x1+r, y1), (x2-r, y2), col, -1)
        cv2.rectangle(img, (x1, y1+r), (x2, y2-r), col, -1)
        for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.circle(img, (cx,cy), r, col, -1)

    @staticmethod
    def _gradient_rect(img, x1, y1, x2, y2, col_top, col_bot):
        """Simple vertical gradient fill."""
        rh = max(1, y2 - y1)
        for dy in range(rh):
            t = dy / rh
            c = tuple(int(col_top[i]*(1-t) + col_bot[i]*t) for i in range(3))
            cv2.line(img, (x1, y1+dy), (x2, y1+dy), c, 1)

    def _section_header(self, panel, y, label, icon_col=None):
        """Section header with left accent bar and optional icon dot."""
        cv2.rectangle(panel, (0, y), (PANEL_W, y+SECTION_H), C_SECTION_BG, -1)
        # gradient accent strip at top of section
        for i in range(3):
            alpha = 1.0 - i / 3
            col = tuple(int(v*alpha) for v in C_BTN_ACTIVE)
            cv2.rectangle(panel, (0, y+i), (PANEL_W, y+i+1), col, -1)
        # Left bar
        cv2.rectangle(panel, (0, y), (3, y+SECTION_H), C_BTN_ACTIVE, -1)
        if icon_col:
            cv2.circle(panel, (12, y+SECTION_H//2), 4, icon_col, -1)
            cv2.circle(panel, (12, y+SECTION_H//2), 4, (200,200,200), 1)
            tx = 22
        else:
            tx = 8
        cv2.putText(panel, label, (tx, y+SECTION_H-6),
                    FONT, FS_SEC, C_SECTION_TEXT, 1, cv2.LINE_AA)
        return y + SECTION_H

    def _button(self, panel, y, label, shortcut, active, is_new=False, dot_col=None):
        """Draw a filter button row. Returns bottom y."""
        x1, y1, x2, y2 = PAD, y+1, PANEL_W-PAD, y+BTN_H-1

        if active:
            self._gradient_rect(panel, x1, y1, x2, y2, C_BTN_ACTIVE, C_BTN_ACTIVE2)
            # Glow border
            cv2.rectangle(panel, (x1-1,y1-1), (x2+1,y2+1), C_BORDER_ACT, 1)
            txt_col = C_ACTIVE_TEXT
            sc_col  = (10, 10, 10)
            # Active indicator arrow on right
            ax = x2 - 6
            ay = (y1 + y2) // 2
            pts = np.array([[ax-5,ay-4],[ax,ay],[ax-5,ay+4]], np.int32)
            cv2.fillPoly(panel, [pts], (10,10,10))
        else:
            self._rounded_rect(panel, x1, y1, x2, y2, C_BTN, r=4)
            cv2.rectangle(panel, (x1,y1), (x2,y2), C_BORDER, 1)
            txt_col = C_TEXT
            sc_col  = (75, 75, 105)

        # Dot indicator
        dot_x = x1 + 10
        dot_y = (y1 + y2) // 2
        if dot_col:
            d_col = (10,10,10) if active else dot_col
            cv2.circle(panel, (dot_x, dot_y), 4, d_col, -1)

        # Shortcut key badge
        badge_x = x1 + 22
        badge_w = 18
        badge_y1 = dot_y - 8; badge_y2 = dot_y + 8
        badge_col = (10,10,10) if active else (45,45,70)
        self._rounded_rect(panel, badge_x, badge_y1, badge_x+badge_w, badge_y2, badge_col, r=2)
        cv2.putText(panel, shortcut, (badge_x+3, badge_y1+12),
                    FONT, 0.30, (255,255,255) if not active else (200,200,200),
                    1, cv2.LINE_AA)

        # Label
        cv2.putText(panel, label, (badge_x+badge_w+5, dot_y+5),
                    FONT, FS, txt_col, 1, cv2.LINE_AA)

        # NEW badge
        if is_new and not active:
            nx = x2 - 30
            ny1 = y1 + 5; ny2 = y2 - 5
            self._rounded_rect(panel, nx, ny1, nx+26, ny2, C_NEW_BADGE, r=3)
            cv2.putText(panel, "NEW", (nx+3, ny1+11),
                        FONT, FS_BADGE, (255,255,255), 1, cv2.LINE_AA)

        return y + BTN_H

    def _bg_button(self, panel, y, label, dot_col, active):
        x1,y1,x2,y2 = PAD, y+1, PANEL_W-PAD, y+BTN_H-1
        if active:
            self._gradient_rect(panel, x1, y1, x2, y2, C_BTN_ACTIVE, C_BTN_ACTIVE2)
            cv2.rectangle(panel, (x1-1,y1-1),(x2+1,y2+1), C_BORDER_ACT, 1)
            txt_col = C_ACTIVE_TEXT
        else:
            self._rounded_rect(panel, x1, y1, x2, y2, C_BTN, r=4)
            cv2.rectangle(panel, (x1,y1),(x2,y2), C_BORDER, 1)
            txt_col = C_TEXT

        dot_x = x1 + 12; dot_y = (y1+y2)//2
        cv2.circle(panel, (dot_x, dot_y), 7, dot_col, -1)
        rim = (10,10,10) if active else (100,100,120)
        cv2.circle(panel, (dot_x, dot_y), 7, rim, 1, cv2.LINE_AA)
        # Shine on dot
        cv2.circle(panel, (dot_x-2, dot_y-2), 2, (220,220,220), -1)

        cv2.putText(panel, label, (x1+26, dot_y+5),
                    FONT, FS, txt_col, 1, cv2.LINE_AA)
        return y + BTN_H

    def _action_button(self, panel, y, label, col_top, col_bot=None):
        if col_bot is None:
            col_bot = tuple(max(0,v-40) for v in col_top)
        x1,y1,x2,y2 = PAD, y+1, PANEL_W-PAD, y+BTN_H-1
        self._gradient_rect(panel, x1, y1, x2, y2, col_top, col_bot)
        cv2.rectangle(panel, (x1,y1),(x2,y2), (200,200,220), 1)
        # Centre text
        (tw,th),_ = cv2.getTextSize(label, FONT, FS, 1)
        tx = (PANEL_W - tw) // 2
        ty = (y1 + y2 + th) // 2
        cv2.putText(panel, label, (tx, ty), FONT, FS, (230,230,230), 1, cv2.LINE_AA)
        return y + BTN_H

    # ── status bar helpers ────────────────────────────────────────────────────
    @staticmethod
    def _fps_bar(panel, y, fps, people):
        """Mini status bar at bottom of logo area."""
        # FPS colour
        if fps >= 20:   fp_col = (0, 220, 80)
        elif fps >= 12: fp_col = (0, 180, 255)
        else:           fp_col = (0,  60, 220)

        # FPS indicator bar
        bar_w = min(PANEL_W-4, int((fps / 30.0) * (PANEL_W-4)))
        cv2.rectangle(panel, (2, y+2), (PANEL_W-2, y+6), (30,30,50), -1)
        cv2.rectangle(panel, (2, y+2), (2+bar_w,   y+6), fp_col, -1)

        cv2.putText(panel, f"{fps:.0f} fps", (4, y+18),
                    FONT, 0.36, fp_col, 1, cv2.LINE_AA)
        p_col = (0,200,255) if people > 0 else (80,80,100)
        cv2.putText(panel, f"People: {people}", (PANEL_W-80, y+18),
                    FONT, 0.33, p_col, 1, cv2.LINE_AA)

    # ── main render ───────────────────────────────────────────────────────────
    def render(self, frame_h, body_key, face_key, bg_idx,
               show_boxes, fps, people, age_stage):
        self._frame_count += 1
        panel = np.zeros((frame_h, PANEL_W, 3), dtype=np.uint8)

        # Background gradient
        for y in range(frame_h):
            r = y / frame_h
            panel[y] = [int(14*(1-r)+10*r), int(14*(1-r)+12*r), int(22*(1-r)+18*r)]

        # Subtle scanlines
        panel[::3, :] = np.clip(panel[::3].astype(np.int16) + 5, 0, 255).astype(np.uint8)

        self.hit_boxes = []
        y = 0

        # ── LOGO BAR ──────────────────────────────────────────────────────────
        self._gradient_rect(panel, 0, 0, PANEL_W, 44, (16,16,30), (10,10,20))
        # Glowing accent line under header
        for i, a in enumerate([0.2, 0.5, 1.0, 0.5, 0.2]):
            cv2.line(panel, (0, 44+i), (PANEL_W, 44+i),
                     tuple(int(v*a) for v in C_BTN_ACTIVE), 1)

        cv2.putText(panel, "FILTER", (6, 20), FONT, 0.55, C_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(panel, "CAM", (76, 20), FONT, 0.55, C_BTN_ACTIVE, 1, cv2.LINE_AA)
        # Version tag
        cv2.putText(panel, "v2.0", (PANEL_W-34, 14), FONT, 0.28, (70,70,100), 1, cv2.LINE_AA)

        self._fps_bar(panel, 24, fps, people)
        y = 50

        # ── BODY FILTERS ──────────────────────────────────────────────────────
        y = self._section_header(panel, y, "BODY FILTER", CAT_BODY)
        for (key, sc, lbl, is_new) in BODY_FILTER_LIST:
            dot = CAT_NEW if is_new else CAT_BODY
            y0 = y
            y  = self._button(panel, y, lbl, sc, key==body_key, is_new, dot)
            self.hit_boxes.append((y0, y, 'body', key))

        y += 2

        # ── FACE EFFECTS ──────────────────────────────────────────────────────
        y = self._section_header(panel, y, "FACE EFFECT", CAT_FACE)
        for (key, sc, lbl) in FACE_EFFECT_LIST:
            is_active = (key == face_key)
            display   = lbl
            if key == ord('p') and is_active:
                ages = ["Baby","Child","Teen","Adult","Old"]
                display = f"Age:{ages[age_stage]}"
            y0 = y
            y  = self._button(panel, y, display, sc, is_active, False, CAT_FACE)
            self.hit_boxes.append((y0, y, 'face', key))

        y += 2

        # ── BACKGROUND ────────────────────────────────────────────────────────
        y = self._section_header(panel, y, "BACKGROUND", CAT_BG)
        for i, (name, dot_col) in enumerate(BG_LIST):
            active = (name=="None" and bg_idx==-1) or (i < len(BG_NAMES) and bg_idx==i)
            y0 = y
            y  = self._bg_button(panel, y, name, dot_col, active)
            val = i if name != "None" else -1
            self.hit_boxes.append((y0, y, 'bg', val))

        y += 4

        # ── ACTIONS ───────────────────────────────────────────────────────────
        y = self._section_header(panel, y, "ACTIONS")

        box_top = C_BOX_BTN_ON if show_boxes else C_BOX_BTN_OFF
        lbl_box = "Boxes: ON" if show_boxes else "Boxes: OFF"
        y0=y; y=self._action_button(panel, y, f"[K] {lbl_box}", box_top)
        self.hit_boxes.append((y0, y, 'boxes', None))

        y0=y; y=self._action_button(panel, y, "[S]  Save Screenshot",
                                    (20, 100, 180), (10, 70, 130))
        self.hit_boxes.append((y0, y, 'save', None))

        y0=y; y=self._action_button(panel, y, "[Q]  Quit",
                                    (140, 30, 30), (100, 20, 20))
        self.hit_boxes.append((y0, y, 'quit', None))

        # ── Bottom accent ─────────────────────────────────────────────────────
        cv2.rectangle(panel, (0, frame_h-2), (PANEL_W, frame_h), C_BTN_ACTIVE, -1)

        # Right edge glow
        for i, a in enumerate([0.08, 0.18, 0.40, 0.80, 1.0]):
            xr = PANEL_W - 1 - i
            if xr >= 0:
                panel[:,xr] = np.clip(
                    panel[:,xr].astype(np.float32)*(1-a) +
                    np.array(C_BTN_ACTIVE, dtype=np.float32)*a, 0, 255
                ).astype(np.uint8)

        return panel

    def hit_test(self, x, y):
        if x < 0 or x >= PANEL_W:
            return None
        for (y1, y2, atype, aval) in self.hit_boxes:
            if y1 <= y < y2:
                return (atype, aval)
        return None

# ─────────────────────────────────────────────────────────────────────────────
# MOUSE STATE
# ─────────────────────────────────────────────────────────────────────────────

class MouseState:
    def __init__(self):
        self.x = 0; self.y = 0
        self.clicked_action = None

def make_mouse_callback(ms: MouseState):
    def callback(event, x, y, flags, param):
        ms.x = x; ms.y = y
        if event == cv2.EVENT_LBUTTONUP:
            ms.clicked_action = (x, y)
    return callback

# ─────────────────────────────────────────────────────────────────────────────
# HUD (overlaid on the video portion)
# ─────────────────────────────────────────────────────────────────────────────

def draw_hud(frame, body_name, face_name, bg_name, fps, people):
    h, w = frame.shape[:2]
    # semi-transparent top bar
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,52), (0,0,0), -1)
    frame = cv2.addWeighted(ov, 0.50, frame, 0.50, 0)

    # Filter name pill
    pill_lbl = f"  {body_name}  "
    (pw,_),_ = cv2.getTextSize(pill_lbl, FONT, 0.52, 1)
    cv2.rectangle(frame, (6,4), (pw+12,26), (0,160,100), -1)
    cv2.putText(frame, pill_lbl, (8,20), FONT, 0.52, (10,10,10), 1, cv2.LINE_AA)

    # Face / bg info
    info = []
    if face_name: info.append(f"Face: {face_name}")
    if bg_name:   info.append(f"BG: {bg_name}")
    info.append(f"People: {people}")
    cv2.putText(frame, "   ".join(info), (8,44), FONT, 0.38, (130,130,155), 1, cv2.LINE_AA)

    # FPS top-right
    fps_col = (0,220,80) if fps>=20 else (0,180,255) if fps>=12 else (0,60,220)
    cv2.putText(frame, f"{fps:.0f}fps", (w-64,20), FONT, 0.52, fps_col, 1, cv2.LINE_AA)

    # Thin accent line under HUD
    for i, a in enumerate([1.0, 0.5, 0.2]):
        cv2.line(frame, (0,52+i), (w,52+i), tuple(int(v*a) for v in (0,200,120)), 1)

    return frame

# ─────────────────────────────────────────────────────────────────────────────
# TOAST NOTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

class Toast:
    def __init__(self):
        self.msg   = ""
        self.until = 0.0
        self.col   = (0,200,130)

    def show(self, msg, col=(0,200,130), duration=2.0):
        self.msg   = msg
        self.until = time.time() + duration
        self.col   = col

    def draw(self, frame):
        if time.time() > self.until:
            return frame
        h, w = frame.shape[:2]
        (tw,th),_ = cv2.getTextSize(self.msg, FONT, 0.55, 1)
        px, py = (w-tw)//2, h-80
        fade = min(1.0, (self.until - time.time()) / 0.4)  # fade out last 0.4s
        ov = frame.copy()
        cv2.rectangle(ov, (px-12,py-th-8), (px+tw+12, py+6), (0,0,0), -1)
        cv2.rectangle(ov, (px-13,py-th-9), (px+tw+13, py+7),
                      tuple(int(v*fade) for v in self.col), 2)
        frame = cv2.addWeighted(frame, 1-0.7*fade, ov, 0.7*fade, 0)
        cv2.putText(frame, self.msg, (px, py),
                    FONT, 0.55,
                    tuple(int(v*fade) for v in self.col),
                    1, cv2.LINE_AA)
        return frame

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cameras = find_camera()
    if not cameras:
        print("No webcam found!"); input("Press Enter to exit..."); return

    cam_index = cameras[0]
    if len(cameras) > 1:
        print(f"Multiple cameras: {cameras}  — Enter index [0]: ", end="", flush=True)
        try:
            cam_index = cameras[int(input().strip())]
        except Exception:
            cam_index = cameras[0]

    try:
        seg  = YOLOSegmentor()
        face = FaceDetector()
    except Exception as e:
        print(f"Model load failed: {e}"); input("Press Enter to exit..."); return

    cap = open_camera(cam_index)
    if not cap.isOpened():
        print(f"Cannot open camera {cam_index}."); input("Press Enter..."); return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    body_filter_key = ord('1')
    face_effect_key = None
    bg_idx          = -1
    show_boxes      = False
    prev_time       = time.time()
    save_dir        = os.path.expanduser("~/Pictures")
    os.makedirs(save_dir, exist_ok=True)

    sidebar  = SidebarUI()
    ms       = MouseState()
    toast    = Toast()
    WIN_NAME = "Filter Cam"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_NAME, make_mouse_callback(ms))

    print("\n" + "="*58)
    print("  Filter Cam v2.0  — Ready!")
    print("  Click LEFT PANEL or use keyboard shortcuts.")
    print("  NEW: [R] Skeleton  [X] Stick Figure  [U] Bubbles")
    print("  Q / Esc to quit.")
    print("="*58 + "\n")

    quit_flag = False

    while not quit_flag:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        t     = time.time()

        # ── YOLO ─────────────────────────────────────────────────────────
        mask, boxes = seg.get_mask_and_boxes(frame)

        # ── Body filter ───────────────────────────────────────────────────
        b_name, b_fn = BODY_FILTERS[body_filter_key]
        filtered = b_fn(frame, mask)

        # ── Virtual background ────────────────────────────────────────────
        if bg_idx >= 0:
            h_f, w_f = frame.shape[:2]
            bg        = get_bg(BG_NAMES[bg_idx], h_f, w_f)
            filtered  = composite(filtered, bg, mask)

        # ── Bounding boxes ────────────────────────────────────────────────
        if show_boxes:
            for i, box in enumerate(boxes):
                x1,y1,x2,y2 = box
                cv2.rectangle(filtered,(x1,y1),(x2,y2),(0,220,140),2)
                cv2.putText(filtered,f"Person {i+1}",(x1+4,y1-6),FONT,0.48,(255,255,255),1)

        # ── Face effect ───────────────────────────────────────────────────
        f_name = None
        if face_effect_key is not None:
            faces = face.detect(frame)
            f_name, f_fn = FACE_EFFECTS[face_effect_key]
            if face_effect_key == ord('p'):
                filtered = effect_age_progression(filtered, faces, t)
            else:
                filtered = f_fn(filtered, faces, t)

        # ── HUD ───────────────────────────────────────────────────────────
        fps       = 1.0 / max(t - prev_time, 1e-6)
        prev_time = t
        bg_name   = BG_NAMES[bg_idx] if bg_idx >= 0 else None
        filtered  = draw_hud(filtered, b_name, f_name, bg_name, fps, len(boxes))
        filtered  = toast.draw(filtered)

        # ── Sidebar ───────────────────────────────────────────────────────
        h_f, w_f = filtered.shape[:2]
        panel = sidebar.render(h_f, body_filter_key, face_effect_key,
                               bg_idx, show_boxes, fps, len(boxes),
                               AGE_PROGRESSION_STAGE[0])

        display = np.hstack([panel, filtered])
        cv2.imshow(WIN_NAME, display)

        # ── Mouse click handling ──────────────────────────────────────────
        if ms.clicked_action is not None:
            cx, cy = ms.clicked_action
            ms.clicked_action = None
            hit = sidebar.hit_test(cx, cy)
            if hit:
                atype, aval = hit
                if atype == 'body':
                    body_filter_key = aval
                    toast.show(f"Filter: {BODY_FILTERS[aval][0]}")
                elif atype == 'face':
                    if face_effect_key == aval:
                        if aval == ord('p'):
                            AGE_PROGRESSION_STAGE[0] = (AGE_PROGRESSION_STAGE[0]+1) % 5
                            ages = ["Baby","Child","Teen","Adult","Elderly"]
                            toast.show(f"Age: {ages[AGE_PROGRESSION_STAGE[0]]}")
                        else:
                            face_effect_key = None
                            toast.show("Face effect: Off", (140,140,160))
                    else:
                        face_effect_key = aval
                        if aval == ord('p'): AGE_PROGRESSION_STAGE[0] = 0
                        toast.show(f"Effect: {FACE_EFFECTS[aval][0]}", (200,80,220))
                elif atype == 'bg':
                    bg_idx = aval
                    name   = BG_NAMES[bg_idx] if bg_idx >= 0 else "None"
                    toast.show(f"Background: {name}", (220,140,20))
                elif atype == 'boxes':
                    show_boxes = not show_boxes
                    toast.show(f"Boxes: {'ON' if show_boxes else 'OFF'}", (20,130,20))
                elif atype == 'save':
                    ts   = time.strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(save_dir, f"filterCam_{ts}.png")
                    cv2.imwrite(path, display)
                    toast.show(f"Saved!", (20,120,200))
                    print(f"Saved → {path}")
                elif atype == 'quit':
                    quit_flag = True

        # ── Keyboard shortcuts ────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in BODY_FILTERS:
            body_filter_key = key
            toast.show(f"Filter: {BODY_FILTERS[key][0]}")
        elif key in FACE_EFFECTS:
            if face_effect_key == key:
                if key == ord('p'):
                    AGE_PROGRESSION_STAGE[0] = (AGE_PROGRESSION_STAGE[0]+1) % 5
                else:
                    face_effect_key = None
            else:
                face_effect_key = key
        elif key == ord('b'):
            bg_idx = (bg_idx + 1) % len(BG_NAMES)
            toast.show(f"Background: {BG_NAMES[bg_idx]}", (220,140,20))
        elif key == ord('n'):
            bg_idx = -1
            toast.show("Background: Off", (140,140,160))
        elif key == ord('k'):
            show_boxes = not show_boxes
        elif key == ord('s'):
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(save_dir, f"filterCam_{ts}.png")
            cv2.imwrite(path, display)
            toast.show(f"Saved!", (20,120,200))
            print(f"Saved → {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Bye!")

if __name__ == "__main__":
    main()