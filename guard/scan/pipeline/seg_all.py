import numpy as np, glob, os
from ultralytics import YOLO
m=YOLO('seg/parts_segmentation.pt'); os.makedirs('labels',exist_ok=True)
for f in sorted(glob.glob('imgs/*.jpg')):
    r=m.predict(f,imgsz=768,conf=0.3,iou=0.7,retina_masks=True,verbose=False)[0]
    H,W=r.orig_shape; L=np.full((H,W),255,np.uint8); S=np.zeros((H,W),np.float32)
    if r.masks is not None:
        for mk,c,s in zip(r.masks.data.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            mk=mk>0.5; upd=mk&(s>S); L[upd]=int(c); S[upd]=s
    np.save(f'labels/{os.path.basename(f)[:-4]}.npy',L)
print('done')
