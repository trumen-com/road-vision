import numpy as np, glob, os, torch
from transformers import pipeline
from PIL import Image
torch.set_num_threads(2)
p=pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf', device='cpu')
os.makedirs('depth',exist_ok=True)
for f in sorted(glob.glob('imgs/*.jpg')):
    out=f'depth/{os.path.basename(f)[:-4]}.npy'
    if os.path.exists(out): continue
    im=Image.open(f).convert('RGB')
    d=p(im)['predicted_depth']; d=d.squeeze().numpy().astype(np.float32)   # relative inverse depth (disparity-like)
    np.save(out,d)
print('done')
