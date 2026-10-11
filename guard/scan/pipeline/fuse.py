import numpy as np, pycolmap, glob, os, json
from PIL import Image
r=pycolmap.Reconstruction('sfm')
names={0:'back-bumper',1:'back-windshield',2:'front-bumper',3:'grille',4:'hood',5:'back-door',6:'back-wheel',7:'back-window',8:'fender',9:'front-door',10:'front-wheel',11:'front-window',12:'headlight',13:'mirror',14:'quarter-panel',15:'rocker-panel',16:'tail-light',17:'license-plate',18:'back-door',19:'back-wheel',20:'back-window',21:'fender',22:'front-door',23:'front-wheel',24:'front-window',25:'headlight',26:'mirror',27:'quarter-panel',28:'rocker-panel',29:'tail-light',30:'roof',31:'trunk',32:'windshield'}
base=sorted(set(names.values())); bidx={n:i for i,n in enumerate(base)}
PLATE=bidx['license-plate']
lut=np.full(256,255,np.uint8)
for k,v in names.items(): lut[k]=bidx[v]
P3=r.points3D
allX,allC,allL,allI=[],[],[],[]
stats=[]
for iid,im in r.images.items():
    n=im.name[:-4]
    if not os.path.exists(f'depth/{n}.npy'): continue
    L=lut[np.load(f'labels/{n}.npy')]; H,W=L.shape
    d=np.array(Image.fromarray(np.load(f'depth/{n}.npy')).resize((W,H),Image.BILINEAR))
    img=np.array(Image.open(f'imgs/{im.name}').convert('RGB'))
    cam=r.cameras[im.camera_id]; Rt=im.cam_from_world().matrix()   # 3x4
    # sparse points seen in this image -> (pixel, depth)
    uv,z=[],[]
    for p2 in im.points2D:
        if p2.has_point3D():
            X=P3[p2.point3D_id].xyz; c=Rt[:,:3]@X+Rt[:,3]
            if c[2]>0: uv.append(p2.xy); z.append(c[2])
    uv=np.array(uv); z=np.array(z)
    if len(z)<50: continue
    ui=np.clip(uv[:,0].astype(int),0,W-1); vi=np.clip(uv[:,1].astype(int),0,H-1)
    dv=d[vi,ui]; oncar=L[vi,ui]!=255
    # robust fit 1/z = a*d + b (RANSAC), prefer points on the car
    A=np.stack([dv,np.ones_like(dv)],1); y=1/z; best=None; rng=np.random.default_rng(iid)
    w_idx=np.where(oncar)[0] if oncar.sum()>30 else np.arange(len(z))
    for _ in range(300):
        s=rng.choice(w_idx,2,replace=False)
        try: coef=np.linalg.solve(A[s],y[s])
        except np.linalg.LinAlgError: continue
        res=np.abs(A[w_idx]@coef-y[w_idx])/y[w_idx]
        inl=w_idx[res<0.05]
        if best is None or len(inl)>len(best[1]): best=(coef,inl)
    coef,inl=best; coef=np.linalg.lstsq(A[inl],y[inl],rcond=None)[0]
    zr=1/np.clip(A[w_idx]@coef,1e-6,None); relerr=np.median(np.abs(zr-z[w_idx])/z[w_idx])
    stats.append((n,len(z),int(oncar.sum()),len(inl),float(relerr)))
    if relerr>0.08 or len(inl)<20: continue
    # back-project car pixels (not plate), stride 3
    vv,uu=np.mgrid[0:H:3,0:W:3]; lab=L[vv,uu]; m=(lab!=255)&(lab!=PLATE)
    # erode mask edges (depth bleeding): require same label 6 px around
    for dy,dx in ((6,0),(-6,0),(0,6),(0,-6)):
        m&= L[np.clip(vv+dy,0,H-1),np.clip(uu+dx,0,W-1)]==lab
    uu,vv,lab=uu[m],vv[m],lab[m]
    inv=coef[0]*d[vv,uu]+coef[1]; ok=inv>1e-3; uu,vv,lab,inv=uu[ok],vv[ok],lab[ok],inv[ok]
    zz=1/inv
    xy=np.stack([uu+0.5,vv+0.5],1)
    rays=np.array(cam.cam_from_img(xy)); Xc=np.concatenate([rays*zz[:,None],zz[:,None]],1)
    Xw=(Xc-Rt[:,3])@Rt[:,:3]   # R^T (Xc - t)
    allX.append(Xw); allC.append(img[vv,uu]); allL.append(lab); allI.append(np.full(len(lab),iid))
json.dump(stats,open('align_stats.json','w'))
X=np.concatenate(allX); C=np.concatenate(allC); Lb=np.concatenate(allL); I=np.concatenate(allI)
print('views used',len(allX),'of',len(stats),'raw points',len(X))
np.savez('fused_raw.npz',X=X,C=C,L=Lb,I=I,base=np.array(base))
