import numpy as np, json
z=np.load('fused_raw.npz'); X,C,L,I,base=z['X'],z['C'],z['L'],z['I'],list(z['base'])
import pycolmap
r=pycolmap.Reconstruction('sfm')
cams=np.array([im.projection_center() for im in r.images.values()]); c0=cams.mean(0)
U,S,Vt=np.linalg.svd(cams-c0); e1,e2,n=Vt
Pall=np.array([p.xyz for p in r.points3D.values()])
if np.median((Pall-c0)@n)>0: n=-n; e2=-e2
up=-n  # points mostly below cameras => ground is along n (negative side is "down" after flip)
# vertical coordinate h = (X-c0)@up ; ground height from sparse points
h_sp=(Pall-c0)@up; 
gz=np.percentile(h_sp,5)
voxel=0.02
key=np.floor(X/voxel).astype(np.int64)
k=(key[:,0]*73856093)^(key[:,1]*19349663)^(key[:,2]*83492791)
order=np.argsort(k,kind='stable'); ks=k[order]
starts=np.r_[0,np.nonzero(np.diff(ks))[0]+1]; ends=np.r_[starts[1:],len(ks)]
out_X,out_C,out_L,out_n=[],[],[],[]
Ls=L[order]; Is=I[order]; Xs=X[order]; Cs=C[order]
nb=len(base)
for s,e in zip(starts,ends):
    if e-s<4: continue
    nv=len(np.unique(Is[s:e]))
    if nv<3: continue
    cnt=np.bincount(Ls[s:e],minlength=nb); lab=cnt.argmax()
    if cnt[lab]<0.5*(e-s): continue
    sel=Ls[s:e]==lab
    out_X.append(Xs[s:e][sel].mean(0)); out_C.append(np.median(Cs[s:e][sel],0)); out_L.append(lab); out_n.append(nv)
X=np.array(out_X); C=np.array(out_C).astype(np.uint8); L=np.array(out_L); NV=np.array(out_n)
print('voxels kept',len(X))
# car frame: up, forward (back parts -> front parts)
front=np.isin(L,[base.index(b) for b in ('hood','grille','front-bumper','headlight','windshield')])
back=np.isin(L,[base.index(b) for b in ('trunk','back-bumper','tail-light','back-windshield')])
fwd=X[front].mean(0)-X[back].mean(0); fwd-= (fwd@up)*up; fwd/=np.linalg.norm(fwd)
wh=np.isin(L,[base.index('front-wheel'),base.index('back-wheel')]); rf=L==base.index('roof')
if (X[wh]@up).mean()>(X[rf]@up).mean(): up=-up
fwd=X[front].mean(0)-X[back].mean(0); fwd-= (fwd@up)*up; fwd/=np.linalg.norm(fwd)
left=np.cross(up,fwd)
ctr=X.mean(0)
loc=np.stack([(X-ctr)@fwd,(X-ctr)@left,(X-ctr)@up],1)
# robust extents for scale
length=np.percentile(loc[:,0],99.5)-np.percentile(loc[:,0],0.5)
scale=4.915/length   # Toyota Camry XV80 published overall length 4,915 mm
loc*=scale
loc[:,0]-= (np.percentile(loc[:,0],99.5)+np.percentile(loc[:,0],0.5))/2
loc[:,1]-= np.median(loc[:,1])
ground=np.percentile(loc[:,2],0.5); loc[:,2]-=ground
print('extent L W H (m):', [round(float(np.percentile(loc[:,i],99.5)-np.percentile(loc[:,i],0.5)),2) for i in range(3)])
np.savez('car_parts.npz',X=loc.astype(np.float32),C=C,L=L,NV=NV,base=np.array(base),scale=scale)
for i,b in enumerate(base):
    m=L==i
    if m.sum(): print(f'{b:16s} {m.sum():7d}  y-mean {loc[m,1].mean():+.2f}')
