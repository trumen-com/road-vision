"""Scan accuracy tuning. Builds the fused part cloud under a parameter set and scores it against published
Camry XV80 figures (length is the only scale input; wheelbase, body width and height are checks), plus surface
thickness on flat-ish panels and multi-view label agreement.
  python3 tune.py '{"mv_tau":0.03,"mv_min":2}'  -> prints JSON metrics; --save writes car_parts.npz
"""
import numpy as np, pycolmap, json, sys, os, time
from PIL import Image
from scipy.spatial import cKDTree
SFM = os.environ.get('SFM', 'sfm')
DEF = dict(ds=2, stride=2, erode=6, align='inv', fit_tol=0.05, max_relerr=0.08, mv_k=6, mv_tau=0.0, mv_min=0,
           voxel=0.02, min_views=3, agree=0.5, conf_mask=None)
P = dict(DEF); P.update(json.loads(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].startswith('{') else {})
SAVE = '--save' in sys.argv
PUB = dict(length=4.915, width=1.84, height=1.445, wheelbase=2.825)
names = {0:'back-bumper',1:'back-windshield',2:'front-bumper',3:'grille',4:'hood',5:'back-door',6:'back-wheel',7:'back-window',8:'fender',9:'front-door',10:'front-wheel',11:'front-window',12:'headlight',13:'mirror',14:'quarter-panel',15:'rocker-panel',16:'tail-light',17:'license-plate',18:'back-door',19:'back-wheel',20:'back-window',21:'fender',22:'front-door',23:'front-wheel',24:'front-window',25:'headlight',26:'mirror',27:'quarter-panel',28:'rocker-panel',29:'tail-light',30:'roof',31:'trunk',32:'windshield'}
base = sorted(set(names.values())); B = {n: i for i, n in enumerate(base)}
lut = np.full(256, 255, np.uint8)
for k, v in names.items(): lut[k] = B[v]
r = pycolmap.Reconstruction(SFM)
ds = P['ds']
views = []
t0 = time.time()
for iid, im in r.images.items():
    n = os.path.basename(im.name)[:-4]
    if not os.path.exists(f'depth/{n}.npy'): continue
    Lf = lut[np.load(f'labels/{n}.npy')]; Hf, Wf = Lf.shape
    H, W = Hf // ds, Wf // ds
    L = Lf[::ds, ::ds][:H, :W]
    d = np.array(Image.fromarray(np.load(f'depth/{n}.npy')).resize((W, H), Image.BILINEAR))
    img = np.array(Image.open(f'imgs/{os.path.basename(im.name)}').convert('RGB').resize((W, H), Image.BILINEAR))
    cam = r.cameras[im.camera_id]; Rt = im.cam_from_world().matrix()
    uv, z = [], []
    for p2 in im.points2D:
        if p2.has_point3D():
            c = Rt[:, :3] @ r.points3D[p2.point3D_id].xyz + Rt[:, 3]
            if c[2] > 0: uv.append(p2.xy); z.append(c[2])
    uv = np.array(uv) / ds; z = np.array(z)
    if len(z) < 50: continue
    ui = np.clip(uv[:, 0].astype(int), 0, W - 1); vi = np.clip(uv[:, 1].astype(int), 0, H - 1)
    dv = d[vi, ui]; oncar = L[vi, ui] != 255
    w_idx = np.where(oncar)[0] if oncar.sum() > 30 else np.arange(len(z))
    if P['align'] == 'inv':      # 1/z = a d + b
        A = np.stack([dv, np.ones_like(dv)], 1); y = 1 / z
    elif P['align'] == 'inv_plane':   # 1/z = a d + b + c u + e v  (corrects a tilt in the relative depth)
        A = np.stack([dv, np.ones_like(dv), ui / W - 0.5, vi / H - 0.5], 1); y = 1 / z
    elif P['align'] == 'inv_quad':    # plus quadratic image terms and a d*u, d*v interaction
        uu0, vv0 = ui / W - 0.5, vi / H - 0.5
        A = np.stack([dv, np.ones_like(dv), uu0, vv0, uu0 * uu0, vv0 * vv0, uu0 * vv0, dv * uu0, dv * vv0], 1); y = 1 / z
    else:                         # z = a/d + b  (affine in depth)
        A = np.stack([1 / np.maximum(dv, 1e-3), np.ones_like(dv)], 1); y = z
    rng = np.random.default_rng(iid); best = None
    for _ in range(300):
        s = rng.choice(w_idx, A.shape[1], replace=False)
        try: coef = np.linalg.solve(A[s], y[s])
        except np.linalg.LinAlgError: continue
        pred = A[w_idx] @ coef
        zp = pred if P['align'] == 'aff' else 1 / np.clip(pred, 1e-6, None)
        inl = w_idx[np.abs(zp - z[w_idx]) / z[w_idx] < P['fit_tol']]
        if best is None or len(inl) > len(best[1]): best = (coef, inl)
    coef = np.linalg.lstsq(A[best[1]], y[best[1]], rcond=None)[0]
    def depth_of(dd, uu=None, vv=None):
        if P['align'] == 'aff': return coef[0] / np.maximum(dd, 1e-3) + coef[1]
        q = coef[0] * dd + coef[1]
        if P['align'] in ('inv_plane', 'inv_quad'):
            if uu is None: vv_, uu_ = np.mgrid[0:H, 0:W]; uu, vv = uu_, vv_
            a_, b_ = uu / W - 0.5, vv / H - 0.5
            q = q + coef[2] * a_ + coef[3] * b_
            if P['align'] == 'inv_quad': q = q + coef[4] * a_ * a_ + coef[5] * b_ * b_ + coef[6] * a_ * b_ + coef[7] * dd * a_ + coef[8] * dd * b_
        return 1 / np.clip(q, 1e-6, None)
    zr = depth_of(dv[w_idx], ui[w_idx], vi[w_idx]); relerr = float(np.median(np.abs(zr - z[w_idx]) / z[w_idx]))
    if relerr > P['max_relerr'] or len(best[1]) < 20: continue
    Z = depth_of(d).astype(np.float32)
    views.append(dict(iid=iid, L=L, Z=Z, img=img, cam=cam, Rt=Rt, H=H, W=W, C=im.projection_center(), relerr=relerr))
align_err = float(np.median([v['relerr'] for v in views]))
# back-project
allX, allC, allL, allV = [], [], [], []
rX, rC, rL, rV = [], [], [], []
centers = np.array([v['C'] for v in views])
for vi_, v in enumerate(views):
    H, W, L, Z = v['H'], v['W'], v['L'], v['Z']; st = P['stride']
    vv, uu = np.mgrid[0:H:st, 0:W:st]; lab = L[vv, uu]; m = (lab != 255) & (lab != B['license-plate'])
    e = max(1, P['erode'] // ds)
    for dy, dx in ((e, 0), (-e, 0), (0, e), (0, -e)):
        m &= L[np.clip(vv + dy, 0, H - 1), np.clip(uu + dx, 0, W - 1)] == lab
    uu, vv, lab = uu[m], vv[m], lab[m]; zz = Z[vv, uu]; ok = (zz > 0) & np.isfinite(zz); uu, vv, lab, zz = uu[ok], vv[ok], lab[ok], zz[ok]
    xy = np.stack([(uu + 0.5) * ds, (vv + 0.5) * ds], 1)
    rays = np.array(v['cam'].cam_from_img(xy)); Xc = np.concatenate([rays * zz[:, None], zz[:, None]], 1)
    Rt = v['Rt']; Xw = (Xc - Rt[:, 3]) @ Rt[:, :3]
    rX.append(Xw); rC.append(v['img'][vv, uu]); rL.append(lab); rV.append(np.full(len(lab), vi_, np.int16))
    if P['mv_min'] > 0:   # multi-view depth consistency against the nearest views
        dist = np.linalg.norm(centers - v['C'], axis=1); nb = [j for j in np.argsort(dist) if j != vi_][:P['mv_k']]
        votes = np.zeros(len(Xw), np.int16)
        for j in nb:
            w = views[j]; Rj = w['Rt']; Xcj = Xw @ Rj[:, :3].T + Rj[:, 3]; front = Xcj[:, 2] > 0.1
            pj = np.full((len(Xw), 2), -1.0); pj[front] = w['cam'].img_from_cam(Xcj[front], check_cheirality=False) / ds
            ju, jv = pj[:, 0].astype(int), pj[:, 1].astype(int); inside = front & (ju >= 0) & (jv >= 0) & (ju < w['W']) & (jv < w['H'])
            zj = np.full(len(Xw), np.nan); zj[inside] = w['Z'][jv[inside], ju[inside]]
            lj = np.full(len(Xw), 255); lj[inside] = w['L'][jv[inside], ju[inside]]
            good = inside & (np.abs(zj - Xcj[:, 2]) / Xcj[:, 2] < P['mv_tau']) & (lj == lab)
            votes += good
        keep = votes >= P['mv_min']; Xw, uu, vv, lab = Xw[keep], uu[keep], vv[keep], lab[keep]
    allX.append(Xw); allC.append(v['img'][vv, uu]); allL.append(lab); allV.append(np.full(len(lab), vi_, np.int16))
def fuse(X, C, L, V, min_views):
    key = np.floor(X / P['voxel']).astype(np.int64); key -= key.min(0)
    kk = (key[:, 0] * 1_000_003 + key[:, 1]) * 1_000_003 + key[:, 2]
    uk, inv = np.unique(kk, return_inverse=True); nvox = len(uk); nb_ = len(base)
    cnt = np.bincount(inv, minlength=nvox)
    lc = np.zeros((nvox, nb_), np.int32); np.add.at(lc, (inv, L), 1)
    lab = lc.argmax(1); agree = lc.max(1) / cnt
    pv = np.unique(inv.astype(np.int64) * 1000 + V); nviews = np.bincount((pv // 1000).astype(np.int64), minlength=nvox)
    sel_pt = (lab[inv] == L)
    s = np.bincount(inv[sel_pt], minlength=nvox); s = np.maximum(s, 1)
    Xm = np.stack([np.bincount(inv[sel_pt], X[sel_pt, i], nvox) / s for i in range(3)], 1)
    Cm = np.stack([np.bincount(inv[sel_pt], C[sel_pt, i].astype(float), nvox) / s for i in range(3)], 1)
    keepv = (cnt >= 4) & (nviews >= min_views) & (agree >= P['agree'])
    return Xm[keepv], Cm[keepv].astype(np.uint8), lab[keepv], nviews[keepv], agree[keepv]
X, C, L, NV, AG = fuse(np.concatenate(allX), np.concatenate(allC), np.concatenate(allL), np.concatenate(allV), P['min_views'])
n_strict = len(X)
if P.get('fill', 0) and P['mv_min'] > 0:
    # fill holes (mostly roof and rear glass, seen only at grazing angles) from the unfiltered depth, but only where
    # the strict pass left no point within fill_cell, and with a higher view count
    Xr, Cr, Lr, NVr, AGr = fuse(np.concatenate(rX), np.concatenate(rC), np.concatenate(rL), np.concatenate(rV), P.get('fill_views', 5))
    cell = P.get('fill_cell', 0.06)
    tr = cKDTree(X); dd, _ = tr.query(Xr, k=1)
    add = dd > cell
    X = np.concatenate([X, Xr[add]]); C = np.concatenate([C, Cr[add]]); L = np.concatenate([L, Lr[add]]); NV = np.concatenate([NV, NVr[add]]); AG = np.concatenate([AG, AGr[add]])
# car frame
cams = np.array([im.projection_center() for im in r.images.values()]); c0 = cams.mean(0)
U, S_, Vt = np.linalg.svd(cams - c0); up = Vt[2]
wh = np.isin(L, [B['front-wheel'], B['back-wheel']]); rf = L == B['roof']
if (X[wh] @ up).mean() > (X[rf] @ up).mean(): up = -up
front = np.isin(L, [B[b] for b in ('hood', 'grille', 'front-bumper', 'headlight', 'windshield')])
back = np.isin(L, [B[b] for b in ('trunk', 'back-bumper', 'tail-light', 'back-windshield')])
fwd = X[front].mean(0) - X[back].mean(0); fwd -= (fwd @ up) * up; fwd /= np.linalg.norm(fwd); left = np.cross(up, fwd)
ctr = X.mean(0)
SP = np.array([p.xyz for p in r.points3D.values()])
def frame(up, fwd):
    fwd = fwd - (fwd @ up) * up; fwd /= np.linalg.norm(fwd); return up, fwd, np.cross(up, fwd)
up, fwd, left = frame(up, fwd)
def tolocal(Y): return np.stack([(Y - ctr) @ fwd, (Y - ctr) @ left, (Y - ctr) @ up], 1)
if P.get('refine', 1):
    for _ in range(2):
        # 1) up from a RANSAC plane through sparse ground points around the car
        loc0 = tolocal(X); sp0 = tolocal(SP); ext = np.percentile(np.abs(loc0[:, :2]), 99, axis=0)
        zlow = np.percentile(loc0[:, 2], 2)
        g = sp0[(np.abs(sp0[:, 0]) < ext[0] + 1.2) & (np.abs(sp0[:, 1]) < ext[1] + 1.2) & ~((np.abs(sp0[:, 0]) < ext[0]) & (np.abs(sp0[:, 1]) < ext[1])) & (np.abs(sp0[:, 2] - zlow) < 0.5 * ext[1])]
        if len(g) > 100:
            rng = np.random.default_rng(1); best = None; tol = 0.01 * ext[0]
            for _ in range(500):
                a, b, c = g[rng.choice(len(g), 3, replace=False)]; nrm = np.cross(b - a, c - a)
                if np.linalg.norm(nrm) < 1e-9: continue
                nrm /= np.linalg.norm(nrm); inl = np.abs((g - a) @ nrm) < tol
                if best is None or inl.sum() > best[1].sum(): best = (nrm, inl)
            gi = g[best[1]]; U_, S__, Vt_ = np.linalg.svd(gi - gi.mean(0)); nl = Vt_[2]
            if nl[2] < 0: nl = -nl
            up_new = nl[0] * fwd + nl[1] * left + nl[2] * up; up, fwd, left = frame(up_new / np.linalg.norm(up_new), fwd)
        # 2) yaw and lateral offset from left/right symmetry of the side panels
        loc0 = tolocal(X)
        sidek = np.isin(L, [B[b] for b in ('front-door', 'back-door', 'fender', 'quarter-panel', 'rocker-panel', 'front-wheel', 'back-wheel')])
        xs, cs = [], []
        ex_ = np.percentile(np.abs(loc0[:, 0]), 99)
        for x0 in np.linspace(-0.6 * ex_, 0.6 * ex_, 13):
            m = sidek & (np.abs(loc0[:, 0] - x0) < 0.05 * ex_)
            yl = loc0[m & (loc0[:, 1] > 0), 1]; yr = loc0[m & (loc0[:, 1] < 0), 1]
            if len(yl) > 30 and len(yr) > 30: xs.append(x0); cs.append((np.percentile(yl, 90) + np.percentile(yr, 10)) / 2)
        if len(xs) >= 4:
            b_, a_ = np.polyfit(xs, cs, 1); th = np.arctan(b_)
            fwd_new = np.cos(th) * fwd + np.sin(th) * left; up, fwd, left = frame(up, fwd_new)
            ctr = ctr + a_ * left
loc = tolocal(X)
lo, hi = np.percentile(loc[:, 0], [0.5, 99.5]); scale = PUB['length'] / (hi - lo)
loc *= scale; loc[:, 0] -= (lo + hi) / 2 * scale; ymid = 0.0 if P.get('refine', 1) else np.median(loc[:, 1]); loc[:, 1] -= ymid
# ground from sparse SfM points around the car (ring 0.3-1.5 m outside the footprint)
sp = tolocal(np.array([p.xyz for p in r.points3D.values()])) * scale; sp[:, 0] -= (lo + hi) / 2 * scale; sp[:, 1] -= ymid
ring = (np.abs(sp[:, 0]) < 3.5) & (np.abs(sp[:, 1]) < 2.0) & ~((np.abs(sp[:, 0]) < 2.5) & (np.abs(sp[:, 1]) < 0.95)) & (sp[:, 2] < np.percentile(loc[:, 2], 5) + 0.3)
ground = float(np.median(sp[ring, 2])) if ring.sum() > 50 else float(np.percentile(loc[:, 2], 0.5))
loc[:, 2] -= ground
# metrics
def cen(names_):
    m = np.isin(L, [B[n] for n in names_]); return loc[m].mean(0) if m.sum() else None
fwh, bwh = cen(['front-wheel']), cen(['back-wheel'])
wheelbase = float(fwh[0] - bwh[0])
side = np.isin(L, [B[b] for b in ('front-door', 'back-door', 'fender', 'quarter-panel')])
width = float(np.percentile(loc[side & (loc[:, 1] > 0), 1], 95) - np.percentile(loc[side & (loc[:, 1] < 0), 1], 5))
width99 = float(np.percentile(loc[side & (loc[:, 1] > 0), 1], 99) - np.percentile(loc[side & (loc[:, 1] < 0), 1], 1))
sym = float(np.median(loc[side & (loc[:, 1] > 0), 1]) + np.median(loc[side & (loc[:, 1] < 0), 1]))
height = float(np.percentile(loc[rf, 2], 99))
# panel thickness: RMS distance to local plane in 15 cm neighbourhoods on hood, roof, doors
th = []
for nm in ('hood', 'roof', 'front-door', 'back-door', 'trunk'):
    Q = loc[L == B[nm]]
    if len(Q) < 200: continue
    tr = cKDTree(Q); rng = np.random.default_rng(0)
    for i in rng.choice(len(Q), min(150, len(Q)), replace=False):
        nb = tr.query_ball_point(Q[i], 0.075)
        if len(nb) < 15: continue
        A = Q[nb] - Q[nb].mean(0); sv = np.linalg.svd(A, compute_uv=False); th.append(sv[-1] / np.sqrt(len(nb)))
thick = float(np.median(th) * 1000)
res = dict(params=P, views=len(views), strict_points=int(n_strict), align_err=round(align_err, 4), points=int(len(loc)),
           wheelbase=round(wheelbase, 3), width=round(width, 3), height=round(height, 3),
           err_wheelbase_mm=round((wheelbase - PUB['wheelbase']) * 1000), err_width_mm=round((width - PUB['width']) * 1000),
           err_height_mm=round((height - PUB['height']) * 1000), width99=round(width99, 3), side_offset_mm=round(sym * 500), thickness_mm=round(thick, 1),
           agreement=round(float(AG.mean()), 3), sec=round(time.time() - t0))
res['score_mm'] = round((abs(res['err_wheelbase_mm']) + abs(res['err_width_mm']) + abs(res['err_height_mm'])) / 3, 1)
print(json.dumps(res))
if SAVE:
    np.savez('car_parts.npz', X=loc.astype(np.float32), C=C, L=L, NV=NV, base=np.array(base), scale=scale)
