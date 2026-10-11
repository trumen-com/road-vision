import numpy as np, json, base64, os, trimesh
from urllib.parse import quote_plus
z=np.load('car_parts.npz'); X,C,L,NV,base=z['X'],z['C'],z['L'],z['NV'],list(z['base'])
CAR="Toyota Camry 2025 (XV80) Hybrid"
pretty={'back-bumper':'Rear bumper','back-windshield':'Rear windshield','front-bumper':'Front bumper','grille':'Front grille','hood':'Hood (bonnet)',
 'back-door':'Rear door','back-wheel':'Rear wheel','back-window':'Rear door glass','fender':'Front fender','front-door':'Front door','front-wheel':'Front wheel',
 'front-window':'Front door glass','headlight':'Headlight','mirror':'Side mirror','quarter-panel':'Rear quarter panel','rocker-panel':'Rocker panel (side sill)',
 'tail-light':'Tail light','roof':'Roof panel','trunk':'Trunk lid','windshield':'Windshield'}
search_term={'front-wheel':'wheel rim','back-wheel':'wheel rim','back-window':'rear door window glass','front-window':'front door window glass','rocker-panel':'rocker panel side skirt'}
paired={'back-door':.3,'back-wheel':.3,'back-window':.3,'fender':.3,'front-door':.3,'front-wheel':.3,'front-window':.3,'headlight':.08,'mirror':.4,'quarter-panel':.3,'rocker-panel':.3,'tail-light':.08}
from scipy import ndimage
def largest_cluster(g, cell=0.06):
    idx=np.where(g)[0]; P=X[idx]; k=np.floor((P-P.min(0))/cell).astype(int)
    grid=np.zeros(k.max(0)+1,bool); grid[tuple(k.T)]=True
    lab,n=ndimage.label(ndimage.binary_dilation(grid,iterations=1))
    if n<=1: return g
    pl=lab[tuple(k.T)]; best=np.bincount(pl)[1:].argmax()+1
    out=np.zeros_like(g); out[idx[pl==best]]=True; return out
parts=[]; pid=np.full(len(X),-1)
for i,b in enumerate(base):
    if b=='license-plate': continue
    m=L==i
    if m.sum()<150: continue
    groups=[('',m)] if b not in paired else [('Left',m&(X[:,1]>paired[b])),('Right',m&(X[:,1]<-paired[b]))]
    for side,g in groups:
        if g.sum()<150: continue
        g=largest_cluster(g)
        P=X[g]; lo,hi=np.percentile(P,2,0),np.percentile(P,98,0)
        name=(side+' '+pretty[b].lower()) if side else pretty[b]
        name=name[0].upper()+name[1:]
        q=f"{CAR} {side.lower()+' ' if side else ''}{search_term.get(b,pretty[b].split(' (')[0].lower())}"
        parts.append({'id':len(parts),'key':(side.lower()+'_' if side else '')+b,'name':name,'side':side.lower() or 'center','points':int(g.sum()),
            'views':int(np.median(NV[g])),'size_m':[round(float(v),2) for v in hi-lo],'center_m':[round(float(v),2) for v in P.mean(0)],
            'search':{'google':'https://www.google.com/search?q='+quote_plus(q+' OEM part number'),
                      'ebay':'https://www.ebay.com/sch/i.html?_nkw='+quote_plus(q),
                      'olx_uz':'https://www.olx.uz/list/q-'+quote_plus(f"Camry 75 {pretty[b].split(' (')[0].lower()}").replace('+','-')+'/',
                      'amazon':'https://www.amazon.com/s?k='+quote_plus(q)}, 'query':q})
        pid[g]=parts[-1]['id']
keep=pid>=0
X,C,P=X[keep],C[keep],pid[keep]
print(len(parts),'parts',keep.sum(),'points')
for p in parts: print(p['id'],p['name'],p['points'],p['size_m'])
T=json.load(open('tune_final.json'))
checks=[{'name':'Wheelbase','scan_m':T['wheelbase'],'published_m':2.825},{'name':'Body width (no mirrors)','scan_m':T['width'],'published_m':1.84},{'name':'Height','scan_m':T['height'],'published_m':1.445}]
meta={'vehicle':CAR,'identified_by':'badge and wrap text visible in the photos (set by hand in this demo)','units':'metres',
      'scale_from':'published overall length 4,915 mm','source':'95 views: 29 s walk-around video (3 fps) + 9 photos',
      'checks':checks,'surface_thickness_mm':T['thickness_mm'],'parts':parts}
os.makedirs('out',exist_ok=True); json.dump(meta,open('out/parts.json','w'),indent=1)
# glTF: one named point-cloud node per part
scene=trimesh.Scene()
for p in parts:
    m=P==p['id']; pc=trimesh.PointCloud(X[m][:,[1,2,0]]*np.array([-1,1,1]), colors=np.c_[C[m],np.full(m.sum(),255)])
    scene.add_geometry(pc,node_name=p['key'],geom_name=p['key'])
scene.export('out/camry_parts.glb'); print('glb',os.path.getsize('out/camry_parts.glb')//1024,'KB')
# viewer payload, subsample to ~150k
rng=np.random.default_rng(0); s=rng.permutation(len(X))[:150000]
Xs,Cs,Ps=X[s],C[s],P[s]
S=float(np.abs(Xs).max())
q=np.round(Xs/S*32767).astype('<i2')
payload={'S':S,'n':len(s),'p':base64.b64encode(q.tobytes()).decode(),'c':base64.b64encode(Cs.astype(np.uint8).tobytes()).decode(),'l':base64.b64encode(Ps.astype(np.uint8).tobytes()).decode(),'meta':meta}
json.dump(payload,open('viewer_data.json','w')); print('payload KB',os.path.getsize('viewer_data.json')//1024)
