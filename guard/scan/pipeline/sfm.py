import pycolmap, time, pathlib
t=time.time(); out=pathlib.Path('sfm'); out.mkdir(exist_ok=True); db=out/'db.db'
if db.exists(): db.unlink()
eo=pycolmap.FeatureExtractionOptions(); eo.max_image_size=1600; eo.sift.max_num_features=6000; eo.num_threads=2
pycolmap.extract_features(db, 'imgs', camera_mode=pycolmap.CameraMode.PER_FOLDER if False else pycolmap.CameraMode.AUTO, extraction_options=eo)
print('features', time.time()-t, flush=True)
pycolmap.match_exhaustive(db)
print('matched', time.time()-t, flush=True)
maps=pycolmap.incremental_mapping(db, 'imgs', out)
for k,m in maps.items(): print(k, m.summary())
best=max(maps.values(), key=lambda m: m.num_reg_images()); best.write(out); best.export_PLY(str(out/'sparse.ply'))
print('done', time.time()-t, best.num_reg_images(), best.num_points3D())
