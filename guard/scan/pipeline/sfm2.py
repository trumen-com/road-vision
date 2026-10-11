import pycolmap, time, pathlib
t=time.time(); out=pathlib.Path('sfm2'); out.mkdir(exist_ok=True); db=out/'db.db'
if db.exists(): db.unlink()
eo=pycolmap.FeatureExtractionOptions(); eo.max_image_size=1600; eo.sift.max_num_features=8000; eo.num_threads=2
pycolmap.extract_features(db,'imgs2',camera_mode=pycolmap.CameraMode.PER_FOLDER,extraction_options=eo)
print('features',time.time()-t,flush=True)
pycolmap.match_exhaustive(db); print('matched',time.time()-t,flush=True)
mo=pycolmap.IncrementalPipelineOptions()
maps=pycolmap.incremental_mapping(db,'imgs2',out,options=mo)
best=max(maps.values(),key=lambda m:m.num_reg_images()); best.write(out)
print(best.summary()); print('done',time.time()-t)
