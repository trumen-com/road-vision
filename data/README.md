# Dev labels

`candidate_labels.json` is our model's own output on the samples, in the ground-truth format.
It is a **starting point for annotation only**: open `tools/labeler.html`, load the sample
videos (the originals `C3896.MP4` ... or the 720p copies in `.proxy/`, same timestamps),
click **Import labels** and pick this file, then fix every event (delete false ones, add missed
ones, correct start/end to the official conventions). Export as `dev_labels.json` here.

`dev_labels.json` (after review) is what `tools/tune.py`, `tools/ablate.py` and
`evaluate.py --gt data/dev_labels.json` use.
