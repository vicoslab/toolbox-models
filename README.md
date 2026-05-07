## Models

A model is defined by a `model.json` file which should contains a JSON schema of an object, where the title and description describe the model, and parameters describe various parameters. There should only be a single parameter definition for each model, even if model supports different routines (such as training, inference and preannotation). The `manifest.json` is the source of truth shared by both the frontend ui and backend routines.

### Training
A model's training routine should expect a dataset defined by a `manifest.json` file, which contains an object with properties `train` and `test`, each of which is a list containing objects (object property naming conventions are not mandated, but can be something like `label`, `image_path`, `mask_path`).

### Inference
Model inference should support both cli and http (i.e. flask+label-studio) inference. `infer.py` should be structured such that any imports used only for http are in the else clause of the `__main__` guard, which should be the last block of code in the file. This way it is possible to delete the else branch and retain only the minimal inference example.
```
# running test inference
uv run infer.py my-input.png -- --weights /path/to/model/weights.pt

# running the inference worker (note: use -- after gunicorn args to pass args to model)
uv run gunicorn --bind :9090 infer:app -- --weights /path/to/model/weights.pt
```

### Contributing

Model installation is not done inside the docker image, but rather by a install script. Installations are done in the `TOOLBOX_CACHE` dir, which may be deleted at any point for any reason. You should fetch publicly accessible sources, where possible.
