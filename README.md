## Models

A model is defined by a `model.json` file which should contains a JSON schema of an object, where the title and description describe the model, and parameters describe various parameters. There should only be a single parameter definition for each model, even if model supports different routines (such as training, inference and preannotation). The `manifest.json` is the source of truth shared by both the frontend ui and backend routines.

### Training
A model's training routine should expect a dataset defined by a `manifest.json` file, which contains an object with properties `train` and `test`, each of which is a list containing objects (object property naming conventions are not mandated, but can be something like `label`, `image_path`, `mask_path`).

### Contributing
Launching a standalone model container can help prevent rebuilds when developing.
```
# launch container with persist dir to access weights from model training
docker run --rm -it --network host --ipc host --device nvidia.com/gpu=all --entrypoint bash --mount type=volume,src=toolbox-persist,dst=/persist toolbox-model-<model-name>

# running the inference worker (note: use -- after gunicorn args to pass args to model)
uv run gunicorn --bind :9090 infer:app -- --weights /path/to/model/weights.pt
```