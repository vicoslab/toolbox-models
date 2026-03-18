## Models

A model is defined by a `model.json` file which should contains a JSON schema of an object, where the title and description describe the model, and parameters describe various parameters. There should only be a single parameter definition for each model, even if model supports different routines (such as training, inference and preannotation). The `manifest.json` is the source of truth shared by both the frontend ui and backend routines.

### Training
A model's training routine should expect a dataset defined by a `manifest.json` file, which contains an object with properties `train` and `test`, each of which is a list containing objects (object property naming conventions are not mandated, but can be something like `label`, `image_path`, `mask_path`).

### Contributing
Launching a standalone model container can help prevent rebuilds when developing, but you won't have access to the main container's filesystem, so you'll probably need another way of loading some weights if you need that.
```
# launch container
docker run --rm -it --network host --ipc host --device nvidia.com/gpu=all --entrypoint bash aibox-model-<model-name>

# running the inference worker manually (in the main docker)
uv run gunicorn --bind :9090 infer:app -- --epochs 1 --weights /opt/apps/mlflow/mlartifacts/1/705ce8125737449f918d345c2d8da5af/artifacts/weights.pt
```