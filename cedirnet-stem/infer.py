import os
import site
import sys
from typing import List, Sequence

site.addsitedir(f'{os.environ["TOOLBOX_CACHE"]}/cedirnet-stem/src')

import modelargs
import numpy as np
import torch
from PIL import Image, ImageDraw

from models import get_center_model, get_model

from annotations import load_stem_image
from base_config import NUM_VECTOR_FIELDS, get_args
from checkpoint import load_compatible_model_state, safe_torch_load
from results import label_studio_vector_result, restore_prediction


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CMD_ARGS = modelargs.parse("./model.json")
WIDTH = CMD_ARGS["width"]
HEIGHT = CMD_ARGS["height"]
SCORE_THRESHOLD = CMD_ARGS["score_threshold"]

ARGS = get_args(width=WIDTH, height=HEIGHT)
ARGS["model"]["kwargs"]["pretrained"] = False
MODEL = get_model(ARGS["model"]["name"], ARGS["model"]["kwargs"])
MODEL.init_output(NUM_VECTOR_FIELDS)
MODEL = torch.nn.DataParallel(MODEL.to(DEVICE), device_ids=[0])
WEIGHTS = CMD_ARGS["model"] or f'{os.environ["TOOLBOX_CACHE"]}/cedirnet-stem/stem_checkpoint.pt'

CENTER_MODEL = get_center_model(
    ARGS["center_model"]["name"],
    ARGS["center_model"]["kwargs"],
    is_learnable=ARGS["center_model"]["use_learnable_center_estimation"],
)
CENTER_MODEL.init_output(NUM_VECTOR_FIELDS)
CENTER_MODEL = torch.nn.DataParallel(CENTER_MODEL.to(DEVICE), device_ids=[0])


def _load_center_state(center_model, state):
    center_state = state.get("center_model_state_dict")
    if not center_state:
        raise ValueError("checkpoint does not contain center_model_state_dict")

    input_key = "module.instance_center_estimator.conv_start.0.weight"
    checkpoint_weights = center_state.get(input_key)
    if checkpoint_weights is not None:
        expected_weights = center_model.module.instance_center_estimator.conv_start[0].weight
        if checkpoint_weights.shape != expected_weights.shape:
            center_state = dict(center_state)
            center_state[input_key] = checkpoint_weights[:, : expected_weights.shape[1], :, :]
    center_model.load_state_dict(center_state, strict=False)


print(f'Loading CeDiRNet-STEM model from "{WEIGHTS}"')
STATE = safe_torch_load(WEIGHTS, map_location=DEVICE)
SKIPPED_MODEL_TENSORS = load_compatible_model_state(MODEL, STATE)
if SKIPPED_MODEL_TENSORS:
    print(
        "Warning: ignored checkpoint tensors not used by the point+radius "
        f"adaptation: {len(SKIPPED_MODEL_TENSORS)}"
    )

if CMD_ARGS.get("localisation"):
    print(f'Loading localisation model from "{CMD_ARGS["localisation"]}"')
    localization_state = safe_torch_load(
        CMD_ARGS["localisation"], map_location=DEVICE
    )
    _load_center_state(CENTER_MODEL, localization_state)
else:
    _load_center_state(CENTER_MODEL, STATE)

MODEL.eval()
CENTER_MODEL.eval()


def load(bf_source, haadf_source=None):
    return np.asarray(load_stem_image(bf_source, haadf_source))


def transform(images: Sequence[np.ndarray]):
    transformed = []
    for image in images:
        resized = Image.fromarray(image).resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)
        array = np.asarray(resized).transpose((2, 0, 1)).copy()
        transformed.append(torch.from_numpy(array).float())
    return torch.stack(transformed).to(DEVICE)


def predict(images: List[np.ndarray]):
    original_sizes = [(image.shape[1], image.shape[0]) for image in images]
    with torch.no_grad():
        output = MODEL(transform(images))
        center_output = CENTER_MODEL(output, detect_centers=True)

    center_predictions = center_output["center_pred"].detach().cpu().numpy()
    radius_predictions = (
        center_output["pred_attributes"]["shape_coef"].detach().cpu().numpy()
    )

    all_centers = []
    all_scores = []
    all_radii = []
    for predictions, radii, original_size in zip(
        center_predictions, radius_predictions, original_sizes
    ):
        valid_indices = np.flatnonzero(predictions[:, 0] == 1)
        if valid_indices.size:
            valid_indices = valid_indices[
                np.argsort(predictions[valid_indices, 4])[::-1]
            ]

        centers_image = []
        scores_image = []
        radii_image = []
        for index in valid_indices:
            score = float(predictions[index, 4])
            if score < SCORE_THRESHOLD:
                continue
            center, radius = restore_prediction(
                center=predictions[index, 1:3],
                radius=float(radii[index, 0]),
                network_size=(WIDTH, HEIGHT),
                original_size=original_size,
            )
            centers_image.append(list(center))
            scores_image.append(score)
            radii_image.append(radius)

        all_centers.append(centers_image)
        all_scores.append(scores_image)
        all_radii.append(radii_image)
    return all_centers, all_scores, all_radii


if __name__ == "__main__":
    haadf_source = sys.argv[2] if len(sys.argv) > 2 else None
    image = load(sys.argv[1], haadf_source)
    centers, scores, radii = predict([image])
    display_image = np.repeat(image[:, :, 1:2], 3, axis=2)
    canvas = Image.fromarray(display_image)
    draw = ImageDraw.Draw(canvas)
    for center, score, radius in zip(centers[0], scores[0], radii[0]):
        x = center[0] * canvas.width
        y = center[1] * canvas.height
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="lime", width=2)
        draw.text((x + radius, y), f"{score:.2f}", fill="lime")
    canvas.save("result.png")
else:
    from typing import Dict, Optional
    from uuid import uuid4

    from flask import request
    from label_studio_ml.api import init_app
    from label_studio_ml.model import LabelStudioMLBase
    from label_studio_ml.response import ModelResponse

    class CeDiRNetSTEM(LabelStudioMLBase):
        def get_results(
            self, centers, scores, radii, width, height, from_name, to_name, label
        ):
            results = [
                label_studio_vector_result(
                    center=center,
                    radius=radius,
                    score=score,
                    original_size=(width, height),
                    from_name=from_name,
                    to_name=to_name,
                    label=label,
                    result_id=str(uuid4())[:8],
                )
                for center, score, radius in zip(centers, scores, radii)
            ]
            return {
                "result": results,
                "model_version": self.get("model_version") or "CeDiRNet-STEM",
                "score": sum(scores) / max(len(scores), 1),
            }

        def predict(
            self,
            tasks: List[Dict],
            context: Optional[Dict] = None,
            **kwargs,
        ) -> ModelResponse:
            from_name, to_name, value = self.get_first_tag_occurence(
                "Labels", "Image"
            )
            labels = next(
                (
                    tag["labels"]
                    for tag in self.parsed_label_config.values()
                    if tag.get("labels")
                ),
                ["Particle"],
            )
            # todo: fix this
            images = [
                load(self.get_local_path(task["data"][value], task_id=task["id"]))
                for task in tasks
            ]
            centers, scores, radii = predict(images)
            predictions = []
            for image, image_centers, image_scores, image_radii in zip(
                images, centers, scores, radii
            ):
                height, width = image.shape[:2]
                predictions.append(
                    self.get_results(
                        image_centers,
                        image_scores,
                        image_radii,
                        width,
                        height,
                        from_name,
                        to_name,
                        labels[0],
                    )
                )
            return ModelResponse(predictions=predictions)

    app = init_app(model_class=CeDiRNetSTEM)

    @app.route("/infer", methods=["POST"])
    def infer():
        if "images" not in request.files:
            return {"centers": [], "scores": [], "radii": []}, 400
        images = request.files.getlist("images")
        if len(images) % 2 != 0:
            return {"error": "Input images must be pairs of STEM images (first BF, then HAADF)"}, 400
        images = [load(*images[i:i+2]) for i in range(0,len(images),2)]
        centers, scores, radii = predict(images)
        return {"centers": centers, "scores": scores, "radii": radii}
