import json
import os
from pathlib import Path
import signal
import site
import sys
import tempfile

site.addsitedir(f'{os.environ["TOOLBOX_CACHE"]}/cedirnet-stem/src')

import mlflow
from mlflow.entities import RunStatus
import modelargs
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

from criterions import get_criterion
from datasets import get_centerdir_dataset
from models import get_center_model, get_model
from utils.evaluation.center_with_shape import CenterShapeEval
from utils.utils import tensor_mask_to_ids, variable_len_collate
from utils.visualize import get_visualizer

from inference.processing import CenterDirProcesser

from base_config import NUM_VECTOR_FIELDS, get_args
from checkpoint import load_compatible_model_state, safe_torch_load


def _parallel(module, device):
    return torch.nn.DataParallel(module.to(device), device_ids=[0])


def _load_model_state(model, state):
    skipped = load_compatible_model_state(model, state)
    if skipped:
        print(
            "Warning: ignored checkpoint tensors not used by the point+radius "
            f"adaptation: {len(skipped)}"
        )


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

def load_split(args, split):
    d_opts = args["dataset"]
    dataset, center_groundtruth = get_centerdir_dataset(d_opts["name"], { **d_opts["kwargs"], "split": split }, d_opts["centerdir_gt_opts"])
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=d_opts["batch_size"],
        shuffle=d_opts.get("shuffle", False) and split == "train",
        drop_last=False,
        num_workers=d_opts["workers"],
        pin_memory=args["cuda"],
        collate_fn=variable_len_collate,
    )
    return loader, center_groundtruth

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args["cuda"] else "cpu")

    def initialize(self):
        args = self.args
        self.loader_train, center_groundtruth = load_split(args, "train")
        self.loader_val, center_groundtruth = load_split(args, "val")
        try:
            self.loader_test, _ = load_split(args, "test")
        except:
            self.loader_test = None

        model = get_model(args["model"]["name"], args["model"]["kwargs"])
        model.init_output(NUM_VECTOR_FIELDS)
        center_model = get_center_model(
            args["center_model"]["name"],
            args["center_model"]["kwargs"],
            is_learnable=args["center_model"]["use_learnable_center_estimation"],
        )
        center_model.init_output(NUM_VECTOR_FIELDS)
        criterion = get_criterion(
            args["loss_type"], args["loss_opts"], model, center_model
        )

        self.model = _parallel(model, self.device)
        self.center_model = _parallel(center_model, self.device)
        self.criterion = _parallel(criterion, self.device)
        self.center_groundtruth = _parallel(center_groundtruth, self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args["model"]["lr"],
            weight_decay=args["model"]["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=args["model"]["lambda_scheduler_fn"](args),
        )

        if args.get("pretrained_model_path"):
            state = safe_torch_load(
                args["pretrained_model_path"], map_location=self.device
            )
            _load_model_state(self.model, state)
            if state.get("center_model_state_dict"):
                _load_center_state(self.center_model, state)

        if args.get("pretrained_center_model_path"):
            state = safe_torch_load(
                args["pretrained_center_model_path"], map_location=self.device
            )
            _load_center_state(self.center_model, state)

    def train_epoch(self, epoch):
        self.model.train()
        self.center_model.train()
        losses_epoch = []
        iterator = tqdm(
            self.loader_train,
            desc=f"{epoch + 1}/{self.args['n_epochs']}",
            dynamic_ncols=True,
        )

        for sample in iterator:
            batch_size = sample["image"].shape[0]
            sample = self.center_groundtruth(
                sample, torch.arange(batch_size, dtype=torch.int32)
            )
            instances = sample["instance"].squeeze(1)
            ignore = sample.get("ignore")
            ignore_mask = ignore > 0 if ignore is not None else None
            difficult = (
                (((ignore & 8) | (ignore & 2)) > 0).squeeze(1)
                if ignore is not None
                else torch.zeros_like(instances)
            )

            self.optimizer.zero_grad()
            self.center_model.zero_grad(set_to_none=True)
            output = self.model(sample["image"])
            center_output = self.center_model(output, **sample)
            center_pred = center_output["center_pred"]
            center_heatmap = center_output["center_heatmap"]
            losses = self.criterion(
                center_output["output"],
                sample,
                centerdir_responses=(center_pred, center_heatmap),
                centerdir_gt=sample["centerdir_groundtruth"],
                ignore_mask=ignore_mask,
                difficult_mask=difficult,
                reduction_dims=(1, 2, 3),
                epoch_percent=epoch / max(self.args["n_epochs"], 1),
                **self.args["loss_w"],
            )
            loss = losses[0].sum()
            loss.backward()
            self.optimizer.step()

            value = float(loss.detach().cpu())
            losses_epoch.append(value)
            iterator.set_postfix(loss=value)

        mean_loss = float(np.mean(losses_epoch))
        mlflow.log_metric("loss", mean_loss, step=epoch)
        return mean_loss

    def checkpoint(self, epoch):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "center_model_state_dict": self.center_model.state_dict(),
        }
        if (artifacts := os.getenv("MLFLOW_ARTIFACTS_DESTINATION")) and (run := mlflow.active_run()):
            filename = os.path.join(artifacts, run.info.experiment_id, run.info.run_id, "artifacts", "checkpoint.pth")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            torch.save(state, filename)
            modelargs.emit_action("Weights", f"mlflow-artifacts:/{run.info.experiment_id}/{run.info.run_id}/artifacts/checkpoint.pth")
        else:
            with tempfile.TemporaryDirectory() as directory:
                filename = os.path.join(directory, "checkpoint.pt")
                torch.save(state, filename)
                mlflow.log_artifact(filename)
                info = mlflow.active_run().info
                modelargs.emit_action("Weights", f"mlflow-artifacts:/{info.experiment_id}/{info.run_id}/artifacts/checkpoint.pth")

    @torch.no_grad()
    def eval(self, root, dataset_it, visualizer):
        os.makedirs(root, exist_ok=True)
        args = self.args

        shape_eval = CenterShapeEval(exp_name="")
        shape_eval.save_str = lambda: "" # monkeypatch to prevent additional subpath in dir name
        pipeline = CenterDirProcesser(self.model, [dict(name=None, checkpoint=None, model=self.center_model)], self.device)
        for im_index,(sample,result) in enumerate(pipeline(dataset_it, self.center_groundtruth)):

            im_shape = sample.get("im_shape", sample["image"].shape[-2:])

            if not (instances_ids := sample.get("instance_ids")):
                instances_ids = tensor_mask_to_ids(sample["instance"])

            centerdir_gt = sample.get("centerdir_groundtruth")
            gt_centers_dict = sample.get("center_dict")

            if (ignore_flags := sample.get("ignore")) is not None:
                # get difficult mask based on ignore flags (VALUE of 8 == difficult flag and VALUE of 2 == truncated flag )
                difficult = (((ignore_flags & 8) | (ignore_flags & 2)) > 0).squeeze()
            else:
                difficult = torch.sparse_coo_tensor(size=im_shape)

            if not (pred_mask_ids_ := result.get("pred_mask_ids")):
                assert (pred_mask_ := result.get("pred_mask")) is not None, "ERROR: cannot evaluate without 'pred_mask' or 'pred_mask_ids'"
                pred_mask_ids_ = tensor_mask_to_ids(pred_mask_)

            predictions_ = result["predictions"]
            scoring_fn = lambda scores: scores[:, 1] # scoring_index = {'center': 1}
            all_scores = predictions_[:, 2:] if len(predictions_) > 0 else []

            scores = scoring_fn(all_scores)
            selected = np.where(scores > 0.5)[0]
            attributes = { k: v[selected] for k, v in result.get("pred_attributes", {}).items() }
            predictions_data = { "center": predictions_[selected,:2], "all_scores": all_scores[selected], "score": scores[selected], **attributes }
            if len(pred_mask_ids_) > 0:
                predictions_data["mask_ids"] = [pred_mask_ids_.get(id+1, set()) for id in selected]
            predictions_data = {attr_name: dict(enumerate(attr_mat)) for attr_name,attr_mat in predictions_data.items()}

            centerdir_gt["image_px_in_nm"] = centerdir_gt.get("image_px_in_nm")
            shape_eval_res = shape_eval.add_image_prediction(sample["im_name"], im_index, im_shape, predictions_data,
                                                            instances_ids, gt_centers_dict, difficult, centerdir_gt,
                                                            return_matched_gt_idx=True)
            
            gt_missed, pred_missed, pred_gt_match, pred_polygon, filename_suffix, pred_gt_match_idx = shape_eval_res
            predictions_data['polygon'] = pred_polygon

            
            plot_predictions_gt_match = [] if len(pred_gt_match) == 0 else np.concatenate((pred_gt_match[:, :1], pred_gt_match_idx[:, :1]), axis=1)
            visualizer(sample, result, predictions_data, plot_predictions_gt_match, difficult, visualizer.impath2name_fn(sample["im_name"]), root)

        metrics = shape_eval.calc_and_display_final_metrics(dataset_it, save_dir=root)
        for name, val in list(metrics.items()):
            if type(val) in [tuple, list, dict] or val is None or np.isnan(val):
                del metrics[name]
        return metrics

    def run(self, run):
        root = os.path.join(os.environ["MLFLOW_ARTIFACTS_DESTINATION"], run.info.experiment_id, run.info.run_id, "artifacts")
        visualizer = get_visualizer("CentersShapeVisualizeTest", { "to_file_only": True, "plot_only": ["image"] })
        for epoch in range(self.args["n_epochs"]):
            self.train_epoch(epoch)
            self.scheduler.step()
            if (epoch + 1) % self.args["save_interval"] == 0 or epoch + 1 == self.args["n_epochs"]:
                self.checkpoint(epoch)
            if (epoch + 1) % self.args["val_interval"] == 0 or epoch + 1 == self.args["n_epochs"]:
                metrics = self.eval(os.path.join(root, "val"), self.loader_val, visualizer)
                mlflow.log_metrics(metrics)
        if self.loader_test:
            self.eval(os.path.join(root, "test"), self.loader_test, visualizer)

def main():
    cmd_args = modelargs.parse("./model.json")
    if not (manifest_path := cmd_args.get("manifest")):
        raise ValueError("CeDiRNet-STEM training requires a manifest")
    manifest_path = Path(manifest_path).resolve()

    args = get_args(
        width=cmd_args["width"],
        height=cmd_args["height"],
        batch_size=cmd_args["batch_size"],
        workers=cmd_args["workers"],
    )

    with open(manifest_path) as f:
        manifest = json.load(f)
    
    if "data" in manifest:
        data_train, data_val = train_test_split(manifest["data"], train_size=0.7)
        manifest = {
            "train": manifest.get("train", []) + data_train,
            "val": manifest.get("val", []) + data_val,
            "test": manifest.get("test", []),
        }
    if len(manifest["train"]) == 0 or len(manifest["val"]) == 0:
        raise ValueError("Passed manifest.json does not contain 'data' or 'train'/'val' attributes.")

    args["dataset"]["kwargs"]["manifest"] = manifest_path
    args["n_epochs"] = cmd_args["epochs"]
    args["save_interval"] = cmd_args["save_interval"]
    args["val_interval"] = cmd_args["val_interval"]
    args["pretrained_model_path"] = cmd_args.get("model") or None
    args["model"]["kwargs"]["pretrained"] = not bool(args["pretrained_model_path"])

    default_localisation = os.path.join(
        os.environ["TOOLBOX_CACHE"], "cedirnet-stem", "localization_checkpoint.pth"
    )
    args["pretrained_center_model_path"] = (
        cmd_args.get("localisation") or default_localisation
    )

    mlflow.set_tracking_uri("http://localhost:8081")
    mlflow.set_experiment("CeDiRNet-STEM")
    with mlflow.start_run(run_name=cmd_args.get("name")) as run:
        def handler(_signal, _frame):
            mlflow.end_run(RunStatus.to_string(RunStatus.KILLED))
            sys.exit(0)

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
        modelargs.emit_action("Experiment", run.info.experiment_id)
        modelargs.emit_action("Run", run.info.run_id)
        mlflow.log_params(json.loads(json.dumps(args, default=lambda _: "<callable>")))

        args["dataset"]["kwargs"]["manifest"] = manifest
        args["dataset"]["kwargs"]["root"] = str(manifest_path.parent)
        trainer = Trainer(args)
        trainer.initialize()
        trainer.run(run)


if __name__ == "__main__":
    main()
