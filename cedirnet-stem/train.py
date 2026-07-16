import json
import os
import signal
import site
import sys
import tempfile

site.addsitedir(f'{os.environ["TOOLBOX_CACHE"]}/cedirnet-stem/src')

import mlflow
import modelargs
import numpy as np
import torch
from mlflow.entities import RunStatus
from tqdm import tqdm

from criterions import get_criterion
from datasets import get_centerdir_dataset
from models import get_center_model, get_model
from utils.utils import variable_len_collate

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


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args["cuda"] else "cpu")

    def initialize(self):
        args = self.args
        dataset, center_groundtruth = get_centerdir_dataset(
            args["train_dataset"]["name"],
            args["train_dataset"]["kwargs"],
            args["train_dataset"]["centerdir_gt_opts"],
        )
        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args["train_dataset"]["batch_size"],
            shuffle=args["train_dataset"]["shuffle"],
            drop_last=False,
            num_workers=args["train_dataset"]["workers"],
            pin_memory=args["cuda"],
            collate_fn=variable_len_collate,
        )

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
            self.loader,
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
        artifacts = os.getenv("MLFLOW_ARTIFACTS_DESTINATION")
        run = mlflow.active_run()
        if artifacts and run:
            filename = os.path.join(
                artifacts,
                run.info.experiment_id,
                run.info.run_id,
                "artifacts",
                "checkpoint.pth",
            )
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            torch.save(state, filename)
        else:
            with tempfile.TemporaryDirectory() as directory:
                filename = os.path.join(directory, "checkpoint.pth")
                torch.save(state, filename)
                mlflow.log_artifact(filename)

    def run(self):
        for epoch in range(self.args["n_epochs"]):
            self.train_epoch(epoch)
            self.scheduler.step()
            if (epoch + 1) % self.args["save_interval"] == 0 or (
                epoch + 1 == self.args["n_epochs"]
            ):
                self.checkpoint(epoch)


def main():
    cmd_args = modelargs.parse("./model.json")
    if not cmd_args.get("manifest"):
        raise ValueError("CeDiRNet-STEM training requires a manifest")

    args = get_args(
        width=cmd_args["width"],
        height=cmd_args["height"],
        batch_size=cmd_args["batch_size"],
        workers=cmd_args["workers"],
    )
    args["train_dataset"]["kwargs"]["manifest"] = cmd_args["manifest"]
    args["n_epochs"] = cmd_args["epochs"]
    args["save_interval"] = cmd_args["save_interval"]
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
        print(f"Experiment {run.info.experiment_id}: Run {run.info.run_id}")
        mlflow.log_params(json.loads(json.dumps(args, default=lambda _: "<callable>")))

        trainer = Trainer(args)
        trainer.initialize()
        trainer.run()


if __name__ == "__main__":
    main()
