import os
import site
if not (INSTALL_DIR := os.getenv("INSTALL_DIR")):
    raise ValueError("Model definition requires newer toolbox version")
site.addsitedir(INSTALL_DIR)
import sys
import tempfile

from dataclasses import dataclass, asdict
import copy
import json
from pathlib import Path
import signal

import mlflow
from mlflow.entities import RunStatus
from mlflow.data.meta_dataset import MetaDataset
from mlflow.data.dataset_source_registry import resolve_dataset_source

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import cv2

import albumentations as A
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import Compose
from pytorch_lightning import LightningDataModule, seed_everything

from torchmetrics import AveragePrecision, Metric
from anomalib.metrics import AUROC, AUPRO

from model.supersimplenet import SuperSimpleNet

from common.visualizer import Visualizer
from common.loss import focal_loss

from anomalib.data.utils import LabelName, Split

from datamodules.base import Supervision
from datamodules.base.datamodule import SSNDataModule, InputNormalizationMethod
from datamodules.base.dataset import SSNDataset

import modelargs

class GenericDataset(SSNDataset):

    def __init__(
        self,
        manifest,
        root: Path,
        supervision: Supervision,
        transform: Compose,
        split: Split,
        flips: bool,
        normal_flips: bool,
        dilate: int | None = None,
        dt: tuple[int, int] | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(
            transform=transform,
            root=root,
            split=split,
            flips=flips,
            normal_flips=normal_flips,
            supervision=supervision,
            debug=debug,
        )
        self.manifest = manifest
        self.vflip = A.VerticalFlip(always_apply=True, p=1.0)
        self.hflip = A.HorizontalFlip(always_apply=True, p=1.0)
        self.rotate = A.Rotate(limit=(180, 180), always_apply=True, p=1.0)

    def make_dataset(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        samples = [{
            "image_path": str(self.root / f["image_path"]),
            "label_index": LabelName.NORMAL if f["label"] == "normal" else LabelName.ABNORMAL,
            **(dict(mask_path=str(self.root / f["mask_path"]), is_segmented=True) if "mask_path" in f else dict(mask_path="", is_segmented=False))
        } for f in self.manifest[self.split.value]]

        normal = [x for x in samples if x["label_index"] == LabelName.NORMAL]
        if len(normal) == 0 and split != "test":
            raise ValueError("Cannot start training without any normal samples")
        return pd.DataFrame(normal), pd.DataFrame([x for x in samples if x["label_index"] == LabelName.ABNORMAL])
    
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["index"] = index
        return item

class Generic(SSNDataModule):

    def __init__(
        self,
        manifest,
        root: Path,
        image_size: tuple[int, int],
        supervision: Supervision = Supervision.FULLY_SUPERVISED,
        normalization: str | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 0,
        seed: int | None = None,
        flips: bool = False,
        normal_flips: bool = False,
        dilate: int | None = None,
        dt: tuple[int, int] | None = None,
        debug: bool = False,
    ) -> None:

        super().__init__(
            root=root,
            image_size=image_size,
            supervision=supervision,
            normalization=normalization,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            seed=seed,
            flips=flips,
        )

        self.train_data = GenericDataset(
            manifest=manifest,
            transform=self.transform_train,
            split=Split.TRAIN,
            root=root,
            flips=flips,
            normal_flips=normal_flips,
            supervision=supervision,
            dilate=dilate,
            dt=dt,
            debug=debug,
        )
        self.val_data = GenericDataset(
            manifest=manifest,
            transform=self.transform_eval,
            split=Split.VAL,
            root=root,
            flips=flips,
            normal_flips=False,
            supervision=supervision,
            debug=debug,
        )
        self.test_data = GenericDataset(
            manifest=manifest,
            transform=self.transform_eval,
            split=Split.TEST,
            root=root,
            flips=flips,
            normal_flips=False,
            supervision=supervision,
            debug=debug,
        )

@dataclass
class Results:
    """Class for keeping track of evaluation scores."""
    anomaly_map: list[np.array]
    gt_mask: list[np.array]
    score: list[float]
    seg_score: list[float]
    label: list[float]
    image_path: list[str]
    mask_path: list[str]

    def __getitem__(self, k):
        return getattr(self, k)

def visualize(save_dir, image_path, image, gt_mask, anomaly_map, score, seg_score, label):
    anomaly_map = anomaly_map.squeeze()
    h, w = anomaly_map.shape
    # images are normed to [0, 1] so we cut off at 0.5
    pred_mask = anomaly_map >= 0.5
    gt_label = "Anomalous" if label.item() else "Normal"

    # adjust height and width as 256 as baseline
    fig, plots = plt.subplots(1, 5, figsize=(w / 256 * 12, h / 256 * 4))
    fig.tight_layout(rect=[0,0,1,0.9])

    panels = [
        (image, {}, "Image"),
        (gt_mask, dict(vmin=0, vmax=1), f"Ground truth.\n{gt_label}"),
        (pred_mask, {}, "Predicted mask"),
        (anomaly_map, {}, f"Anomaly map\nNorm."),
        (anomaly_map, dict(vmin=0, vmax=1), f"Anomaly map.\nScore: {round(score.item(), 4)}\nSScore: {round(seg_score.item(), 4)}"),
    ]
    for plot, (im, kwargs, title) in zip(plots, panels):
        plot.axis("off")
        plot.imshow(im, **kwargs)
        plot.set_title(title)

    total_path = save_dir / gt_label
    total_path.mkdir(exist_ok=True, parents=True)

    subdir_name = Path(image_path).parent.name
    plot_name = f"{subdir_name}_{Path(image_path).stem}.png"
    mlflow.log_figure(fig, total_path / plot_name)
    plt.close("all")

def train(
    model: SuperSimpleNet,
    config,
    datamodule: LightningDataModule,
    device: str,
    image_metrics: dict[str, Metric],
    pixel_metrics: dict[str, Metric],
    th: float = 0.5,
):
    epochs = config["epochs"]
    model.to(device)
    optimizer, scheduler = model.get_optimizers()

    model.train()
    train_loader = datamodule.train_dataloader()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        with tqdm(
            total=len(train_loader),
            desc=str(epoch) + "/" + str(epochs),
            miniters=int(1),
            unit="batch",
        ) as prog_bar:
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()

                image_batch = batch["image"].to(device)

                # best downsampling proposed by DestSeg
                mask = batch["mask"].to(device).type(torch.float32)
                mask = F.interpolate(
                    mask.unsqueeze(1),
                    size=(model.fh, model.fw),
                    mode="bilinear",
                    align_corners=False,
                )
                mask = torch.where(
                    mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
                )

                label = batch["label"].to(device).type(torch.float32)

                anomaly_map, score, mask, label = model.forward(
                    image_batch, mask, label
                )

                # adjusted truncated l1: mask + flipped sign (ano->pos, good->neg)
                normal_scores = anomaly_map[mask == 0]
                anomalous_scores = anomaly_map[mask > 0]
                true_loss = torch.clip(normal_scores + th, min=0)
                fake_loss = torch.clip(-anomalous_scores + th, min=0)

                if len(true_loss):
                    true_loss = true_loss.mean()
                else:
                    true_loss = 0
                if len(fake_loss):
                    fake_loss = fake_loss.mean()
                else:
                    fake_loss = 0

                loss = (
                    true_loss
                    + fake_loss
                    + focal_loss(torch.sigmoid(anomaly_map), mask)
                    + focal_loss(torch.sigmoid(score), label)
                )

                loss.backward()

                if config["clip_grad"]:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                else:
                    norm = None

                optimizer.step()

                total_loss += loss.detach().cpu().item()

                output = {
                    "batch_loss": np.round(loss.data.cpu().detach().numpy(), 5),
                    "avg_loss": np.round(total_loss / (i + 1), 5),
                }
                if norm is not None:
                    output["norm"] = norm

                prog_bar.set_postfix(**output)
                prog_bar.update(1)

        if (epoch + 1) % config["eval_step_size"] == 0:
            splits = [("train", datamodule.train_dataloader), ("val", datamodule.val_dataloader), ("test", datamodule.test_dataloader)]
            for split, loader_fn in splits:
                if split in config["eval_splits"]:
                    results = eval(
                        save_dir=Path("visualisations") / str(epoch) / split,
                        model=model,
                        loader=loader_fn(),
                        device=device,
                        image_metrics=image_metrics,
                        pixel_metrics=pixel_metrics,
                        normalize=True,
                        display_size=config["display_size"],
                    )
                    mlflow.log_metrics({ f"{split}/{k}": v for k, v in results.items() }, epoch)
        mlflow.log_metrics({ f"train/{k}": v for k, v in output.items() }, epoch)
        scheduler.step()


@torch.no_grad()
def eval(
    save_dir: Path,
    model: SuperSimpleNet,
    loader,
    device: str,
    image_metrics: dict[str, Metric],
    pixel_metrics: dict[str, Metric],
    normalize: bool = True,
    display_size: float | int = 5,
):
    model.to(device)
    model.eval()

    # for anomaly map max as image score
    seg_image_metrics = {}

    for m_name, metric in image_metrics.items():
        metric.cpu()
        metric.reset()

        seg_image_metrics[f"seg-{m_name}"] = copy.deepcopy(metric)

    for metric in pixel_metrics.values():
        metric.cpu()
        metric.reset()

    results = {
        "anomaly_map": [],
        "gt_mask": [],
        "score": [],
        "seg_score": [],
        "label": [],
        "image_path": [],
        "mask_path": [],
    }
    indices = []
    for batch in tqdm(loader, position=0, leave=True, desc="eval"):
        image_batch = batch["image"].to(device)
        anomaly_map, anomaly_score = model.forward(image_batch)

        anomaly_map = anomaly_map.detach().cpu()
        anomaly_score = anomaly_score.detach().cpu()

        results["anomaly_map"].append(anomaly_map.detach().cpu())
        results["gt_mask"].append(batch["mask"].detach().cpu())

        results["score"].append(torch.sigmoid(anomaly_score))
        results["seg_score"].append(
            anomaly_map.reshape(anomaly_map.shape[0], -1).max(dim=1).values
        )
        results["label"].append(batch["label"].detach().cpu())

        results["image_path"].extend(batch["image_path"])
        results["mask_path"].extend(batch["mask_path"])
        indices.extend(batch["index"])

    results = Results(
        anomaly_map = torch.cat(results["anomaly_map"]),
        score = torch.cat(results["score"]),
        seg_score = torch.cat(results["seg_score"]),
        gt_mask = torch.cat(results["gt_mask"]).type(torch.int8),
        label = torch.cat(results["label"]),
        image_path = results["image_path"],
        mask_path = results["mask_path"],
    )

    # normalize
    if normalize:
        results.anomaly_map = (
            results.anomaly_map - results.anomaly_map.flatten().min()
        ) / (
            results.anomaly_map.flatten().max()
            - results.anomaly_map.flatten().min()
        )
        results.score = (results.score - results.score.min()) / (
            results.score.max() - results.score.min()
        )
        results.seg_score = (results.seg_score - results.seg_score.min()) / (
            results.seg_score.max() - results.seg_score.min()
        )

    results_dict = {}
    for name, metric in image_metrics.items():
        if not name.startswith("AP-"):
            metric.update(results)
        else:
            metric.update(results.score, results.label)
        results_dict[name] = metric.to(device).compute().item()
        metric.to("cpu")

    for name, metric in pixel_metrics.items():
        try:
            # avoid nan in early stages
            am = results.anomaly_map
            am[am != am] = 0
            results.anomaly_map = am

            if not name.startswith("AP-"):
                metric.update(results)
            else:
                metric.update(results.anomaly_map.squeeze(1), results.gt_mask)
            results_dict[name] = metric.to(device).compute().item()
        except RuntimeError:
            # AUPRO in some cases with early predictions crashes cuda, so just skip it in that case
            results_dict[name] = 0
        metric.to("cpu")

    n = display_size if type(display_size) == int else int(display_size * len(loader.dataset))
    indices = { results.image_path[i]: (i, idx) for i, idx in enumerate(indices) }
    for image_path, (i, idx) in tqdm(indices.items(), desc="visualise"):
        im = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        if flip := loader.dataset.get_flip_augmentation(idx):
            im = flip(image=im)["image"]
        visualize(save_dir, image_path, im, *[getattr(results, prop)[i] for prop in ["gt_mask", "anomaly_map", "score", "seg_score", "label"]])
    score_dict = {}
    # save both segscore and score to json
    for img_path, score, seg_score, label in zip(
        results.image_path,
        results.score,
        results.seg_score,
        results.label,
    ):
        kind = "bad" if label == 1 else "good"
        if kind not in score_dict:
            score_dict[kind] = {}
        score_dict[kind][Path(img_path).stem] = {
            "score": score.item(),
            "seg_score": seg_score.item(),
        }

    mlflow.log_text(json.dumps(score_dict), save_dir / "image_scores.json")
    mlflow.log_text(json.dumps(results_dict), save_dir / "results.json")

    return results_dict

def train_and_eval(model, datamodule, config, device):
    mlflow.set_tracking_uri("http://localhost:8081")
    mlflow.set_experiment("SuperSimpleNet")
    dataset = MetaDataset(source=resolve_dataset_source(config["manifest"]), name=config["manifest"].removesuffix("/manifest.json"))
    with mlflow.start_run(run_name=config.get("name")) as run:
        def handler(sig, frame):
            mlflow.end_run(RunStatus.to_string(RunStatus.KILLED))
            sys.exit(0)
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

        modelargs.emit_action("Experiment", run.info.experiment_id)
        modelargs.emit_action("Run", run.info.run_id)
        mlflow.create_external_model(name="SuperSimpleNet", params=config)
        mlflow.log_params(config)
        mlflow.log_input(dataset)
        args = {
            "model": model,
            "device": device,
            "image_metrics": {
                "I-AUROC": AUROC(fields=["score", "label"]), # , prefix="image_"
                "AP-det": AveragePrecision(task="binary"),
            },
            "pixel_metrics": {
                "P-AUROC": AUROC(fields=["anomaly_map", "gt_mask"]), # , prefix="pixel_"
                "AUPRO": AUPRO(fields=["anomaly_map", "gt_mask"]), # , prefix="pixel_"
                "AP-loc": AveragePrecision(task="binary"),
            },
        }

        train(**args, datamodule=datamodule, config=config)

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            model.save_model(p)
            mlflow.log_artifact(p / "weights.pt")
            modelargs.emit_action("Weights", f"mlflow-artifacts:/{run.info.experiment_id}/{run.info.run_id}/artifacts/weights.pt")
        
        test_loader = datamodule.test_dataloader()
        if len(test_loader) > 0:
            eval(save_dir=Path("visualisations") / str(config['epochs'] - 1) / split, **args, loader=test_loader, normalize=True, display_size=config["display_size"])

if __name__ == "__main__":
    base_config = modelargs.parse('./model.json')

    manifest_path = Path(base_config["manifest"]).absolute()
    root = manifest_path.parent
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    if "data" in manifest:
        data_train, data_val = train_test_split(manifest["data"], train_size=0.7, stratify=[x.get("label", "") for x in manifest["data"]])
        manifest = {
            "train": manifest.get("train", []) + data_train,
            "val": manifest.get("val", []) + data_val,
            "test": manifest.get("test", []),
        }
    if len(manifest["train"]) == 0 or len(manifest["val"]) == 0:
        raise ValueError("Passed manifest.json does not contain 'data' or 'train'/'val' attributes.")

    unsupervised = True
    weakly_supervised = True
    fully_supervised = True
    for split in ["train", "val", "test"]:
        old_len = len(manifest[split])
        manifest[split] = [x for x in manifest[split] if "label" in x]
        if len(manifest[split]) != old_len:
            print(f"Warning: dropped {old_len - len(manifest[split])} items in split '{split}' due to missing label")

        for i,item in enumerate(manifest[split]):
            if "mask_path" in item:
                weakly_supervised = False
            if item["label"] != "normal":
                unsupervised = False
                if "mask_path" not in item:
                    fully_supervised = False

    if unsupervised:
        supervision = Supervision.UNSUPERVISED
    elif weakly_supervised:
        supervision = Supervision.WEAKLY_SUPERVISED
    elif fully_supervised:
        supervision = Supervision.FULLY_SUPERVISED
    else:
        supervision = Supervision.MIXED_SUPERVISION

    modelargs.emit_action("Info", json.dumps(dict(title="Supervision", description="Supervision refers to detected dataset structure. Click to learn more.", value=str(supervision), link="https://arxiv.org/pdf/2508.19060")))
    if supervision == Supervision.UNSUPERVISED:
        config = {
            **base_config,
            "num_workers": 8,
            "overlap": True,  # makes no difference, just faster if false to avoid computation
            "stop_grad": True,
            "clip_grad": False,
            "layers": [ "layer2", "layer3" ],
        }
    else:
        config = {
            **base_config,
            "num_workers": 1,
            "overlap": False,
            "perlin_thr": 0.6,
            "stop_grad": False,
            "clip_grad": True,
            "layers": [ "layer2", "layer3" ],
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(config["seed"], workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    image_size = [config["height"], config["width"]]
    model = SuperSimpleNet(image_size=image_size, config=config)
    if config['weights']:
        model.load_model(config['weights'])

    datamodule = Generic(
        manifest,
        root=root,
        image_size=image_size,
        train_batch_size=config["batch"],
        eval_batch_size=config["batch"],
        num_workers=config["num_workers"],
        supervision=supervision,
        seed=config["seed"],
        flips=config["flips"],
    )
    datamodule.setup()

    results = train_and_eval(
        model=model, datamodule=datamodule, config=config, device=device
    )
