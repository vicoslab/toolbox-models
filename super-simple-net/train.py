import os
import site
site.addsitedir(f'{os.environ["TOOLBOX_CACHE"]}/super-simple-net')
import sys
import tempfile

from dataclasses import dataclass
import copy
import json
from pathlib import Path

import mlflow
from mlflow.entities import RunStatus

from tqdm import tqdm
import numpy as np
import pandas as pd

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

    def make_dataset(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        samples = [{
            "image_path": str(self.root / f["image_path"]),
            "label_index": LabelName.NORMAL if f["label"] == "normal" else LabelName.ABNORMAL,
            **(dict(mask_path=str(self.root / f["mask_path"]), is_segmented=True) if "mask_path" in f else dict(mask_path="", is_segmented=False))
        } for f in self.manifest[self.split.value]]

        return pd.DataFrame([x for x in samples if x["label_index"] == LabelName.NORMAL]), \
               pd.DataFrame([x for x in samples if x["label_index"] == LabelName.ABNORMAL])

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


def train(
    model: SuperSimpleNet,
    epochs: int,
    datamodule: LightningDataModule,
    device: str,
    image_metrics: dict[str, Metric],
    pixel_metrics: dict[str, Metric],
    th: float = 0.5,
    clip_grad: bool = True,
    eval_step_size: int = 4,
):
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

                if clip_grad:
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

            if (epoch + 1) % eval_step_size == 0:
                results = test(
                    model=model,
                    datamodule=datamodule,
                    device=device,
                    image_metrics=image_metrics,
                    pixel_metrics=pixel_metrics,
                    normalize=True,
                )
                mlflow.log_metrics({**results, **output}, epoch)
            else:
                mlflow.log_metrics(output, epoch)
        scheduler.step()


@torch.no_grad()
def test(
    model: SuperSimpleNet,
    datamodule: LightningDataModule,
    device: str,
    image_metrics: dict[str, Metric],
    pixel_metrics: dict[str, Metric],
    normalize: bool = True,
    image_save_path: Path = None,
    score_save_path: Path = None,
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

    test_loader = datamodule.test_dataloader()
    results = {
        "anomaly_map": [],
        "gt_mask": [],
        "score": [],
        "seg_score": [],
        "label": [],
        "image_path": [],
        "mask_path": [],
    }
    for batch in tqdm(test_loader, position=0, leave=True, desc="eval"):
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
                metric.update(results.anomaly_map, results.gt_mask.type(torch.float32))
            results_dict[name] = metric.to(device).compute().item()
        except RuntimeError:
            # AUPRO in some cases with early predictions crashes cuda, so just skip it in that case
            results_dict[name] = 0
        metric.to("cpu")

    Visualizer(Path("vis")).visualize(results)
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

    mlflow.log_text(json.dumps(score_dict), "image_scores.json")
    mlflow.log_text(json.dumps(results_dict), "results.json")

    return results_dict

def train_and_eval(model, datamodule, config, device):
    mlflow.set_tracking_uri("http://localhost:8081")
    mlflow.set_experiment("SuperSimpleNet")
    with mlflow.start_run(run_name=config.get("name")) as run:
        def handler(sig, frame):
            mlflow.end_run(RunStatus.to_string(RunStatus.KILLED))
            sys.exit(0)
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

        print(f"Experiment {run.info.experiment_id}: Run {run.info.run_id}")
        mlflow.log_params(config)
        args = {
            "model": model,
            "datamodule": datamodule,
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

        train(
            **args,
            epochs=config["epochs"],
            clip_grad=config["clip_grad"],
            eval_step_size=config["eval_step_size"],
        )

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            model.save_model(p)
            mlflow.log_artifact(p / "weights.pt")
        
        test(**args, normalize=True)

if __name__ == "__main__":
    base_config = modelargs.parse('./model.json')

    manifest_path = Path(base_config["manifest"]).absolute()
    root = manifest_path.parent
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    if not ("train" in manifest and "test" in manifest):
        if "data" in manifest:
            gen = np.random.default_rng(seed=3999924951)
            data = gen.permutation(manifest["data"])
            pivot = round(0.7*len(data))
            manifest = {
                "train": data[:pivot],
                "test": data[pivot:],
            }
        else:
            raise ValueError("Passed manifest.json does not contain 'data' or 'train'/'test' attributes.")
    
    implicit_normal = False
    fully_supervised = True
    weakly_supervised = True
    for split in ["train", "test"]:
        for i,item in enumerate(manifest[split]):
            if "label" not in item:
                implicit_normal = True
                manifest[split][i]["label"] = "normal"
            if "mask_path" in item:
                weakly_supervised = False
            elif item["label"] != "normal":
                fully_supervised = False
    
    if implicit_normal:
        print("Warning: used 'normal' as label for image because no image-level annotation was supplied")

    if fully_supervised:
        supervision = Supervision.FULLY_SUPERVISED
    elif weakly_supervised:
        supervision = Supervision.WEAKLY_SUPERVISED
    else:
        supervision = Supervision.MIXED_SUPERVISION

    if supervision != Supervision.FULLY_SUPERVISED:
        config = {
            **base_config,
            "num_workers": 8,
            "overlap": True,  # makes no difference, just faster if false to avoid computation
            "flips": False,  # makes no difference, just faster if false to avoid computation
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
            "flips": True,
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
