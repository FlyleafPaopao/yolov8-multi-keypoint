# Ultralytics YOLO 🚀, GPL-3.0 license
import sys
from copy import copy
import numpy as np

import torch
import torch.nn as nn
sys.path.append('../../../..')
from ultralytics.nn.tasks import DetectionModel
from ultralytics.yolo import v8
from ultralytics.yolo.data import build_dataloader
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import DEFAULT_CFG, colorstr
from ultralytics.yolo.utils.loss import BboxLoss
from ultralytics.yolo.utils.ops import xywh2xyxy
from ultralytics.yolo.utils.plotting import plot_images, plot_results
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.yolo.utils.torch_utils import de_parallel


# BaseTrainer python usage
class DetectionTrainer(BaseTrainer):

    def get_dataloader(self, dataset_path, batch_size, mode="train", rank=0):
        # TODO: manage splits differently
        # calculate stride - check if model is initialized
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return create_dataloader(path=dataset_path,
                                 imgsz=self.args.imgsz,
                                 batch_size=batch_size,
                                 stride=gs,
                                 hyp=vars(self.args),
                                 augment=mode == "train",
                                 cache=self.args.cache,
                                 pad=0 if mode == "train" else 0.5,
                                 rect=self.args.rect,
                                 rank=rank,
                                 workers=self.args.workers,
                                 close_mosaic=self.args.close_mosaic != 0,
                                 prefix=colorstr(f'{mode}: '),
                                 shuffle=mode == "train",
                                 seed=self.args.seed)[0] if self.args.v5loader else \
            build_dataloader(self.args, batch_size, img_path=dataset_path, stride=gs, rank=rank, mode=mode)[0] #yolo/data/build.py

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        batch["keypoints"] = batch["keypoints"].cpu()
        batch["batch_idx"] = batch["batch_idx"].cpu()
        batch["cls"] = batch["cls"].cpu()
        batch["bboxes"] = batch["bboxes"].cpu()

        return batch

    def set_model_attributes(self):
        # nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        # print(str(cfg))
        model = DetectionModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', '  lkptv_loss', ' lkpt_loss'
        return v8.detect.DetectionValidator(self.test_loader,
                                            save_dir=self.save_dir,
                                            logger=self.console,
                                            args=copy(self.args))

    def criterion(self, preds, batch):
        if not hasattr(self, 'compute_loss'):
            self.compute_loss = Loss(de_parallel(self.model))
        return self.compute_loss(preds, batch)

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        plot_images(images=batch["img"],
                    batch_idx=batch["batch_idx"],
                    cls=batch["cls"].squeeze(-1),
                    bboxes=batch["bboxes"],
                    points=batch["keypoints"],
                    paths=batch["im_file"],
                    fname=self.save_dir / f"train_batch{ni}.jpg")

    def plot_metrics(self):
        plot_results(file=self.csv)  # save results.png

class WingLoss(nn.Module):
    def __init__(self, w=10, e=2):
        super(WingLoss, self).__init__()
        self.w = w
        self.e = e
        self.C = self.w - self.w * np.log(1 + self.w / self.e)

    def forward(self, x, t, sigma=1):
        weight = torch.ones_like(t)
        weight[torch.where(t == -1)] = 0
        diff = weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < self.w).float()
        y = flag * self.w * torch.log(1 + abs_diff / self.e) + (1 - flag) * (abs_diff - self.C)
        return y.sum()

class KPTLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=1.0):
        super(KPTLoss, self).__init__()
        self.loss_fcn = WingLoss()  # nn.SmoothL1Loss(reduction='sum')
        self.alpha = alpha

    def forward(self, pred, truel, mask):
        loss = self.loss_fcn(pred * mask, truel * mask)
        return loss / (torch.sum(mask) + 10e-14)


# Criterion class for computing training losses
class Loss:

    def __init__(self, model):  # model must be de-paralleled


        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.na = 1
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.kpt_num = m.kpt_num
        self.no_kpt_num = m.no_kpt_num
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device
        self.kpt_label = True

        self.use_dfl = m.reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0, kpt_num=self.kpt_num)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        BCE_kptv = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h.obj_pw], device=device))
        self.BCE_kptv = BCE_kptv
        self.kptloss = KPTLoss()

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def preprocess2(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5 + self.kpt_num * 2, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5 + self.kpt_num * 2, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            if self.kpt_num != 0:
                scale_tensor_point = torch.zeros(2 * self.kpt_num, device=self.device)
                for i in range(self.kpt_num):
                    scale_tensor_point[[2*i,2*i+1]] = scale_tensor[[0,1]].float()
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
            out[..., 5:] = out[..., 5:].mul_(scale_tensor_point)
        return out




    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl
        batch["keypoints"] = batch["keypoints"].to(self.device).detach()
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores,k_points = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc, self.no_kpt_num), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        k_points = k_points.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        x_anchor = anchor_points[..., 0].unsqueeze(0).unsqueeze(-1).repeat(1, 1, self.kpt_num)
        y_anchor = anchor_points[..., 1].unsqueeze(0).unsqueeze(-1).repeat(1, 1, self.kpt_num)

        batch["keypoints"] = batch["keypoints"].cpu()
        batch["batch_idx"] = batch["batch_idx"].cpu()
        batch["cls"] = batch["cls"].cpu()
        batch["bboxes"] = batch["bboxes"].cpu()

        # targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"], batch["keypoints"].view(-1, self.kpt_num * 2)), 1)

        targets = self.preprocess2(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes, gt_points = targets.split((1, 4, 2*self.kpt_num), 2)  # cls, xyxy

        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, target_points, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype), (k_points.detach() * stride_tensor).type(gt_points.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes,gt_points, mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)
        target_points /= stride_tensor


        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
        # point loss
        if self.kpt_label:
            pkpt_x = k_points[..., ::3] * 2. - 0.5 + x_anchor
            pkpt_y = k_points[..., 1::3] * 2. - 0.5 + y_anchor
            pkpt_score = k_points[..., 2::3]
            # mask
            kpt_mask = (target_points[..., 0::2] != 0)
            loss[3] += self.BCE_kptv(pkpt_score[fg_mask], kpt_mask.float()[fg_mask])*20
            loss[4] += (self.kptloss(target_points[..., 0::2][fg_mask], pkpt_x[fg_mask], kpt_mask[fg_mask]) + self.kptloss(target_points[..., 1::2][fg_mask], pkpt_y[fg_mask],
                                                                                     kpt_mask[fg_mask])) *40


        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.cls
        loss[4] *= self.hyp.kpt

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl,lkptv, lkpt)


def train(cfg=DEFAULT_CFG, use_python=True): #cfg的yaml文件是ultralytics/yolo/cfg/default.yaml
    model = cfg.model or "yolov8s.pt"   #不需要修改yolo v8中的相关参数时可以使用
    model_yaml = "../../../../ultralytics/models/v8/yolov8s.yaml"  #需要修改yolo v8中的相关参数时使用
    data = cfg.data or "coco128.yaml"  # or yolo.ClassificationDataset("mnist")，在ultralytics/yolo/data/datasets
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device)

    if use_python:
        from ultralytics import YOLO
        YOLO(model_yaml).train(**args)
    else:
        trainer = DetectionTrainer(overrides=args)
        trainer.train()



if __name__ == "__main__":
    train()
