# Ultralytics YOLO 🚀, GPL-3.0 license

import os
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.append('../../../..')
from ultralytics.yolo.data import build_dataloader
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import DEFAULT_CFG, colorstr, ops, yaml_load
from ultralytics.yolo.utils.checks import check_file, check_requirements
from ultralytics.yolo.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.yolo.utils.plotting import output_to_target, plot_images
from ultralytics.yolo.utils.torch_utils import de_parallel
import  warnings
warnings.filterwarnings("ignore")
class DetectionValidator(BaseValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, logger=None, args=None):
        super().__init__(dataloader, save_dir, pbar, logger, args)
        self.args.task = 'detect'
        self.data_dict = yaml_load(check_file(self.args.data), append_filename=True) if self.args.data else None
        self.is_coco = False
        self.class_map = None
        self.metrics = DetMetrics(save_dir=self.save_dir)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes", "keypoints"]:
            batch[k] = batch[k].to(self.device)

        nb = len(batch["img"])
        self.lb = [torch.cat([batch["cls"], batch["bboxes"]], dim=-1)[batch["batch_idx"] == i]
                   for i in range(nb)] if self.args.save_hybrid else []  # for autolabelling

        return batch

    def init_metrics(self, model):
        head = model.model[-1] if self.training else model.model.model[-1]
        val = self.data.get('val', '')  # validation path
        self.is_coco = isinstance(val, str) and val.endswith(f'coco{os.sep}val2017.txt')  # is COCO dataset
        self.class_map = ops.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.args.save_json |= self.is_coco and not self.training  # run on final val if training COCO
        self.nc = head.nc
        self.names = model.names
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc)
        self.seen = 0
        self.jdict = []
        self.stats = []

    def get_desc(self):
        return ('%22s' + '%11s' * 6 + '%15s' * 2 + '%15s' * 17) % (
        'Class', 'Images', 'Instances', 'Box(P', "R", "mAP50", "mAP50-95", "AVG_distance", "MAX_distance",
        "interval(0,1]", "interval(1,2]", "interval(2,3]", "interval(3,4]", "interval(4,5]", "interval(5,6]",
        "interval(6,7]", "interval(7,8]", "interval(8,9]", "interval(9,10]", "interval(10,+∞]", "kpt_p", "kpt_r",
        "kpt_p_x", "kpt_r_x", "kpt_p_y", "kpt_r_y)")

    def postprocess(self, preds):
        conf =0.25
        iou = 0.7
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        labels=self.lb,
                                        multi_label=True,
                                        agnostic=self.args.single_cls,
                                        max_det=self.args.max_det)
        return preds

    def update_metrics(self, preds, batch):
        # Metrics
        for si, pred in enumerate(preds):
            idx = batch["batch_idx"] == si
            cls = batch["cls"][idx]
            bbox = batch["bboxes"][idx]
            keypoints = batch["keypoints"][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            shape = batch["ori_shape"][si]
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            self.seen += 1
            keypoints = keypoints.to(self.device)
            h, w = batch["img"].shape[2:]
            ratio_pad = torch.zeros(4, device=self.device)
            number = torch.zeros(1, device=self.device)
            number[0] = len(keypoints)
            ratio_pad[0],ratio_pad[1], ratio_pad[2], ratio_pad[3] = w, h, batch["ratio_pad"][si][1][0],batch["ratio_pad"][si][1][1]
            tpoints = keypoints * torch.tensor((w, h), device=self.device)
            tpoints = tpoints.view(-1, DEFAULT_CFG.kpt_num * 2)

            if npr == 0:
                if nl:
                    self.stats.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1), *torch.zeros((18, 2 * DEFAULT_CFG.kpt_num), device=self.device), tpoints,preds, ratio_pad,number))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            ops.scale_boxes(batch["img"][si].shape[1:], predn[:, :4], shape,
                            ratio_pad=batch["ratio_pad"][si])  # native-space pred
            ops.scale_points(batch["img"][si].shape[1:], predn[:, 6:], shape,
                            ratio_pad=batch["ratio_pad"][si])  # native-space pred

            # Evaluate
            if nl:
                height, width = batch["img"].shape[2:]
                bbox = bbox.to(self.device)

                tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                    (width, height, width, height), device=self.device)  # target boxes

                ops.scale_boxes(batch["img"][si].shape[1:], tbox, shape,
                                ratio_pad=batch["ratio_pad"][si])  # native-space labels
                tbox =tbox.cpu()
                cls = cls.cpu()
                tpoints = tpoints.cpu()
                labelsn = torch.cat((cls, tbox, tpoints), 1)  # native-space labels
                pred = pred.cpu()
                correct_bboxes, iou = self._process_batch(predn, labelsn)
                i = np.argsort(-pred[:, 4])
                pred_cls = pred[:, 5]
                pre_point = pred[:, 6:]
                pre_point,pred_cls = pre_point[i],pred_cls[i]

                kpt = pre_point.reshape(pre_point.shape[0], -1, 3)
                kpt = kpt[:, :, [0, 1]]
                kpt = kpt.reshape(pre_point.shape[0], -1)
                kpts = np.zeros((tpoints.shape[0], tpoints.shape[1]))
                target_cls = cls.squeeze(-1).reshape(-1, 1)


                correct_class = target_cls == pred_cls
                for u in range(len(correct_class)):
                    max = 0
                    for o in range(len(correct_class[0])):
                        if correct_class[u][o] and max < iou[u][o]:
                            max = iou[u][o]
                            kpts[u] = kpt[o]

                # TODO: maybe remove these `self.` arguments as they already are member variable
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn[:,:5])
            self.stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1), torch.tensor(kpts), tpoints, ratio_pad, number))  # (conf, pcls, tcls)

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            # if self.args.save_txt:
            #    save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')

    def get_stats(self):

        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy
        if len(stats) and stats[0].any():
            self.metrics.process(*stats)
        self.nt_per_class = np.bincount(stats[-5].astype(int), minlength=self.nc)  # number of targets per class
        return self.metrics.results_dict

    def print_results(self):
        pf = '%22s' + '%11i' * 2 + '%11.3g' * 4 + '%15.3g' * 2 + '%15.3g' * (len(self.metrics.keys) - 5)  # print format
        self.logger.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            self.logger.warning(
                f'WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels')

        pf2 = '%22s' + '%11i' * 2 + '%11.3g' * 4 + '%15.3s' * 2 + '%15.3s' * (len(self.metrics.keys) - 5)
        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                self.logger.info(pf2 % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i)))

        if self.args.plots:
            self.confusion_matrix.plot(save_dir=self.save_dir, names=list(self.names.values()))

    def _process_batch(self, detections, labels):
        """
        Return correct prediction matrix
        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        iou = box_iou(labels[:, 1:5], detections[:, :4])
        correct = np.zeros((detections.shape[0], self.iouv.shape[0])).astype(bool)
        detections = detections.cpu()

        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(self.iouv)):
            x = torch.where((iou >= self.iouv[i]) & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                    1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=detections.device), iou

    def get_dataloader(self, dataset_path, batch_size):
        # TODO: manage splits differently
        # calculate stride - check if model is initialized
        gs = max(int(de_parallel(self.model).stride if self.model else 0), 32)
        return create_dataloader(path=dataset_path,
                                 imgsz=self.args.imgsz,
                                 batch_size=batch_size,
                                 stride=gs,
                                 hyp=vars(self.args),
                                 cache=False,
                                 pad=0.5,
                                 rect=True,
                                 workers=self.args.workers,
                                 prefix=colorstr(f'{self.args.mode}: '),
                                 shuffle=False,
                                 seed=self.args.seed)[0] if self.args.v5loader else \
            build_dataloader(self.args, batch_size, img_path=dataset_path, stride=gs, mode="val")[0]

    def plot_val_samples(self, batch, ni):
        plot_images(batch["img"],
                    batch["batch_idx"],
                    batch["cls"].squeeze(-1),
                    batch["bboxes"],
                    batch["keypoints"],
                    paths=batch["im_file"],
                    fname=self.save_dir / f"val_batch{ni}_labels.jpg",
                    names=self.names)

    def plot_predictions(self, batch, preds, ni):
        plot_images(batch["img"],
                    *output_to_target(preds, max_det=10000),
                    paths=batch["im_file"],
                    fname=self.save_dir / f'val_batch{ni}_pred.jpg',
                    names=self.names)  # pred

    def pred_to_json(self, predn, filename):
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append({
                'image_id': image_id,
                'category_id': self.class_map[int(p[5])],
                'bbox': [round(x, 3) for x in b],
                'score': round(p[4], 5)})

    def eval_json(self, stats):
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data['path'] / "annotations/instances_val2017.json"  # annotations
            pred_json = self.save_dir / "predictions.json"  # predictions
            self.logger.info(f'\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...')
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements('pycocotools>=2.0.6')
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                eval = COCOeval(anno, pred, 'bbox')
                if self.is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = eval.stats[:2]  # update mAP50-95 and mAP50
            except Exception as e:
                self.logger.warning(f'pycocotools unable to run: {e}')
        return stats


def val(cfg=DEFAULT_CFG, use_python=True):
    model = "../../../../ultralytics/yolo/v8/detect/runs/detect/train4/weights/best.pt"
    data = cfg.data or "coco128.yaml"

    args = dict(model=model, data=data)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).val(**args)
    else:
        validator = DetectionValidator(args=args)
        validator(model=args['model'])


if __name__ == "__main__":
    val()
