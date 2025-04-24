# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import numpy as np
import os
import PIL.Image as Image
import torch
from torch.nn import functional as F
from detectron2.data import MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.evaluation import SemSegEvaluator
from detectron2.utils.visualizer import ColorMode

# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import torch

from baselines.utils.visualizer import CustomVisualizer

class CATSegEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
        post_process_func=None,
        visualize=True,
    ):
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )
        meta = MetadataCatalog.get(dataset_name)
        try:
            self._evaluation_set = meta.evaluation_set
        except AttributeError:
            self._evaluation_set = None
        self.post_process_func = (
            post_process_func
            if post_process_func is not None
            else lambda x, **kwargs: x
        )
        self.visualize = visualize
        
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)

        
        if self.visualize:
            self.vis_path = os.path.join(self._output_dir, "visualization")
            PathManager.mkdirs(self.vis_path)
        
        self.meta = meta
        self.ignore_label = meta.ignore_label
        self.device = "cuda"

    def reset(self):
        super().reset()
        self._conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        )
        self._conf_matrix_pred_all = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        )

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output_org in zip(inputs, outputs):
            sem_seg_output, sem_seg_output_all = output_org["sem_seg"], output_org["sem_seg_all"]

            output = self.post_process_func(sem_seg_output, image=np.array(Image.open(input["file_name"])))
            output_all = self.post_process_func(sem_seg_output_all, image=np.array(Image.open(input["file_name"])))
            
            output = output.argmax(dim=0).to(self._cpu_device)
            output_all = output_all.argmax(dim=0).to(self._cpu_device)

            gt_classes = input["obj_part_instances"].gt_classes
            gt_masks = input["obj_part_instances"].gt_masks
            eval_image_size = tuple(output.shape[-2:])

            if len(gt_masks) == 0:
                gt = np.zeros_like(pred) + self._ignore_label
            else:
                gt = np.zeros_like(gt_masks[0], dtype=np.float) + self._ignore_label
                for i in range(len(gt_classes)):
                    gt[gt_masks[i] == True] = gt_classes[i]
                gt = F.interpolate(torch.tensor(gt).unsqueeze(0).unsqueeze(0), size=eval_image_size, mode='nearest').squeeze()
                gt = gt.int().numpy()
            
            output[gt == self._ignore_label] = self.meta.ignore_label

            pred = np.array(output, dtype=np.int)
            pred_all = np.array(output_all, dtype=np.int)
                
            pred[pred == self._ignore_label] = self._num_classes
            pred_all[pred_all == self._ignore_label] = self._num_classes
            pred_all[(gt == self._ignore_label)] = self._num_classes
            
            gt[gt == self._ignore_label] = self._num_classes 
            
            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._conf_matrix_pred_all += np.bincount(
                (self._num_classes + 1) * pred_all.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix_pred_all.size,
            ).reshape(self._conf_matrix_pred_all.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))
            
            if self.visualize:
                ext = os.path.splitext(input["file_name"])[1]
                input_img_tensor = F.interpolate(input["image"].unsqueeze(0), size=eval_image_size, mode='bilinear').squeeze()
                input_img_npy = input_img_tensor.permute(1, 2, 0).int().numpy()
                
                visualizer_pred = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)
                visualizer_pred_all = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)
                visualizer_gt = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)
                
                vis_pred = visualizer_pred.draw_sem_seg(pred)
                vis_pred.save(os.path.join(self.vis_path, os.path.basename(input["file_name"])))
                
                vis_pred_all = visualizer_pred_all.draw_sem_seg(np.array(output_all, dtype=np.int))
                vis_pred_all.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, "_all.jpg")))

                vis_gt = visualizer_gt.draw_sem_seg(gt)
                vis_gt.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, "_gt.jpg")))
    

    def calculate_metrics(self, conf_matrix, num_classes, class_names, include_last_class=False):
        acc = np.full(num_classes, np.nan, dtype=np.float64)
        iou = np.full(num_classes, np.nan, dtype=np.float64)
        recall = np.full(num_classes, np.nan, dtype=np.float64)

        if include_last_class:
            tp = conf_matrix.diagonal().astype(np.float64)
            pos_gt = np.sum(conf_matrix, axis=0).astype(np.float64)
            pos_pred = np.sum(conf_matrix, axis=1).astype(np.float64)
        else:
            tp = conf_matrix.diagonal()[:-1].astype(np.float64)
            pos_gt = np.sum(conf_matrix[:-1, :-1], axis=0).astype(np.float64)
            pos_pred = np.sum(conf_matrix[:-1, :-1], axis=1).astype(np.float64)

        class_weights = pos_gt / np.sum(pos_gt)

        recall_valid = pos_gt > 0
        acc[recall_valid] = tp[recall_valid] / pos_gt[recall_valid]

        union = pos_gt + pos_pred - tp
        iou_valid = (pos_gt + pos_pred) > 0

        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        recall[recall_valid] = tp[recall_valid] / pos_gt[recall_valid]

        macc = np.nanmean(acc)
        miou = np.nanmean(iou)
        fiou = np.nansum(iou * class_weights)
        pacc = np.nansum(tp) / np.nansum(pos_gt)
        mRecall = np.nanmean(recall)

        res = {
            "mIoU": 100 * miou,
            "fwIoU": 100 * fiou,
            "mACC": 100 * macc,
            "pACC": 100 * pacc,
            "mRecall": 100 * mRecall,
        }
        
        for i, name in enumerate(class_names):
            res[f"IoU-{name}"] = 100 * iou[i]
            res[f"ACC-{name}"] = 100 * acc[i]
            res[f"Recall-{name}"] = 100 * recall[i]

        return res

    def evaluate(self):
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            conf_matrix_pred_all_list = all_gather(self._conf_matrix_pred_all)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

            self._conf_matrix_pred_all = np.zeros_like(self._conf_matrix_pred_all)
            for conf_matrix_pred_all in conf_matrix_pred_all_list:
                self._conf_matrix_pred_all += conf_matrix_pred_all


        res = self.calculate_metrics(self._conf_matrix, self._num_classes, self._class_names, include_last_class=False)
        res_pred_all = self.calculate_metrics(self._conf_matrix_pred_all, self._num_classes + 1, self._class_names, include_last_class=True)

        if self._evaluation_set is not None:
            for set_name, set_inds in self._evaluation_set.items():
                set_inds = np.array(set_inds, dtype=int)
                mask = np.zeros(len(self._class_names), dtype=bool)
                mask[set_inds] = True

                subset_iou_valid = mask & np.array([res[f"IoU-{self._class_names[i]}"] > 0 for i in range(len(self._class_names))])

                if np.any(subset_iou_valid):
                    miou = np.nanmean([res[f"IoU-{self._class_names[i]}"] for i in set_inds if subset_iou_valid[i]])
                    mrecall = np.nanmean([res[f"Recall-{self._class_names[i]}"] for i in set_inds if subset_iou_valid[i]])
                    pacc = np.nansum([res[f"ACC-{self._class_names[i]}"] for i in set_inds if subset_iou_valid[i]]) / np.sum(subset_iou_valid)

                    res[f"mIoU-{set_name}"] = miou
                    res[f"mRecall-{set_name}"] = mrecall
                    res[f"pACC-{set_name}"] = pacc

                # Calculate for inverse mask (unbase classes)
                inv_mask = ~mask
                subset_iou_valid_inv = inv_mask & np.array([res[f"IoU-{self._class_names[i]}"] > 0 for i in range(len(self._class_names))])

                if np.any(subset_iou_valid_inv):
                    miou_inv = np.nanmean([res[f"IoU-{self._class_names[i]}"] for i in range(len(self._class_names)) if subset_iou_valid_inv[i]])
                    mrecall_inv = np.nanmean([res[f"Recall-{self._class_names[i]}"] for i in range(len(self._class_names)) if subset_iou_valid_inv[i]])
                    pacc_inv = np.nansum([res[f"ACC-{self._class_names[i]}"] for i in range(len(self._class_names)) if subset_iou_valid_inv[i]]) / np.sum(subset_iou_valid_inv)

                    res[f"mIoU-un{set_name}"] = miou_inv
                    res[f"mRecall-un{set_name}"] = mrecall_inv
                    res[f"pACC-un{set_name}"] = pacc_inv

                # Repeat the same for res_pred_all
                subset_iou_valid_pred_all = mask & np.array([res_pred_all[f"IoU-{self._class_names[i]}"] > 0 for i in range(len(self._class_names))])

                if np.any(subset_iou_valid_pred_all):
                    miou_pred_all = np.nanmean([res_pred_all[f"IoU-{self._class_names[i]}"] for i in set_inds if subset_iou_valid_pred_all[i]])
                    mrecall_pred_all = np.nanmean([res_pred_all[f"Recall-{self._class_names[i]}"] for i in set_inds if subset_iou_valid_pred_all[i]])
                    pacc_pred_all = np.nansum([res_pred_all[f"ACC-{self._class_names[i]}"] for i in set_inds if subset_iou_valid_pred_all[i]]) / np.sum(subset_iou_valid_pred_all)

                    res_pred_all[f"mIoU-{set_name}"] = miou_pred_all
                    res_pred_all[f"mRecall-{set_name}"] = mrecall_pred_all
                    res_pred_all[f"pACC-{set_name}"] = pacc_pred_all

                subset_iou_valid_pred_all_inv = inv_mask & np.array([res_pred_all[f"IoU-{self._class_names[i]}"] > 0 for i in range(len(self._class_names))])

                if np.any(subset_iou_valid_pred_all_inv):
                    miou_pred_all_inv = np.nanmean([res_pred_all[f"IoU-{self._class_names[i]}"] for i in range(len(self._class_names)) if subset_iou_valid_pred_all_inv[i]])
                    mrecall_pred_all_inv = np.nanmean([res_pred_all[f"Recall-{self._class_names[i]}"] for i in range(len(self._class_names)) if subset_iou_valid_pred_all_inv[i]])
                    pacc_pred_all_inv = np.nansum([res_pred_all[f"ACC-{self._class_names[i]}"] for i in range(len(self._class_names)) if subset_iou_valid_pred_all_inv[i]]) / np.sum(subset_iou_valid_pred_all_inv)

                    res_pred_all[f"mIoU-un{set_name}"] = miou_pred_all_inv
                    res_pred_all[f"mRecall-un{set_name}"] = mrecall_pred_all_inv
                    res_pred_all[f"pACC-un{set_name}"] = pacc_pred_all_inv

        if 'mIoU-base' in res and 'mIoU-unbase' in res:
            res['h-IoU'] = 2 * (res['mIoU-base'] * res['mIoU-unbase']) / (res['mIoU-base'] + res['mIoU-unbase']) if (res['mIoU-base'] + res['mIoU-unbase']) != 0 else np.nan
        if 'mRecall-base' in res and 'mRecall-unbase' in res:
            res['h-Recall'] = 2 * (res['mRecall-base'] * res['mRecall-unbase']) / (res['mRecall-base'] + res['mRecall-unbase']) if (res['mRecall-base'] + res['mRecall-unbase']) != 0 else np.nan

        if 'mIoU-base' in res_pred_all and 'mIoU-unbase' in res_pred_all:
            res_pred_all['h-IoU'] = 2 * (res_pred_all['mIoU-base'] * res_pred_all['mIoU-unbase']) / (res_pred_all['mIoU-base'] + res_pred_all['mIoU-unbase']) if (res_pred_all['mIoU-base'] + res_pred_all['mIoU-unbase']) != 0 else np.nan
        if 'mRecall-base' in res_pred_all and 'mRecall-unbase' in res_pred_all:
            res_pred_all['h-Recall'] = 2 * (res_pred_all['mRecall-base'] * res_pred_all['mRecall-unbase']) / (res_pred_all['mRecall-base'] + res_pred_all['mRecall-unbase']) if (res_pred_all['mRecall-base'] + res_pred_all['mRecall-unbase']) != 0 else np.nan

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            file_path_pred_all = os.path.join(self._output_dir, "sem_seg_evaluation_all.pth")

            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)

            with PathManager.open(file_path_pred_all, "wb") as f:
                torch.save(res_pred_all, f)

        results = OrderedDict({"oracle_obj": res, "pred_all": res_pred_all})
        self._logger.info(results)
        return results
