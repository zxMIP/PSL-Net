import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32
from mmdet.core import bbox2roi, bbox_overlaps
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads.cascade_roi_head import CascadeRoIHead

class GroupRoIHead(CascadeRoIHead):
    def __init__(self, *args, pos_iou_thrs=[0.5, 0.6, 0.7], num_attempts=1, **kwargs):
        self.pos_iou_thrs = pos_iou_thrs
        super(GroupRoIHead, self).__init__(*args, **kwargs)
        self.num_attempts = num_attempts
        self._init_dy_groupconv()

    def compute_consistency_loss(self, bbox_pred_1, bbox_pred_2):
            bbox_diff = bbox_pred_1 - bbox_pred_2
            consistency_loss = torch.norm(bbox_diff, p=2, dim=1)
            return consistency_loss.mean()

    def horizontal_flip(self, img):
            return torch.flip(img, [-1])

    def flip_annotations(self, gt_points, img_width):
            x, y = gt_points
            center_x = img_width // 2
            flipped_x = center_x + (center_x - x)
            flipped_point = [flipped_x, y]
            return flipped_point

    def generate_bboxes(self, points, img_metas):

        x = [self.extract_feat(img) for img in img_metas]
        feat_sizes = [item.size()[-2:] for item in x]
        mlvl_points = self.gen_points(feat_sizes, dtype=x[0].dtype, device=x[0].device)

        rela_coods_list = self.get_relative_coordinate(mlvl_points, points)
        mlti_assign_results = self.point_assign(mlvl_points, points)
        rpn_losses, results_list = self.rpn_forward_train(
            x,
            img_metas,
            gt_bboxes=None,
            gt_labels=None,
            gt_bboxes_ignore=None,
            assign_results=mlti_assign_results
        )

        pred_bboxes, pred_scores = self._rpn_post_process(results_list)

        detached_x = [item.detach() for item in x]
        for conv in self.projection_convs:
            detached_x = [F.relu(conv(item)) for item in detached_x]

        roi_results = self.roi_head.simple_test(
            detached_x,
            pred_bboxes,
            img_metas,
            rela_coods_list=rela_coods_list
        )

        final_bboxes = []
        for bboxes, _ in roi_results:
            final_bboxes.append(bboxes)

        return final_bboxes

    def _init_dy_groupconv(self):
        self.compress_feat = nn.Conv2d(258, 256, 3, stride=1, padding=1)
        self.cls_embedding = nn.Embedding(80, 256)
        self.generate_params = nn.ModuleList([
            nn.Linear(256, 1 * 1 * 256 * 256) for _ in range(self.num_stages)
        ])
        self.avg_pool = nn.AvgPool2d((7, 7))
        self.compress = nn.Linear(256 + 256, 256)
        self.group_norm = nn.GroupNorm(32, 256)

    def _bbox_forward(self,
                      stage,
                      x,
                      rois,
                      coord_feats,
                      group_size=None,
                      rois_per_image=None,
                      gt_labels=None,
                      **kwargs):
        start = 0
        param_list = []
        ori_pool_rois_feats = self.avg_pool(rois).squeeze()
        for img_id in range(len(gt_labels)):
            num_rois = rois_per_image[img_id]
            end = num_rois + start
            pool_rois_feats = ori_pool_rois_feats[start:end]
            start = end
            bag_size = group_size[img_id]
            label = gt_labels[img_id]
            num_gt = len(label)
            bag_embeds = self.cls_embedding.weight[label]
            pool_rois_feats = pool_rois_feats.view(bag_size, num_gt, 256)
            pool_rois_feats = pool_rois_feats.mean(0)
            pool_rois_feats = torch.cat([bag_embeds, pool_rois_feats], dim=-1)
            pool_rois_feats = self.compress(pool_rois_feats)

            params = self.generate_params[stage](pool_rois_feats)
            conv_weight = params.view(num_gt, 256, 256, 1, 1)
            conv_weight = conv_weight.reshape(num_gt * 256, 256, 1, 1)
            param_list.append(conv_weight)
        params = torch.cat(param_list, dim=0)

        return params

    def _first_coord_pooling(self, coord_feats, proposal_list):
        rois_list = []
        start_index = 0

        for img_id, bag_bboxes in enumerate(proposal_list):
            bag_size, num_gt, _ = bag_bboxes.size()
            roi_index = torch.arange(start_index,
                                     start_index + num_gt,
                                     device=bag_bboxes.device)
            roi_index = roi_index[None, :, None].repeat(bag_size, 1, 1).float()
            bag_bboxes = torch.cat([roi_index, bag_bboxes], dim=-1)
            bag_rois = bag_bboxes.view(-1, 5)
            rois_list.append(bag_rois)
            start_index += num_gt

        rois = torch.cat(rois_list, 0).contiguous()
        self.roi_index = rois[:, :1]
        self.cood_roi_extractor = copy.deepcopy(self.bbox_roi_extractor[0])
        self.cood_roi_extractor.out_channels = 2
        self.coord_feats = coord_feats[:self.cood_roi_extractor.num_inputs]

        coord_feats = self.cood_roi_extractor(
            coord_feats[:self.cood_roi_extractor.num_inputs], rois)

        return coord_feats

    def _not_first_coord_pooling(self, rois):
        rois = torch.cat([self.roi_index, rois], dim=-1)
        coord_feats = self.cood_roi_extractor(
            self.coord_feats[:self.cood_roi_extractor.num_inputs], rois)
        return coord_feats

    def instance_assign(self, stage, anchors, gt_bboxes, gt_labels, group_size):

        repeat_gts = []
        repeat_labels = []
        num_gts = 0
        group_size_each_gt = []
        for gt, label, bag_size in zip(gt_bboxes, gt_labels, group_size):
            num_gts += len(gt)
            gt = gt[None, :, :].repeat(bag_size, 1, 1)
            label = label[None, :].repeat(bag_size, 1)
            repeat_gts.append(gt.view(-1, 4))
            repeat_labels.append(label.view(-1))
            group_size_each_gt.extend(gt.size(1) * [bag_size])

        repeat_gts = torch.cat(repeat_gts, dim=0)
        repeat_labels = torch.cat(repeat_labels, dim=0)

        self.repeat_labels = repeat_labels

        match_quality_matrix = bbox_overlaps(anchors,
                                             repeat_gts,
                                             is_aligned=True)

        pos_mask = match_quality_matrix > self.pos_iou_thrs[stage]
        targets_weight = match_quality_matrix.new_ones(len(pos_mask))

        bbox_targets = self.bbox_head[stage].bbox_coder.encode(
            anchors, repeat_gts)
        all_labels = torch.ones_like(
            repeat_labels) * self.bbox_head[0].num_classes
        all_labels[pos_mask] = repeat_labels[pos_mask]

        pos_bbox_targets = bbox_targets[pos_mask]

        center_points = []
        for gt_bbox, num_attempts, gt_label in zip(gt_bboxes, self.num_attempts, gt_labels):
            x_center = (gt_bbox[0] + gt_bbox[2]) / 2
            y_center = (gt_bbox[1] + gt_bbox[3]) / 2
            center_points.append([x_center, y_center, gt_label])
        center_points = torch.tensor(center_points, dtype=torch.float32, device=gt_bboxes.device)

        random_points = []
        for gt_bbox, num_attempts, gt_label in zip(gt_bboxes, self.num_attempts, gt_labels):
            x_min, y_min, x_max, y_max = gt_bbox
            for _ in range(num_attempts):
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                random_points.append([x, y, gt_label])
        random_points = torch.tensor(random_points, dtype=torch.float32, device=gt_bboxes.device)

        return pos_mask, pos_bbox_targets, all_labels, targets_weight, center_points, random_points

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             stage,
             cls_score,
             bbox_pred,
             rois,
             bbox_targets,
             labels,
             pos_mask,
             reduction_override=None):

        label_weights = torch.ones_like(labels)
        bbox_weights = torch.ones_like(bbox_targets)
        losses = dict()
        avg_factor = max(pos_mask.sum(), pos_mask.new_ones(1).sum())

        loss_cls_ = self.bbox_head[stage].loss_cls(
            cls_score,
            labels,
            label_weights,
            avg_factor=avg_factor,
            reduction_override=reduction_override)

        losses['loss_cls'] = loss_cls_
        losses['acc'] = accuracy(cls_score, labels)
        losses['avg_pos'] = pos_mask.sum()
        pos_inds = pos_mask
        if pos_inds.any():
            if self.bbox_head[stage].reg_decoded_bbox:
                bbox_pred = self.bbox_head[stage].bbox_coder.decode(
                    rois[:, 1:], bbox_pred)
            if self.bbox_head[stage].reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0),
                                               4)[pos_inds.type(torch.bool)]
            else:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), -1,
                    4)[pos_inds.type(torch.bool),
                       labels[pos_inds.type(torch.bool)]]
            losses['loss_bbox'] = self.bbox_head[stage].loss_bbox(
                pos_bbox_pred,
                bbox_targets,
                bbox_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
        else:
            losses['loss_bbox'] = bbox_pred.sum() * 0

        return losses


    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore=None, rela_coods_list=None, gt_points=None, stage=None,
                      feat_rois=None, group_size=None, rois_per_image=None, coord_feats=None, cls_score=None, bbox_pred=None,
                      rois=None, bbox_targets=None, labels=None, pos_mask=None, flipped_re_bbox=None, **kwargs):

        center_points, random_points = self.instance_assign(stage, feat_rois[:, 1:], gt_bboxes, gt_labels, group_size)

        center_bbox_targets = self.generate_bboxes(img_metas, center_points[:, :2])

        bbox_results_center = self._bbox_forward(
            stage,
            x,
            feat_rois,
            coord_feats,
            group_size,
            rois_per_image,
            gt_labels=gt_labels,
            gt_points=gt_points,
            img_metas=img_metas,
            bbox_targets=center_bbox_targets
        )

        random_bbox_targets = self.generate_bboxes(img_metas, random_points[:, :2])

        bbox_results_random = self._bbox_forward(
            stage,
            x,
            feat_rois,
            coord_feats,
            group_size,
            rois_per_image,
            gt_labels=gt_labels,
            gt_points=gt_points,
            img_metas=img_metas,
            bbox_targets=random_bbox_targets
        )

        consistency_loss_center = self.compute_consistency_loss(bbox_results_center['bbox_pred'], bbox_results_random['bbox_pred'])
        pseudo_bbox_targets = self.generate_bboxes(img_metas, gt_points[:, :2])
        pseudo_bbox_results = self._bbox_forward(
            stage,
            x,
            feat_rois,
            coord_feats,
            group_size,
            rois_per_image,
            gt_labels=gt_labels,
            gt_points=gt_points,
            img_metas=img_metas,
            bbox_targets=pseudo_bbox_targets

        )

        flipped_images = [self.horizontal_flip(img) for img in img_metas]
        flipped_annotations = [self.flip_annotations(ann, img_info['width']) for ann, img_info in zip(gt_points, img_metas)]

        flipped_bbox_targets = self.generate_bboxes(img_metas, flipped_annotations[:, :2])
        pseudo_bbox_results_flipped = self._bbox_forward(stage, x, feat_rois, coord_feats, group_size, rois_per_image,
                                                         gt_labels=gt_labels, gt_points=flipped_annotations, img_metas=flipped_bbox_targets)

        img_width = flipped_images.size(2)
        flipped_re_bbox = pseudo_bbox_results_flipped['bbox_pred'].clone()
        flipped_re_bbox[:, 0] = img_width - pseudo_bbox_results_flipped['bbox_pred'][:, 2]
        flipped_re_bbox[:, 2] = img_width - pseudo_bbox_results_flipped['bbox_pred'][:, 0]
        consistency_loss_flip = self.compute_consistency_loss(pseudo_bbox_results['bbox_pred'], flipped_re_bbox['bbox_pred'])

        group_loss = self.loss(self, stage, cls_score, bbox_pred, rois, bbox_targets, labels, pos_mask, reduction_override=None)

        group_size = [item.size(0) for item in proposal_list]
        wh_each_level = [item.shape[-2:] for item in x]
        num_img = len(rela_coods_list)
        num_level = len(rela_coods_list[0])
        all_num_gts = 0
        format_rela_coods_list = []
        for img_id in range(num_img):
            real_coods = rela_coods_list[img_id]
            mlvl_coord_list = []
            for level in range(num_level):
                format_coords = real_coods[level]
                num_gt = format_coords.size(0)
                all_num_gts += num_gt
                format_coords = format_coords.view(num_gt,
                                                   *wh_each_level[level],
                                                   2).permute(0, 3, 1, 2)
                mlvl_coord_list.append(format_coords)
            format_rela_coods_list.append(mlvl_coord_list)

        mlvl_concate_coods = []
        for level in range(num_level):
            mlti_img_cood = [
                format_rela_coods_list[img_id][level]
                for img_id in range(num_img)
            ]
            concat_coods = torch.cat(mlti_img_cood, dim=0).contiguous()
            mlvl_concate_coods.append(concat_coods)

        losses = dict()

        rois_per_image = [
            item.size(0) * item.size(1) for item in proposal_list
        ]

        rois = None

        for stage in range(self.num_stages):
            if stage == 0:
                coord_feats = self._first_coord_pooling(
                    mlvl_concate_coods, proposal_list)
                feat_rois = bbox2roi(
                    [item.view(-1, 4).detach() for item in proposal_list])
            else:
                coord_feats = self._not_first_coord_pooling(rois)
                feat_rois = rois.split(rois_per_image, dim=0)
                feat_rois = bbox2roi(
                    [item.view(-1, 4).detach() for item in feat_rois])

            pos_mask, pos_bbox_targets, pos_labels, \
                targets_reweight = self.instance_assign(
                     stage, feat_rois[:, 1:], gt_bboxes, gt_labels, group_size)

            bbox_results = self._bbox_forward(
                stage,
                x,
                feat_rois,
                coord_feats,
                group_size,
                rois_per_image,
                gt_labels=gt_labels,
                gt_points=gt_points,
                img_metas=img_metas,
            )

            single_stage_loss = self.loss(stage, bbox_results['cls_score'],
                                          bbox_results['bbox_pred'], feat_rois,
                                          pos_bbox_targets, pos_labels,
                                          pos_mask)
            single_stage_loss['avg_pos'] = single_stage_loss[
                'avg_pos'] / float(all_num_gts) * 5

            for name, value in single_stage_loss.items():
                losses[f's{stage}.{name}'] = (value *
                                              self.stage_loss_weights[stage]
                                              if 'loss' in name else value)

            if stage < self.num_stages - 1:
                with torch.no_grad():
                    rois = self.bbox_head[stage].bbox_coder.decode(
                        feat_rois[:, 1:],
                        bbox_results['bbox_pred'],
                    )

        return flipped_images, flipped_annotations, flipped_bbox_targets, pseudo_bbox_results_flipped, flipped_re_bbox, \
               consistency_loss_flip, consistency_loss_center, group_loss

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    rela_coods_list=None,
                    labels=None,
                    gt_points=None,
                    **kwargs):

        num_images = len(proposal_list)
        group_size = [item.size(0) for item in proposal_list]
        ms_scores = []
        wh_each_level = [item.shape[-2:] for item in x]
        num_img = len(rela_coods_list)
        num_level = len(rela_coods_list[0])
        format_rela_coods_list = []

        repeat_labels = []
        for label, bag_size in zip(labels, group_size):
            label = label[None, :].repeat(bag_size, 1)
            repeat_labels.append(label.view(-1))
        self.repeat_labels = torch.cat(repeat_labels, dim=0)

        for img_id in range(num_img):
            real_coods = rela_coods_list[img_id]
            mlvl_coord_list = []
            for level in range(num_level):
                format_coords = real_coods[level]
                num_gt = format_coords.size(0)
                format_coords = format_coords.view(num_gt,
                                                   *wh_each_level[level],
                                                   2).permute(0, 3, 1, 2)
                mlvl_coord_list.append(format_coords)
            format_rela_coods_list.append(mlvl_coord_list)
        mlvl_concate_coods = []

        for level in range(num_level):
            mlti_img_cood = [
                format_rela_coods_list[img_id][level]
                for img_id in range(num_img)
            ]
            concat_coods = torch.cat(mlti_img_cood, dim=0).contiguous()
            mlvl_concate_coods.append(concat_coods)

        rois_per_image = [
            item.size(0) * item.size(1) for item in proposal_list
        ]
        for stage in range(self.num_stages):
            self.current_stage = stage
            if stage == 0:
                coord_feats = self._first_coord_pooling(
                    mlvl_concate_coods, proposal_list)
                feat_rois = bbox2roi(
                    [item.view(-1, 4).detach() for item in proposal_list])
            else:
                coord_feats = self._not_first_coord_pooling(
                    torch.cat(proposal_list, dim=0))
                feat_rois = proposal_list
                feat_rois = bbox2roi(
                    [item.view(-1, 4).detach() for item in feat_rois])

            bbox_results = self._bbox_forward(
                stage,
                x,
                feat_rois,
                coord_feats,
                rois_per_image=rois_per_image,
                group_size=group_size,
                gt_labels=labels,
                gt_points=gt_points,
                img_metas=img_metas,
            )

            bbox_preds = bbox_results['bbox_pred']

            if self.bbox_head[-1].loss_cls.use_sigmoid:
                cls_score = bbox_results['cls_score'].sigmoid()
                num_classes = cls_score.size(-1)
            else:
                cls_score = bbox_results['cls_score'].softmax(-1)
                num_classes = cls_score.size(-1) - 1
            cls_score = cls_score[:, :num_classes]

            decode_bboxes = []
            all_scores = []

            for img_id in range(num_images):
                img_shape = img_metas[img_id]['img_shape']
                img_mask = feat_rois[:, 0] == img_id
                temp_rois = feat_rois[img_mask]
                temp_bbox_pred = bbox_preds[img_mask]
                bboxes = self.bbox_head[stage].bbox_coder.decode(
                    temp_rois[..., 1:], temp_bbox_pred, max_shape=img_shape)
                temp_scores = cls_score[img_mask]
                bboxes = bboxes.view(group_size[img_id], -1, 4)
                temp_scores = temp_scores.view(group_size[img_id], -1,
                                               num_classes)
                decode_bboxes.append(bboxes)
                all_scores.append(temp_scores)

            ms_scores.append(all_scores)
            proposal_list = [item.view(-1, 4) for item in decode_bboxes]

        ms_scores = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_images)
        ]

        pred_bboxes = []
        pred_scores = []
        pred_labels = []
        for img_id in range(num_images):
            all_class_scores = ms_scores[img_id]
            if all_class_scores.numel():
                repeat_label = labels[img_id][None].repeat(
                    group_size[img_id], 1)
                scores = torch.gather(all_class_scores, 2,
                                      repeat_label[..., None]).squeeze(-1)

                num_gt = decode_bboxes[img_id].shape[1]
                dets, keep = batched_nms(
                    decode_bboxes[img_id].view(-1, 4), scores.view(-1),
                    repeat_label.view(-1),
                    dict(max_num=1000, iou_threshold=self.iou_threshold))
                num_pred = len(keep)
                gt_index = keep % num_gt
                arrange_gt_index = torch.arange(num_gt,
                                                device=keep.device)[:, None]
                keep_matrix = gt_index == arrange_gt_index
                temp_index = torch.arange(-num_pred,
                                          end=0,
                                          step=1,
                                          device=keep.device)
                keep_matrix = keep_matrix * temp_index

                value_, index = keep_matrix.min(dim=-1)
                dets = dets[index]
                pred_bboxes.append(dets[:, :4])
                pred_scores.append(dets[:, -1])
                pred_labels.append(labels[img_id])

            else:
                pred_bboxes.append(all_class_scores.new_zeros(0, 4))
                pred_scores.append(all_class_scores.new_zeros(0))
                pred_labels.append(all_class_scores.new_zeros(0))

        return pred_bboxes, pred_scores, pred_labels


