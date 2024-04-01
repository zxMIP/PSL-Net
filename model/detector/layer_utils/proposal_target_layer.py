from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from model.detector.model.config import cfg
from model.detector.model.bbox_transform import bbox_transform
from model.detector.utils.bbox import bbox_overlaps

import torch

def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):

    all_rois = rpn_rois
    all_scores = rpn_scores

    if cfg.TRAIN.USE_GT:
        zeros = rpn_rois.new_zeros(gt_boxes.shape[0], 1)
        all_rois = torch.cat((all_rois, torch.cat(
            (zeros, gt_boxes[:, :-1]), 1)), 0)
        all_scores = torch.cat((all_scores, zeros), 0)

    num_images = 1
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = int(round(cfg.TRAIN.FG_FRACTION * rois_per_image))

    labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
        all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image,
        _num_classes)

    rois = rois.view(-1, 5)
    roi_scores = roi_scores.view(-1)
    labels = labels.view(-1, 1)
    bbox_targets = bbox_targets.view(-1, _num_classes * 4)
    bbox_inside_weights = bbox_inside_weights.view(-1, _num_classes * 4)
    bbox_outside_weights = (bbox_inside_weights > 0).float()

    beta = 0.75
    mu = 8
    lambda_value = 2
    iou = calculate_iou(all_rois, gt_boxes)
    s = calculate_classification_scores(all_scores)

    t_values = beta * iou + (1 - beta) * s
    c_plus = torch.exp(mu * t_values)
    Wpos = t_values * c_plus
    Wneg = (iou ** lambda_value) * (1 - t_values)

    regression_loss = calculate_regression_loss(all_rois, gt_boxes, Wpos)

    classification_loss = calculate_classification_loss(rpn_scores, Wpos, Wneg)

    alpha = 0.5
    total_loss_value = alpha * regression_loss + (1 - alpha) * classification_loss

    return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, rpn_scores, total_loss_value, regression_loss, classification_loss


def calculate_iou(rois, gt_boxes):

    x1 = torch.max(rois[:, 0].unsqueeze(1), gt_boxes[:, 0])
    y1 = torch.max(rois[:, 1].unsqueeze(1), gt_boxes[:, 1])
    x2 = torch.min(rois[:, 2].unsqueeze(1), gt_boxes[:, 2])
    y2 = torch.min(rois[:, 3].unsqueeze(1), gt_boxes[:, 3])

    intersection = torch.clamp(x2 - x1 + 1, min=0) * torch.clamp(y2 - y1 + 1, min=0)

    area_roi = (rois[:, 2] - rois[:, 0] + 1) * (rois[:, 3] - rois[:, 1] + 1)
    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)

    iou = intersection / (area_roi.unsqueeze(1) + area_gt - intersection)

    return iou


def calculate_classification_scores(rpn_scores):

    scores = rpn_scores

    return scores


def get_anchor_indices_within_radius(anchors, gt_center, radius=2.0):

    anchor_centers = (anchors[:, :2] + anchors[:, 2:]) / 2
    distances = np.sqrt(np.sum((anchor_centers - gt_center) ** 2, axis=1))
    anchor_indices = np.where(distances <= radius)[0]
    return anchor_indices


def get_anchor_indices_without_radius(anchors, gt_boxes, radius=2.0):
    anchor_centers = (anchors[:, :2] + anchors[:, 2:]) / 2
    distances = np.sqrt(np.sum((anchor_centers - gt_boxes) ** 2, axis=1))
    anchor_indices = np.where((distances > radius) & (anchor_centers[:, 0] >= gt_boxes[:, 0]) & (anchor_centers[:, 1] >= gt_boxes[:, 1]) &
                              (anchor_centers[:, 0] <= gt_boxes[:, 2]) & (anchor_centers[:, 1] <= gt_boxes[:, 3]))[0]
    return anchor_indices


def smooth_l1_loss(prediction, target, sigma=1.0):

    diff = prediction - target
    abs_diff = np.abs(diff)
    smooth_l1_loss = np.where(abs_diff < (1.0 / sigma ** 2), 0.5 * (sigma * diff) ** 2, abs_diff - 0.5 / sigma ** 2)
    return smooth_l1_loss


def calculate_regression_loss(anchors, gt_boxes, Wpos):

    anchor_indices_within_radius = get_anchor_indices_within_radius(anchors, gt_boxes)
    regression_losses = smooth_l1_loss(anchors[anchor_indices_within_radius], gt_boxes)
    weighted_regression_losses = regression_losses * Wpos[anchor_indices_within_radius]
    return torch.sum(weighted_regression_losses)


def calculate_classification_loss(scores, Wpos, Wneg, anchors, gt_boxes):

    anchor_indices_within_radius = get_anchor_indices_within_radius(anchors, gt_boxes)
    anchor_indices_without_radius = get_anchor_indices_without_radius(anchors, gt_boxes)
    individual_classification_loss_within_radius = -Wpos[anchor_indices_within_radius] * torch.log(
        scores[anchor_indices_within_radius]) - Wneg[anchor_indices_within_radius] * torch.log(
        1 - scores[anchor_indices_within_radius])
    focal_loss_without_radius = FocalLoss(scores[anchor_indices_without_radius], gamma=2)
    total_classification_loss = torch.sum(individual_classification_loss_within_radius) + focal_loss_without_radius

    return total_classification_loss


def _get_bbox_regression_labels(bbox_target_data, num_classes):

    clss = bbox_target_data[:, 0]
    bbox_targets = clss.new_zeros(clss.numel(), 4 * num_classes)
    bbox_inside_weights = clss.new_zeros(bbox_targets.shape)
    inds = (clss > 0).nonzero().view(-1)
    if inds.numel() > 0:
        clss = clss[inds].contiguous().view(-1, 1)
        dim1_inds = inds.unsqueeze(1).expand(inds.size(0), 4)
        dim2_inds = torch.cat(
            [4 * clss, 4 * clss + 1, 4 * clss + 2, 4 * clss + 3], 1).long()
        bbox_targets[dim1_inds, dim2_inds] = bbox_target_data[inds][:, 1:]
        bbox_inside_weights[dim1_inds, dim2_inds] = bbox_targets.new(
            cfg.TRAIN.BBOX_INSIDE_WEIGHTS).view(-1, 4).expand_as(dim1_inds)

    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        targets = ((targets - targets.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS)) /
                   targets.new(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return torch.cat([labels.unsqueeze(1), targets], 1)


def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image,
                 rois_per_image, num_classes):

    overlaps = bbox_overlaps(all_rois[:, 1:5].data, gt_boxes[:, :4].data)
    max_overlaps, gt_assignment = overlaps.max(1)
    labels = gt_boxes[gt_assignment, [4]]

    fg_inds = (max_overlaps >= cfg.TRAIN.FG_THRESH).nonzero().view(-1)
      bg_inds = ((max_overlaps < cfg.TRAIN.BG_THRESH_HI) + (
        max_overlaps >= cfg.TRAIN.BG_THRESH_LO) == 2).nonzero().view(-1)

    if fg_inds.numel() == 0 and bg_inds.numel() == 0:
        to_replace = all_rois.size(0) < rois_per_image
        bg_inds = torch.from_numpy(npr.choice(np.arange(0, all_rois.size(0)), size=int(rois_per_image), replace=to_replace)).long().cuda()
        fg_rois_per_image = 0
    elif fg_inds.numel() > 0 and bg_inds.numel() > 0:
        fg_rois_per_image = min(fg_rois_per_image, fg_inds.numel())
        fg_inds = fg_inds[torch.from_numpy(
            npr.choice(
                np.arange(0, fg_inds.numel()),
                size=int(fg_rois_per_image),
                replace=False)).long().to(gt_boxes.device)]
        bg_rois_per_image = rois_per_image - fg_rois_per_image
        to_replace = bg_inds.numel() < bg_rois_per_image
        bg_inds = bg_inds[torch.from_numpy(
            npr.choice(
                np.arange(0, bg_inds.numel()),
                size=int(bg_rois_per_image),
                replace=to_replace)).long().to(gt_boxes.device)]
    elif fg_inds.numel() > 0:
        to_replace = fg_inds.numel() < rois_per_image
        fg_inds = fg_inds[torch.from_numpy(
            npr.choice(
                np.arange(0, fg_inds.numel()),
                size=int(rois_per_image),
                replace=to_replace)).long().to(gt_boxes.device)]
        fg_rois_per_image = rois_per_image
    elif bg_inds.numel() > 0:
        to_replace = bg_inds.numel() < rois_per_image
        bg_inds = bg_inds[torch.from_numpy(
            npr.choice(
                np.arange(0, bg_inds.numel()),
                size=int(rois_per_image),
                replace=to_replace)).long().to(gt_boxes.device)]
        fg_rois_per_image = 0
    else:
        import pdb
        pdb.set_trace()

    keep_inds = torch.cat([fg_inds, bg_inds], 0)
    labels = labels[keep_inds].contiguous()
    labels[int(fg_rois_per_image):] = 0
    rois = all_rois[keep_inds].contiguous()
    roi_scores = all_scores[keep_inds].contiguous()

    bbox_target_data = _compute_targets(
        rois[:, 1:5].data, gt_boxes[gt_assignment[keep_inds]][:, :4].data,
        labels.data)

    bbox_targets, bbox_inside_weights = \
      _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, roi_scores, bbox_targets, bbox_inside_weights
