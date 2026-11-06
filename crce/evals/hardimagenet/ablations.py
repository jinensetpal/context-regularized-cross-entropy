"""
The code for Hard-ImageNet evals was taken from the Hard-ImageNet codebase:
    https://github.com/mmoayeri/HardImageNet/blob/a39e3f996f70eb538ff71829b80781bac617a2d4/ablations.py
"""

from tqdm import tqdm
import numpy as np
import torch
import scipy

'''
We implement multiple methods for ablating away the segmented object.
'''


def replace_with_gray(img, mask, keep_shape=True):
    gray = torch.ones_like(mask) * 0.5
    if keep_shape:
        out = img * (1-mask) + gray * mask
    else:
        out = []
        for i in range(img.shape[0]):
            bbox = get_bbox(mask[i, 0])
            out.append(img[i] * (1 - bbox) + gray[i] * bbox)
        out = torch.stack(out)
    return out


def get_corners(arr):
    on_pixels = np.where(arr != 0)
    x_max, y_max = [np.max(on_pixels[i]) for i in [0, 1]]
    x_min, y_min = [np.min(on_pixels[i]) for i in [0, 1]]
    return x_min, x_max, y_min, y_max


def get_bbox(arr, expand=False):
    out = np.zeros_like(arr)
    if arr.sum() > 0:
        x_min, x_max, y_min, y_max = get_corners(arr)
        out[x_min:x_max, y_min:y_max] = 1
    return out


def trim_tile(img, mask, x1, x2, y1, y2, dir):
    _, h, w = img.shape
    x1, y1 = [max(a, 0) for a in [x1, y1]]
    x2, y2 = [min(a, d) for a, d in zip([x2, y2], [h, w])]
    if mask[:, y1:y2, x1:x2].sum() == 0:
        out = img[:, y1:y2, x1:x2]
        size = (x2-x1) * (y2-y1)
    else:
        # find first instance of other object
        if dir == 'left':
            is_there_obj_by_col = mask[0, y1:y2, x1:x2].sum(1)
            # we take the sum from col i leftwards (towards bbox) looking for lowest i where all leftward sums are 0 (no object)
            sum_moving_right = [sum(is_there_obj_by_col[i:]) for i in range(x2-x1)]
            furthest_we_can_go = sum_moving_right.index(0) if 0 in sum_moving_right else 0
            out = img[:, y1:y2, (x1+furthest_we_can_go):x2]
            size = furthest_we_can_go * (y2 - y1)
        elif dir == 'right':
            is_there_obj_by_col = mask[0, y1:y2, x1:x2].sum(0)
            # now its sum from box to col i
            sum_moving_left = [sum(is_there_obj_by_col[:i]) for i in range(x2 - x1, 0, -1)]
            furthest_we_can_go = sum_moving_left.index(0) if 0 in sum_moving_left else 0
            out = img[:, y1:y2, x1:(x1+furthest_we_can_go)]
            size = furthest_we_can_go * (y2-y1)
        elif dir == 'up':   # actually down bc images have increasing y going downwards but whatever
            is_there_obj_by_col = mask[0, y1:y2, x1:x2].sum(0)
            # now its sum from box to row i
            sum_moving_down = [sum(is_there_obj_by_col[:i]) for i in range(y2-y1, 0, -1)]
            furthest_we_can_go = sum_moving_down.index(0) if 0 in sum_moving_down else 0
            out = img[:, y1:(y1+furthest_we_can_go), x1:x2]
            size = (x2-x1) * furthest_we_can_go
        elif dir == 'down':
            is_there_obj_by_col = mask[0, y1:y2, x1:x2].sum(1)
            # we take the sum from row i upwards (towards bbox) looking for lowest i where all upward sums are 0 (no object)
            sum_moving_up = [sum(is_there_obj_by_col[i:]) for i in range(y2-y1)]
            furthest_we_can_go = sum_moving_up.index(0) if 0 in sum_moving_up else 0
            out = img[:, (y1+furthest_we_can_go):y2, x1:x2]
            size = (x2-x1) * furthest_we_can_go

    return out, size


def repeat_tile_to_fill_bbox(tile, bbox_w, bbox_h, dir):
    out = torch.zeros(3, bbox_h, bbox_w)

    _, tile_h, tile_w = tile.shape
    if dir == 'right':
        num_tile_copies = bbox_w // tile_w
        for i in range(num_tile_copies):
            out[:, :, i * tile_w:min(bbox_w, (i + 1) * tile_w)] = tile
        if bbox_w % tile_w != 0:
            out[:, :, (tile_w * num_tile_copies):] = tile[:, :, :(bbox_w % tile_w)]
    elif dir == 'left':
        num_tile_copies = bbox_w // tile_w
        for i in range(num_tile_copies):
            out[:, :, max(0, bbox_w - (i + 1) * tile_w):(bbox_w - i * tile_w)] = tile
        if bbox_w % tile_w != 0:
            out[:, :, :(bbox_w % tile_w)] = tile[:, :, -1 * (bbox_w % tile_w):]
    if dir == 'up':
        num_tile_copies = bbox_h // tile_h
        for i in range(num_tile_copies):
            out[:, i*tile_h:min(bbox_h, (i+1)*tile_h), :] = tile
        if bbox_h % tile_h != 0:
            out[:, (tile_h * num_tile_copies):, :] = tile[:, :(bbox_h % tile_h), :]
    elif dir == 'down':
        num_tile_copies = bbox_h // tile_h
        for i in range(num_tile_copies):
            out[:, max(0, bbox_h-(i+1)*tile_h):(bbox_h-i*tile_h), :] = tile
        if bbox_h % tile_h != 0:
            out[:, :(bbox_h % tile_h), :] = tile[:, -1*(bbox_h % tile_h):, :]
    return out


def largest_adjacent_tile(img, mask, bbox):
    '''
    Given a bounding box bbox, we check the four adjacent boxes of the same size.
    Each adjacent tile is cut off either at the image boundary or if there is another
    instance of the object (identified via mask). We then return the largest tile.
    '''
    x_min, x_max, y_min, y_max = get_corners(bbox)
    right_tile, size_r = trim_tile(img, mask, x_max, 2*x_max-x_min, y_min, y_max, dir='right')
    left_tile, size_l = trim_tile(img, mask, 2*x_min-x_max, x_min, y_min, y_max, dir='left')
    up_tile, size_u = trim_tile(img, mask, x_min, x_max, y_max, 2*y_max-y_min, dir='up')
    down_tile, size_d = trim_tile(img, mask, x_min, x_max, 2*y_min-y_max, y_max, dir='down')

    max_size_ind = np.argmax([size_r, size_l, size_u, size_d])
    biggest_tile = [right_tile, left_tile, up_tile, down_tile][max_size_ind]

    dirs = ['right', 'left', 'up', 'down']
    out = repeat_tile_to_fill_bbox(biggest_tile, (x_max-x_min), (y_max-y_min), dirs[max_size_ind])
    return out


def tile(img, mask):
    labels, num_features = scipy.ndimage.label(mask[0, 0])
    for j in range(1, 1 + num_features):
        labels2 = labels.copy()
        labels2[labels2 != j] = 0
        if labels2.sum() > 0:
            bbox = get_bbox(labels2)
            if bbox.sum() > 0:
                tile = largest_adjacent_tile(img[0], mask[0], bbox)
                x_min, x_max, y_min, y_max = get_corners(bbox)
                img[:, :, x_min:x_max, y_min:y_max] = tile.swapaxes(1, 2)
    return img


def eval_under_ablation(model, ablation, dset, bs=32, shuffle=True):
    model = model.eval().cuda()
    loader = torch.utils.data.DataLoader(dset, batch_size=bs, shuffle=shuffle, num_workers=8, pin_memory=True)

    cc, ctr = 0, 0
    true_class_confidences = []
    for img, mask, label in tqdm(loader):
        ablated_img = ablation(img, mask)
        ablated_img, label = [x.cuda() for x in [ablated_img, label]]
        logits = model(ablated_img)
        cc += (logits.argmax(1) == label).sum()
        ctr += img.shape[0] if bs > 1 else 1

        probs = torch.softmax(logits, 1)
        true_class_confidences.extend(probs[:, label].flatten().detach().cpu().tolist())

    return (cc/ctr).item(), true_class_confidences
