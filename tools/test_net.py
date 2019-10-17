# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from charnet.modeling.model import CharNet
import cv2, os
import numpy as np
import argparse
from charnet.config import cfg
import matplotlib.pyplot as plt


def save_word_recognition(word_instances, image_id, save_root, separator=chr(31)):
    with open('{}/{}.txt'.format(save_root, image_id), 'wt') as fw:
        for word_ins in word_instances:
            if len(word_ins.text) > 0:
                fw.write(separator.join([str(_) for _ in word_ins.word_bbox.astype(np.int32).flat]))
                fw.write(separator)
                fw.write(word_ins.text)
                fw.write('\n')


def resize(im, size):
    h, w, _ = im.shape
    scale = max(h, w) / float(size)
    image_resize_height = int(round(h / scale / cfg.SIZE_DIVISIBILITY) * cfg.SIZE_DIVISIBILITY)
    image_resize_width = int(round(w / scale / cfg.SIZE_DIVISIBILITY) * cfg.SIZE_DIVISIBILITY)
    scale_h = float(h) / image_resize_height
    scale_w = float(w) / image_resize_width
    im = cv2.resize(im, (image_resize_width, image_resize_height), interpolation=cv2.INTER_LINEAR)
    return im, scale_w, scale_h, w, h


def vis(img, word_instances):
    img_word_ins = img.copy()
    for word_ins in word_instances:
        word_bbox = word_ins.word_bbox
        cv2.polylines(img_word_ins, [word_bbox[:8].reshape((-1, 2)).astype(np.int32)],
                      True, (0, 255, 0), 2)
        cv2.putText(
            img_word_ins,
            '{}'.format(word_ins.text),
            (word_bbox[0], word_bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
    return img_word_ins


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test")

    parser.add_argument("config_file", help="path to config file", type=str)
    parser.add_argument("image_dir", type=str)
    parser.add_argument("results_dir", type=str)

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    print(cfg)

    charnet = CharNet()
    charnet.load_state_dict(torch.load(cfg.WEIGHT))
    charnet.eval()
    charnet.cuda()

    for im_name in sorted(os.listdir(args.image_dir)):
        print("Processing {}...".format(im_name))
        im_file = os.path.join(args.image_dir, im_name)
        im_original = cv2.imread(im_file)
        im, scale_w, scale_h, original_w, original_h = resize(im_original, size=cfg.INPUT_SIZE)
        with torch.no_grad():
            char_bboxes, char_scores, word_instances = charnet(im, scale_w, scale_h, original_w, original_h)
            save_word_recognition(
                word_instances, os.path.splitext(im_name)[0],
                args.results_dir, cfg.RESULTS_SEPARATOR
            )
