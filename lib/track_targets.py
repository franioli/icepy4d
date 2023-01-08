"""
MIT License

Copyright (c) 2022 Francesco Ioli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import matplotlib.pyplot as plt

import time
import cv2

from copy import deepcopy
from tqdm import tqdm

from lib.matching.templatematch import TemplateMatch, Stats
from lib.read_config import parse_yaml_cfg
from lib.utils.inizialize_variables import Inizialization

cfg_file = "config/config_base.yaml"
cfg = parse_yaml_cfg(cfg_file)
init = Inizialization(cfg)
init.inizialize_belpy()
cams = init.cams
images = init.images
targets = init.targets

cam_id = 0
epoch = 0
dt = 1
roi_buffer = 128
targets_to_use = ["F2"]  # , "F11"

template_width = 16
search_width = 64

debug_viz = False
debug = True

t_est = {}
diff = {}
diff_noCC = {}


class TrackTargets:
    def __init__(self) -> None:
        pass


if True:
    t = targets[epoch].extract_image_coor_by_label(targets_to_use, cam_id).squeeze()
    t_int = np.round(t).astype(int)
    roi = [
        int(t_int[0]) - roi_buffer,
        int(t_int[1]) - roi_buffer,
        int(t_int[0]) + roi_buffer,
        int(t_int[1]) + roi_buffer,
    ]
    t_roi = np.array([t[0] - roi[0], t[1] - roi[1]])
    t_roi_int = np.round(t_roi).astype(int)

# for epoch in tqdm(cfg.proc.epoch_to_process):  # tqdm(range(1, 2)):  #

#     t = targets[epoch].extract_image_coor_by_label(targets_to_use, cam_id).squeeze()

#     t_int = np.round(t).astype(int)
#     roi = [
#         int(t_int[0]) - roi_buffer,
#         int(t_int[1]) - roi_buffer,
#         int(t_int[0]) + roi_buffer,
#         int(t_int[1]) + roi_buffer,
#     ]
#     t_roi = np.array([t[0] - roi[0], t[1] - roi[1]])
#     t_roi_int = np.round(t_roi).astype(int)

#     A = images[cams[cam_id]][0][roi[1] : roi[3], roi[0] : roi[2]]
#     B = images[cams[cam_id]][epoch][roi[1] : roi[3], roi[0] : roi[2]]

#     #  Viz template on starting image
#     if debug_viz:
#         template_coor = [
#             (t_roi_int[0] - template_width, t_roi_int[1] - template_width),
#             (t_roi_int[0] + template_width, t_roi_int[1] + template_width),
#         ]
#         win_name = "template"
#         img = cv2.cvtColor(deepcopy(A), cv2.COLOR_BGR2RGB)
#         cv2.circle(img, (t_roi_int[0], t_roi_int[1]), 0, (0, 255, 0), -1)
#         cv2.rectangle(img, template_coor[0], template_coor[1], (0, 255, 0), 1)
#         cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
#         cv2.imshow(win_name, img)
#         cv2.waitKey()
#         cv2.destroyAllWindows()

#     r = OC(
#         cv2.cvtColor(A, cv2.COLOR_RGB2GRAY),
#         cv2.cvtColor(B, cv2.COLOR_RGB2GRAY),
#         np.array(t_roi[0]),
#         np.array(t_roi[1]),
#         TemplateWidth=template_width,
#         SearchWidth=search_width,
#     )

#     t_est[epoch] = np.array([t[0] + r.du, t[1] + r.dv])

#     if debug:
#         t_meas = targets[epoch].extract_image_coor_by_label(targets_to_use, cam_id)[
#             0
#         ]
#         diff[epoch] = t_meas - t_est[epoch]
#         diff_noCC[epoch] = (
#             t_meas
#             - targets[0].extract_image_coor_by_label(targets_to_use, cam_id)[0]
#         )

#         img = cv2.imread(images[cams[cam_id]].get_image_path(epoch))
#         cv2.drawMarker(
#             img,
#             (
#                 np.round(t_est[epoch][0]).astype(int),
#                 np.round(t_est[epoch][1]).astype(int),
#             ),
#             (255, 0, 0),
#             cv2.MARKER_CROSS,
#             1,
#         )
#         cv2.imwrite("tmp/" + images[cams[cam_id]][epoch], img)
#         # with Image.open(images[cams[cam_id]].get_image_path(epoch)) as im:
#         #     draw = ImageDraw.Draw(im)
#         #     draw.ellipse(list(np.concatenate((t_est[epoch],t_est[epoch]))), outline=(255,0,0), width=1)
#         #     im.save('test.jpg', "JPEG")

#     if debug_viz:
#         fig, ax = plt.subplots(1, 2)
#         ax[0].imshow(A)
#         ax[0].scatter(t_roi[0], t_roi[1], s=50, c="r", marker="+")
#         ax[0].set_aspect("equal")
#         ax[1].imshow(B)
#         ax[1].scatter(t_roi[0] + r.du, t_roi[1] + r.dv, s=50, c="r", marker="+")
#         ax[1].set_aspect("equal")

# diff = np.stack((diff.values()))
# nans = np.isnan(diff[:, 0])
# print(f"{sum(nans)} invalid features")
# print_stats(diff[~nans, :])

# print("Without CC tracking:")
# diff_noCC = np.stack((diff_noCC.values()))
# print_stats(diff_noCC)

# print("Done")
