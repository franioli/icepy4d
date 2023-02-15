import cv2
import torch
import logging


def process_resize(w, h, resize):
    assert len(resize) > 0 and len(resize) <= 2
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # # Issue warning if resolution is too small or too large.
    # if max(w_new, h_new) < 160:
    #     logging.warning("Warning: input resolution is very small, results may vary")
    # elif max(w_new, h_new) > 2000:
    #     logging.warning("Warning: input resolution is very large, results may vary")

    return w_new, h_new


def frame2tensor(frame, device):
    return torch.from_numpy(frame / 255.0).float()[None, None].to(device)


def read_image(path, device, resize, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype("float32"), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype("float32")

    tensor = frame2tensor(image, device)
    return image, tensor, scales
