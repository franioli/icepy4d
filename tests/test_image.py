import pytest
import numpy as np
import os

from pathlib import Path

from icepy4d.classes.images import ImageDS, Image


def test_image_ds(data_dir):
    images = ImageDS(data_dir / "img/cam1")
    assert isinstance(images, ImageDS), "Unable to build an image datastore"
    assert images[0] == "IMG_2637.jpg", "Unable to retrieve image name from image DS"
    assert (
        images.get_image_stem(0) == "IMG_2637"
    ), "Unable to retrieve image stem from image DS"
    assert (
        images.get_image_path(0) == data_dir / "img/cam1/IMG_2637.jpg"
    ), "Unable to retrieve image path from image DS"
    assert (
        images.get_image_date(0) == "2022:05:01"
    ), "Unable to retrieve image date from exif"
    assert (
        images.get_image_time(0) == "14:01:15"
    ), "Unable to retrieve image time from exif"
    # Test ImageDS iterator
    assert (
        next(images) == data_dir / "img/cam1/IMG_2637.jpg"
    ), "Unable to iterate over ImageDS datastore"


def test_image_class(data_dir):
    images = ImageDS(data_dir / "img/cam1")
    assert isinstance(
        images.read_image(0), Image
    ), "Unable to read image from Image DS as Image object"
    img = images.read_image(0)
    assert (
        img.date == "2022:05:01"
    ), "Unable to retrieve image date from exif with Image object"
    assert (
        img.time == "14:01:15"
    ), "Unable to retrieve image time from exif with Image object"
    # TODO: add tests on build K from sensor database
