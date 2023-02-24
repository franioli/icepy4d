# Image and ImageDS classes

You can easily import Icepy classes by

```python
import icepy.classes as icepy_classes
```

and directly access to the Image and ImageDS classes by

```python
icepy_classes.Image
```

::: icepy.classes.images.Image
    handler: python
    options:
      members:
        - height
        - width
        - name
        - stem
        - path
        - parent
        - extension
        - exif
        - date
        - time
        - value
        - get_datetime
        - read_image
        - reset_image
        - read_exif
        - extract_patch
        - get_intrinsics_from_exif
        - undistort_image
      show_root_heading: true
      show_source: true

::: icepy.classes.images.ImageDS
    handler: python
    options:
      members:
        - __len__
        - __contains__
        - __getitem__
        - __next__
        - reset_imageds
        - read_image_list
        - read_image
        - read_dates
        - get_image_path
        - get_image_stem
        - get_image_date
        - get_image_time
        - write_exif_to_csv
      show_root_heading: true
      show_source: true
