# Point and Points classes

You can easily import icepy4d classes by

```python
import icepy4d.classes as icepy4d_classes
```

and directly access to the Image and ImageDS classes by

```python
icepy4d_classes.Point
```

::: icepy4d.classes.points.Point
    handler: python
    options:
      members:
        - X
        - Y
        - Z
        - track_id
        - coordinates
        - color
        - project
      members_order: "alphabetical"
      show_root_heading: true
      show_source: true

::: icepy4d.classes.points.Points
    handler: python
    options:
      members:
        - __len__
        - __getitem__
        - __contains__
        - __delitem__
        - __next__
        - num_points
        - last_track_id
        - get_track_ids
        - append_point
        - set_last_track_id
        - append_points_from_numpy
        - to_numpy
        - colors_to_numpy
        - to_point_cloud
        - reset_points
        - filter_point_by_mask
        - filter_points_by_index
        - get_points_by_index
        - save_as_txt
        - save_as_pickle
      show_root_heading: true
      show_source: true
