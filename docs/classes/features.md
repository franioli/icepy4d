# Feature and Features classes

You can easily import icepy4d classes by

```python
import icepy4d.classes as icepy4d_classes
```

and directly access to the Image and ImageDS classes by

```python
icepy4d_classes.Feature
```

::: icepy4d.classes.features.Feature
    handler: python
    options:
      members:
        - x
        - y
        - xy
        - track_id
        - descr
        - score
      members_order: "alphabetical"
      show_root_heading: true
      show_source: true

::: icepy4d.classes.features.Features
    handler: python
    options:
      members:
        - __len__
        - __getitem__
        - __contains__
        - __delitem__
        - __next__
        - num_features
        - last_track_id
        - get_track_ids
        - append_feature
        - set_last_track_id
        - append_features_from_numpy
        - to_numpy
        - kpts_to_numpy
        - descr_to_numpy
        - scores_to_numpy
        - get_features_as_dict
        - reset_fetures
        - filter_feature_by_mask
        - filter_feature_by_index
        - get_feature_by_index
        - save_as_txt
        - save_as_pickle
      show_root_heading: true
      show_source: true
