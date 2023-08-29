# Import required standard modules


# Import required icepy4d4D modules
from icepy4d import classes as icepy4d_classes
from icepy4d.classes.epoch import Epoch
from icepy4d import matching
from icepy4d import io
from icepy4d.utils import initialization as inizialization


# Define processing for single epoch
def process_epoch(epoch, epoch_id, epoch_dict, cfg, logger, timer) -> Epoch:
    cams = cfg.cams
    epochdir = cfg.paths.results_dir / epoch_dict[epoch_id]
    match_dir = epochdir / "matching"

    if cfg.proc.load_existing_results:
        try:
            epoch = Epoch.read_pickle(epochdir / f"{epoch_dict[ep]}.pickle")

            # Compute reprojection error
            io.write_reprojection_error_to_file(cfg.residuals_fname, epoches[ep])

            # Save focal length to file
            io.write_cameras_to_file(cfg.camera_estimated_fname, epoches[ep])

            return epoch

        except:
            logger.error(
                f"Unable to load epoch {epoch_dict[ep]} from pickle file. Creating new epoch..."
            )
            epoch = inizialization.initialize_epoch(
                cfg=cfg, images=images, epoch_id=ep, epoch_dir=epochdir
            )

    else:
        epoch = inizialization.initialize_epoch(
            cfg=cfg, images=images, epoch_id=ep, epoch_dir=epochdir
        )

    # Matching
    matching_quality = matching.Quality.HIGH
    tile_selection = matching.TileSelection.PRESELECTION
    tiling_grid = [4, 3]
    tiling_overlap = 200
    geometric_verification = matching.GeometricVerification.PYDEGENSAC
    geometric_verification_threshold = 1
    geometric_verification_confidence = 0.9999

    matcher = matching.SuperGlueMatcher(cfg.matching)
    matcher.match(
        epoch.images[cams[0]].value,
        epoch.images[cams[1]].value,
        quality=matching_quality,
        tile_selection=tile_selection,
        grid=tiling_grid,
        overlap=tiling_overlap,
        do_viz_matches=True,
        do_viz_tiles=False,
        save_dir=match_dir,
        geometric_verification=geometric_verification,
        threshold=geometric_verification_threshold,
        confidence=geometric_verification_confidence,
    )

    f = {cam: icepy4d_classes.Features() for cam in cams}
    f[cams[0]].append_features_from_numpy(
        x=matcher.mkpts0[:, 0],
        y=matcher.mkpts0[:, 1],
        descr=matcher.descriptors0,
        scores=matcher.scores0,
    )
    f[cams[1]].append_features_from_numpy(
        x=matcher.mkpts1[:, 0],
        y=matcher.mkpts1[:, 1],
        descr=matcher.descriptors1,
        scores=matcher.scores1,
    )
    epoch.features = f

    timer.update("matching")

    # Relative orientation
