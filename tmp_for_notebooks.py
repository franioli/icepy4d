# Bakcup of old code for making explanatory notebooks

    # Load existing epcoh
    if LOAD_EXISTING_SOLUTION:
        path = f"{epochdir}/{epoch_dict[ep]}.pickle"
        logging.info(f"Loading epoch from {path}")
        epoch = Epoch.read_pickle(path, ignore_errors=True)
        if epoch is not None:
            epoches.add_epoch(epoch)
            logging.info("Epoch loaded.")

            # matches_fig_dir = "res/fig_for_paper/matches_fig"
            # make_matching_plot(epoch, epoch, matches_fig_dir, show_fig=False)

            del epoch
            continue
        else:
            logging.error("Unable to import epoch.")
    else:
        # Create new epoch
        epoch = initializer.init_epoch(epoch_id=ep, epoch_dir=epochdir)
        epoches.add_epoch(epoch)

        # NOTE: Move this part of code to a notebook for an example of how to create a new epoch
        # im_epoch: icepy4d_classes.ImagesDict = {
        #     cam: icepy4d_classes.Image(images[cam].get_image_path(ep)) for cam in cams
        # }

        # # Load targets
        # target_paths = [
        #     cfg.georef.target_dir / (im_epoch[cam].stem + cfg.georef.target_file_ext)
        #     for cam in cams
        # ]
        # targ_ep = icepy4d_classes.Targets(
        #     im_file_path=target_paths,
        #     obj_file_path=cfg.georef.target_dir / cfg.georef.target_world_file,
        # )

        # # Load cameras
        # cams_ep: icepy4d_classes.CamerasDict = {}
        # for cam in cams:
        #     calib = icepy4d_classes.Calibration(
        #         cfg.paths.calibration_dir / f"{cam}.txt"
        #     )
        #     cams_ep[cam] = calib.to_camera()

        # cams_ep = {
        #     cam: icepy4d_classes.Calibration(
        #         cfg.paths.calibration_dir / f"{cam}.txt"
        #     ).to_camera()
        #     for cam in cams
        # }

        # # init empty features and points
        # feat_ep = {cam: icepy4d_classes.Features() for cam in cams}
        # pts_ep = icepy4d_classes.Points()

        # epoch = Epoch(
        #     im_epoch[cams[0]].datetime,
        #     images=im_epoch,
        #     cameras=cams_ep,
        #     features=feat_ep,
        #     points=pts_ep,
        #     targets=targ_ep,
        #     point_cloud=None,
        #     epoch_dir=epochdir,
        # )
        # epoches.add_epoch(epoch)

        # del im_epoch, cams_ep, feat_ep, pts_ep, targ_ep, target_paths

    # Perform matching and tracking
    if cfg.proc.do_matching:
        if DO_PRESELECTION:
            if cfg.proc.do_tracking and ep > cfg.proc.epoch_to_process[0]:
                epoch.features = tracking_base(
                    images,
                    epoches[ep - 1].features,
                    cams,
                    epoch_dict,
                    ep,
                    cfg.tracking,
                    epochdir,
                )

            epoch.features = match_by_preselection(
                images,
                epoch.features,
                cams,
                ep,
                cfg.matching,
                match_dir,
                n_tiles=4,
                n_dist=1.5,
                viz_results=True,
                fast_viz=True,
            )
        else:
            features_old = MatchingAndTracking(
                cfg=cfg,
                epoch=ep,
                images=images,
                features=features_old,
                epoch_dict=epoch_dict,
            )
            epoch.features = features_old[ep]

        # Run additional matching on selected patches:
        if DO_ADDITIONAL_MATCHING:
            logging.info("Performing additional matching on user-specified patches")
            im_stems = [epoch.images[cam].stem for cam in cams]
            sg_opt = {
                "weights": cfg.matching.weights,
                "keypoint_threshold": 0.0001,
                "max_keypoints": 8192,
                "match_threshold": 0.2,
                "force_cpu": False,
            }
            for i, patches_lim in enumerate(PATCHES):
                find_matches_on_patches(
                    images=images,
                    patches_lim=patches_lim,
                    epoch=ep,
                    features=epoch.features,
                    cfg=sg_opt,
                    do_geometric_verification=True,
                    geometric_verification_threshold=10,
                    viz_results=True,
                    fast_viz=True,
                    viz_path=match_dir
                    / f"{im_stems[0]}_{im_stems[1]}_matches_patch_{i}.png",
                )

            # Run again geometric verification
            geometric_verification(
                epoch.features,
                threshold=cfg.matching.pydegensac_threshold,
                confidence=cfg.matching.pydegensac_confidence,
            )
            logging.info("Matching by patches completed.")

            # For debugging
            # for cam in cams:
            #     epoch.features[cam].plot_features(images[cam].read_image(ep).value)
    else:
        try:
            epoch.features = load_matches_from_disk(match_dir)
        except FileNotFoundError as err:
            logging.exception(err)
            logging.warning("Performing new matching and tracking...")
            features_old = MatchingAndTracking(
                cfg=cfg,
                epoch=ep,
                images=images,
                features=features_old,
                epoch_dict=epoch_dict,
            )
            epoch.features = features_old[ep]

    timer.update("matching")


    """ SfM """

    logging.info(f"Reconstructing epoch {ep}...")

    # --- Space resection of Master camera ---#
    # At the first ep, perform Space resection of the first camera
    # by using GCPs. At all other epoches, set camera 1 EO equal to first one.
    if cfg.proc.do_space_resection and ep == 0:
        """Initialize Single_camera_geometry class with a cameras object"""
        space_resection = sfm.Space_resection(epoch.cameras[cams[0]])
        space_resection.estimate(
            epoch.targets.get_image_coor_by_label(cfg.georef.targets_to_use, cam_id=0)[
                0
            ],
            epoch.targets.get_object_coor_by_label(cfg.georef.targets_to_use)[0],
        )
        # Store result in camera 0 object
        epoch.cameras[cams[0]] = space_resection.camera
