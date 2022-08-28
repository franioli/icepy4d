
# %% Plot targets
# #  Plot with matplotlib
# cam = 1
# i = 0
# _plot_style = dict(markersize=5, markeredgewidth=2,
#                     markerfacecolor='none', markeredgecolor='r',
#                     marker='x', linestyle='none')

# xy = targets.get_im_coord(cam)[i]
# img = cv2.cvtColor(images[cam][i], cv2.COLOR_BGR2RGB)
# fig, ax = plt.subplots()
# ax.imshow(img) 
# ax.plot(xy[0], xy[1], **_plot_style)
    
# %% Save reconstruction
# with h5py.File("test.hdf5", "w") as f:
#     dset = f.create_dataset("features", data=features)
    
    
# dset = h5py.File("test.hdf5", "r")
# dset.close()


#%% Test cameras in Open3d
# print("Testing camera in open3d ...")
# #  width, height, fx, fy, cx, cyi
# intrisics = o3d.camera.PinholeCameraIntrinsic(ref_cams[cam0].width,
#                                               ref_cams[cam0].heigth,
#                                               ref_cams[cam0].K[0,0],
#                                               ref_cams[cam0].K[1,1],
#                                               ref_cams[cam0].K[0,2],
#                                               ref_cams[cam0].K[1,2],
#                                               )
# # o3d.io.write_pinhole_camera_intrinsic("test.json", x)
# # y = o3d.io.read_pinhole_camera_intrinsic("test.json")
# # print(y)
# # print(np.asarray(y.intrinsic_matrix))

# x = o3d.camera.PinholeCameraParameters()
# x.intrinsic = intrisics
# x.extrinsic = cameras[cam0][epoch].extrinsics
# print(x.intrinsic)
# print(x.extrinsic)

# # o3d.visualization.draw_geometries([pcd[epoch], x])

# %% MESH 
# epoch = 0
# mesh = []

# pcd[epoch]

# alpha = 10
# print(f"alpha={alpha:.3f}")
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd[epoch], alpha)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


# pcd[epoch].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=10))
# radii = [1, 3, 5, 10]
# rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     pcd[epoch], o3d.utility.DoubleVector(radii))
# o3d.visualization.draw_geometries([pcd[epoch], rec_mesh])


# print('run Poisson surface reconstruction')
# with o3d.utility.VerbosityContextManager(
#         o3d.utility.VerbosityLevel.Debug) as cm:
#     pcd[epoch].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=10))
#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
#         pcd[epoch], depth=9)
    
# print(mesh)
# o3d.visualization.draw_geometries([mesh],
#                                   zoom=0.664,
#                                   front=[-0.4761, -0.4698, -0.7434],
#                                   lookat=[1.8900, 3.2596, 0.9284],
#                                   up=[0.2304, -0.8825, 0.4101])

#%% 
# vis = o3d.visualization.VisualizerWithKeyCallback()
# vis.create_window()
# vis.get_render_option().background_color = np.asarray([0.4, 0.4, 0.4])
# view_ctl = vis.get_view_control()
# vis.add_geometry(pcd[0])

# Rx = o3d.geometry.Geometry3D.get_rotation_matrix_from_axis_angle(np.array([1., 0., 0.], 
#                                                                           dtype='float64')*np.pi)
# T = np.eye(4)
# T[0:3,0:3] = Rx 
# pose = T @ cameras[cam0][0].pose
# cam = view_ctl.convert_to_pinhole_camera_parameters()
# cam.extrinsic = pose # where T is your matrix
# view_ctl.convert_from_pinhole_camera_parameters(cam)
# vis.run()
# vis.destroy_window()

#%% Viz point cloud with cameras
#TODO: make wrapper around point cloud plot with cameras
# epoch = 0
# cam_syms = []
# cam_colors = [[1,0,0],[0,0,1]]
# for i, cam in enumerate(cam_names):
#     cam_syms.append(make_camera_pyramid(cameras[cam][epoch], 
#                                         cam_colors[i],
#                                         focal_len_scaled=30,
#                                         ) )
# # o3d.visualization.draw_geometries([pcd[epoch], cam_syms[0], cam_syms[1]])


# viewer = o3d.visualization.Visualizer()
# viewer.create_window()
# viewer.add_geometry(pcd[epoch])
# # for geometry in geometries:
# #     viewer.add_geometry(geometry)
# opt = viewer.get_render_option()
# opt.show_coordinate_frame = True
# opt.background_color = np.asarray([0.2, 0.2, 0.2])
# viewer.run()
# viewer.destroy_window()

#%% Rotating RS
# epoch = 0
# c0, c1 = cameras[cam0][epoch], cameras[cam1][epoch]

# # Perform rotation of 180deg around X axis   
# if rotate_RS:
#     ang = np.pi
#     Rx = o3d.geometry.Geometry3D.get_rotation_matrix_from_axis_angle(np.array([1., 0., 0.], 
#                                                                               dtype='float64')*ang)
#     # Rotate point clouds
#     pcd[epoch].rotate(Rx)
    
#     # Rotatate Cameras and update projection matrixes
#     T = np.eye(4)
#     T[0:3,0:3] = Rx 
#     c0.extrinsics = T @ c0.extrinsics
#     c0.update_camera_from_extrinsics()
#     c1.extrinsics = T @ c1.extrinsics
#     c1.update_camera_from_extrinsics()
    
#     # for cam in cam_names:
#         # cameras[cam][epoch].R = np.dot(Rx, cameras[cam][epoch].R)
#         # cameras[cam][epoch].t = np.dot(Rx, cameras[cam][epoch].t)
#         # cameras[cam][epoch].compose_P()
#         # cameras[cam][epoch].C_from_P() 
#     print('Reference system rotated by 180 degrees around X axis')
    
# cam_syms = []
# cam_colors = [[1,0,0],[0,0,1]]
# for i, cam in enumerate(cam_names):
#     cam_syms.append(make_camera_pyramid(cameras[cam][epoch],
#                                         color=cam_colors[i],
#                                         focal_len_scaled=30,
#                                         ))
# o3d.visualization.draw_geometries([pcd[epoch], cam_syms[0], cam_syms[1]])
