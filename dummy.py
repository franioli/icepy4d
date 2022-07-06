
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
