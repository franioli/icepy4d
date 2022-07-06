import open3d as o3d

from pathlib import Path

from lib.misc import create_directory

def create_point_cloud(points3d, points_col=None, path=None, *scalar_fied):
    ''' Function to create a point cloud object by using Open3D library.
    Input:  (nx3, float32) array of points 3D.
            (nx3, float32) array of color of each point. 
                Colors are defined in [0,1] range as float numbers. 
            Path were to save the point cloud to disk in ply format. 
                If path is None, the point cloud is not saved to disk.
            Scalar fields: to be implemented.
            #TODO: implement scalar fields.
    Return: Open3D point cloud object
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3d)
    if points_col is not None:
        pcd.colors = o3d.utility.Vector3dVector(points_col)
    if path is not None:
        write_ply(pcd, path)
    return pcd

def write_ply(pcd, path):
    ''' Write point cloud to disk as .ply

    Parameters
    ----------
    pcd : O3D point cloud
    out_path : Path or str of output ply

    Returns: None
    '''
    path = Path(path)
    create_directory(path.parent) 
    o3d.io.write_point_cloud(str(path), pcd)  
    
    
def display_pc_inliers(cloud, ind):
    ''' Display a O3D point cloud, separating inliers from outliers 
    (e.g. after a SOR filter)
    Parameters
    ----------
    cloud : O3D obejct
        Point cloud with n points.
    ind : (nx1) List of int
        List of indices of the inlier points
    Returns
    -------
    None.
    '''
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])