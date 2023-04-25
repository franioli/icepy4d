# %%
from pathlib import Path
import open3d as o3d

# import cloudComPy as cc  # import the CloudComPy module


def save_to_las():
    STEP = 10

    root_path = Path.cwd()
    path = "res/point_clouds"
    pattern = "dense_*.ply"
    dest_dir = Path("pcd_potree")

    dest_dir.mkdir(exist_ok=True)
    path = Path(root_path) / path

    files = sorted(Path(path).glob(f"{pattern}"))
    for i, file in enumerate(files):
        if i % STEP != 0:
            continue
        print(f"Processing {file}")
        pcd = cc.loadPointCloud(str(file))
        ret = cc.SavePointCloud(pcd, str(dest_dir / f"{file.stem}.las"))
        cc.deleteEntity(pcd)


def subsample():
    MIN_DIST = 1.0

    pcd_path = "/home/francesco/phd/open-data-zenodo/belvedere/pcd"
    pattern = "*.laz"
    dest_dir = "/home/francesco/phd/open-data-zenodo/belvedere/pcd_potree"

    pcd_path = Path(pcd_path)
    files = pcd_path.glob(f"{pattern}")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)

    for i, file in enumerate(files):
        print(f"Processing {file.stem}...", end="")
        pcd = cc.loadPointCloud(str(file))
        print("loaded...", end="")
        params = cc.SFModulationParams()
        pcd_subsampled = cc.CloudSamplingTools.resampleCloudSpatially(
            pcd, MIN_DIST, params
        )
        (pcd_subsampled, res) = pcd.partialClone(pcd_subsampled)
        print("subsampled...", end="")
        ret = cc.SavePointCloud(pcd_subsampled, str(dest_dir / f"{file.stem}.las"))
        print("saved")
        cc.deleteEntity(pcd)
        cc.deleteEntity(pcd_subsampled)


if __name__ == "__main__":
    import icepy4d.visualization as vis
    from icepy4d.classes import PointCloud

    root_path = Path.cwd()
    path = "res/point_clouds"
    pattern = "dense_*.ply"
    dest_dir = Path("pcd_potree")

    dest_dir.mkdir(exist_ok=True)
    path = Path(root_path) / path

    files = sorted(Path(path).glob(f"{pattern}"))

    id = 100
    # pcd = o3d.io.read_point_cloud(str(files[id]))
    # o3d.visualization.draw_geometries([pcd])
    pcd = PointCloud(pcd_path=str(files[id]))
    vis.display_point_cloud(pcd)
    # subsample()
