import Metashape as ps
import Metashape
import copy
from pathlib import Path

chunk = ps.app.document.chunk
cameras = chunk.cameras
markers = chunk.markers

camera = cameras[0]

# T_chunk_2_world
T = chunk.transform.matrix

for camera in cameras:
    # Pose in chunk reference system
    pose = camera.transform

    # Pose in world rs
    pose_w = T * pose
    Cw = pose_w.translation()

    Rw = pose_w.rotation().inv()
    A = Metashape.Matrix.Rotation(Rw) * Metashape.Matrix.Translation(Cw)
    tw = -A.translation()

    # E = Metashape.Matrix.Rotation(Rw) * Metashape.Matrix.Translation(tw)
    fname = Path("phd/belpy") / f"{camera.label}_extrinsics.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for x in range(3):
            f.write(f"{Rw.row(x)[0]} {Rw.row(x)[1]} {Rw.row(x)[2]} {tw[x]}\n")
        f.write(f"{0.0} {0.0} {0.0} {1.0}\n")
        f.close()

# # R,t in chunk reference system
# R = camera.transform.rotation().t()
# t = -(R * camera.transform.translation())

# # Extrinsics (E) in chunk reference system
# E = Metashape.Matrix.Diag([1, 1, 1, 1])
# E = E.Rotation(R) * E.Translation(t)

# # Extrinsics in world reference system
# Ew = T * E
# tw = E.translation()
# Rw = E.rotation()


# position = camera.reference.location
# orientation = ps.Vector([-83.85, 79.86, 174.14])
# position = chunk.crs.unproject(position)

# Projections
P = chunk.transform.matrix.inv().mulp(Metashape.Vector([49.6488, 192.0875, 71.7466]))
p = camera.project(P)
