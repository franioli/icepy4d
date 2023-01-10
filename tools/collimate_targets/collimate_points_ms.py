import Metashape
import os


def write_markers_one_cam_per_file() -> None:
    """Write Marker image coordinates to csv file (one file per camera),
    named with camera label.
    Each file is formatted as follows:
    marker1, x, y
    marker2, x, y
    ...
    markerM, x, y

    """
    output_dir = Metashape.app.getExistingDirectory()

    doc = Metashape.app.document
    chunk = doc.chunk

    for camera in chunk.cameras:

        # Write header to file
        fname = os.path.join(output_dir, camera.label + ".csv")
        file = open(fname, "w")
        file.write("label,x,y\n")

        for marker in chunk.markers:
            projections = marker.projections  # list of marker projections
            marker_name = marker.label

            for cur_cam in marker.projections.keys():
                if cur_cam == camera:
                    x, y = projections[cur_cam].coord

                    # subtract 0.5 px to image coordinates (metashape image RS)
                    x -= 0.5
                    y -= 0.5

                    # writing output to file
                    file.write(f"{marker_name},{x:.4f},{y:.4f}\n")
        file.close()

    print("All targets exported successfully")


label = "Scripts/Export targets Belpy"
Metashape.app.addMenuItem(label, write_markers_one_cam_per_file)
print(f"To execute this script press {label}")
