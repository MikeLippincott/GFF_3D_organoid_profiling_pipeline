import numpy
import skimage.measure


def calulate_surface_area(label_object, props, spacing):

    # this seems less elegant than you might wish, given that regionprops returns a slice,
    # but we need to expand the slice out by one voxel in each direction, or surface area freaks out
    volume = label_object[
        max(props["bbox-0"][0] - 1, 0) : min(
            props["bbox-3"][0] + 1, label_object.shape[0]
        ),
        max(props["bbox-1"][0] - 1, 0) : min(
            props["bbox-4"][0] + 1, label_object.shape[1]
        ),
        max(props["bbox-2"][0] - 1, 0) : min(
            props["bbox-5"][0] + 1, label_object.shape[2]
        ),
    ]
    volume_truths = volume == 1
    verts, faces, _normals, _values = skimage.measure.marching_cubes(
        volume_truths,
        method="lewiner",
        spacing=spacing,
        level=0,
    )
    surface_area = skimage.measure.mesh_surface_area(verts, faces)

    return surface_area


def measure_3D_area_size_shape(label_object, spacing):
    """Computing the measurements for a single map of objects"""

    desired_properties = [
        "area",
        "major_axis_length",
        "minor_axis_length",
        "bbox",
        "centroid",
        "bbox_area",
        "extent",
        "euler_number",
        "equivalent_diameter",
    ]

    props = skimage.measure.regionprops_table(
        label_object, properties=desired_properties
    )

    # SurfaceArea
    surfacearea = calulate_surface_area(label_object, props, spacing)

    features_to_record = {
        "VOLUME": props["area"],
        "SURFACE_AREA": surfacearea,
        "MAJOR_AXIS_LENGTH": props["major_axis_length"],
        "MINOR_AXIS_LENGTH": props["minor_axis_length"],
        "CENTER_X": props["centroid-2"],
        "CENTER_Y": props["centroid-1"],
        "CENTER_Z": props["centroid-0"],
        "BBOX_VOLUME": props["bbox_area"],
        "MIN_X": props["bbox-2"],
        "MAX_X": props["bbox-5"],
        "MIN_Y": props["bbox-1"],
        "MAX_Y": props["bbox-4"],
        "MIN_Z": props["bbox-0"],
        "MAX_Z": props["bbox-3"],
        "EXTENT": props["extent"],
        "EULER_NUMBER": props["euler_number"],
        "EQUIVALENT_DIAMETER": props["equivalent_diameter"],
    }
    return features_to_record
