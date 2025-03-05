import numpy
import skimage


def measure_area_size_shape(label_object, spacing):
    """Computing the measurements for a single map of objects"""
    # labels = objects.segmented
    # nobjects = len(objects.indices)

    desired_properties = [
        "area",
        # "surface_areas",
        "major_axis_length",
        "minor_axis_length",
        # "centroid-2",
        # "centroid-1",
        # "centroid-0",
        "bbox_area",
        # "bbox-2",
        # "bbox-5",
        # "bbox-1",
        # "bbox-4",
        # "bbox-0",
        # "bbox-3",
        "extent",
        "euler_number",
        "equivalent_diameter",
        "solidity",
    ]

    props = skimage.measure.regionprops_table(
        label_object, properties=desired_properties
    )

    # SurfaceArea
    surface_areas = numpy.zeros(len(props["label"]))
    for index, label in enumerate(props["label"]):
        # this seems less elegant than you might wish, given that regionprops returns a slice,
        # but we need to expand the slice out by one voxel in each direction, or surface area freaks out
        volume = label_object[
            max(props["bbox-0"][index] - 1, 0) : min(
                props["bbox-3"][index] + 1, label_object.shape[0]
            ),
            max(props["bbox-1"][index] - 1, 0) : min(
                props["bbox-4"][index] + 1, label_object.shape[1]
            ),
            max(props["bbox-2"][index] - 1, 0) : min(
                props["bbox-5"][index] + 1, label_object.shape[2]
            ),
        ]
        volume = volume == label
        verts, faces, _normals, _values = skimage.measure.marching_cubes(
            volume,
            method="lewiner",
            spacing=spacing,
            level=0,
        )
        surface_areas[index] = skimage.measure.mesh_surface_area(verts, faces)

    features_to_record = {
        "VOLUME": props["area"],
        "SURFACE_AREA": surface_areas,
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
        "SOLIDITY": props["solidity"],
    }
    return features_to_record
