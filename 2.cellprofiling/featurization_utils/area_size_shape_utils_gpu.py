import cucim.skimage.measure
import cupy
import cupyx
import cupyx.scipy.ndimage
import numpy
import scipy
import skimage
import skimage.measure
from loading_classes import ImageSetLoader, ObjectLoader


def calulate_surface_area(label_object, props, spacing):

    # this seems less elegant than you might wish, given that regionprops returns a slice,
    # but we need to expand the slice out by one voxel in each direction, or surface area freaks out
    surface_areas = []
    for index, label in enumerate(props["label"]):

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
        volume_truths = volume > 0
        verts, faces, _normals, _values = cucim.skimage.measure.marching_cubes(
            volume_truths,
            method="lewiner",
            spacing=spacing,
            level=0,
        )
        surface_areas.append(cucim.skimage.measure.mesh_surface_area(verts, faces))

    return surface_areas


def measure_3D_area_size_shape_gpu(image_set_loader, object_loader):
    label_object = cupy.asarray(object_loader.objects)
    spacing = image_set_loader.spacing
    unique_objects = object_loader.object_ids
    desired_properties = [
        "area",
        "bbox",
        "centroid",
        "bbox_area",
        "extent",
        "euler_number",
        "equivalent_diameter",
        "solidity",
    ]

    props = cucim.skimage.measure.regionprops_table(
        label_object, properties=desired_properties
    )
    props["label"] = unique_objects
    features_to_record = {
        "object_id": props["label"],
        "VOLUME": props["area"],
        "CENTER.X": props["centroid-2"],
        "CENTER.Y": props["centroid-1"],
        "CENTER.Z": props["centroid-0"],
        "BBOX.VOLUME": props["bbox_area"],
        "MIN.X": props["bbox-2"],
        "MAX.X": props["bbox-5"],
        "MIN.Y": props["bbox-1"],
        "MAX.Y": props["bbox-4"],
        "MIN.Z": props["bbox-0"],
        "MAX.Z": props["bbox-3"],
        "EXTENT": props["extent"],
        "EULER.NUMBER": props["euler_number"],
        "EQUIVALENT.DIAMETER": props["equivalent_diameter"],
    }
    try:
        features_to_record["SURFACE.AREA"] = calulate_surface_area(
            label_object=label_object,
            props=props,
            spacing=spacing,
        )
    except:
        features_to_record["SURFACE.AREA"] = cupy.nan
    return features_to_record
