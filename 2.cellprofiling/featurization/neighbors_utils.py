import numpy
import skimage.measure
from loading_classes import ObjectLoader


def neighbors_expand_box(
    min_coor,
    max_coord,
    current_min,
    current_max,
    expand_by,
):
    if current_min - expand_by < min_coor:
        current_min = min_coor
    else:
        current_min -= expand_by
    if current_max + expand_by > max_coord:
        current_max = max_coord
    else:
        current_max += expand_by
    return current_min, current_max


# crop the image to the bbox of the mask
def crop_3D_image(image, bbox):
    z1, y1, x1, z2, y2, x2 = bbox
    return image[z1:z2, y1:y2, x1:x2]


def measure_3D_number_of_neighbors(
    object_loader: ObjectLoader,
    distance_threshold: int = 10,
    anisotropy_factor: int = 10,
):
    label_object = object_loader.objects
    labels = object_loader.object_ids
    # set image global min and max coordinates
    image_global_min_coord_z = 0
    image_global_min_coord_y = 0
    image_global_min_coord_x = 0
    image_global_max_coord_z = label_object.shape[0]
    image_global_max_coord_y = label_object.shape[1]
    image_global_max_coord_x = label_object.shape[2]

    neighbors_out_dict = {
        "object_id": [],
        "NEIGHBORS_ADJACENT": [],
        f"NEIGHBORS_{distance_threshold}": [],
    }
    for index, label in enumerate(labels):
        selected_label_object = label_object.copy()
        selected_label_object[selected_label_object != label] = 0
        props_label = skimage.measure.regionprops_table(
            selected_label_object, properties=["bbox"]
        )
        # get the number of neighbors for each object
        distance_x_y = distance_threshold
        distance_z = numpy.ceil(distance_threshold / anisotropy_factor).astype(int)
        # find how many other indexes are within a specified distance of the object
        # first expand the mask image by a specified distance
        z_min, y_min, x_min, z_max, y_max, x_max = (
            props_label["bbox-0"][0],
            props_label["bbox-1"][0],
            props_label["bbox-2"][0],
            props_label["bbox-3"][0],
            props_label["bbox-4"][0],
            props_label["bbox-5"][0],
        )

        new_z_min, new_z_max = neighbors_expand_box(
            min_coor=image_global_min_coord_z,
            max_coord=image_global_max_coord_z,
            current_min=z_min,
            current_max=z_max,
            expand_by=distance_z,
        )
        new_y_min, new_y_max = neighbors_expand_box(
            min_coor=image_global_min_coord_y,
            max_coord=image_global_max_coord_y,
            current_min=y_min,
            current_max=y_max,
            expand_by=distance_x_y,
        )
        new_x_min, new_x_max = neighbors_expand_box(
            min_coor=image_global_min_coord_x,
            max_coord=image_global_max_coord_x,
            current_min=x_min,
            current_max=x_max,
            expand_by=distance_x_y,
        )
        bbox = (new_z_min, new_y_min, new_x_min, new_z_max, new_y_max, new_x_max)
        croppped_neighbor_image = crop_3D_image(image=label_object, bbox=bbox)

        n_neighbors_by_distance = (
            len(numpy.unique(croppped_neighbor_image[croppped_neighbor_image > 0])) - 1
        )
        neighbors_out_dict["object_id"].append(label)
        neighbors_out_dict["NEIGHBORS_ADJACENT"].append(n_neighbors_by_distance)
        neighbors_out_dict[f"NEIGHBORS_{distance_threshold}"].append(
            n_neighbors_by_distance
        )

    return neighbors_out_dict
