import numpy
import scipy.ndimage
import skimage.segmentation


def get_outline(mask):
    outline = numpy.zeros_like(mask)
    for z in range(mask.shape[0]):
        outline[z] = skimage.segmentation.find_boundaries(mask[z])
    return outline


def measure_3D_intensity(object_loader):
    image_object = object_loader.image
    label_object = object_loader.objects
    labels = object_loader.object_ids
    ranges = len(labels)

    output_dict = {
        "object_id": [],
        "feature_name": [],
        "channel": [],
        "compartment": [],
        "value": [],
    }
    for index, label in enumerate(labels):
        selected_label_object = label_object.copy()
        selected_image_object = image_object.copy()

        selected_label_object[selected_label_object != label] = 0
        selected_image_object[selected_label_object == 0] = 0
        non_zero_pixels_object = selected_image_object[selected_image_object > 0]

        mask_outlines = get_outline(selected_label_object)
        mesh_z, mesh_y, mesh_x = numpy.mgrid[
            0 : selected_image_object.shape[0],
            0 : selected_image_object.shape[1],
            0 : selected_image_object.shape[2],
        ]

        ranges = len(numpy.unique(selected_label_object))

        # calculate the integrated intensity
        integrated_intensity = scipy.ndimage.sum(
            selected_image_object,
            selected_label_object,
        )
        # calculate the volume
        volume = numpy.sum(selected_label_object)

        # calculate the mean intensity
        mean_intensity = integrated_intensity / volume
        # calculate the standard deviation
        std_intensity = numpy.std(non_zero_pixels_object)
        # min intensity
        min_intensity = numpy.min(non_zero_pixels_object)
        # max intensity
        max_intensity = numpy.max(non_zero_pixels_object)
        # lower quartile
        lower_quartile_intensity = numpy.percentile(non_zero_pixels_object, 25)
        # upper quartile
        upper_quartile_intensity = numpy.percentile(non_zero_pixels_object, 75)
        # median intensity
        median_intensity = numpy.median(non_zero_pixels_object)
        # max intensity location
        max_z, max_y, max_x = scipy.ndimage.maximum_position(
            image_object,
        )  # z, y, x
        cm_x = scipy.ndimage.mean(mesh_x)
        cm_y = scipy.ndimage.mean(mesh_y)
        cm_z = scipy.ndimage.mean(mesh_z)
        i_x = scipy.ndimage.sum(mesh_x)
        i_y = scipy.ndimage.sum(mesh_y)
        i_z = scipy.ndimage.sum(mesh_z)
        # calculate the center of mass
        cmi_x = i_x / integrated_intensity
        cmi_y = i_y / integrated_intensity
        cmi_z = i_z / integrated_intensity
        # calculate the center of mass distance
        diff_x = cm_x - cmi_x
        diff_y = cm_y - cmi_y
        diff_z = cm_z - cmi_z
        # mass displacement
        mass_displacement = numpy.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
        # mean aboslute deviation
        mad_intensity = numpy.mean(numpy.abs(non_zero_pixels_object - mean_intensity))
        edge_count = scipy.ndimage.sum(mask_outlines)
        integrated_intensity_edge = numpy.sum(image_object[mask_outlines > 0])
        mean_intensity_edge = integrated_intensity_edge / edge_count
        std_intensity_edge = numpy.std(image_object[mask_outlines > 0])
        min_intensity_edge = numpy.min(image_object[mask_outlines > 0])
        max_intensity_edge = numpy.max(image_object[mask_outlines > 0])
        measurements_dict = {
            "INTEGRATED.INTENSITY": integrated_intensity,
            "VOLUME": volume,
            "MEAN.INTENSITY": mean_intensity,
            "STD.INTENSITY": std_intensity,
            "MIN.INTENSITY": min_intensity,
            "MAX.INTENSITY": max_intensity,
            "LOWER.QUARTILE.INTENSITY": lower_quartile_intensity,
            "UPPER.QUARTILE.INTENSITY": upper_quartile_intensity,
            "MEDIAN.INTENSITY": median_intensity,
            "MAX.Z": max_z,
            "MAX.Y": max_y,
            "MAX.X": max_x,
            "CM.X": cm_x,
            "CM.Y": cm_y,
            "CM.Z": cm_z,
            "I.X": i_x,
            "I.Y": i_y,
            "I.Z": i_z,
            "CMI.X": cmi_x,
            "CMI.Y": cmi_y,
            "CMI.Z": cmi_z,
            "DIFF.X": diff_x,
            "DIFF.Y": diff_y,
            "DIFF.Z": diff_z,
            "MASS.DISPLACEMENT": mass_displacement,
            "MAD.INTENSITY": mad_intensity,
            "EDGE.COUNT": edge_count,
            "INTEGRATED.INTENSITY.EDGE": integrated_intensity_edge,
            "MEAN.INTENSITY.EDGE": mean_intensity_edge,
            "STD.INTENSITY.EDGE": std_intensity_edge,
            "MIN.INTENSITY.EDGE": min_intensity_edge,
            "MAX.INTENSITY.EDGE": max_intensity_edge,
        }

        for feature_name, value in measurements_dict.items():

            output_dict["object_id"].append(label)
            output_dict["feature_name"].append(feature_name)
            output_dict["channel"].append(object_loader.channel)
            output_dict["compartment"].append(object_loader.compartment)
            output_dict["value"].append(value)
    return output_dict
