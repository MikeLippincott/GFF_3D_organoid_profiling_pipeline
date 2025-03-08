import numpy
import scipy.ndimage
import skimage.segmentation


def get_outline(mask):
    outline = numpy.zeros_like(mask)
    for z in range(mask.shape[0]):
        outline[z] = skimage.segmentation.find_boundaries(mask[z])
    return outline


def measure_3D_intensity(image_object, label_object):

    non_zero_pixels_object = image_object[image_object > 0]
    mask_outlines = get_outline(label_object)
    mesh_z, mesh_y, mesh_x = numpy.mgrid[
        0 : image_object.shape[0],
        0 : image_object.shape[1],
        0 : image_object.shape[2],
    ]

    # calculate the integrated intensity
    integrated_intensity = numpy.sum(image_object)
    # calculate the volume
    volume = numpy.sum(label_object)
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

    return {
        "integrated_intensity": integrated_intensity,
        "volume": volume,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "lower_quartile_intensity": lower_quartile_intensity,
        "upper_quartile_intensity": upper_quartile_intensity,
        "median_intensity": median_intensity,
        "max_z": max_z,
        "max_y": max_y,
        "max_x": max_x,
        "cm_x": cm_x,
        "cm_y": cm_y,
        "cm_z": cm_z,
        "i_x": i_x,
        "i_y": i_y,
        "i_z": i_z,
        "cmi_x": cmi_x,
        "cmi_y": cmi_y,
        "cmi_z": cmi_z,
        "diff_x": diff_x,
        "diff_y": diff_y,
        "diff_z": diff_z,
        "mass_displacement": mass_displacement,
        "mad_intensity": mad_intensity,
        "edge_count": edge_count,
        "integrated_intensity_edge": integrated_intensity_edge,
        "mean_intensity_edge": mean_intensity_edge,
        "std_intensity_edge": std_intensity_edge,
        "min_intensity_edge": min_intensity_edge,
        "max_intensity_edge": max_intensity_edge,
    }
