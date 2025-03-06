from typing import Dict

import numpy
import scipy
import skimage


def granularity_feature(length, image_name):
    C_GRANULARITY = "Granularity_%s_%s"
    return C_GRANULARITY % (length, image_name)


def measure_3D_granularity(
    image_object: numpy.ndarray,
    label_object: numpy.ndarray,
    radius: int = 20,
    granular_spectrum_length: int = 5,
    subsample_size: float = 0.25,
    image_name: str = "image",
) -> Dict[str, float]:

    pixels = image_object.copy()
    mask = label_object.copy()

    # begin by subsampling the image
    new_shape = numpy.array(pixels.shape)
    new_shape = new_shape * subsample_size
    k, i, j = (
        numpy.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(float)
        / subsample_size
    )
    pixels = scipy.ndimage.map_coordinates(pixels, (k, i, j), order=1)
    mask = scipy.ndimage.map_coordinates(mask.astype(float), (k, i, j)) > 0.9

    back_shape = new_shape * subsample_size
    k, i, j = (
        numpy.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(float)
        / subsample_size
    )
    back_pixels = scipy.ndimage.map_coordinates(pixels, (k, i, j), order=1)
    back_mask = scipy.ndimage.map_coordinates(mask.astype(float), (k, i, j)) > 0.9

    footprint = skimage.morphology.ball(radius, dtype=bool)

    back_pixels_mask = numpy.zeros_like(back_pixels)
    back_pixels_mask[back_mask == True] = back_pixels[back_mask == True]
    back_pixels = skimage.morphology.erosion(back_pixels_mask, footprint=footprint)

    back_pixels_mask = numpy.zeros_like(back_pixels)
    back_pixels_mask[back_mask == True] = back_pixels[back_mask == True]

    back_pixels = skimage.morphology.dilation(back_pixels_mask, footprint=footprint)
    k, i, j = numpy.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(
        float
    )
    k *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
    i *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
    j *= float(back_shape[2] - 1) / float(new_shape[2] - 1)
    back_pixels = scipy.ndimage.map_coordinates(back_pixels, (k, i, j), order=1)
    pixels -= back_pixels
    pixels[pixels < 0] = 0
    startmean = numpy.mean(pixels[mask])
    ero = pixels.copy()

    # Mask the test image so that masked pixels will have no effect
    # during reconstruction
    ero[~mask] = 0
    currentmean = startmean
    startmean = max(startmean, numpy.finfo(float).eps)
    footprint = skimage.morphology.ball(1, dtype=bool)
    statistics = [image_name]
    feature_measurments = {}
    for i in range(1, granular_spectrum_length + 1):
        prevmean = currentmean
        ero_mask = numpy.zeros_like(ero)
        ero_mask[mask is True] = ero[mask is True]
        ero = skimage.morphology.erosion(ero_mask, footprint=footprint)
        rec = skimage.morphology.reconstruction(ero, pixels, footprint=footprint)
        currentmean = numpy.mean(rec[mask])
        gs = (prevmean - currentmean) * 100 / startmean
        statistics += ["%.2f" % gs]
        feature = granularity_feature(i, image_name=image_name)
        print(feature, gs)
        # Restore the reconstructed image to the shape of the
        # original image so we can match against object labels
        orig_shape = image_object.shape
        k, i, j = numpy.mgrid[
            0 : orig_shape[0], 0 : orig_shape[1], 0 : orig_shape[2]
        ].astype(float)
        k *= float(new_shape[0] - 1) / float(orig_shape[0] - 1)
        i *= float(new_shape[1] - 1) / float(orig_shape[1] - 1)
        j *= float(new_shape[2] - 1) / float(orig_shape[2] - 1)
        rec = scipy.ndimage.map_coordinates(rec, (k, i, j), order=1)
        feature_measurments[feature] = rec
        granularity_features = {}
        gss = startmean
        for granularity_metric in feature_measurments:
            new_mean = scipy.ndimage.mean(feature_measurments[granularity_metric])
            gss = (gss - new_mean) * 100 / startmean
            granularity_features[granularity_metric] = gss
    return granularity_features
