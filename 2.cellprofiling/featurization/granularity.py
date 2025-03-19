from typing import Dict

import numpy
import scipy
import skimage
from loading_classes import ObjectLoader


def granularity_feature(length):
    C_GRANULARITY = "GRANULARITY.%s"
    return C_GRANULARITY % (length)


class ObjectRecord:
    def __init__(self, object_loader, object_index):
        self.object_index = object_index
        self.labels = object_loader.objects.copy()
        # select the object
        self.labels[self.labels != object_index] = 0
        self.image = object_loader.image.copy()
        self.image[self.labels != object_index] = 0

        # self.labels[self.labels == object_index] = 1
        # self.labels[~object_index] = 0

        self.nobjects = len(numpy.unique(self.labels))
        if self.nobjects != 0:
            self.range = numpy.arange(1, numpy.max(self.labels) + 1)
            self.current_mean = scipy.ndimage.mean(self.image, self.labels)

            self.start_mean = numpy.maximum(self.current_mean, numpy.finfo(float).eps)


def measure_3D_granularity(
    object_loader: ObjectLoader,
    radius: int = 20,
    granular_spectrum_length: int = 5,
    subsample_size: float = 0.25,
    image_name: str = "image",
) -> Dict[str, float]:

    image_object = object_loader.image
    label_object = object_loader.label_image
    # radius=10
    radius = 10
    # granular_spectrum_length=16
    granular_spectrum_length = 16
    subsample_size = 0.25
    image_name = "nuclei"

    pixels = image_object.copy()
    mask = label_object.copy()

    # begin by downsampling the image
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

    object_records = [
        ObjectRecord(object_loader, object_index)
        for _, object_index in enumerate(object_loader.object_ids)
    ]

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
    object_measurements = {"object_id": [], "feature": [], "value": []}
    for i in range(1, granular_spectrum_length + 1):
        prevmean = currentmean
        ero_mask = numpy.zeros_like(ero)
        ero_mask[mask == True] = ero[mask == True]
        ero = skimage.morphology.erosion(ero_mask, footprint=footprint)
        rec = skimage.morphology.reconstruction(ero, pixels, footprint=footprint)
        currentmean = numpy.mean(rec[mask])
        gs = (prevmean - currentmean) * 100 / startmean
        statistics += ["%.2f" % gs]
        feature = granularity_feature(i)
        feature_measurments[feature] = gs
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
        for object_record in object_records:
            assert isinstance(object_record, ObjectRecord)
            if object_record.nobjects > 0:
                new_mean = scipy.ndimage.mean(rec, object_record.labels)
                gss = (
                    (object_record.current_mean - new_mean)
                    * 100
                    / object_record.start_mean
                )
                object_record.current_mean = new_mean
            else:
                gss = numpy.zeros((0,))
            object_measurements["object_id"].append(object_record.object_index)
            object_measurements["feature"].append(feature)
            object_measurements["value"].append(gss)
    return object_measurements
