import logging

import matplotlib.pyplot as plt
import numpy
import skimage.io
import skimage.measure
from area_size_shape import measure_3D_area_size_shape
from colocalization import (
    calculate_3D_colocalization,
    prepare_two_images_for_colocalization,
)
from granularity import measure_3D_granularity
from intensity import measure_3D_intensity
from loading_classes import ImageSetLoader
from neighbors import measure_3D_number_of_neighbors
from texture import measure_3D_texture

logging.basicConfig(level=logging.INFO)


class ColocalizationTwoObject_Loader:
    def __init__(
        self,
        image_set_loader: ImageSetLoader,
        image1,
        label_image1,
        object1,
        image2,
        label_image2,
        object2,
        compartment,
    ):
        self.image_set_loader = image_set_loader
        self.image1 = image1
        self.label_image1 = label_image1
        self.compartment = compartment
        self.object1 = object1
        self.image2 = image2
        self.label_image2 = label_image2
        self.object2 = object2
        self.retrieve_objects()

    def retrieve_objects(self):
        self.image_object1 = self.image1.copy()
        self.image_object2 = self.image2.copy()
        self.label_object1 = numpy.zeros_like(self.label_image1)
        self.label_object2 = numpy.zeros_like(self.label_image2)
        # get the object mask area in the image
        # get just the mask of interest
        self.label_object1[self.label_image1 == 0] = 0
        self.label_object1[self.label_image1 == self.object1] = self.object1
        self.image_object1[self.label_object1 == 0] = 0

        self.label_object2[self.label_image2 == 0] = 0
        self.label_object2[self.label_image2 == self.object2] = self.object2
        self.image_object2[self.label_object2 == 0] = 0


class ColocalizationFeaturization:
    def __init__(
        self,
        image_set_loader: ImageSetLoader,
        two_object_loader: ColocalizationTwoObject_Loader,
    ):
        self.image_set_loader = image_set_loader
        self.object_loader = two_object_loader
        self.calculate_colocalization_features()

    def calculate_colocalization_features(self):
        (
            self.croppped_image_1,
            self.croppped_image_2,
        ) = prepare_two_images_for_colocalization(
            label_object1=self.object_loader.label_object1,
            label_object2=self.object_loader.label_object2,
            image_object1=self.object_loader.image_object1,
            image_object2=self.object_loader.image_object2,
        )
        self.features = {}
        self.features["colocalization"] = calculate_3D_colocalization(
            croppped_image_1=self.croppped_image_1,
            croppped_image_2=self.croppped_image_2,
            thr=15,
            fast_costes="Accurate",
        )
        logging.info(
            f"Calculated Colocalization features for {self.object_loader.compartment} {self.object_loader.object1} and {self.object_loader.object2}"
        )
        return self.features

    def process_features_for_output(self):
        features = {}
        for key, value in self.features.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    features[f"{key}_{sub_key}"] = sub_value
            else:
                features[key] = value
        return features

    def get_features(self):
        return self.features
