import logging
import pathlib

import numpy
import skimage.io
import skimage.measure
from area_size_shape import measure_3D_area_size_shape
from colocalization import calculate_3D_colocalization
from granularity import measure_3D_granularity
from intensity import measure_3D_intensity
from neighbors import measure_3D_number_of_neighbors
from texture import measure_3D_texture

logging.basicConfig(level=logging.INFO)


class ImageSetLoader:
    def __init__(
        self, image_set_path: pathlib.Path, spacing: tuple, channel_mapping: dict
    ):
        self.spacing = spacing
        files = sorted(image_set_path.glob("*"))
        files = [f for f in files if f.suffix in [".tif", ".tiff"]]

        # Load images into a dictionary
        self.image_set_dict = {}
        for f in files:
            for key, value in channel_mapping.items():
                if value in f.name:
                    self.image_set_dict[key] = skimage.io.imread(f)

        self.retrieve_image_attributes()
        self.get_unique_objects_in_compartments()

    def retrieve_image_attributes(self):
        self.unique_objects = {}
        for key, value in self.image_set_dict.items():
            if "mask" in key:
                self.unique_objects[key] = numpy.unique(value)

    def get_unique_objects_in_compartments(self):
        self.unique_objects = {}
        compartments = self.get_compartments()
        for compartment in compartments:
            self.unique_objects[compartment] = numpy.unique(
                self.image_set_dict[compartment]
            )

    def get_image(self, key):
        return self.image_set_dict[key]

    def get_image_names(self):
        return [x for x in self.image_set_dict.keys() if "mask" not in x]

    def get_compartments(self):
        return [x for x in self.image_set_dict.keys() if "mask" in x]

    def get_anisotropy(self):
        return self.spacing[0] / self.spacing[1]


class ObjectLoader:
    def __init__(self, image, label_image, channel_name, compartment_name, label_index):
        self.image = image
        self.label_index = label_index
        self.label_image = label_image
        self.channel = channel_name
        self.compartment = compartment_name
        self.objects = skimage.measure.label(image)
        self.object_ids = numpy.unique(self.objects)[1:]
        self.object_count = len(self.object_ids)
        self.retrieve_objects()
        self.generate_mesh()

    def retrieve_objects(self):
        self.image_object = self.image.copy()
        self.label_object = numpy.zeros_like(self.label_image)
        # get the object mask area in the image
        # get just the mask of interest
        self.label_object[self.label_object == 0] = 0
        self.label_object[self.label_image == self.label_index] = self.label_index

        self.image_object[self.label_object == 0] = 0

    def generate_mesh(self):
        self.mesh_z, self.mesh_y, self.mesh_x = numpy.mgrid[
            0 : self.label_object.shape[0],
            0 : self.label_object.shape[1],
            0 : self.label_object.shape[2],
        ]


class Featurization:
    def __init__(
        self,
        image_set_loader: ImageSetLoader,
        object_loader: ObjectLoader,
        neighbors_distance_threshold,
    ):
        self.image_set_loader = image_set_loader
        self.object_loader = object_loader
        self.neighbors_distance_threshold = neighbors_distance_threshold
        self.calculate_single_object_features()

    def calculate_single_object_features(self):
        self.features = {}

        self.features["area_size_shape"] = measure_3D_area_size_shape(
            label_object=self.object_loader.label_object,
            spacing=self.image_set_loader.spacing,
        )
        logging.info(
            f"Calculated Area Size Shape features for {self.object_loader.compartment} {self.object_loader.label_index}"
        )
        self.features["granularity"] = measure_3D_granularity(
            image_object=self.object_loader.image_object,
            label_object=self.object_loader.label_object,
            radius=20,
            granular_spectrum_length=5,
            subsample_size=0.25,
            image_name=self.object_loader.channel,
        )
        logging.info(
            f"Calculated Granularity features for {self.object_loader.compartment} {self.object_loader.label_index}"
        )
        self.features["intensity"] = measure_3D_intensity(
            image_object=self.object_loader.image_object,
            label_object=self.object_loader.label_object,
        )
        self.features[
            f"neighbors_{self.neighbors_distance_threshold}"
        ] = measure_3D_number_of_neighbors(
            label_object=self.object_loader.label_object,
            label_object_all=self.object_loader.label_image,
            distance_threshold=self.neighbors_distance_threshold,
            anisptropy_factor=self.image_set_loader.get_anisotropy(),
        )
        logging.info(
            f"Calculated Neighbors features for {self.object_loader.compartment} {self.object_loader.label_index}"
        )
        self.features["neighbors_adjacent"] = measure_3D_number_of_neighbors(
            label_object=self.object_loader.label_object,
            label_object_all=self.object_loader.label_image,
            distance_threshold=1,
            anisptropy_factor=self.image_set_loader.get_anisotropy(),
        )
        logging.info(
            f"Calculated Adjacent Neighbors features for {self.object_loader.compartment} {self.object_loader.label_index}"
        )

        # self.features["texture"] = measure_3D_texture(
        #     image = self.object_loader.image_object,
        #     distance = 100,
        # )

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
