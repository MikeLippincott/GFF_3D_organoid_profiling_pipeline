import logging
import pathlib

import numpy
import skimage.io
import skimage.measure

logging.basicConfig(level=logging.INFO)


class ImageSetLoader:
    def __init__(
        self, image_set_path: pathlib.Path, spacing: tuple, channel_mapping: dict
    ):
        self.spacing = spacing
        self.anisotropy_factor = self.spacing[0] / self.spacing[1]
        self.image_set_name = image_set_path.name
        files = sorted(image_set_path.glob("*"))
        files = [f for f in files if f.suffix in [".tif", ".tiff"]]

        # Load images into a dictionary
        self.image_set_dict = {}
        for f in files:
            for key, value in channel_mapping.items():
                if value in f.name:
                    self.image_set_dict[key] = skimage.io.imread(f)

        self.retrieve_image_attributes()
        self.get_compartments()
        self.get_image_names()
        self.get_unique_objects_in_compartments()

    def retrieve_image_attributes(self):
        self.unique_objects = {}
        for key, value in self.image_set_dict.items():
            if "mask" in key:
                self.unique_objects[key] = numpy.unique(value)

    def get_unique_objects_in_compartments(self):
        self.unique_compartment_objects = {}
        for compartment in self.compartments:
            self.unique_compartment_objects[compartment] = numpy.unique(
                self.image_set_dict[compartment]
            )
            # remove the 0 label
            self.unique_compartment_objects[compartment] = [
                x for x in self.unique_compartment_objects[compartment] if x != 0
            ]

    def get_image(self, key):
        return self.image_set_dict[key]

    def get_image_names(self):
        self.image_names = [
            x for x in self.image_set_dict.keys() if x not in self.compartments
        ]

    def get_compartments(self):
        self.compartments = [
            x
            for x in self.image_set_dict.keys()
            if "Nuclei" in x or "Cell" in x or "Cytoplasm" in x or "Organoid" in x
        ]

    def get_anisotropy(self):
        return self.spacing[0] / self.spacing[1]


class ObjectLoader:
    def __init__(self, image, label_image, channel_name, compartment_name):
        self.image = image
        self.label_image = label_image
        self.channel = channel_name
        self.compartment = compartment_name
        self.objects = skimage.measure.label(label_image)
        self.object_ids = numpy.unique(self.objects)
        # drop the 0 label
        self.object_ids = self.object_ids[1:]


class TwoObjectLoader:
    def __init__(
        self, image_set_loader: ImageSetLoader, compartment, channel1, channel2
    ):
        self.image_set_loader = image_set_loader
        self.compartment = compartment
        self.label_image = self.image_set_loader.image_set_dict[compartment].copy()
        self.image1 = self.image_set_loader.image_set_dict[channel1].copy()
        self.image2 = self.image_set_loader.image_set_dict[channel2].copy()
        self.object_ids = image_set_loader.unique_compartment_objects[compartment]
