from ieat.utils import resize, normalize_img, color_quantize_np

import os


from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import cv2
import logging
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
logger = logging.getLogger()

REPO_PATH = '/Users/Tony/OneDrive - Nexus365/Societies/OxAI/Repos/ieat'

# # Code adapted from
# https://colab.research.google.com/github/apeguero1/image-gpt/blob/master/Transformers_Image_GPT.ipynb
# - thanks to the author


class EmbeddingExtractor:
    """Extracts embeddings from images with a pre-trained model."""
    def __init__(self, model_name, from_cache=False, cuda=False):
        """
        Parameters
        ----------
        model_name : str
            A name for this model, used for caching.
        from_cache : bool
            Whether to used cached embeddings.
        """
        self.device = torch.device("cuda" if cuda else "cpu")
        self.from_cache = from_cache
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """
        Loads the model, from the web or from the filesystem.
        """
        raise NotImplementedError

    def extract_dir(self, d, file_types=('.jpg', '.jpeg', '.png', '.webp'), batch_size=None, visualize=False, **extract_params):
        """
        Extracts embeddings from images in a directory.
        Parameters
        ----------
        d : str
            path to a directory of images
        file_types : list[str]
            list of acceptable file extensions for images
        batch_size : int
            number of images processed at a time - helps when you have limited memory
        visualize : bool
            whether to display the images after pre-processing
        extract_params : dict
            additional parameters for extraction

        Returns
        -------
        encs : pd.DataFrame
            a Pandas dataframe of features - see `EmbeddingExtractor.extract`
        """
        embedding_path = os.path.join(REPO_PATH, self._make_embedding_path(d))
        image_paths = [
            os.path.join(d, f) for f in os.listdir(d)
            if os.path.splitext(f)[1] in file_types
        ]
        if self.from_cache and os.path.exists(embedding_path):
            print("Loading embeddings for %s from file" % os.path.basename(d))
            encs = pd.read_csv(embedding_path, index_col=0).set_index("img")
            if visualize:
                self.process_samples(image_paths, visualize=True)
        else:
            print("Extracting embeddings for %s" % os.path.basename(d))
            if self.model is None:
                self.load_model()
            start = time.time()
            # do extraction in batches to save memory
            encs = self.extract(
                image_paths,
                batch_size=batch_size,
                output_path=embedding_path,
                visualize=visualize,
                **extract_params
            )
            finish = time.time()
            print(f'finished extracting {os.path.basename(d)} of {len(image_paths)} images in {finish - start:.1f}s')
        return encs

    def extract(self, image_paths, batch_size=None, output_path=None, visualize=False, **extract_kwargs):
        """
        Extracts features from a set of image paths.

        Parameters
        ----------
        image_paths : list
            a list of paths to images to extract features for
        batch_size : int or None
            number of images processed at a time - helps when you have limited memory; if None, use just one batch
        output_path : str or None
            path to save a CSV cache file with the extracted features; if none, don't cache
        gpu : bool
            whether to use GPU (True) or CPU (False)
        visualize : bool
            whether to display the images after pre-processing
        extract_kwargs : dict
            additional parameters for extraction

        Returns
        -------
        encs : pd.DataFrame
            data frame of features, indexed by the original image path
        """

        if batch_size is None:
            batch_size = len(image_paths)

        with torch.no_grad():  # saves some memory
            batches = [image_paths[i:i+batch_size] for i in range(0, len(image_paths), batch_size)]
            # model specific context extraction
            encs = pd.concat([
                pd.DataFrame(
                    self._extract_context(self.process_samples(batch, visualize=visualize), **extract_kwargs)
                )
                for batch in batches
            ])

            encs["img"] = [os.path.basename(path) for path in image_paths]

            # DEPRECATED - NOW THAT CACHE IS STORED BY CATEGORY
            # df["category"] = [os.path.basename(os.path.dirname(path)) for path in image_paths]

            if output_path is not None:
                # add the image names to the CSV file
                encs.to_csv(output_path)

            return encs.set_index("img")

    def process_samples(self, image_paths, visualize=False):
        """
        Pre-process the image samples for embedding extraction.
        Parameters
        ----------
        image_paths : list[str]
            list of image paths to pre-process
        visualize : bool
            whether to display the images after pre-processing
        Returns
        -------
        list
            list of processed images, usually as `list[np.ndarray]`
        """
        raise NotImplementedError

    def _extract_context(self, samples, **extract_kwargs) -> np.ndarray:
        raise NotImplementedError

    def _make_embedding_path(self, d):
        return "embeddings/{}_{}_{}.csv".format(
            os.path.basename(os.path.dirname(d)),
            os.path.basename(d),
            self._make_param_path()
        )

    def _make_param_path(self):
        raise NotImplementedError

    @staticmethod
    def visualize(images, paths):
        """
        Visualize some preprocessed images.

        Parameters
        ----------
        images : list[tensors]
            the images, as matrices
        paths : list[str]
            list of the original image paths, so we can get the parent directory
        """
        print(os.path.basename(os.path.dirname(paths[0])))
        f, axes = plt.subplots(1, len(images), dpi=300)
        for img, ax in zip(images, axes):
            ax.axis('off')
            ax.imshow(img.numpy())
        plt.show()






class ResnetExtractor(EmbeddingExtractor):

    def __init__(self, supervised, blurred, model='resnet-50', cuda=False, layer='default', layer_output_size=2048,  **parent_params):
        """
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """

        super().__init__(model, **parent_params)
        self.device = torch.device("cuda" if cuda else "cpu")
        self.model_name = model
        self.images = None
        self.supervised = supervised
        self.blurred = blurred
        self.layer=layer
        self.layer_output_size = layer_output_size

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        self.extracting = None

    def load_model(self):
        self.model, self.extraction_layer = self._get_model_and_layer(self.model_name, self.layer)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.extracting = self.model.backbone.body

    def _extract_context(self, images, visualize=False):
        """ Get vector embedding from image paths
        :param images: tensor of images
        :returns: Numpy ndarray
        """
        #embeddings = torch.zeros(len(images), self.layer_output_size, 1, 1)
        # get embeddings from the last layer of resnet
        assert self.extracting is not None
        start = time.time()
        embeddings = self.extracting(images)[self.extraction_layer]
        assert embeddings is not None
        pool = nn.AdaptiveAvgPool2d((1,1))
        embeddings = pool(embeddings)
        finish = time.time()
        print(f'finished one batch in {finish - start:.1f}s')
        return embeddings.numpy()[:, :, 0, 0]



    def process_samples(self, image_paths, visualize=False):

        # convert image_paths to images that are RGB and resized to 1024
        images=[cv2.resize(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB),(1024, 1024)) for image_path in image_paths]
        # convert to a list of tensors
        image_tensors=[self.normalize(self.to_tensor(im)) for im in images]
        #image_tensors = [self.to_tensor(cv2.resize(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB),(1024, 1024))) for image_path in image_paths]
        images_stacked = torch.stack(image_tensors).to(self.device)

        if visualize:
           self.visualize(image_tensors, image_paths)

        #return image_tensors
        return images_stacked



    def _get_model_and_layer(self, model_name, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-50'
        :param layer: layer as a string for resnet-50 or int for alexnet
        :returns: pytorch model, selected layer
        """

        if model_name == 'resnet-50':
            path = os.path.join(REPO_PATH, f"models/{self.supervised}_{self.blurred}")
            model = torch.load(path)
            if layer == 'default':
                layer = '3'
            else:
                layer = str(layer-1)
            return model, layer

        else:
            raise KeyError('Model %s was not found' % model_name)


    def _make_param_path(self):
        return f"{self.supervised}_{self.blurred}"


