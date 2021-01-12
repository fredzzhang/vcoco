"""
V-COCO dataset in Python3

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json

from typing import Optional, List, Callable, Tuple, Any
from pocket.data import ImageDataset

class VCOCO(ImageDataset):
    """
    V-COCO dataset

    Parameters:
    -----------
    root: str
        Root directory where images are saved.
    anno_file: str
        Path to json annotation file.
    transform: callable
        A function/transform that  takes in an PIL image and returns a transformed version.
    target_transform: callble
        A function/transform that takes in the target and transforms it.
    transforms: callable
        A function/transform that takes input sample and its target as entry and 
        returns a transformed version.
    """
    def __init__(self, root: str, anno_file: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None) -> None:
        super().__init__(root, transform, target_transform, transforms)
        with open(anno_file, 'r') as f:
            anno = json.load(f)

        self.num_object_cls = None
        self.num_action_cls = 24

        self._anno_file = anno_file

        # Compute metadata
        self._compute_metatdata(anno)

    def __len__(self) -> int:
        """Return the number of images"""
        return len(self._image_ids)

    def __getitem__(self, i: int) -> Tuple[Any, Any]:
        """
        Parameters:
        -----------
        i: int
            The index to an image.
        
        Returns:
        --------
        image: Any
            Input Image. By default, when relevant transform arguments are None,
            the image is in the form of PIL.Image.
        target: Any
            The annotation associated with the given image. By default, when
            relevant transform arguments are None, the taget is a dict with the
            following keys:
                boxes_h: List[list]
                    Human bouding boxes in a human-object pair encoded as the top
                    left and bottom right corners
                boxes_o: List[list]
                    Object bounding boxes corresponding to the human boxes
                actions: List[int]
                    Ground truth action class for each human-object pair
                objects: List[int]
                    Object category index for each object in human-object pairs. The
                    indices follow the COCO2014 (91 classes) standard
                file_name: str
                    Name of the image file
        """
        image = self.load_image(os.path.join(
            self._root, self.filename(i)
        ))
        target = self._anno[i]
        return self._transforms(image, target)

    def __repr__(self) -> str:
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=\"' + repr(self._root)
        reprstr += '\", anno_file=\"'
        reprstr += repr(self._anno_file)
        reprstr += '\")'
        # Ignore the optional arguments
        return reprstr

    @property
    def annotations(self) -> List[dict]:
        return self._anno

    @property
    def actions(self) -> List[str]:
        return self._actions

    @property
    def objects(self) -> List[str]:
        return NotImplementedError

    def filename(self, idx: int) -> str:
        """Return the image file name given the index"""
        return self._anno[idx]['file_name']

    def image_size(self, idx: int) -> Tuple[int, int]:
        """Return the size (width, height) of an image"""
        return self.load_image(os.path.join(
            self._root,
            self.filename(idx)
        )).size

    def _compute_metatdata(self, f: dict) -> None:
        self._anno = f['annotations']
        self._actions = f['classes']
        self._image_ids = f['images']