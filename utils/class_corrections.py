import numpy as np
import torch

from utils.general import bbox_io1

# Adjust class if io1 of contained class and container class >= threshold
BBOX_IO1_THRESHOLD = 0.7
# If A is inside B, then it becomes C
CONTAINED_CLASS_MAPPING = {
    # "brace cracked/broken"
    0: {
        12: 0,  # inside "entire brace" -> "brace cracked/broken"
        13: 2,  # inside "entire crossarm" -> "crossarm cracked/broken"
        14: 1,  # inside "entire pole" -> "cracked pole"
    },
    # "crossarm cracked/broken"
    2: {
        12: 0,  # inside "entire brace" -> "brace cracked/broken"
        13: 2,  # inside "entire crossarm" -> "crossarm cracked/broken"
        14: 1,  # inside "entire pole" -> "cracked pole"
    },
    # "cracked pole"
    1: {
        12: 0,  # inside "entire brace" -> "brace cracked/broken"
        13: 2,  # inside "entire crossarm" -> "crossarm cracked/broken"
        14: 1,  # inside "entire pole" -> "cracked pole"
    },
    # Testing with FPL model of AEP images
    # "Polymer Insulator"
    9: {
        5: 8,  # inside "Steel Crossarm" -> "Polymer Dead-end Insulator"
        19: 8,  # inside "Wood Pole" -> "Polymer Dead-end Insulator"
    }
}


def get_corrected_class(cls, xyxy, det, names):
    """
    The class of a detected object could depend on whether it's found (mostly)
    inside the bounding box of another object. Correct it if necessary.
    """

    # Is this class one of the ones that might be contained in bounding box of another class?
    if int(cls) in CONTAINED_CLASS_MAPPING:

        # Get just detections for the outer classes
        # A detection has the format x, y, x, y, conf, class
        mask = torch.as_tensor([int(elem) in CONTAINED_CLASS_MAPPING[int(cls)] for elem in det[:, 5]])
        outer_classes_detections = det[mask]

        # See if bounding box xyxy is (mostly) inside one of the outer_classes_boxes
        if len(outer_classes_detections):
            outer_classes_boxes = np.array(
                [np.array(xyxy) for (*xyxy, conf, cls) in outer_classes_detections])
            io1 = bbox_io1(xyxy, outer_classes_boxes)

            # If it's somehow contained in multiple objects, use one that it's most inside of
            values, indices = io1.max(0)

            # Does the class need to be adjusted?
            if values.item() >= BBOX_IO1_THRESHOLD:
                # A detection has the format x, y, x, y, conf, class
                outer_class = int(outer_classes_detections[indices.item()][5])
                message = '{} detected inside {}'.format(names[int(cls)], names[int(outer_class)])

                new_cls = torch.tensor(CONTAINED_CLASS_MAPPING[int(cls)][int(outer_class)])
                if new_cls == cls:
                    message += ': no change'
                else:
                    message += ': changing {} to {}'.format(names[int(cls)], names[int(new_cls)])
                    cls = new_cls

                print(message)

    return cls


def hide_container_object(cls):
    """
    If this class is one of the container objects that's used to correct the class of objects found inside it,
    it should get hidden if desired
    """
    container_classes_keys = [list(CONTAINED_CLASS_MAPPING[key].keys()) for key in list(CONTAINED_CLASS_MAPPING.keys())]
    return int(cls) in set(np.concatenate(container_classes_keys).flat)
