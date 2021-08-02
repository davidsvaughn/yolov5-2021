import hashlib
import json
from typing import List, Tuple
from jsonizable import Jsonizable

# pylint: disable=unsubscriptable-object
Segmentation = List[List[Tuple[float, float]]]
BoundingBox = List[Tuple[float, float]]
Flags = List[str]


class COCOCategory(Jsonizable):
    """
    A JSON-serializable COCO-compliant category descriptor.
    """

    # pylint: disable=missing-class-docstring
    class Meta:
        schema = dict(
            supercategory=str,
            id=int,
            name=str
        )

    supercategory: str = None
    name: str = None
    id: int = None


class InferenceMetrics(Jsonizable):
    """
    Inference metrics, such as detection confidence, etc.
    """

    # pylint: disable=missing-class-docstring
    class Meta:
        schema = dict(
            confidence=float,
        )

    # Detection confidence.
    confidence: float = None


class InferenceResult(Jsonizable):
    """
    A single detection result.
    """

    # pylint: disable=missing-class-docstring
    class Meta:
        schema = dict(
            metrics=InferenceMetrics,
            category=COCOCategory,
            bbox=list,
            segmentation=list
        )

    # Detection metrics.
    metrics: InferenceMetrics = None

    # Category of this detection, in COCO format.
    category: COCOCategory = None

    # Bounding box of the detection.
    bbox: BoundingBox = None

    # Segmentation mask of the detection.
    segmentation: Segmentation = None

    def md5(self) -> str:
        """
        Creates an MD5 hash of the detection result.

        Returns:
            str: MD5 hash of the detection result.
        """
        return hashlib.md5(json.dumps(self.write()).encode('utf-8')).hexdigest()


class InferenceResponse(Jsonizable):
    """
    Inference response object. This contains some metadata about the original request,
    along with a single detection result and metrics. This is the final format that gets
    sent along to Precision Analytics.
    """

    # pylint: disable=missing-class-docstring
    class Meta:
        schema = dict(
            createDate=str,
            flags=list,
            id=str,
            deduplicationId=str,
            modelId=str,
            paType=str,
            resourceId=str,
            source=str,
            type=str,
            status=str,
            result=InferenceResult
        )

    # Unique ID of the detection in PA.
    id: str = None

    # Deduplication ID. This is an MD5 hash of the result object.
    # This ID will always be the same for the same detection polygon and class.
    deduplicationId: str = None

    # The unqiue ID of the resource image in PA that was processed and had something detected.
    resourceId: str = None

    # The unique ID of the model used to generate the detection or to be trained with the user created detection.
    modelId: str = None

    # The date the detection was created.
    createDate: str = None

    # String flags indicating the findings detected.
    flags: Flags = None

    # The source of an annotation.
    source: str = None

    # The type of detection made.
    # Possible values are: AI, Analyst, PreAnalyst
    type: str = "AI"

    # Status of the detection.
    # Possible values are: Pending, Processing, Processed
    status: str = None

    # Example: "ML::Detection"
    paType: str = None

    # Actual inference result.
    result: InferenceResult = None
