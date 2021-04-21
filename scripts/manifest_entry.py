from typing import Any, List
from jsonizable import Jsonizable


class ManifestEntry(Jsonizable):
    """
    This class defines the training data manifest entry.
    """

    # pylint: disable=missing-class-docstring
    class Meta:
        schema = {
            "s3Url": str,
            "annotations": list,
            "datasets": [str],
        }

    # S3 URL of the image resource.
    s3Url: str = None

    # Which datasets this image belongs to (train/val/test).
    datasets: List[str] = None

    # A list of annotations (format depends on model).
    annotations: List[Any] = None

    def image_file_name(self) -> str:
        """
        Returns the file name of the image resource.
        """
        return self.s3Url.split("/").pop()

    def resource_name(self) -> str:
        """
        Returns the name of the resource.
        """
        return self.image_file_name().split(".")[0]
