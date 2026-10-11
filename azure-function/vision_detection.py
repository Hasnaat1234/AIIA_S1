import os
import time
from dataclasses import dataclass

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from detection import CONFIDENCE_THRESHOLD, PERSON_CLASS_ID, InvalidImageError


@dataclass
class VisionResult:
    detections: list
    persons: int
    avg_confidence: float
    inference_ms: float


class VisionDetector:
    """Counts people with Azure AI Vision (Image Analysis 4.0, PEOPLE feature)."""

    def __init__(self) -> None:
        self.client = ImageAnalysisClient(
            endpoint=os.environ["VISION_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["VISION_KEY"]),
        )

    def detect(self, image_data: bytes) -> VisionResult:
        started_at = time.perf_counter()
        try:
            result = self.client.analyze(
                image_data=image_data,
                visual_features=[VisualFeatures.PEOPLE],
            )
        except HttpResponseError as error:
            raise InvalidImageError(error.message) from error
        inference_ms = (time.perf_counter() - started_at) * 1000
        people = result.people.list if result.people else []
        detections = []
        for person in people:
            if person.confidence < CONFIDENCE_THRESHOLD:
                continue
            box = person.bounding_box
            detections.append(
                {
                    "class": PERSON_CLASS_ID,
                    "confidence": person.confidence,
                    "box": {
                        "x1": box.x,
                        "y1": box.y,
                        "x2": box.x + box.width,
                        "y2": box.y + box.height,
                    },
                }
            )
        confidences = [d["confidence"] for d in detections]
        avg = sum(confidences) / len(confidences) if confidences else 0.0
        return VisionResult(detections, len(detections), avg, inference_ms)
