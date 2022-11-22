# -*- coding: utf-8 -*-
# File: tessocr.py

# Copyright 2021 Dr. Janis Meyer. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tesseract OCR engine for text extraction
"""
from os import environ
from typing import Any, Dict, List, Optional, Union
from ..utils.metacfg import set_config_by_yaml
from google.cloud import vision

import numpy as np

from ..utils.detection_types import ImageType, Requirement
from ..utils.file_utils import get_google_cloud_requirements
from .base import DetectionResult, ObjectDetector, PredictorBase
from ..utils.settings import LayoutType, ObjectTypes
from dotenv import load_dotenv

import cv2

load_dotenv()

GOOGLE_SERVICE_ACCOUNT_PATH = environ["GOOGLE_SERVICE_ACCOUNT_PATH"]
environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_SERVICE_ACCOUNT_PATH
class GoogleCloudVisionError(RuntimeError):
    """
    Google Cloud Vision Error
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__()
        self.status = status
        self.message = message
        self.args = (status, message)


class VisionOcrDetector(ObjectDetector):
    # TODO: Write good document
    def __init__(
        self,
        path_yaml: str
    ):
        # print(GOOGLE_SERVICE_ACCOUNT_PATH)
        self.name = "google_vision"
        self.client = vision.ImageAnnotatorClient()
        
        hyper_param_config = set_config_by_yaml(path_yaml)
        
        self.path_yaml = path_yaml
        self.config = hyper_param_config

        if self.config.LINES:
            self.categories = {"1": LayoutType.word, "2": LayoutType.line}
        else:
            self.categories = {"1": LayoutType.word}
            
    def predict(self, np_img: ImageType) -> List[DetectionResult]:
        results = self._predict_text(np_img)
        return results

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_google_cloud_requirements()]

    def clone(self) -> PredictorBase:
        return self.__class__(self.path_yaml, self.config_overwrite)
    
    def possible_categories(self) -> List[ObjectTypes]:
        if self.config.LINES:
            return [LayoutType.word, LayoutType.line]
        return [LayoutType.word]

    def _predict_text(self, np_img: ImageType) -> List[DetectionResult]:
        content=cv2.imencode(".jpg",np_img)[1].tobytes()
        
        image = vision.Image(content=content)

        # response_label = client.label_detection(image)

        response = self.client.text_detection(image)
        texts = response.text_annotations
        results = self._text_response_to_dict(texts)
        all_results = []
        for result in results:
            if int(result["conf"]) != -1:
                word = DetectionResult(
                    box=[result["left"], result["top"], result["left"] + result["width"], result["top"] + result["height"]],
                    score=result["conf"] / 100,
                    text=result["text"],
                    block=str(result["block_num"]),
                    line=str(result["line_num"]),
                    class_id=1,
                    class_name=LayoutType.word,
                )
                all_results.append(word)
        return all_results

    def _text_response_to_dict(self, texts: str) -> Dict[str, List[Union[str, int, float]]]:
        results = []
        for index, item in enumerate(texts[1:]):
            result = {}
            result["left"] = item.bounding_poly.vertices[0].x
            result["top"] = item.bounding_poly.vertices[0].y
            result["width"] = item.bounding_poly.vertices[1].x - item.bounding_poly.vertices[0].x
            result["height"] = item.bounding_poly.vertices[3].y - item.bounding_poly.vertices[0].y
            result["conf"] = item.score
            result["text"] = item.description
            result["block_num"] = index
            result["line_num"] = 1 if item.description else 0
            results.append(result)
        return results

