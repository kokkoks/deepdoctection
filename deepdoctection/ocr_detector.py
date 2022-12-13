from deepdoctection.utils.systools import get_configs_dir_path
from deepdoctection.analyzer.dd import _maybe_copy_config_to_cache, _auto_select_lib_and_device
from deepdoctection.utils.metacfg import set_config_by_yaml
from deepdoctection.extern.model import ModelCatalog, ModelDownloadManager
from deepdoctection.extern.d2detect import D2FrcnnDetector
from deepdoctection.pipe.layout import ImageLayoutService
from deepdoctection.pipe.cell import SubImageLayoutService
from deepdoctection.utils.settings import CellType, LayoutType
from deepdoctection.pipe.refine import TableSegmentationRefinementService
from deepdoctection.pipe.segment import TableSegmentationService
from deepdoctection.extern.base import DetectionResult
from deepdoctection.extern.visionocr import VisionOcrDetector
from deepdoctection.pipe.text import TextExtractionService, TextOrderService
from deepdoctection.pipe.common import MatchingService
from deepdoctection.mapper.misc import to_image
from deepdoctection.datapoint.image import Image
from typing import Optional, List
# from IPython.core.display import HTML
# import os
from deepdoctection.pipe.common import PageParsingService
import numpy as np
import pandas as pd

_DD_ONE = "deepdoctection/configs/conf_dd_one.yaml"
_VISION = "deepdoctection/configs/conf_vision.yaml"

class OCRDetector():
    def __init__(self):
        lib, device = _auto_select_lib_and_device()
        dd_one_config_path = _maybe_copy_config_to_cache(_DD_ONE)
        _maybe_copy_config_to_cache(_VISION)

        self.cfg = set_config_by_yaml(dd_one_config_path)
        self.cfg.freeze(freezed=False)
        self.cfg.LIB = lib
        self.cfg.DEVICE = device
        self.cfg.TAB = True
        self.cfg.TAB_REF = True
        self.cfg.OCR = True
        self.cfg.freeze()

        self.layout = self._init_image_layout_service(self.cfg)
        self.cell = self._init_cell_layout_service(self.cfg)
        self.item = self._init_item_layout_service(self.cfg)
        self.text_extraction = self._init_text_extraction_service(self.cfg)
        self.table_segmentation = self._init_table_segmentation_service(self.cfg)
        self.table_segmentation_refinement = TableSegmentationRefinementService()
        self.text_matching = self._init_text_matching_service(self.cfg)
        self.text_ordering = self._init_text_order_service(self.cfg)
        self.page_parser = self._init_page_parser_service()

    def predict(self, np_img: np.ndarray, table_detection_results: List[List[str]] = None) -> pd.DataFrame:
        image = to_image(np_img)

        
        self.layout.dp_manager.datapoint = image
        if table_detection_results:
            detection_result_list = self._convert_bbox_to_detection_list(table_detection_results)
            self.layout.serve(image, detect_result_list=detection_result_list)
        else:
            self.layout.serve(image)

        self.cell.dp_manager.datapoint = self.layout.dp_manager.datapoint
        self.cell.serve(image)

        self.item.dp_manager.datapoint = self.cell.dp_manager.datapoint
        self.item.serve(image)

        self.text_extraction.dp_manager.datapoint = self.item.dp_manager.datapoint
        self.text_extraction.serve(self.item.dp_manager.datapoint)

        self.table_segmentation.dp_manager.datapoint = self.text_extraction.dp_manager.datapoint
        self.table_segmentation.serve(self.text_extraction.dp_manager.datapoint)

        self.table_segmentation_refinement.dp_manager.datapoint = self.table_segmentation.dp_manager.datapoint
        self.table_segmentation_refinement.serve(self.table_segmentation.dp_manager.datapoint)

        self.text_matching.dp_manager.datapoint = self.table_segmentation_refinement.dp_manager.datapoint
        self.text_matching.serve(image)

        self.text_ordering.dp_manager.datapoint = self.text_matching.dp_manager.datapoint
        self.text_ordering.serve(image)

        page = self.page_parser.pass_datapoint(self.text_ordering.dp_manager.datapoint)
        df = None
        try:
            df = pd.read_html(page.tables[0].html)[0]
        except:
            df = None
        return df, page

    def _convert_bbox_to_detection_list(self, bboxes: List[List[float]]) -> List[DetectionResult]:
        # {'0': <LayoutType.text>, '1': <LayoutType.title>, '2': <LayoutType.list>, '3': <LayoutType.table>, '4': <LayoutType.figure>}
        # [[100.0, 100.0, 200.0, 200.0, 0.98]]
        detection_result_list = []
        for bbox in bboxes:
            x1, y1, x2, y2, conf = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
            detection_result = DetectionResult(
                box=[x1, y1, x2, y2], 
                class_id=4, 
                score=conf, 
                mask=None, 
                absolute_coords=True, 
                class_name=LayoutType.table, text=None, block=None, line=None, uuid=None)
            detection_result_list.append(detection_result)
        return detection_result_list
        
    def _init_image_layout_service(self, cfg) -> ImageLayoutService:
        layout_config_path = ModelCatalog.get_full_path_configs(cfg.CONFIG.D2LAYOUT)
        layout_weights_path = ModelDownloadManager.maybe_download_weights_and_configs(cfg.WEIGHTS.D2LAYOUT)
        profile = ModelCatalog.get_profile(cfg.WEIGHTS.D2LAYOUT)
        categories_layout = profile.categories
        d_layout = D2FrcnnDetector(layout_config_path, layout_weights_path, categories_layout, device=cfg.DEVICE)
        layout = ImageLayoutService(d_layout, to_image=True, crop_image=True)
        return layout

    def _init_cell_layout_service(self, cfg) -> SubImageLayoutService:
        cell_config_path = ModelCatalog.get_full_path_configs(cfg.CONFIG.D2CELL)
        cell_weights_path = ModelDownloadManager.maybe_download_weights_and_configs(cfg.WEIGHTS.D2CELL)
        profile = ModelCatalog.get_profile(cfg.WEIGHTS.D2CELL)
        categories_cell = profile.categories
        d_cell = D2FrcnnDetector(cell_config_path, cell_weights_path, categories_cell, device=cfg.DEVICE)
        
        cell = SubImageLayoutService(d_cell, LayoutType.table, {1: 6}, True)
        return cell

    def _init_item_layout_service(self, cfg) -> SubImageLayoutService:
        item_config_path = ModelCatalog.get_full_path_configs(cfg.CONFIG.D2ITEM)
        item_weights_path = ModelDownloadManager.maybe_download_weights_and_configs(cfg.WEIGHTS.D2ITEM)
        profile = ModelCatalog.get_profile(cfg.WEIGHTS.D2ITEM)
        categories_item = profile.categories
        d_item = D2FrcnnDetector(item_config_path, item_weights_path, categories_item, device=cfg.DEVICE)

        item = SubImageLayoutService(d_item, LayoutType.table, {1: 7, 2: 8}, True)
        return item

    def _init_table_segmentation_service(self, cfg) -> TableSegmentationService:
        return TableSegmentationService(
            cfg.SEGMENTATION.ASSIGNMENT_RULE,
            cfg.SEGMENTATION.IOU_THRESHOLD_ROWS if cfg.SEGMENTATION.ASSIGNMENT_RULE in ["iou"] else cfg.SEGMENTATION.IOA_THRESHOLD_ROWS,
            cfg.SEGMENTATION.IOU_THRESHOLD_COLS if cfg.SEGMENTATION.ASSIGNMENT_RULE in ["iou"] else cfg.SEGMENTATION.IOA_THRESHOLD_COLS,
            cfg.SEGMENTATION.FULL_TABLE_TILING,
            cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS,
            cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS,
        )

    def _init_text_extraction_service(self, cfg) -> TextExtractionService:

        ocr_config_path = get_configs_dir_path() / cfg.CONFIG.VISION_OCR

        d_vision_ocr = VisionOcrDetector(ocr_config_path)
        text = TextExtractionService(d_vision_ocr)

        return text
    
    def _init_text_matching_service(self, cfg) -> MatchingService:
        match = MatchingService(
            parent_categories=cfg.WORD_MATCHING.PARENTAL_CATEGORIES,
            child_categories=LayoutType.word,
            matching_rule=cfg.WORD_MATCHING.RULE,
            threshold=cfg.WORD_MATCHING.IOU_THRESHOLD
            if cfg.WORD_MATCHING.RULE in ["iou"]
            else cfg.WORD_MATCHING.IOA_THRESHOLD,
        )
        return match

    def _init_text_order_service(self, cfg) -> TextOrderService:
        order = TextOrderService(
            text_container=LayoutType.word,
            floating_text_block_names=[LayoutType.title, LayoutType.text, LayoutType.list],
            text_block_names=[
                LayoutType.title,
                LayoutType.text,
                LayoutType.list,
                LayoutType.cell,
                CellType.header,
                CellType.body,
            ],
        )
        return order

    def _init_page_parser_service(self) -> PageParsingService:
        return PageParsingService(
            text_container=LayoutType.word,
            text_block_names=[LayoutType.title, LayoutType.text, LayoutType.list, LayoutType.table],
        )
