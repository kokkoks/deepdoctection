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
from deepdoctection.utils.settings import LayoutType, WordType, CellType, Relationships
import copy
from typing import Optional, List
from deepdoctection.pipe.common import PageParsingService
import numpy as np
import pandas as pd
import cv2 

_DD_ONE = "deepdoctection/configs/conf_dd_one.yaml"
_VISION = "deepdoctection/configs/conf_vision.yaml"

class OCRDetector():
    def __init__(self):
        lib, device = _auto_select_lib_and_device()
        dd_one_config_path = _DD_ONE
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
        bgr_np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        dp_image = to_image(bgr_np_img)

        self.layout.dp_manager.datapoint = dp_image
        if table_detection_results:
            detection_result_list = self._convert_bbox_to_detection_list(table_detection_results)
            self.layout.serve(dp_image, detect_result_list=detection_result_list)
        else:
            self.layout.serve(dp_image)

        self.cell.dp_manager.datapoint = self.layout.dp_manager.datapoint
        self.cell.serve(self.layout.dp_manager.datapoint)

        self.item.dp_manager.datapoint = dp_image
        self.item.serve(dp_image)

        self.text_extraction.dp_manager.datapoint = self.item.dp_manager.datapoint
        self.text_extraction.serve(self.item.dp_manager.datapoint)

        self.table_segmentation.dp_manager.datapoint = dp_image
        raw_table_segment_list = self.table_segmentation.serve(dp_image)

        self.table_segmentation_refinement.dp_manager.datapoint = dp_image
        self.table_segmentation_refinement.serve(dp_image)
        
        self.text_matching.dp_manager.datapoint = dp_image
        self.text_matching.serve(dp_image)
        
        self.text_ordering.dp_manager.datapoint = dp_image
        self.text_ordering.serve(dp_image)

        page = self.page_parser.pass_datapoint(dp_image)

        dfs = [self._convert_table_segments_to_df(dp_image, raw_table_segments) for raw_table_segments in raw_table_segment_list]

        return page, dfs

    def _convert_table_segments_to_df(self, dp_image, table_segments):
        _table_segments = copy.deepcopy(table_segments)
        sorted_table_segments = sorted(_table_segments, key = lambda x: (int(x.row_num), int(x.col_num)))
        max_row = max([int(item.row_num) for item in sorted_table_segments])
        max_col = max([int(item.col_num) for item in sorted_table_segments])
        table_arr = [[None for i in range(max_col)] for j in range(max_row)]

        for table_segment in sorted_table_segments:
            anns = dp_image.get_annotation(annotation_ids=table_segment.annotation_id)
            for ann in anns:
                text_container_ann_ids = ann.get_relationship(Relationships.child)
                text_container_anns = dp_image.get_annotation(
                    annotation_ids=text_container_ann_ids,
                    category_names=LayoutType.word,
                )
                _row = int(table_segment.row_num) - 1
                _col = int(table_segment.col_num) - 1
                text = self._join_words(text_container_anns)
                table_arr[_row][_col] = text
        return pd.DataFrame(table_arr)

    def _join_words(self, text_container_anns, space_ratio = 23) -> str:
        result = ""
        for idx, text_container_ann in enumerate(text_container_anns):
            word_bbox = text_container_ann.bounding_box
            prev_word_bbox = text_container_anns[idx-1].bounding_box
            word = text_container_ann.sub_categories[WordType.characters].value
            x1, y1, x2, y2 = word_bbox.ulx, word_bbox.uly, word_bbox.lrx, word_bbox.lry
            prev_x2, prev_y2 = prev_word_bbox.lrx, prev_word_bbox.lry
            should_add_space = (x1 - prev_x2)/(y2-y1)*100 > space_ratio
            if idx == 0:
                result += word
            elif y1 > prev_y2:
                result += "\n" + word
            elif should_add_space:
                result += " " + word
            else:
                result += word
        return result

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
        layout_config_path = cfg.CONFIG.D2LAYOUT
        layout_weights_path = cfg.WEIGHTS.D2LAYOUT
        profile = ModelCatalog.get_profile(cfg.WEIGHTS.D2LAYOUT)
        categories_layout = profile.categories
        d_layout = D2FrcnnDetector(layout_config_path, layout_weights_path, categories_layout, device=cfg.DEVICE)
        layout = ImageLayoutService(d_layout, to_image=True, crop_image=True)
        return layout

    def _init_cell_layout_service(self, cfg) -> SubImageLayoutService:
        cell_config_path = cfg.CONFIG.D2CELL
        cell_weights_path = cfg.WEIGHTS.D2CELL
        profile = ModelCatalog.get_profile(cfg.WEIGHTS.D2CELL)
        categories_cell = profile.categories
        d_cell = D2FrcnnDetector(cell_config_path, cell_weights_path, categories_cell, device=cfg.DEVICE)
        
        cell = SubImageLayoutService(d_cell, LayoutType.table, {1: 6}, True)
        return cell

    def _init_item_layout_service(self, cfg) -> SubImageLayoutService:
        item_config_path = cfg.CONFIG.D2ITEM
        item_weights_path = cfg.WEIGHTS.D2ITEM
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

        d_vision_ocr = VisionOcrDetector(cfg.CONFIG.VISION_OCR)
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
