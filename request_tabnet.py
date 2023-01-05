import requests
import numpy as np
import cv2
from PIL import Image
import os
from glob import glob
import json
from pdf2image import convert_from_path
import base64
from typing import Dict, List, Tuple
from datetime import datetime

def np_img_to_b64(img: np.array) -> str:
    
    _, buffer = cv2.imencode('.jpg', img)
    base64_byte_image = base64.b64encode(buffer)
    base64_str_image = base64_byte_image.decode("utf_8")
    
    return base64_str_image

def request_bboxes(b64_str_img: str, url: str) -> Tuple[List, List, List]:
    """
    return borderless bboxes, cell bboxes, border bboxes
    """
    image_payload = {'image': b64_str_img}
    response = requests.post(url, json = image_payload)
    try:
        response_bboxes = json.loads(response.text)
        bboxes_result = response_bboxes["result"]
        return bboxes_result[0], bboxes_result[1], bboxes_result[2]
    except:
        return [], [], []

input_root_dir = "dataset/po_cpf/CPFSAP_1"
output_path = "output_cascade_tabnet/result.json"
pdf_paths = sorted(glob(f"{input_root_dir}/*.pdf"))
crop_table_url = "https://ptvn-document-ai-table-extraction-tzprdgk55a-as.a.run.app/api/extract_tables"

with open(output_path, "r") as f:
    try:
        result_dict = json.load(f)
    except json.JSONDecodeError as e:
        result_dict = {}
        
existing_files = result_dict.keys()
existing_file_path = [f"{input_root_dir}/{fn}" for fn in existing_files]

filtered_pdf_paths = [path for path in pdf_paths if path not in existing_file_path]

print(len(pdf_paths), len(filtered_pdf_paths))

start_time = datetime.now()
for idx, path in enumerate(filtered_pdf_paths):
    filename = os.path.basename(path)

    result_dict[filename] = []

    images = convert_from_path(path)
    for page_idx, image in enumerate(images):
        rgb_np_img = np.array(image)
        bgr_np_img = cv2.cvtColor(rgb_np_img, cv2.COLOR_RGB2BGR)

        base64_str_image = np_img_to_b64(bgr_np_img)

        borderless_bboxes, cell_bboxes, border_bboxes = request_bboxes(base64_str_image, crop_table_url)
        if borderless_bboxes or cell_bboxes or border_bboxes:
            bboxes_result = {
                "page": page_idx,
                "borderless_bboxes": borderless_bboxes,
                "cell_bboxes": cell_bboxes,
                "border_bboxes": border_bboxes
            }
            result_dict[filename].append(bboxes_result)
    with open(output_path, "w") as f:
        json.dump(result_dict, f)
    print(f"index: {idx}, filename: {filename} is finished")
    print(f"page count: {len(images)}, time: {datetime.now() - start_time}")
    print("-"*20)