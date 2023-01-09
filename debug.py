import cv2
import numpy as np
from deepdoctection.ocr_detector import OCRDetector
from pdf2image import convert_from_path
import pandas as pd
pd.set_option("max_colwidth", None)
from IPython.display import display
from glob import glob
from PIL import Image
from IPython.core.display import HTML

# paths = sorted(glob("dataset/raw_po_cpall/*.pdf"))
# path = "../ocr_test/CPALL/CPALL/effd964ca5b54828ba52df2ced8bd5c3.png"
# bgr_image_as_np_array = cv2.imread(path)
# path = "../ocr_test/CPALL/CPALL/a426d83bd4b2452db47ee6695d5b3e3f.pdf"

# path = "../ocr_test/CPALL/CPALL/c02b482f3d244073aa610ab9866ff154.pdf"
# path = "../ocr_test/CPALL/CPALL/c6c87df29bde46c8a78486e34f9bd6eb.pdf"
# path = "../ocr_test/CPALL/CPALL/8cac0123ec774ae3b21d22931cac04fd.pdf"

# path = "00062_ad3f61fa98774eb5a236717720f6f2e0.pdf"
# path = paths[1]
path = "dataset/raw_po_cpall/5520fe2cf8ea4200a9658c1c275263ab.pdf"
# path = paths[3]
images = convert_from_path(path)
print(len(images))
# bgr_image_as_np_array = np.array(images[6])
# bgr_image_as_np_array = np.array(images[4])
bgr_image_as_np_array = np.array(images[5])

ocr_detector = OCRDetector()

page, dfs = ocr_detector.predict(bgr_image_as_np_array)
print(dfs[0])