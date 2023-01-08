from pdf2image import convert_from_path as convert
from pathlib import Path
import sys

if len(sys.argv) < 2:
    filename = input("Filename: ")
else:
    filename = sys.argv[1]

prefix = Path(filename).stem
for idx, page in enumerate(convert(filename)):
    page.save(f"{prefix}_p{idx+1}.png", "PNG")