from rembg import remove
from PIL import Image, ImageOps
from argparse import ArgumentParser
from os import listdir
from os.path import isfile, join, splitext

parser = ArgumentParser(description='imgprep script to remove backgrounds and optionally convert to grayscale')
parser.add_argument('--in', dest='in_dir', help='Set input directory', default='./')
parser.add_argument('--out', dest='out_dir', help='Set output directory', default=None)
parser.add_argument('-mono', dest='mono', help='Enable conversion to monochrome', action='store_true', default=False)
args = parser.parse_args()

if args.out_dir is None:
    args.out_dir = args.in_dir

files = [f for f in listdir(args.in_dir) if isfile(join(args.in_dir, f))]

for f in files:
    ext = splitext(f)[-1].lower()
    in_path = join(args.in_dir, f)
    out_path = join(args.out_dir, f)
    if ext == '.png':
        in_img = Image.open(in_path)
        out_img = remove(in_img)
        if args.mono:
            out_img = out_img.convert('LA').convert('RGBA')
            #out_img = ImageOps.grayscale(out_img)
        out_img.save(out_path)
        print("finished " + in_path)
