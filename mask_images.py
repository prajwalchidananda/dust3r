import numpy as np
import cv2
import glob
import argparse


def mask_fill(image, mask, fill_value=(0, 0, 0)):
    image = image.copy()
    image = image.astype("float32")
    mask = mask.astype("float32")
    mask = mask / 255
    fill_value = np.array(fill_value, dtype="float32")
    image = image * (1 - mask[:, :, None]) + fill_value * mask[:, :, None]
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, nargs="+")
    parser.add_argument("--masks", type=str, nargs="+")
    parser.add_argument("--outdir", type=str, default="io/output/")
    parser.add_argument("--fill_value", type=int, nargs=3, default=(0, 0, 0))
    parser.add_argument("--blur_kernel", type=int, default=5)
    parser.add_argument("--dilate_kernel", type=int, default=5)
    args = parser.parse_args()

    if len(args.images) != len(args.masks):
        raise ValueError("Images and masks must have the same length")

    if len(args.images) == 1:
        image_paths = sorted(glob.glob(args.images[0]))
        mask_paths = sorted(glob.glob(args.masks[0]))
    else:
        image_paths = args.images
        mask_paths = args.masks

    for image_path, mask_path in zip(image_paths, mask_paths):
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate_kernel, args.dilate_kernel)))
        mask = cv2.GaussianBlur(mask, (args.blur_kernel, args.blur_kernel), 0)

        image = mask_fill(image, mask, fill_value=args.fill_value)

        cv2.imwrite(f"{args.outdir}/{image_path.split('/')[-1]}", image)