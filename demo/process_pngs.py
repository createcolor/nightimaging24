import argparse
from tqdm import tqdm
from raw_prc_pipeline.pipeline import PipelineExecutor, RawProcessingPipelineDemo
from pathlib import Path
import cv2
import numpy as np
from raw_prc_pipeline import expected_landscape_img_height, expected_landscape_img_width, expected_img_ext
from utils import fraction_from_json, json_read


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo script for processing PNG images with given metadata files.')
    parser.add_argument('-p', '--png_dir', type=Path, required=True,
                        help='Path of the directory containing PNG images with metadata files')
    parser.add_argument('-o', '--out_dir', type=Path, default=None,
                        help='Path to the directory where processed images will be saved. Images will be saved in JPG format.')
    parser.add_argument('-ie', '--illumination_estimation', type=str, default='gw',
                        help='Options for illumination estimation algorithms: "gw", "wp", "sog", "iwp".')
    parser.add_argument('-tm', '--tone_mapping', type=str, default='Flash',
                        help='Options for tone mapping algorithms: "Base", "Flash", "Storm", "Linear", "Drago", "Mantiuk", "Reinhard".')
    parser.add_argument('-n', '--denoising_flg', action='store_false',
                        help='Denoising flag. By default resulted images will be denoised with some default parameters.')
    parser.add_argument('-m', '--camera_matrix', type=float, nargs=9, default=[ 1.06835938, -0.29882812, -0.14257812, 
                                                                                 -0.43164062,  1.35546875,  0.05078125, 
                                                                                 -0.1015625, 0.24414062, 0.5859375], 
                                                                                 help='Mean color matrix of Hauwei Mate 40 Pro')
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.png_dir

    return args


class PNGProcessingDemo():
    def __init__(self, ie_method, tone_mapping, denoising_flg, camera_matrix):
        self.camera_matrix = camera_matrix
        self.pipeline_demo = RawProcessingPipelineDemo(
            illumination_estimation=ie_method, denoise_flg=denoising_flg, tone_mapping=tone_mapping, 
            out_landscape_height=expected_landscape_img_height, out_landscape_width=expected_landscape_img_width)

    def __call__(self, png_path: Path, out_path: Path):
        # parse raw img
        raw_image = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        # parse metadata
        metadata = json_read(png_path.with_suffix('.json'), object_hook=fraction_from_json)

        # executing img pipelinex
        pipeline_exec = PipelineExecutor(
            raw_image, metadata, self.pipeline_demo)
        # process img
        output_image = pipeline_exec()

        # save results
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), output_image, [
                    cv2.IMWRITE_JPEG_QUALITY, 100])


def main(png_dir, out_dir, illumination_estimation, tone_mapping, denoising_flg, camera_matrix):
    out_dir.mkdir(exist_ok=True)

    png_paths = list(png_dir.glob('*.png'))
    out_paths = [
        out_dir / png_path.with_suffix(expected_img_ext).name for png_path in png_paths]

    png_processor = PNGProcessingDemo(illumination_estimation, tone_mapping, denoising_flg, camera_matrix)
        
    for png_path, out_path in tqdm(zip(png_paths, out_paths), total=len(png_paths)):
        png_processor(png_path, out_path)


if __name__ == '__main__':
    args = parse_args()
    main(**vars((args)))
