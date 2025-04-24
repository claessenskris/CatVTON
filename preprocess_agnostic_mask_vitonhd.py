import argparse
import os
import json

from huggingface_hub import snapshot_download

from diffusers.image_processor import VaeImageProcessor
from model.cloth_masker import AutoMasker

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of Preprocess Agnostic Mask")
    parser.add_argument(
        "--data_root_path", 
        type=str, 
        required=True,
        help="Path to the dataset to evaluate."
    )
    parser.add_argument(
        "--repo_path",
        type=str,
        default="zhengchong/CatVTON",
        help=(
            "The Path or repo name of CatVTON. "
        ),
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def main(args):
    args.repo_path = snapshot_download(repo_id=args.repo_path)

    mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(args.repo_path, "DensePose"),
        schp_ckpt=os.path.join(args.repo_path, "SCHP"),
        device='cuda', 
    )

    assert os.path.exists(pair_txt := os.path.join(args.data_root_path,
                                                   'test_pairs_unpaired.txt')), f"File {pair_txt} does not exist."
    with open(pair_txt, 'r') as f:
        lines = f.readlines()
    args.data_root_path = os.path.join(args.data_root_path, 'test')
    output_dir = os.path.join(args.data_root_path, 'agnostic-mask')
    for line in lines:
        person_img, cloth_img = line.strip().split(" ")
        if os.path.exists(os.path.join(output_dir, person_img.replace('.jpg', '.png'))):
            continue
        cloth_img_without_ext = os.path.splitext(cloth_img)[0]
        cloth_img_json = os.path.join(args.data_root_path, 'cloth', f"{cloth_img_without_ext}.json")
        try:
            with open(cloth_img_json, 'r') as read_file:
                json_record = json.load(read_file)
            cloth_type = json_record['cloth_type']
        except FileNotFoundError:
            continue
        mask = automasker(
            os.path.join(args.data_root_path, 'image', person_img),
            cloth_type
        )['mask']
        mask = mask_processor.blur(mask, blur_factor=9)
        mask.save(os.path.join(output_dir, person_img.replace('.jpg', '_mask.png')))
        
if __name__ == "__main__":
    args = parse_args()
    main(args)
