import argparse
import os

from huggingface_hub import snapshot_download
from tqdm import tqdm

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
    cloth_type = 'upper'
    for line in lines:
        person_img, _ = line.strip().split(" ")
        if os.path.exists(os.path.join(output_dir, person_img.replace('.jpg', '.png'))):
            continue
        mask = automasker(
            os.path.join(args.data_root_path, 'image', person_img),
            cloth_type
        )['mask']
        mask.save(os.path.join(output_dir, person_img.replace('.jpg', '.png')))

if __name__ == "__main__":
    args = parse_args()
    main(args)
