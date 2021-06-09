import argparse
import os

import h5py
from tqdm import tqdm

from scenecollisionnet.utils import (
    MeshLoader,
    ProcessKillingExecutor,
    process_mesh,
)


def process_mesh_timeout(*args, **kwargs):
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate mesh dataset")
    parser.add_argument(
        "meshes_dir",
        type=str,
        help="path to the Shapenet mesh dataset directory",
    )
    parser.add_argument(
        "grasps_dir",
        type=str,
        help="path to ACRONYM grasp dataset directory of hdf5 files",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="output dataset path",
    )

    args = parser.parse_args()
    mesh_dir = args.meshes_dir
    grasps_dir = args.grasps_dir
    out_dir = args.output_dir

    if not os.path.exists(mesh_dir):
        print("Input directory does not exist!")
    mesh_loader = MeshLoader(mesh_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_ds_file = os.path.join(out_dir, "object_info.hdf5")

    inputs = []
    for f in tqdm(os.listdir(grasps_dir), desc="Generating Inputs"):
        mc, mk, ms = os.path.splitext(f)[0].split("_")
        try:
            in_path = mesh_loader.get_path(mk)
        except ValueError:
            continue
        out_path = os.path.join(
            out_dir,
            mc,
            os.path.splitext(os.path.basename(in_path))[0] + ".obj",
        )
        inputs.append(
            (in_path, out_path, float(ms), os.path.join(grasps_dir, f))
        )

    executor = ProcessKillingExecutor(max_workers=8)
    generator = executor.map(
        process_mesh,
        inputs,
        timeout=120,
        callback_timeout=process_mesh_timeout,
    )

    categories = {}
    with h5py.File(out_ds_file, "a") as f:
        f.create_group("meshes")
        f.create_group("categories")
        for mesh_info in tqdm(
            generator, total=len(inputs), desc="Processing Meshes"
        ):
            if mesh_info is not None:
                mk, minfo = mesh_info
                f["meshes"].create_group(mk)
                for key in minfo:
                    f["meshes"][mk][key] = minfo[key]

                if minfo["category"] not in categories:
                    categories[minfo["category"]] = [mk]
                else:
                    categories[minfo["category"]].append(mk)

        for c in categories:
            f["categories"][c] = categories[c]
