import argparse
import os
from shutil import copyfile

import h5py
import numpy as np
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
        help="path to the mesh dataset directory",
    )
    parser.add_argument(
        "--dataset_file", type=str, help="path to existing dataset hdf5"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="output dataset path",
    )

    args = parser.parse_args()
    mesh_dir = args.meshes_dir
    dataset_file = args.dataset_file
    out_dir = args.output_dir

    if not os.path.exists(mesh_dir):
        print("Input directory does not exist!")
    mesh_loader = MeshLoader(mesh_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_ds_file = os.path.join(out_dir, "object_info.hdf5")
    if dataset_file is not None:
        copyfile(dataset_file, out_ds_file)

        mesh_keys, mesh_scales, mesh_cats = [], [], []
        with h5py.File(out_ds_file, "r") as f:
            for mk in f["meshes"]:
                mesh_keys.append("_".join(mk.split("_")[:-1]))
                mesh_scales.append(f["meshes"][mk]["scale"][()])
                mesh_cats.append(f["meshes"][mk]["category"].asstr()[()])
            mesh_stps = np.zeros(len(mesh_keys), dtype=bool)
    else:
        mesh_keys = mesh_loader.meshes()
        mesh_scales = np.ones(len(mesh_keys))
        mesh_cats = [
            mk.split("~")[0] if len(mk.split("~")) > 1 else ""
            for mk in mesh_keys
        ]
        mesh_stps = np.ones(len(mesh_keys), dtype=bool)

    inputs = []
    for mk, ms, mc, mp in tqdm(
        zip(mesh_keys, mesh_scales, mesh_cats, mesh_stps),
        total=len(mesh_keys),
        desc="Generating Inputs",
    ):
        try:
            in_path = mesh_loader.get_path(mk)
        except ValueError:
            continue
        out_path = os.path.abspath(
            os.path.join(
                out_dir,
                mc,
                os.path.splitext(os.path.basename(in_path))[0] + ".obj",
            )
        )
        inputs.append((in_path, out_path, ms, None, mp))

    executor = ProcessKillingExecutor(max_workers=8)
    generator = executor.map(
        process_mesh,
        inputs,
        timeout=120,
        callback_timeout=process_mesh_timeout,
    )

    with h5py.File(out_ds_file, "a") as f:
        if "meshes" not in f:
            f.create_group("meshes")

        categories = {}
        if "categories" not in f:
            f.create_group("categories")

        for mesh_info in tqdm(
            generator, total=len(inputs), desc="Processing Meshes"
        ):
            if mesh_info is not None:
                mk, minfo = mesh_info
                if mk not in f["meshes"]:
                    f["meshes"].create_group(mk)
                for key in minfo:
                    if key in f["meshes"][mk]:
                        del f["meshes"][mk][key]
                    f["meshes"][mk][key] = minfo[key]

                if minfo["category"] not in categories:
                    categories[minfo["category"]] = [mk]
                elif mk not in categories[minfo["category"]]:
                    categories[minfo["category"]].append(mk)

        for c in categories:
            if c in f["categories"]:
                del f["categories"][c]
            f["categories"][c] = categories[c]
