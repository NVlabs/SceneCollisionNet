import argparse
import os
import os.path as osp
import sys
from timeit import default_timer as timer

import imageio
import matplotlib as mpl
import numpy as np
import pygifsicle
import torch
import torch.nn as nn
import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader

mpl.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
from autolab_core import BinaryClassificationResult, YamlConfig, keyboard_input

from scenecollisionnet.collision_models import SceneCollisionNet
from scenecollisionnet.datasets import BenchmarkSceneCollisionDataset

sns.set()


def benchmark(
    device, model, criterion, test_loader, iterations, out_path, vis=True
):
    model.eval()
    time_start = timer()

    if vis:
        vis_path = osp.join(out_path, "vis")
        if not osp.exists(vis_path):
            os.mkdir(vis_path)

    test_loss = 0
    preds, trues = np.array([]), np.array([])
    data_loader = iter(test_loader)
    passes = []
    coll_times = []
    for batch_idx in tqdm.trange(
        iterations,
        desc="Benchmarking",
    ):
        data = next(data_loader)
        coll_time = data[-1]
        coll_times.append(coll_time)
        if vis:
            scene_manager, obj_mesh, obj_pose = data[:3]
            data = [Variable(d.float().to(device)) for d in data[3:-1]]
        else:
            data = [Variable(d.float().to(device)) for d in data[:-1]]
        scene, obj, trans, rots, coll = data
        fw_pass_start = timer()
        with torch.no_grad():
            out = model(scene, obj, trans, rots).squeeze(dim=-1)
        passes.append(timer() - fw_pass_start)

        coll = data[-1].type(out.type())
        loss = criterion(out, coll)

        pred_batch = torch.sigmoid(out).cpu().numpy().flatten()
        true_batch = coll.cpu().numpy().flatten()
        preds = np.append(preds, pred_batch)
        trues = np.append(trues, true_batch)
        test_loss += loss.item()

        if vis:
            obj_pose = obj_pose.numpy()
            trans = trans.cpu().numpy()
            rots = rots.cpu().numpy()
            scene = scene[0, :, :3].cpu().numpy()
            obj = obj[0, :, :3].cpu().numpy()
            obj_centroid = obj.mean(axis=0)
            with imageio.get_writer(
                osp.join(vis_path, "vis_{:d}.gif".format(batch_idx)),
                mode="I",
                duration=0.25,
            ) as writer:
                for i, tr, rot, pr, gt in zip(
                    np.arange(len(trans)), trans, rots, pred_batch, true_batch
                ):
                    mesh_trans = tr - (obj_centroid - obj_pose[:3, 3])
                    mesh_tf = np.eye(4)
                    mesh_tf[:3, 3] = mesh_trans
                    if (rot != 0).any():
                        b1 = rot[:3]
                        b2 = rot[3:] - b1.dot(rot[3:]) * b1
                        b2 /= np.linalg.norm(b2)
                        b3 = np.cross(b1, b2)
                        mesh_tf[:3, :3] = np.stack((b1, b2, b3), axis=-1)
                    new_pose = mesh_tf @ obj_pose

                    # Render images with full meshes
                    pred_color = plt.get_cmap("hsv")(0.3 * (1.0 - pr))[:-1]
                    scene_manager.add_object(
                        "pred_coll_obj",
                        obj_mesh,
                        pose=new_pose,
                        color=pred_color,
                    )
                    pred_im, _ = scene_manager._renderer.render_rgbd()
                    scene_manager.remove_object("pred_coll_obj")
                    gt_color = plt.get_cmap("hsv")(0.3 * (1.0 - gt))[:-1]
                    scene_manager.add_object(
                        "gt_coll_obj", obj_mesh, pose=new_pose, color=gt_color
                    )
                    gt_im, _ = scene_manager._renderer.render_rgbd()
                    scene_manager.remove_object("gt_coll_obj")

                    # Render images with wireframe meshes and point clouds
                    scene_dists = np.linalg.norm(
                        scene_manager._renderer._camera_node.matrix[:3, 3]
                        - scene,
                        axis=1,
                    )
                    scene_colors = (scene_dists - scene_dists.min()) / (
                        scene_dists.max() - scene_dists.min()
                    )
                    for n in scene_manager._renderer._node_dict:
                        scene_manager._renderer._node_dict[n].mesh.primitives[
                            0
                        ].material.wireframe = True
                    scene_manager._renderer.add_points(
                        scene,
                        "scene_points",
                        color=plt.get_cmap("hsv")(scene_colors),
                        radius=0.005,
                    )

                    scene_manager.add_object(
                        "pred_coll_obj",
                        obj_mesh,
                        pose=new_pose,
                        color=pred_color,
                    )
                    pred_pts_im, _ = scene_manager._renderer.render_rgbd()
                    scene_manager.remove_object("pred_coll_obj")
                    scene_manager.add_object(
                        "gt_coll_obj", obj_mesh, pose=new_pose, color=gt_color
                    )
                    gt_pts_im, _ = scene_manager._renderer.render_rgbd()
                    scene_manager.remove_object("gt_coll_obj")
                    scene_manager._renderer.remove_object("scene_points")
                    for n in scene_manager._renderer._node_dict:
                        scene_manager._renderer._node_dict[n].mesh.primitives[
                            0
                        ].material.wireframe = False

                    top_row = np.concatenate(
                        (
                            np.pad(
                                gt_im.data,
                                ((0, 10), (0, 10), (0, 0)),
                                constant_values=np.iinfo(np.uint8).max,
                            ),
                            np.pad(
                                pred_im.data,
                                ((0, 10), (0, 0), (0, 0)),
                                constant_values=np.iinfo(np.uint8).max,
                            ),
                        ),
                        axis=1,
                    )
                    bot_row = np.concatenate(
                        (
                            np.pad(
                                gt_pts_im.data,
                                ((0, 0), (0, 10), (0, 0)),
                                constant_values=np.iinfo(np.uint8).max,
                            ),
                            np.pad(
                                pred_pts_im.data,
                                ((0, 0), (0, 0), (0, 0)),
                                constant_values=np.iinfo(np.uint8).max,
                            ),
                        ),
                        axis=1,
                    )
                    full_gif_im = np.concatenate((top_row, bot_row), axis=0)
                    writer.append_data(full_gif_im)
                scene_manager._renderer._renderer.delete()
                del scene_manager
            pygifsicle.optimize(
                osp.join(vis_path, "vis_{:d}.gif".format(batch_idx))
            )
    test_loss /= iterations

    # sort by probs
    accs = []
    tprs = []
    f1s = []
    taus = np.linspace(0, 1, 21, endpoint=False)
    for t in taus:
        b = BinaryClassificationResult(preds, trues, threshold=t)
        accs.append(b.accuracy)
        f1s.append(b.f1_score)
        tprs.append(b.tpr)

    np.savez_compressed(
        osp.join(out_path, "accuracy_curve.npz"),
        taus=taus,
        accs=accs,
        f1s=f1s,
        tprs=tprs,
    )

    ax = sns.lineplot(x=taus, y=accs, label="Accuracy")
    ax = sns.lineplot(x=taus, y=f1s, label="F1")
    ax = sns.lineplot(x=taus, y=tprs, label="TPR")
    ax.legend()
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.ylim([0.5, 1.0])
    plt.xlim([-0.01, 1.01])
    plt.savefig(osp.join(out_path, "accuracy_curve.png"))

    bcr = BinaryClassificationResult(preds, trues)
    with open(osp.join(out_path, "results.txt"), "w") as f:
        elapsed_time = timer() - time_start
        log = (
            ["Images: {:d}".format(iterations)]
            + ["Queries: {:d}".format(iterations * len(coll.flatten()))]
            + ["Loss: {:.5f}".format(test_loss)]
            + ["Accuracy: {:.3f}".format(bcr.accuracy)]
            + ["F1 Score: {:.3f}".format(bcr.f1_score)]
            + ["AP Score: {:.3f}".format(bcr.ap_score)]
            + [
                "FW Pass Time: {:.4f} +- {:.4f} s".format(
                    np.mean(passes), np.std(passes)
                )
            ]
            + ["Time: {:.2f} s".format(elapsed_time)]
            + [
                "FCL Time: {:.3f} +- {:.3f} s".format(
                    np.mean(coll_times), np.std(coll_times)
                )
            ]
        )
        log = map(str, log)
        f.write("\n".join(log))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Benchmark a 3D CollisionNet")
    parser.add_argument(
        "--cfg",
        type=str,
        default="cfg/benchmark_scenecollisionnet.yaml",
        help="config file with benchmarking params",
    )
    args = parser.parse_args()

    config = YamlConfig(args.cfg)

    # Create output directory for model
    out = osp.join(config["output"], config["model"]["name"])
    if osp.exists(out):
        response = keyboard_input(
            "A benchmark folder exists at {}. Overwrite?".format(out),
            yesno=True,
        )
        if response.lower() == "n":
            os.exit()
    else:
        os.makedirs(out)
    config.save(osp.join(out, "benchmark.yaml"))

    # Check whether GPU available
    if not torch.cuda.is_available():
        print("No CUDA available!")
        sys.exit(0)
    device = torch.device(config["device"])

    # 1. Model
    model_path = osp.join(config["model"]["path"], config["model"]["name"])
    train_cfg = YamlConfig(osp.join(model_path, "train.yaml"))
    model = SceneCollisionNet(
        bounds=train_cfg["model"]["bounds"],
        vox_size=train_cfg["model"]["vox_size"],
    )
    checkpoint = torch.load(osp.join(model_path, "model.pth.tar"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)

    # 2. Dataset
    kwargs = {
        "num_workers": 0
        if config["vis"]
        else (
            config["num_workers"]
            if "num_workers" in config
            else os.cpu_count()
        ),
        "pin_memory": True,
        "worker_init_fn": lambda _: np.random.seed(),
    }
    for k, v in config["dataset"].items():
        train_cfg["dataset"][k] = v
    test_set = BenchmarkSceneCollisionDataset(
        **train_cfg["dataset"],
        **train_cfg["camera"],
        bounds=train_cfg["model"]["bounds"],
        vis=config["vis"]
    )
    test_loader = DataLoader(test_set, batch_size=None, **kwargs)

    benchmark(
        device=device,
        model=model,
        criterion=nn.BCEWithLogitsLoss(),
        test_loader=test_loader,
        iterations=config["iterations"],
        out_path=out,
        vis=config["vis"],
    )
