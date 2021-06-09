import os

import numpy as np
import trimesh.transformations as tra
from isaacgym import gymapi


def from_rpy(a, e, th):
    rt = tra.euler_matrix(a, e, th)
    q = tra.quaternion_from_matrix(rt)
    q = np.roll(q, -1)
    return gymapi.Quat(q[0], q[1], q[2], q[3])


def gym_pose_to_matrix(pose):
    q = [pose["r"][3], pose["r"][0], pose["r"][1], pose["r"][2]]
    trans = tra.quaternion_matrix(q)
    trans[:3, 3] = [pose["p"][i] for i in range(3)]

    return trans


def write_urdf(
    obj_name,
    obj_path,
    output_folder,
):
    content = open("resources/urdf.template").read()
    content = content.replace("NAME", obj_name)
    content = content.replace("MEAN_X", "0.0")
    content = content.replace("MEAN_Y", "0.0")
    content = content.replace("MEAN_Z", "0.0")
    content = content.replace("SCALE", "1.0")
    content = content.replace("COLLISION_OBJ", obj_path)
    content = content.replace("GEOMETRY_OBJ", obj_path)
    urdf_path = os.path.abspath(
        os.path.join(output_folder, obj_name + ".urdf")
    )
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    open(urdf_path, "w").write(content)
    return urdf_path
