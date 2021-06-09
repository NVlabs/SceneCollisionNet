import numpy as np
import trimesh.transformations as tra

GRASP_FRAME_INIT_OFFSET = tra.quaternion_matrix([0.707, 0, 0, -0.707])
GRASP_FRAME_INIT_OFFSET[:3, 3] = [0, 0, 0.0]
GRASP_FRAME_FINAL_OFFSET = tra.quaternion_matrix([0.707, 0, 0, -0.707])
GRASP_FRAME_FINAL_OFFSET[:3, 3] = [0, 0, 0.1]

TABLE_LABEL = 0
ROBOT_LABEL = np.iinfo(np.uint8).max
DEPTH_CLIP_RANGE = 3.0

ROBOT_Q_INIT = np.array(
    [
        -1.22151887,
        -1.54163973,
        -0.3665906,
        -2.23575787,
        0.5335327,
        1.04913162,
        -0.14688508,
        0.04,
        0.04,
    ]
)
