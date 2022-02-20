import numpy as np


def convert_w_t_c(RT_ctow):
    """
    input RT is transform from camera to world
    o3d requires world to camera
    """
    print("TODO1: INCORRECT CODE: RT_wtoc doesn't have 1 at 3,3 position")
    RT_wtoc = np.zeros((RT_ctow.shape))
    RT_wtoc[0:3,0:3] = RT_ctow[0:3,0:3].T
    RT_wtoc[0:3,3] = - RT_ctow[0:3,0:3].T @ RT_ctow[0:3,3]
    return RT_wtoc