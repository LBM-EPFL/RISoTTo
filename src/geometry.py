import torch as pt


def superpose(xyz_ref, xyz):
    # centering
    t = pt.mean(xyz, dim=1).unsqueeze(1)
    t_ref = pt.mean(xyz_ref, dim=1).unsqueeze(1)

    # SVD decomposition
    U, _, Vt = pt.linalg.svd(pt.matmul(pt.transpose(xyz_ref-t_ref,1,2), xyz-t))

    # reflection matrix
    Z = pt.zeros(U.shape, device=xyz.device) + pt.eye(U.shape[1], U.shape[2], device=xyz.device).unsqueeze(0)
    Z[:,-1,-1] = pt.linalg.det(U) * pt.linalg.det(Vt)

    R = pt.matmul(pt.transpose(Vt,1,2), pt.matmul(Z, pt.transpose(U,1,2)))

    return pt.matmul(xyz-t, R)+t_ref


