import torch as pt


def superpose(xyz_ref, xyz):
    # centering
    t = pt.mean(xyz, dim=1).unsqueeze(1)
    t_ref = pt.mean(xyz_ref, dim=1).unsqueeze(1)

    # SVD decomposition
    U, S, Vt = pt.linalg.svd(pt.matmul(pt.transpose(xyz_ref-t_ref,1,2), xyz-t))

    # reflection matrix
    Z = pt.zeros(U.shape, device=xyz.device) + pt.eye(U.shape[1], U.shape[2], device=xyz.device).unsqueeze(0)
    Z[:,-1,-1] = pt.linalg.det(U) * pt.linalg.det(Vt)

    R = pt.matmul(pt.transpose(Vt,1,2), pt.matmul(Z, pt.transpose(U,1,2)))

    return xyz_ref-t_ref, pt.matmul(xyz-t, R)


def compute_rmsd(xyz0, xyz1):
    # superpose
    xyz1, xyz0 = superpose(xyz0.view(1,-1,3), xyz1.view(1,-1,3))

    # compute rmsd
    rmsd = pt.sqrt(pt.mean(pt.sum(pt.square(xyz0-xyz1), dim=2)))

    return rmsd


