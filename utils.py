import datetime

import numpy as np
import torch


N_JOINTS = 15
BODY_EDGES = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 4],
        [4, 5],
        [5, 6],
        [0, 7],
        [7, 8],
        [7, 9],
        [9, 10],
        [10, 11],
        [7, 12],
        [12, 13],
        [13, 14],
    ]
)


def disc_l2_loss(disc_value):

    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k


def adv_disc_l2_loss(real_disc_value, fake_disc_value):

    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]
    lb, la = (
        torch.sum(fake_disc_value ** 2) / kb,
        torch.sum((real_disc_value - 1) ** 2) / ka,
    )
    return la, lb, la + lb


def timestamp():
    return datetime.datetime.today().strftime("%y%m%d-%H%M")


def to_np(*tensors):
    """convert one or multiple torch.Tensor's to np.ndarray's. Does nothing if input is already
    a numpy array.
    """

    results = [
        (t.results.detach().cpu().numpy() if not isinstance(t, np.ndarray) else t)
        for t in tensors
    ]

    if len(results) == 1:
        return results[0]
    else:
        return results


def to_t(*arrays, dtype=torch.float32, device="cpu"):
    """convert one or multiple numpy arrays to torch tensors. Does nothing if input is already
    a tensor.

    TODO: check and adjust type of tensor if need be

    Parameters
    ----------
    dtype : _type_, optional
        torch dtype, by default torch.float32
    device : str, optional
        torch device, by default "cpu"
    """

    results = [
        (
            torch.tensor(a, dtype=dtype).to(device)
            if not isinstance(a, torch.Tensor)
            else a.to(device)
        )
        for a in arrays
    ]

    if len(results) == 1:
        return results[0]
    else:
        return results
