import torch

import numpy as np
import torch
import torch.nn.functional
from e3nn import o3
from e3nn.util import jit
from scipy.spatial.transform import Rotation as R

from mace import data, modules, tools
from mace.tools import torch_geometric
from mace.modules.models import ScaleShiftMACE
from mace.tools.scripts_utils import extract_config_mace_model

# set default dtype
torch.set_default_dtype(torch.float64)

table = tools.AtomicNumberTable([1, 6, 7, 8, 9, 15, 16, 17, 35, 53])
config = data.Configuration(
    atomic_numbers=np.array([8, 1, 1]),
    positions=np.array(
        [
            [0.0, -2.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    forces=np.array(
        [
            [0.0, -1.3, 0.0],
            [1.0, 0.2, 0.0],
            [0.0, 1.1, 0.3],
        ]
    ),
    energy=-1.5,
)

atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=3.0)

data_loader = torch_geometric.dataloader.DataLoader(
    dataset=[atomic_data],
    batch_size=1,
    shuffle=True,
    drop_last=False,
)
batch = next(iter(data_loader))
print(batch)


def main(model_path: str, new_model_path: str) -> ScaleShiftMACE:
    old_model = torch.load(model_path)
    model_config = extract_config_mace_model(old_model)
    print(model_config)

    # init the new ScaleShiftMACE
    model = ScaleShiftMACE(**model_config)
    model.load_state_dict(old_model.state_dict())
    torch.save(model, new_model_path)

    old_energy = old_model(batch.to_dict(), training=True)
    new_energy = model(batch.to_dict(), training=True)

    print("Old energy:", old_energy)
    print("New energy:", new_energy)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--old_model", "-o", type=str)
    parser.add_argument("--new_model", "-n", type=str)
    args = parser.parse_args()

    main(args.old_model, args.new_model)
