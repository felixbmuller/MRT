
import torch
import numpy as np
import pandas as pd
import fire
from tqdm import tqdm

import data as D
import inference as I
import utils as U

# current bug (in my opinion): one of the tensors in calc_loss contains a batch dimension, the other does not

def eval_training_progress(sample_size=800):
    """run with 5 cpus and ~3gb ram

    Parameters
    ----------
    sample_size : int, optional
        _description_, by default 800
    """

    with torch.no_grad():

        all_datasets = {
            "mocap": D.TESTDATA("mocap"),
            "mupots": D.TESTDATA("mupots"),
            "train_mocap": D.DATA(),
        }

        all_losses = []

        for model_idx in range(4, 100, 5):

            model = I.init_model(f"saved_model/{model_idx}.model")

            print("Loaded model ", model_idx)

            for data_type in ["mocap", "mupots", "train_mocap"]:

                print("  Evaluating on ", data_type)

                torch.manual_seed(42)
                loader = torch.utils.data.DataLoader(all_datasets[data_type], batch_size=1, shuffle=True)

                for jjj, data in enumerate(loader):

                    if jjj >= sample_size:
                        break

                    pred, actual = I.infer(model, *data)

                    l1, l2, l3 = I.calc_loss(pred, actual)

                    all_losses.append((model_idx, data_type, l1, l2, l3))

        df = pd.DataFrame(all_losses, columns=["model", "dataset", "l1", "l2", "l3"])
        df.index.name = "idx"

        print("Storing dataframe")
        df.to_csv(f"output/all_losses_ssize{sample_size}_{U.timestamp()}.csv")

        
if __name__ == "__main__":
    fire.Fire(eval_training_progress)


            