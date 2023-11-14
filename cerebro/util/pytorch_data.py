import os
import gc
import glob
import torch
import pandas as pd
from torch.utils.data import IterableDataset


class GeneralPytorchDataset(IterableDataset):
    def __init__(self, mode, pickled_data_path):
        self.remaining_files = glob.glob(os.path.join(pickled_data_path, "*.pkl"))
        self.completed_files = []
        self.local_index = 0
        self.current_df = None
        self.mode = mode

    def __iter__(self):
        # initialize first file

        gc.collect()
        torch.cuda.empty_cache()

        f = self.remaining_files.pop(0)
        self.completed_files.append(f)
        self.local_index = 0
        self.current_df = pd.read_pickle(f)
        self.current_df = self.current_df.reset_index(drop=True)

        return self

    def __next__(self):
        # load next and remove current dataframe if current index is more than the size of the current dataframe
        # TODO: fix the case when a file will have 0 records !!
        self.local_index += 1

        if self.local_index >= len(self.current_df.index):
            # completed all files
            if not self.remaining_files:
                raise StopIteration
            else:
                f = self.remaining_files.pop(0)
                self.completed_files.append(f)
                self.local_index = 0
                self.current_df = pd.read_pickle(f)
                self.current_df = self.current_df.reset_index(drop=True)
                gc.collect()

        input_tensor = self.current_df["input_tensor"][self.local_index]
        output_tensor = self.current_df["output_tensor"][self.local_index]
        row_id =  self.current_df["id"][self.local_index]

        return input_tensor, output_tensor, row_id

    def __len__(self):
        # parse through all files to determine total number of examples

        total_length = 0
        all_files = self.remaining_files + self.completed_files

        for f in all_files:
            df = pd.read_pickle(f)
            total_length += len(df.index)

        return total_length
