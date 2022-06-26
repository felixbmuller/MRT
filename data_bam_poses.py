import logging

import torch.utils.data as data
import numpy as np

from bam_poses.data.person import Person
from bam_poses.data.group import Group

FPS = 25  # frame rate in BAM poses, the same as we use


class BamPosesData(data.Dataset):
    def __init__(self, train: bool = True):
        """Load BAMp data via the bam_poses API, split it into non-overlapping 4 second slices and 
        transform it to be compatible with the MRT model.

        Parameters
        ----------
        train : bool, optional
            whether to load the training (A, B, C) or test data (D), by default True
        """

        if train:
            logging.info("Loading training data")

            posesA, masksA = Group(Person.load_for_dataset(dataset="A")).get_splits(
                length=4 * FPS, non_overlapping=True
            )
            posesB, masksB = Group(Person.load_for_dataset(dataset="B")).get_splits(
                length=4 * FPS, non_overlapping=True
            )
            posesC, masksC = Group(Person.load_for_dataset(dataset="C")).get_splits(
                length=4 * FPS, non_overlapping=True
            )

            self.poses = np.concatenate((posesA, posesB, posesC), axis=0)
            self.masks = np.concatenate((masksA, masksB, masksC), axis=0)

        else:
            logging.info("Loading test data")

            self.poses, self.masks = Group(
                Person.load_for_dataset(dataset="D")
            ).get_splits(length=4 * FPS, non_overlapping=True)

        self.n_scenes = self.poses.shape[0]
        self.n_persons = self.poses.shape[1]
        self.n_frames = self.poses.shape[2]

        logging.info("Filtering and reshaping joints")

        # drop eyes and ground truth nose to reduce the number of joints from 18 to 15
        # fmt: off
        use_joints = [0, # nose 
               3, 4, # ears
               5, 6, 7, 8, 9, 10, # arms
               11, 12, 13, 14, 15, 16]
        # fmt: on
        self.poses = self.poses[:, :, :, use_joints, :]

        # reshape to combine joints and dimensions
        self.n_joint_dim = len(use_joints) * 3
        self.poses = self.poses.reshape(
            self.n_scenes,
            self.n_persons,
            self.n_frames,
            self.n_joint_dim,
        )

        logging.info("Data loading complete")

    def __getitem__(self, iscene):
        """Return the given scene split into a 1 sec input and 3 sec output sequence.

        Parameters
        ----------
        iscene : bool
            scene id between 0 and len(self)

        Returns
        -------
        in_seq
            input sequence of shape (self.n_persons, FPS, self.n_joint_dim)
        out_seq
            output sequence of shape (self.n_persons, FPS * 2 + 1, self.n_joint_dim)
        """

        # select all fully visible persons
        people = [
            self.poses[iscene, iperson]
            for iperson in range(self.n_persons)
            if np.mean(self.masks[iscene, iperson]) > 0.999
        ]

        in_seq = np.full((self.n_persons, FPS, self.n_joint_dim), np.nan)
        out_seq = np.full((self.n_persons, FPS * 2 + 1, self.n_joint_dim), np.nan)

        # add all fully present people with one frame overlap
        for iperson, person in enumerate(people):
            in_seq[iperson] = person[:FPS]
            out_seq[iperson] = person[FPS - 1 :]

        return in_seq, out_seq

    def __len__(self):
        return self.n_scenes

