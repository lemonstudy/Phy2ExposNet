from __future__ import print_function, division
import os
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models

class RualDataset(Dataset):
    """
    Prepared USC loader with historical file splits and optional rotation grouping.
      Train: 15000
      Val:    2000
      Test:   2000
    The local USC_2 directory contains 19016 images; 16 remain unused.

    Directory:
      root/
        city_map/
        tx_map/
        power_map/

    inputs : (2, H, W)  -> [city_map, tx_map]
    target : (1, H, W)  -> power_map
    """

    def __init__(self,
                 dir_dataset="data/USC/",
                 phase="train",          # "train" | "val" | "test"
                 seed=42,
                 thresh=0.0,
                 transform=transforms.ToTensor(),
                 exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
                 split_mode="legacy"):

        assert phase in ["train", "val", "test"]

        self.dir_dataset = dir_dataset
        self.dir_city  = os.path.join(dir_dataset, "city_map")
        self.dir_tx    = os.path.join(dir_dataset, "tx_map")
        self.dir_power = os.path.join(dir_dataset, "power_map")

        self.phase = phase
        self.seed = seed
        self.thresh = thresh
        self.transform = transform
        self.exts = exts
        if split_mode not in ("grouped", "legacy"):
            raise ValueError(f"Unknown USC split mode: {split_mode}")
        self.split_mode = split_mode

        names = [
            f for f in os.listdir(self.dir_power)
            if os.path.isfile(os.path.join(self.dir_power, f))
            and f.lower().endswith(exts)
        ]
        names.sort()

        valid = []
        for fn in names:
            if (os.path.exists(os.path.join(self.dir_city, fn)) and
                os.path.exists(os.path.join(self.dir_tx, fn))):
                valid.append(fn)

        if len(valid) < 19000:
            raise ValueError(
                f"Prepared USC split requires at least 19000 matched samples; found {len(valid)}. "
                "The local experiment used USC_2 (19016 images), not the raw 4754-sample archive. "
                "See the Datasets section in README.md before preparing or changing this split."
            )
        rng = np.random.RandomState(seed)
        if split_mode == "legacy":
            idx = np.arange(len(valid))
            rng.shuffle(idx)
            valid = [valid[i] for i in idx]
            split_names = {
                "train": valid[:15000],
                "val": valid[15000:17000],
                "test": valid[17000:19000],
            }
        else:
            groups = defaultdict(list)
            for filename in valid:
                stem = Path(filename).stem
                if not stem.isdigit() or int(stem) < 1:
                    raise ValueError("Grouped USC splitting requires the verified USC_2 numeric filenames.")
                groups[(int(stem) - 1) // 4].append(filename)
            for group, filenames in groups.items():
                expected = set(range(4 * group + 1, 4 * group + 5))
                if len(filenames) != 4 or {int(Path(f).stem) for f in filenames} != expected:
                    raise ValueError(f"Incomplete rotation group {group}; expected all four copies.")
            group_ids = np.array(sorted(groups), dtype=np.int64)
            rng.shuffle(group_ids)
            boundaries = {"train": (0, 3750), "val": (3750, 4250), "test": (4250, 4750)}
            split_names = {
                name: [filename for group in group_ids[start:end]
                       for filename in sorted(groups[group], key=lambda f: int(Path(f).stem))]
                for name, (start, end) in boundaries.items()
            }
        self.names = split_names[phase]
        self.group_ids = sorted({(int(Path(f).stem) - 1) // 4 for f in self.names}) if all(Path(f).stem.isdigit() for f in self.names) else []


    def __len__(self):
        return len(self.names)

    @staticmethod
    def _read_gray(path):
        img = io.imread(path)
        if img.ndim == 3:
            img = img[..., 0]
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        return img

    def __getitem__(self, idx):
        fn = self.names[idx]

        p_city  = os.path.join(self.dir_city, fn)
        p_tx    = os.path.join(self.dir_tx, fn)
        p_power = os.path.join(self.dir_power, fn)

        city = self._read_gray(p_city)
        tx   = self._read_gray(p_tx)
        y    = self._read_gray(p_power)

        city = (city > 0.5).astype(np.float32)

        tx   = (tx > 0.5).astype(np.float32)

        if self.thresh > 0:
            mask = y < self.thresh
            y = y.copy()
            y[mask] = self.thresh
            y = (y - self.thresh) / (1.0 - self.thresh)

        inputs = np.stack([city, tx], axis=2).astype(np.float32)     # (H,W,2)
        target = np.expand_dims(y, axis=2).astype(np.float32)        # (H,W,1)

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)     # (2,H,W)
            target = self.transform(target).type(torch.float32)     # (1,H,W)

        return inputs, target

class RadioUNet_c(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="data/RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.2,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
            
        """

        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)

            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=500
        elif phase=="val":
            self.ind1=501
            self.ind2=600
        elif phase=="test":
            self.ind1=601
            self.ind2=699
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code

        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 

        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1

        name1 = str(dataset_map_ind) + ".png"

        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"

        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   

        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))

        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/255
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 

            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256

        if self.thresh>0:
            mask = image_gain < self.thresh
            image_gain[mask]=self.thresh
            image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
            image_gain=image_gain/(1-self.thresh)

        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx], axis=2)        

        else: #cars

            image_buildings=image_buildings/256
            image_Tx=image_Tx/256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)

        return [inputs, image_gain]


def get_dataset(name, phase, data_dir=None, usc_split="legacy"):
    """Select a supplied loader without changing its split or preprocessing."""
    if name not in ("radiomapseer", "usc"):
        raise ValueError(f"Unknown dataset: {name}")
    if data_dir is None:
        data_dir = "data/USC" if name == "usc" else "data/RadioMapSeer"
    directory = os.fspath(data_dir).rstrip("/\\") + "/"
    if name == "usc":
        return RualDataset(phase=phase, dir_dataset=directory, split_mode=usc_split)
    return RadioUNet_c(phase=phase, dir_dataset=directory)


def split_metadata(dataset):
    """Record split identity so a checkpoint's training protocol stays explicit."""
    if isinstance(dataset, RualDataset):
        identity = dataset.names
        protocol = "usc_" + dataset.split_mode
        groups = dataset.group_ids
    else:
        groups = [int(i) + 1 for i in dataset.maps_inds[dataset.ind1:dataset.ind2 + 1]]
        identity = [f"{i}:{dataset.numTx}" for i in groups]
        protocol = "radiomapseer_map"
    digest = hashlib.sha256("\n".join(identity).encode("utf-8")).hexdigest()
    return {"protocol": protocol, "seed": getattr(dataset, "seed", 42), "samples": len(dataset), "original_groups": len(groups), "membership_sha256": digest}


def save_run_metadata(checkpoint, arguments, datasets):
    record = {"arguments": vars(arguments), "splits": {name: split_metadata(ds) for name, ds in datasets.items()}}
    Path(checkpoint).with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
