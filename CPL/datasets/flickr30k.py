import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class flickr30k(DatasetBase):

    dataset_dir = "flickr30k"
    
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        num_shots = cfg.DATASET.NUM_SHOTS
        txt_file_suffix = f"{num_shots}shots"

        classnames = []
        with open(
            os.path.join(
                self.dataset_dir, f"classnames-{txt_file_suffix}.txt"
            ),
            "r",
        ) as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}

        train = self.read_data(cname2lab, f"train-{txt_file_suffix}.txt")
        val = self.read_data(cname2lab, f"val-{txt_file_suffix}.txt")
        test = self.read_data(cname2lab, f"test-{txt_file_suffix}.txt")

        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots) # 1 shot always 
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        with open(
            os.path.join(self.dataset_dir, f"test-{txt_file_suffix}.txt"), "r"
        ) as f:
            lines = f.readlines()
            train_shot = (len(lines)) - 1000 
            
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, train_shot, subsample=subsample)  # subsample = fewshot for training; = unseen for testing

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split("*")
                imname = line[0]
                classname = " ".join(line[1:])
                impath = os.path.join(self.image_dir, imname)

                label = cname2lab[classname]
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items