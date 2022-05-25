from genericpath import exists
import torch
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.append('../../../GenderBias/src/')
from DataUtils import balance_dataset

from tqdm import tqdm
import os
from pathlib import Path
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_probing_data(data_file, balanced=None):
    data = torch.load(data_file)
    # cat = data["categories"]
    X, y = data["X"], data["y"]

    z = data["z"]
    original_z = data["z"]

    if balanced in ("oversampled", "subsampled"):
        X, y, z = balance_dataset(X, y, z, oversampling=True if balanced=="oversampled" else False)

    # y = torch.tensor(y).long()
    X = torch.tensor(X)

    dataset = TensorDataset(X)
    # original_y = data[split]["original_y"]
    # n_labels = len(np.unique(original_y))

    def preprocess_probing_data(X, z):
        z[z == 'F'] = 1
        z[z == 'M'] = 0
        z = z.astype(int)
        X, z = X, torch.tensor(z).long()

        return TensorDataset(X, z)
    
    return preprocess_probing_data(dataset.tensors[0], z)

def get_general_set(data_file):
    return torch.load(data_file)

def get_data(data_file, type_, balanced=None):
    if type_ == "probing":
        return get_probing_data(data_file, balanced)
    else:
        return get_general_set(data_file)

def get_vectors(model, data, save_file, all_layers=False):
    if os.path.exists(save_file):
        print("Exists.")
        vectors = torch.load(save_file)
    else:
        vectors = []
        labels = []
        genders = []
        
        X = torch.tensor(data['X']).to(device)
        y = data['y']
        z = data['z']
        masks = torch.tensor(data['masks']).to(device)
        model = model.to(device)
        # _, counts = np.unique(z, return_counts=True)
        # print('Gender ratios:', [x/sum(counts) for x in counts])

        with torch.no_grad():
            model.eval()

            for i, x in enumerate(tqdm(X)):
                input_ids = x.to(device)
                mask = masks[i].to(device)
                with torch.no_grad():
                    if not all_layers:
                        v = model(input_ids.unsqueeze(0), attention_mask=mask.unsqueeze(0)).last_hidden_state[:, 0, :][0].cpu().detach().numpy()
                    else:
                        v = model(input_ids.unsqueeze(0), masks[i].unsqueeze(0),
                                output_hidden_states=True, return_dict=True)
                        v = torch.cat([h[:, 0].cpu().detach().unsqueeze(1) \
                                    for h in v.hidden_states[1:]], 0).numpy()
                
                vectors.append(v)
                labels.append(y[i])
                genders.append(z[i])

            vectors = np.array(vectors)
            labels = np.array(labels)
            genders = np.array(genders)
        
        vectors = {"X": vectors, "y": labels, "z": genders}
        os.makedirs('/'.join(save_file.split('/')[:-1]), exist_ok=True)
        torch.save(vectors, save_file)
    
    return vectors

def get_text_and_labels(seed, datatype, num_samples=100000, gender_balanced=True):

    if os.path.exists(f"data/biosbias/BIOS_train_{datatype}_seed={seed}.pkl") and \
    os.path.exists(f"data/biosbias/BIOS_valid_{datatype}_seed={seed}.pkl") and \
    os.path.exists(f"data/biosbias/BIOS_test_{datatype}_seed={seed}.pkl"):
        train_dict = pickle.load(open(f"data/biosbias/BIOS_train_{datatype}_seed={seed}.pkl", "rb"))
        dev_dict = pickle.load(open(f"data/biosbias/BIOS_valid_{datatype}_seed={seed}.pkl", "rb"))
        test_dict = pickle.load(open(f"data/biosbias/BIOS_test_{datatype}_seed={seed}.pkl", "rb"))
        X_train, z_train = train_dict["X"], train_dict["z"]
        X_valid, z_valid = dev_dict["X"], dev_dict["z"]
        X_test, z_test = test_dict["X"], test_dict["z"]

        return X_train, z_train, X_valid, z_valid, X_test, z_test

    f = open('data/biosbias/BIOS.pkl', 'rb')
    ds = pickle.load(f)

    labels = []
    genders = []
    inputs = []

    for r in tqdm(ds):
        if datatype == "scrubbed":
            sent = r["bio"]  # no start_pos needed
        else:
            sent = r["raw"][r["start_pos"]:]

        inputs.append(sent)
        labels.append(r["title"])
        genders.append(r["gender"])
    
    inputs = np.array(inputs)
    labels = np.array(labels)
    genders = np.array(genders)

    if num_samples != -1:
        if gender_balanced:
            indices = []
            all_genders = np.unique(genders)
            num_samples_per_gender = num_samples//len(all_genders)
            for g in all_genders:
                ids_to_pick = np.array(range(len(inputs)))[genders==g][:num_samples_per_gender]
                indices.extend(ids_to_pick)
            inputs = inputs[indices]
            labels = labels[indices]
            genders = genders[indices]
        else:
            inputs = inputs[:num_samples]

    X_train_valid, X_test, y_train_valid, y_test, z_train_valid, z_test = train_test_split(
        inputs, labels, genders, random_state=seed, stratify=labels, test_size=0.25)

    X_train, X_valid, y_train, y_valid, z_train, z_valid = train_test_split(
        X_train_valid, y_train_valid, z_train_valid, random_state=seed, stratify=y_train_valid,
        test_size=0.133)
    
    pickle.dump({"X": X_train, "y": y_train, "z": z_train}, open(f"data/biosbias/BIOS_train_{datatype}_seed={seed}.pkl", "wb"))
    pickle.dump({"X": X_valid, "y": y_valid, "z": z_valid}, open(f"data/biosbias/BIOS_valid_{datatype}_seed={seed}.pkl", "wb"))
    pickle.dump({"X": X_test, "y": y_test, "z": z_test}, open(f"data/biosbias/BIOS_test_{datatype}_seed={seed}.pkl", "wb"))
    
    return X_train, z_train, X_valid, z_valid, X_test, z_test