import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.SeqUtils import seq1
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==========================
# CONFIG
# ==========================
PDB_FOLDER = "/mnt/d/WSL_projects/protein_project/pdbs"  # Change path if needed
CHAIN_ID = "A"
MIN_LEN = 30
MAX_LEN = 400
processed_files = 0
skipped_files = 0

# DSSP 8-state -> Q3 mapping
DSSP_TO_Q3 = {
    "H": "H", "G": "H", "I": "H",  # Helix-like
    "E": "E", "B": "E",            # Beta-like
    "T": "C", "S": "C", "-": "C"   # Coil-like
}


# ==========================
# Functions for processing data
# ==========================
def clean_sequence(seq: str | None) -> str | None:
    if seq is None or len(seq) < MIN_LEN:
        return None
    return seq[:MAX_LEN]


def extract_chain_residues(structure, chain_id="A"):
    model = next(structure.get_models())
    if chain_id not in model:
        return []
    chain = model[chain_id]
    residues = [res for res in chain if res.id[0] == " "]  # Standard residues only
    return residues


def extract_sequence_from_residues(residues):
    return "".join(seq1(r.resname, custom_map={"MSE": "M"}) for r in residues)


def extract_dssp_q3_labels(pdb_path: str, chain_id="A"):
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)

        residues = extract_chain_residues(structure, chain_id=chain_id)
        if not residues:
            return None, None

        seq = extract_sequence_from_residues(residues)
        seq = clean_sequence(seq)
        if seq is None:
            return None, None

        model = next(structure.get_models())
        dssp = DSSP(model, pdb_path)

        labels = []
        for r in residues[:len(seq)]:
            key = (chain_id, r.id)
            if key not in dssp:
                labels.append("C")
                continue

            dssp_8 = dssp[key][2]
            labels.append(DSSP_TO_Q3.get(dssp_8, "C"))

        if len(labels) != len(seq):
            return None, None

        return seq, labels

    except Exception:
        return None, None


# ==========================
# Sliding Window Encoding
# ==========================

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD = "X"
AA_LIST = AMINO_ACIDS + PAD
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

WINDOW_SIZE = 7
HALF_W = WINDOW_SIZE // 2


def one_hot_aa(aa):
    vec = [0] * len(AA_LIST)
    vec[AA_TO_IDX.get(aa, AA_TO_IDX[PAD])] = 1
    return vec


def sliding_window_encode(seq, labels):
    """
    Returns:
        X_windows: list of feature vectors
        y_windows: list of labels (center residue)
    """
    padded_seq = PAD * HALF_W + seq + PAD * HALF_W

    X_windows = []
    y_windows = []

    for i in range(len(seq)):
        window = padded_seq[i : i + WINDOW_SIZE]

        window_vec = []
        for aa in window:
            window_vec.extend(one_hot_aa(aa))

        X_windows.append(window_vec)
        y_windows.append(labels[i])

    return X_windows, y_windows


def convert_labels_to_numeric(labels):
    label_map = {'H': 0, 'E': 1, 'C': 2}
    return [label_map[label] for label in labels]


# ==========================
# Prepare data for machine learning
# ==========================
def prepare_data():
    global processed_files, skipped_files

    X_all = []
    y_all = []

    pdb_files = os.listdir(PDB_FOLDER)

    for fname in pdb_files:
        if not fname.endswith(".pdb"):
            continue

        pdb_path = os.path.join(PDB_FOLDER, fname)
        seq, labels = extract_dssp_q3_labels(pdb_path, chain_id=CHAIN_ID)

        if seq is None:
            skipped_files += 1
            continue

        processed_files += 1

        numeric_labels = convert_labels_to_numeric(labels)
        X_win, y_win = sliding_window_encode(seq, numeric_labels)

        X_all.extend(X_win)
        y_all.extend(y_win)

    if len(X_all) == 0:
        print("No valid data found.")
        return None, None

    print(f"✔ Successfully processed PDB files: {processed_files}")
    print(f"✔ Skipped PDB files: {skipped_files}")
    print(f"✔ Total windows: {len(y_all)}")
    print(f"✔ Feature vector size: {len(X_all[0])}")

    return np.array(X_all), np.array(y_all)


# ==========================
# Train and Save Machine Learning Model
# ==========================
import joblib

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = SVC(
        kernel="rbf",
        gamma="scale",
        C=1.0,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "svm_q3_model.pkl")
    print("✔ Trained model saved as svm_q3_model.pkl")


    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"SVM Accuracy (Sliding Window): {acc * 100:.2f}%")

    return model

# ==========================
# Main Execution
# ==========================
def main():
    # Prepare the data
    X, y = prepare_data()
    if X is None or y is None:
        return

    # Train the model
    train_model(X, y)


if __name__ == "__main__":
    main()