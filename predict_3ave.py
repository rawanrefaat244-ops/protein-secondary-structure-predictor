import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.SeqUtils import seq1

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "svm_q3_model.pkl"
PDB_PATH = "3ave.pdb"
CHAIN_ID = "A"


# To match training settings exactly
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD = "X"
AA_LIST = AMINO_ACIDS + PAD
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

WINDOW_SIZE = 7
HALF_W = WINDOW_SIZE // 2

DSSP_TO_Q3 = {
    "H": "H", "G": "H", "I": "H",
    "E": "E", "B": "E",
    "T": "C", "S": "C", "-": "C"
}

INV_LABEL_MAP = {0: "H", 1: "E", 2: "C"}  # numeric -> letter


# --------------------------
# Load trained model
# --------------------------
def load_model(model_path=MODEL_PATH):
    model = joblib.load(model_path)
    return model


# --------------------------
# Extract 3AVE sequence (primary structure) from PDB
# --------------------------
def extract_sequence_from_pdb(pdb_path=PDB_PATH, chain_id=CHAIN_ID):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("3AVE", pdb_path)
    model = structure[0]

    if chain_id not in model:
        raise ValueError(f"Chain {chain_id} not found in {pdb_path}")

    chain = model[chain_id]
    residues = [r for r in chain if r.id[0] == " "]
    seq = "".join(seq1(r.resname, custom_map={"MSE": "M"}) for r in residues)
    return seq


# --------------------------
# Sliding-window encoding (same as training)
# --------------------------
def one_hot_aa(aa):
    vec = [0] * len(AA_LIST)
    vec[AA_TO_IDX.get(aa, AA_TO_IDX[PAD])] = 1
    return vec


def sliding_window_encode_sequence(seq):
    padded = PAD * HALF_W + seq + PAD * HALF_W
    X = []

    for i in range(len(seq)):
        window = padded[i:i + WINDOW_SIZE]
        window_vec = []
        for aa in window:
            window_vec.extend(one_hot_aa(aa))
        X.append(window_vec)

    return np.array(X)


# --------------------------
# Predict Q3 labels for 3AVE
# --------------------------
def predict_q3(model, seq):
    X = sliding_window_encode_sequence(seq)
    y_pred = model.predict(X)  # returns 0/1/2
    pred_labels = "".join(INV_LABEL_MAP[int(v)] for v in y_pred)
    return pred_labels


# --------------------------
# Get DSSP true Q3 labels for evaluation on 3AVE
# --------------------------
def extract_true_dssp_q3(pdb_path=PDB_PATH, chain_id=CHAIN_ID):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("3AVE", pdb_path)
    model = structure[0]

    dssp = DSSP(model, pdb_path)
    labels = []

    # DSSP keys are (chain_id, residue_id)
    for key in dssp.keys():
        if key[0] != chain_id:
            continue
        ss8 = dssp[key][2]
        labels.append(DSSP_TO_Q3.get(ss8, "C"))

    return "".join(labels)


def q3_score(true_labels, pred_labels):
    if len(true_labels) != len(pred_labels):
        raise ValueError("True and predicted lengths do not match.")

    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    return correct / len(true_labels)

def clean_jpred_prediction(jpred_raw):

    return jpred_raw.replace("-", "C")


# --------------------------
# MAIN
# --------------------------
def main():
    # Load model
    model = load_model(MODEL_PATH)

    # Extract sequence
    seq = extract_sequence_from_pdb(PDB_PATH, CHAIN_ID)
    print("\nPrimary sequence of 3AVE:")
    print(seq)
    print(f"\nSequence length: {len(seq)}")

    # Ground Truth (DSSP)
    true = extract_true_dssp_q3(PDB_PATH, CHAIN_ID)
    print("\nTrue DSSP Q3 (H/E/C):")
    print(true)

    # SVM Prediction
    svm_pred = predict_q3(model, seq)
    print("\nSVM Predicted Q3 (H/E/C):")
    print(svm_pred)

    svm_score = q3_score(true, svm_pred)
    print(f"\n✔ Q3 score of trained SVM on 3AVE: {svm_score * 100:.2f}%")

    # JPred prediction
    jpred_raw = "-----EEEEE-----EEEE-----EEEEEEEEE------EEEEEE----E-------EEE------EEEEEEEEE---------EEEEEEEE--------EEEEEE-------EEEEE-----------EEEEEEEE------EEEEEE----EEE------E------EEEEEEEEEE---------EEEEEEEE------EEEEEE---"
    jpred_pred = clean_jpred_prediction(jpred_raw)
    print("\nJPred Predicted Q3 (H/E/C):")
    print(jpred_pred)

    jpred_score = q3_score(true, jpred_pred)
    print(f"\n✔ Q3 score of JPred on 3AVE: {jpred_score * 100:.2f}%")

if __name__ == "__main__":
    main()
