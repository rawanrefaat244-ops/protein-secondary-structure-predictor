# Secondary Structure Predictor for 3AVE

This project predicts the secondary structure of the human IgG1 Fc fragment protein structure **3AVE** using a custom Support Vector Machine (SVM) model.

The model is trained on external PDB structures and then evaluated on the test protein **3AVE** by comparing the predicted secondary structure labels against DSSP-derived ground truth labels.

## Project Objective

The goal of this project is to predict the three-state secondary structure of the 3AVE protein chain:

- `H` = Helix
- `E` = Beta strand
- `C` = Coil

The final performance is measured using the **Q3 score**, which represents the percentage of residues whose predicted secondary structure matches the true DSSP-assigned secondary structure.

## Protein Context

**PDB ID:** 3AVE  
**Protein:** Human IgG1 Fc fragment  
**Chain used:** A  

The 3AVE structure represents an immunoglobulin-like Fc fragment, which is expected to contain a high proportion of beta-strand secondary structure due to its beta-sandwich fold.

## Project Workflow

The project has two main scripts:

### 1. `trained_svm.py`

This script trains an SVM model for protein secondary structure prediction.

It performs the following steps:

1. Loads external PDB structures from the training folder.
2. Extracts amino acid sequences and DSSP secondary structure labels.
3. Converts DSSP 8-state labels into three Q3 classes (H, E, C).
4. Splits complete proteins into training (80%) and testing (20%) sets.
5. Independently generates sliding windows for each set.
6. Applies one-hot encoding to produce 147-dimensional feature vectors.
7. Trains an RBF-SVM classifier and evaluates its performance.
8. Saves the trained model as `svm_q3_model.pkl`.

**Important:** The split is performed at the protein level before sliding-window generation to prevent overlapping windows from the same protein appearing in both training and testing sets.

### 2. `predict_3ave.py`

This script uses the trained SVM model to predict the secondary structure of 3AVE.

It performs the following steps:

1. Loads the trained model file `svm_q3_model.pkl`.
2. Extracts the primary amino acid sequence from `3ave.pdb`.
3. Encodes the sequence using the same sliding-window method used during training.
4. Predicts Q3 labels for each residue.
5. Extracts the true DSSP Q3 labels from `3ave.pdb`.
6. Calculates the Q3 score for the trained SVM model.
7. Compares the result against a JPred baseline prediction.

## Repository Structure

```text
secondary-structure-predictor/
│
├── trained_svm.py
├── predict_3ave.py
├── 3ave.fasta
├── 3ave.pdb
├── requirements.txt
├── .gitignore
└── README.md
```

## Important Note About the Trained Model

The trained model file:

```text
svm_q3_model.pkl
```

is required to run `predict_3ave.py`, but it is not included in this repository because of its file size.

To generate it locally, run:

```bash
python trained_svm.py
```

After training, the model will be saved automatically as:

```text
svm_q3_model.pkl
```

Then you can run the prediction script.

## Training Data Note

The training script expects a local folder of PDB files.

Inside `trained_svm.py`, the training folder is defined as:

```python
PDB_FOLDER = "/home/lenovo555/protein_project/pdbs"
```

If you run this project on another machine, update this path to the folder that contains your training PDB files.

Example:

```python
PDB_FOLDER = "/path/to/your/pdb/files"
```

The script processes chain `A` from each PDB file and skips files that cannot be processed by DSSP or do not meet the sequence length criteria.

## Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The main Python dependencies are:

```text
biopython
joblib
numpy
scikit-learn
```

This project also requires **DSSP** to be installed and accessible from the system, because Biopython uses DSSP to assign secondary structure labels from PDB files.

## How to Run the Project

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Train the SVM model

Before running prediction, train the model:

```bash
python trained_svm.py
```

Example training output:

```text
✔ Successfully processed PDB files: 83
✔ Skipped PDB files: 17
✔ Total proteins: 83
Train proteins: 66
Test proteins: 17
Train windows: 13599
Test windows: 3549
✔ Trained model saved as svm_q3_model.pkl
SVM Accuracy: 84.93%
```

### 4. Predict secondary structure for 3AVE

After generating `svm_q3_model.pkl`, run:

```bash
python predict_3ave.py
```

Example prediction output:

```text
Primary sequence of 3AVE:
LLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSRDELTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLS

Sequence length: 211

✔ Q3 score of trained SVM on 3AVE: 72.51%

✔ Q3 score of JPred on 3AVE: 71.09%
```

## Model Method

The model uses a **sliding-window encoding** approach.

For each residue, the model looks at a window of neighboring amino acids:

```text
Window size = 7
```

Each amino acid is represented using one-hot encoding. The model then predicts the secondary structure label of the center residue.

The final feature vector size is:

```text
147
```

This comes from:

```text
7 positions × 21 amino acid symbols
```

where the 21 symbols include the 20 standard amino acids plus the padding symbol `X`.

## DSSP to Q3 Mapping

DSSP originally provides 8-state secondary structure labels. In this project, they are converted into 3-state Q3 labels:

| DSSP Label | Q3 Label | Meaning |
|---|---|---|
| `H`, `G`, `I` | `H` | Helix-like |
| `E`, `B` | `E` | Beta-like |
| `T`, `S`, `-` | `C` | Coil-like |

## Results

### Internal Evaluation

Two splitting strategies were evaluated while keeping the SVM hyperparameters unchanged.

| Evaluation Method | Q3 Accuracy |
|---|---:|
| Original random window-level split | 81.49% |
| Updated protein-level split | 84.93% |

The original pipeline generated overlapping sliding windows before randomly splitting the resulting samples into training and testing sets.

This allowed highly overlapping windows from the same protein to appear in both sets, potentially producing an overly optimistic evaluation.

To address this issue, the pipeline was updated to split complete protein sequences before applying sliding-window encoding.

The updated model was trained on 66 proteins and evaluated on 17 held-out proteins.

**Note:** The internal test sets differ between the two experiments. Therefore, the increase from 81.49% to 84.93% should not be interpreted as a direct improvement in generalization performance.

### External Evaluation on 3AVE

| Method | Q3 Accuracy |
|---|---:|
| Original SVM | 72.51% |
| Updated SVM | 72.51% |
| JPred baseline | 71.09% |

Both SVM versions correctly predicted 153 out of 211 residues in 3AVE.

Although the original and updated models produced different residue-level predictions, their overall Q3 accuracy remained unchanged.

The updated SVM achieved a slightly higher Q3 score than the JPred baseline on this test protein.

### Key Finding

The primary improvement was methodological rather than predictive.

Splitting at the protein level prevents windows from the same protein from appearing in both training and testing sets, providing a more appropriate evaluation of performance on held-out proteins.

However, sequence similarity between different proteins was not explicitly controlled.

## Files Required to Run Prediction

To run `predict_3ave.py`, the following files must be present in the project root directory:

```text
predict_3ave.py
3ave.pdb
svm_q3_model.pkl
```

`3ave.pdb` is included in the repository.  
`svm_q3_model.pkl` must be generated locally by running `trained_svm.py`.

## Limitations

This is an educational machine learning project for protein secondary structure prediction.

Current limitations include:

- The model uses a simple sliding-window encoding.
- It does not use evolutionary profiles such as PSSM or HMM features.
- It does not use deep learning or protein language model embeddings.
- The model performance depends heavily on the quality and diversity of the training PDB dataset.
- The current evaluation focuses on one test structure: 3AVE.
- The training dataset path is currently configured locally and should be updated before running on another machine.
- Although the protein-level split prevents overlapping windows from the same protein appearing in both sets, homologous or highly similar proteins may still be distributed across training and testing sets.
- The model uses a fixed local window of seven amino acids, limiting its ability to capture long-range sequence dependencies.

## Future Improvements

Possible improvements include:

- Use a larger and more diverse training dataset.
- Add PSSM-based evolutionary features.
- Add cross-validation across PDB structures.
- Evaluate on multiple independent test proteins.
- Try Random Forest, Gradient Boosting, or neural network models.
- Compare against additional public secondary structure prediction tools.
- Organize input/output folders for larger-scale experiments.

## Technologies Used

- Python
- Biopython
- DSSP
- NumPy
- Scikit-learn
- Joblib
- Support Vector Machine (SVM)
