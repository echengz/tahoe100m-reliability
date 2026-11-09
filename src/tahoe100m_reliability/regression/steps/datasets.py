# ============================================================
# DATASET CONFIGURATION — Wei et al. Cellular Context
# ============================================================
# Each dataset needs:
#   context_col:  obs column identifying the cellular context
#   condition_col: obs column identifying control vs stimulated
#   ctrl_label:   value in condition_col that marks control cells

CELLULAR_DATASETS = {
    # ---- Kang et al. (IFN-beta stimulation of PBMCs) ----
    "kangCrossCell": {
        "context_col": "cell_type",
        "condition_col": "condition",
        "ctrl_label": "control",
    },
    "kangCrossPatient": {
        "context_col": "sample_id",
        "condition_col": "perturbation",
        "ctrl_label": "control",
    },

    # ---- Haber et al. (intestinal epithelium, infections) ----
    "Haber": {
        "context_col": "cell_type",
        "condition_col": "condition",
        "ctrl_label": "Control",
    },

    # ---- Afriat et al. (liver infection) ----
    # Use condition1 (Pericentral, Periportal — 2 zones) to match Wei et al.'s
    # benchmark aggregation level (the unit on which they report performance).
    "Afriat": {
        "context_col": "condition1",
        "condition_col": "perturbation",
        "ctrl_label": "control",
    },

    # ---- McFarland et al. (cancer cell lines + drugs) ----
    "McFarland": {
        "context_col": "cell_line",
        "condition_col": "perturbation",
        "ctrl_label": "control",
    },

    # ---- Parekh et al. (TF overexpression) ----
    "Parekh": {
        "context_col": "cell_type",
        "condition_col": "perturbation",
        "ctrl_label": "CTRL",
    },

    # ---- TCDD (dioxin exposure, liver cell types) ----
    "TCDD": {
        "context_col": "celltype",
        "condition_col": "perturbation",
        "ctrl_label": "control",
    },

    # ---- Cross-patient (cancer patients + drugs) ----
    "crossPatient": {
        "context_col": "patient",
        "condition_col": "perturbation",
        "ctrl_label": "control",
    },

    # ---- Cross-species (LPS stimulation) ----
    # Use pre-pooled condition1 (4 species pooled across individuals) and
    # condition2 (LPS vs control, pooled across LPS time-points) to match Wei.
    "crossSpecies": {
        "context_col": "condition1",
        "condition_col": "condition2",
        "ctrl_label": "control",
    },

    # ---- Kaggle/NeurIPS Cross-Cell (drugs x cell types) ----
    "KaggleCrossCell": {
        "context_col": "cell_type",
        "condition_col": "perturbation",
        "ctrl_label": "control",
    },

    # ---- Kaggle/NeurIPS Cross-Patient (drugs x patients) ----
    "KaggleCrossPatient": {
        "context_col": "donor_id",
        "condition_col": "perturbation",
        "ctrl_label": "control",
    },

    # ---- Sciplex3 (drugs x cell lines) ----
    # Use condition2 (drug name without dose) to match Wei's evaluation unit;
    # per-dose reliability would block re-evaluation merge with their results.
    "sciplex3": {
        "context_col": "cell_line",
        "condition_col": "condition2",
        "ctrl_label": "control",
    },
}


# Common alternative names to try during auto-discovery
CONTEXT_COL_CANDIDATES = [
    "cell_type", "celltype", "cell_line", "cellLine", "cell_line_id",
    "patient", "donor", "species", "sample", "batch",
    "outSample", "outsample",
]


CONDITION_COL_CANDIDATES = [
    "condition", "perturbation", "treatment", "stim", "stimulation",
    "drug", "compound", "perturb",
]


CTRL_LABEL_CANDIDATES = [
    "control", "Control", "ctrl", "CTRL",
    "DMSO", "dmso", "Vehicle", "vehicle",
    "Uninfected", "uninfected", "untreated", "Untreated",
    "normal", "Normal", "baseline", "Baseline",
]
