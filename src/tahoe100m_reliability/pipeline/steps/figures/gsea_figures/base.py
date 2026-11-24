"""
Base and configs for for GSEA figure generation.
"""
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt


HALLMARK_GROUPINGS = {
    "Apical Junction":"Cellular Component",
    "Apical Surface":"Cellular Component",
    "Peroxisome":"Cellular Component",

    "Adipogenesis":"Development",
    "Angiogenesis":"Development",
    "Epithelial Mesenchymal Transition":"Development",
    "Myogenesis":"Development",
    "Pancreas Beta Cells":"Development",
    "Spermatogenesis":"Development",

    "DNA Repair":"DNA Damage",
    "UV Response Dn":"DNA Damage",
    "UV Response Up":"DNA Damage",

    "Allograft Rejection":"Immune",
    "Coagulation":"Immune",
    "Complement":"Immune",
    "Inflammatory Response":"Immune",
    "Interferon Alpha Response":"Immune",
    "Interferon Gamma Response":"Immune",

    "Bile Acid Metabolism":"Metabolic",
    "Cholesterol Homeostasis":"Metabolic",
    "Fatty Acid Metabolism":"Metabolic",
    "Glycolysis":"Metabolic",
    "heme Metabolism":"Metabolic",
    "Oxidative Phosphorylation":"Metabolic",
    "Reactive Oxygen Species Pathway":"Metabolic",
    "Xenobiotic Metabolism":"Metabolic",

    "Androgen Response":"Pathway (Signaling)",
    "Estrogen Response Early":"Pathway (Signaling)",
    "Estrogen Response Late":"Pathway (Signaling)",
    "Hedgehog Signaling":"Pathway (Signaling)",
    "IL-2/STAT5 Signaling":"Pathway (Signaling)",
    "IL-6/JAK/STAT3 Signaling":"Pathway (Signaling)",
    "KRAS Signaling Dn":"Pathway (Signaling)",
    "KRAS Signaling Up":"Pathway (Signaling)",
    "mTORC1 Signaling":"Pathway (Signaling)",
    "Notch Signaling":"Pathway (Signaling)",
    "PI3K/AKT/mTOR  Signaling":"Pathway (Signaling)",
    "Protein Secretion":"Pathway (Signaling)",
    "TGF-beta Signaling":"Pathway (Signaling)",
    "TNF-alpha Signaling via NF-kB":"Pathway (Signaling)",
    "Wnt-beta Catenin Signaling":"Pathway (Signaling)",

    "Apoptosis":"Proliferation (Cell Cycle)",
    "E2F Targets":"Proliferation (Cell Cycle)",
    "G2-M Checkpoint":"Proliferation (Cell Cycle)",
    "Mitotic Spindle":"Proliferation (Cell Cycle)",
    "Myc Targets V1":"Proliferation (Cell Cycle)",
    "Myc Targets V2":"Proliferation (Cell Cycle)",
    "p53 Pathway":"Proliferation (Cell Cycle)",

    "Hypoxia":"Stress",
    "Unfolded Protein Response":"Stress"
}

# cell lines flagged by Tahoe-100M to be of low quality
FLAGGED_CELL_LINES = ['CVCL_1531', 'CVCL_1517', 'CVCL_1571']


class GSEAHeatmaps:
    """Base class for GSEA heatmaps."""

    def __init__(
        self,
        output_dir: Path,
        organ_by_cell_line: dict[str, Any],
        fdr_threshold: float = 0.05,
        non_significant_alpha: float = 0.40,
        cmap='coolwarm'
    ) -> None:
        """Initializer."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.organ_by_cell_line = organ_by_cell_line

        self.cmap = plt.get_cmap(cmap)
        self.fdr_threshold = 0.05
        self.non_significant_alpha = 0.40
