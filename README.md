**Data Availability**
All multi-omics data is sourced from the UCSC Xena Datahub. You can find direct links to the cohorts used in this study at Figshare (DOI: 10.6084/m9.figshare.30566096).

For your chosen cancer type (e.g., LGG), download the mRNA, miRNA, Methylation, and Survival datasets. Save them in a folder named /data using the following exact naming convention:
- LGG_mRNA.tsv
- LGG_miRNA.tsv
- LGG_METH.tsv
- LGG_sur.tsv

**How to Run**
1. Environment Setup: Ensure you have Python 3.9+ installed

   Install dependencies with the following command
   ```bash
   pip install -r requirements.txt
   
3. Configuration: Open main_reproduce.py and update the cancer_name variable (Line 12) to match your data prefix     (e.g., cancer_name = "LGG").

4. Execution: Run the scripts with the following commands:

   ```bash
   # Reproduce the original CA-CAE pipeline
   python main_reproduce.py

   # Run the extended model (Standard Autoencoder + RSF feature selection)
   python main_new.py
   
5. Outputs
   - Significant features from the Lasso-Cox setup are saved in /features for the reproduced model, and /features_new for the extension model.
   - Visualizations: Kaplan-Meier survival plots (.pdf) and cluster assignments (.csv) are saved in /result for the reproduced result, and    /result_new for the extension result.
