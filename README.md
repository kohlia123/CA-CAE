**Data Availability**
All multi-omics data is sourced from the UCSC Xena Datahub. You can find direct links to the cohorts used in this study at Figshare (DOI: 10.6084/m9.figshare.30566096).

The data for additional cancers used in this project that are not in the original study can be found at the following links
- [ESCA](https://xenabrowser.net/datapages/?cohort=TCGA%20Esophageal%20Cancer%20(ESCA)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443)
- [KIRP](https://xenabrowser.net/datapages/?cohort=GDC%20TCGA%20Kidney%20Papillary%20Cell%20Carcinoma%20(KIRP)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443)
- [LIHC](https://xenabrowser.net/datapages/?cohort=TCGA%20Liver%20Cancer%20(LIHC)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443)

**How to Run**
1. Environment Setup: On the SCC, load Python 3.9+ module and install requirements. Before running the model on the SCC, make sure to use a node with a GPU.
   ```bash
   qrsh -P ds596 -l gpus=1 -l gpu_type=A100
   ```

   ```bash
   module load python3/3.10.5
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```


2. Download files 
For your chosen cancer type (e.g., LGG), download the mRNA, miRNA, Methylation, and Survival datasets (Make sure to download IlluminaHiseq, Methylation450k, and Curated survival data). Save them in a folder named /data using the following exact naming convention:
- LGG_mRNA.tsv 
- LGG_miRNA.tsv 
- LGG_METH.tsv
- LGG_sur.tsv
   
3. Execution: Run the script with the following command
   ```bash
   python main_reproduce.py --cancer <CANCER_TYPE_HERE>
   
4. Outputs
   - Significant features from the Lasso-Cox setup are saved in /features.
   - Visualizations: Kaplan-Meier survival plots (.pdf) and cluster assignments (.csv) are saved in /result.
