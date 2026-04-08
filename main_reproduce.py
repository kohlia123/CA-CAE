import os
import pandas as pd
import numpy as np
import warnings
from CACAE import Model, Process
from CACAE.utils import ClusterProcessor, do_km_plot
from CACAE.Survive_select import survive_select

warnings.filterwarnings("ignore")

# 1. Setup Project Constants
cancer_name = "LGG"

# Create necessary directories
for folder in ['sorted_data', 'features', 'result']:
    os.makedirs(folder, exist_ok=True)

# 2. Define File Paths
path_mirna = f"data/{cancer_name}_miRNA.tsv"
path_mrna = f"data/{cancer_name}_mRNA.tsv"
path_meth = f"data/{cancer_name}_METH.tsv"
path_sur = f"data/{cancer_name}_sur.tsv"

# 3. Load and Transpose Data
# index_col=0 ensures 'sample' or 'miRNA_ID' becomes the index
# .T is CRITICAL because GDC files have genes as rows, but we need them as columns
print("Loading and transposing data...")
miRNA = pd.read_table(path_mirna, sep='\t', index_col=0).T
RNA = pd.read_table(path_mrna, sep='\t', index_col=0).T
Meth = pd.read_table(path_meth, sep='\t', index_col=0).T
survive = pd.read_table(path_sur, sep='\t', index_col=0)

# 4. Align Samples across all 4 files
# This ensures we only use patients who have all omics types + survival data
common_samples = RNA.index.intersection(miRNA.index).intersection(Meth.index).intersection(survive.index)

RNA = RNA.loc[common_samples].fillna(0)
miRNA = miRNA.loc[common_samples].fillna(0)
Meth = Meth.loc[common_samples].fillna(0)
survive = survive.loc[common_samples]
survive = survive[['OS.time', 'OS']].astype(float)

print(f"Alignment complete. Analyzing {len(common_samples)} matching patient samples.")

survive = survive.dropna(subset=['OS.time', 'OS'])

common_samples = survive.index.intersection(RNA.index)
RNA = RNA.loc[common_samples]
miRNA = miRNA.loc[common_samples]
Meth = Meth.loc[common_samples]
survive = survive.loc[common_samples]

print(f"Post-cleaning: {len(common_samples)} samples remain with valid survival data.")

# 5. Data Preprocessing (Standard Deviation & Correlation Sorting)
print("Processing RNA data...")
RNA_processor = Process.DataProcessor(RNA)
RNA_sorted = RNA_processor.sort_corr(5000) # Must be multiple of 100
RNA_sorted.to_csv(os.path.join('sorted_data', f'{cancer_name}_sorted_mRNA.csv'), index=True)

print("Processing miRNA data...")
miRNA_processor = Process.DataProcessor(miRNA)
miRNA_sorted = miRNA_processor.sort_corr(100)
miRNA_sorted.to_csv(os.path.join('sorted_data', f'{cancer_name}_sorted_miRNA.csv'), index=True)

print("Processing Methylation data...")
Meth_processor = Process.DataProcessor(Meth)
Meth_sorted = Meth_processor.sort_corr(2000)
Meth_sorted.to_csv(os.path.join('sorted_data', f'{cancer_name}_sorted_Meth.csv'), index=True)

# 6. Initialize and Train CA-CAE Models
print("Training Autoencoders (this may take a few minutes)...")
RNA_model = Model.CACAE(RNA_sorted.shape[1])
miRNA_model = Model.CACAE(miRNA_sorted.shape[1])
Meth_model = Model.CACAE(Meth_sorted.shape[1])

RNA_model.fit(RNA_sorted)
miRNA_model.fit(miRNA_sorted)
Meth_model.fit(Meth_sorted)

# 7. Extract Hidden Features
print("Extracting features from the Hidden Layer...")
RNA_feat = pd.DataFrame(RNA_model.extract_feature(RNA_sorted), index=common_samples)
miRNA_feat = pd.DataFrame(miRNA_model.extract_feature(miRNA_sorted), index=common_samples)
Meth_feat = pd.DataFrame(Meth_model.extract_feature(Meth_sorted), index=common_samples)

# Combine all extracted features into one matrix
flatten = pd.concat([RNA_feat, miRNA_feat, Meth_feat], axis=1)

# 8. Survival-Based Feature Selection (Lasso-Cox)
print("Selecting survival-related features...")
# Using p-value threshold of 0.05
SURVIVE_SELECT = survive_select(survive, flatten, 0.05)
SURVIVE_SELECT.to_csv(os.path.join('features', f'{cancer_name}_features.csv'))

# 9. Clustering and Evaluation
print("Performing K-Means clustering and generating KM-Plot...")
cp = ClusterProcessor(SURVIVE_SELECT, survive)
cp.compute_indexes(3) # Change as needed for each cancer type
p_value, clusters = cp.LogRankp(3)

# 10. Generate Final Visualizations
do_km_plot(survive, pvalue=p_value.p_value, cindex=None, cancer_type=cancer_name, model_name='CA-CAE')

print(f"Analysis for {cancer_name} complete. Results saved in 'result/' folder.")