import os
import pandas as pd
import numpy as np
import warnings
from CACAE import Model_new, Process
from CACAE.utils_new import ClusterProcessor, do_km_plot
from CACAE.Survive_select_new import survive_select

warnings.filterwarnings("ignore")

# 1. Setup Project Constants
cancer_name = "SARC" 

# Create necessary directories - Updated to include result_new and features_new
for folder in ['sorted_data', 'result_new', 'features_new']:
    os.makedirs(folder, exist_ok=True)

# 2. Define File Paths
path_mirna = f"data/{cancer_name}_miRNA.tsv"
path_mrna = f"data/{cancer_name}_mRNA.tsv"
path_meth = f"data/{cancer_name}_METH.tsv"
path_sur = f"data/{cancer_name}_sur.tsv"

# 3. Load and Transpose Data
print("Loading and transposing data...")
miRNA = pd.read_table(path_mirna, sep='\t', index_col=0).T
RNA = pd.read_table(path_mrna, sep='\t', index_col=0).T
Meth = pd.read_table(path_meth, sep='\t', index_col=0).T
survive = pd.read_table(path_sur, sep='\t', index_col=0)

# 4. Align Samples
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

print(f"Post-cleaning: {len(common_samples)} samples remain.")

# 5. Data Preprocessing
print("Processing data layers...")
RNA_processor = Process.DataProcessor(RNA)
RNA_sorted = RNA_processor.sort_corr(5000)
RNA_sorted.to_csv(os.path.join('sorted_data', f'{cancer_name}_sorted_mRNA.csv'), index=True)

miRNA_processor = Process.DataProcessor(miRNA)
miRNA_sorted = miRNA_processor.sort_corr(100)
miRNA_sorted.to_csv(os.path.join('sorted_data', f'{cancer_name}_sorted_miRNA.csv'), index=True)

Meth_processor = Process.DataProcessor(Meth)
Meth_sorted = Meth_processor.sort_corr(2000)
Meth_sorted.to_csv(os.path.join('sorted_data', f'{cancer_name}_sorted_Meth.csv'), index=True)

# 6. Initialize and Train Standard AE Models (Using Model_new)
print("Training Standard Autoencoders...")
# Note: Using Model_new.StandardAE as imported in line 6
RNA_model = Model_new.StandardAE(RNA_sorted.shape[1])
miRNA_model = Model_new.StandardAE(miRNA_sorted.shape[1])
Meth_model = Model_new.StandardAE(Meth_sorted.shape[1])

RNA_model.fit(RNA_sorted)
miRNA_model.fit(miRNA_sorted)
Meth_model.fit(Meth_sorted)

# 7. Extract Hidden Features
print("Extracting features from the Hidden Layer...")
RNA_feat = pd.DataFrame(RNA_model.extract_feature(RNA_sorted), index=common_samples)
miRNA_feat = pd.DataFrame(miRNA_model.extract_feature(miRNA_sorted), index=common_samples)
Meth_feat = pd.DataFrame(Meth_model.extract_feature(Meth_sorted), index=common_samples)

flatten = pd.concat([RNA_feat, miRNA_feat, Meth_feat], axis=1)

# 8. Survival-Based Feature Selection (RSF - Using Survive_select_new)
print("Selecting survival-related features using RSF...")
SURVIVE_SELECT = survive_select(survive, flatten, top_n=50)
# Updated save path to features_new
SURVIVE_SELECT.to_csv(os.path.join('features_new', f'{cancer_name}_features.csv'))

# 9. Clustering and Evaluation
print("Performing K-Means clustering...")
cp = ClusterProcessor(SURVIVE_SELECT, survive)
cp.compute_indexes(3) # Change as needed for each cancer type
p_value, clusters = cp.LogRankp(3) # Adjust to match compute_indexes

# 10. Generate Final Visualizations
do_km_plot(survive, pvalue=p_value.p_value, cindex=None, cancer_type=cancer_name, model_name='StandardAE-RSF', output_dir='result_new')


print(f"Analysis for {cancer_name} complete. Results saved in 'result_new/' folder.")