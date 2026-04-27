import pandas as pd
from sklearn.cluster import KMeans
from lifelines.statistics import multivariate_logrank_test
import os
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class ClusterProcessor:
    def __init__(self, data, sur_data):
        self.data = data
        self.sur_data = sur_data
        self.logrank_cache = {}  

    def KmeansCluster(self, nclusters):
        K_mod = KMeans(n_clusters=nclusters, random_state=42) # Added random_state for consistency
        K_mod.fit(self.data)
        return K_mod.predict(self.data)

    def LogRankp(self, nclusters):
        if nclusters in self.logrank_cache:
            return self.logrank_cache[nclusters]

        clusters = self.KmeansCluster(nclusters)
        self.sur_data['Type'] = clusters
        pvalue = multivariate_logrank_test(self.sur_data['OS.time'], self.sur_data['Type'], self.sur_data['OS'])
        self.logrank_cache[nclusters] = (pvalue, clusters)
        return pvalue, clusters

    def compute_indexes(self, maxclusters):
        for i in range(2, maxclusters+1):
            pvalue, clusters = self.LogRankp(i)
            # Use cosine metric as in your original file for multi-omics consistency
            silhouette = silhouette_score(self.data, clusters, metric='cosine')
            ch_score = calinski_harabasz_score(self.data, clusters)
            db_score = davies_bouldin_score(self.data, clusters)
            
            print(f"Number of clusters: {i}")
            print(f"Silhouette Score: {silhouette:.4f}")
            print(f"Calinski-Harabasz Index: {ch_score:.4f}")
            print(f"Davies-Bouldin Index: {db_score:.4f}")
            print(f"P-value: {pvalue.p_value:.2e}")

def do_km_plot(survive_data, pvalue, cindex, cancer_type, model_name, output_dir='result_new'):
    from lifelines import KaplanMeierFitter
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    values = np.asarray(survive_data['Type'])
    events = np.asarray(survive_data['OS'])
    times = np.asarray(survive_data['OS.time'])
    
    df = pd.DataFrame({'Type': values, 'OS': events, 'OS.time': times})

    # Save CSV to the specific output_dir
    output_filename = os.path.join(output_dir, f'{cancer_type}_clusters.csv')
    df.to_csv(output_filename, index=False)
    
    sns.set(style='ticks', context='notebook', font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    kaplan = KaplanMeierFitter()
    for label in sorted(set(values)):
        mask = (values == label)
        kaplan.fit(times[mask], event_observed=events[mask], label=f'cluster {label}')
        kaplan.plot_survival_function(ax=ax, ci_alpha=0)
    
    ax.legend(loc=1, frameon=False)
    ax.set_xlabel('days')
    ax.set_ylabel('Survival Probability')
    
    title = f'{model_name}\nCancer: {cancer_type}  p-value: {pvalue:.1e}'
    if cindex:
        title += f'  Cindex: {cindex:.2f}'
    
    ax.set_title(title, fontsize=18, fontweight='bold')

    # Save PDF to the specific output_dir
    fig.savefig(os.path.join(output_dir, f'{cancer_type}_{model_name}.pdf'), dpi=300)
    plt.close(fig) # Close plot to free up memory