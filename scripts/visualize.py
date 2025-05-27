import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

FEATURE_DIR = "features/binance_top300"
COINS_TO_VISUALIZE_LIST = [
    "FIDAUSDT",
    "DODOUSDT",
    "WLDUSDT",
    "XLMUSDT",
    "KAVAUSDT"
]

DBSCAN_EPS_RAW = 0.1
DBSCAN_MIN_SAMPLES_RAW = 3
DBSCAN_EPS_COMPRESSED = 0.5
DBSCAN_MIN_SAMPLES_COMPRESSED = 5

PCA_COMPONENTS_FOR_VIZ = 2

IMAGE_SAVE_DIR = "visualization_output"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

try:
    results_df = pd.read_csv('clustering_results/clustering_comparison_results.csv')
    print("clustering_comparison_results.csv loaded successfully.")
except FileNotFoundError:
    print("Error: clustering_results/clustering_comparison_results.csv not found.")
    print("Please generate clustering results for all coins first.")
    exit()

def load_and_prepare_data(coin_name, feature_type):
    if feature_type == 'raw':
        data_path = os.path.join(FEATURE_DIR, f"X_{coin_name}.npy")
    elif feature_type == 'compressed_ae':
        data_path = os.path.join(FEATURE_DIR, f"Z_ae_{coin_name}.npy")
    else:
        raise ValueError("feature_type must be 'raw' or 'compressed_ae'")

    y_path = os.path.join(FEATURE_DIR, f"y_{coin_name}.npy")

    if not os.path.exists(data_path) or not os.path.exists(y_path):
        print(f"Error: Data files for {coin_name} ({feature_type}) not found.")
        return None, None, None

    data = np.load(data_path)
    labels = np.load(y_path)

    if len(data) < 5:
        print(f"Warning: {coin_name} ({feature_type}): Insufficient number of samples ({len(data)} samples).")
        return None, None, None

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    pca_viz = PCA(n_components=PCA_COMPONENTS_FOR_VIZ)
    data_2d_viz = pca_viz.fit_transform(scaled_data)

    return scaled_data, data_2d_viz, labels

def perform_and_visualize_clustering(coin_name, X_scaled, X_2d_viz, y_moonshot, title_prefix,
                                     kmeans_n_clusters, dbscan_eps, dbscan_min_samples):
    kmeans_labels = None
    kmeans_score = np.nan
    if kmeans_n_clusters >= 2 and kmeans_n_clusters < len(X_scaled):
        try:
            kmeans = KMeans(n_clusters=kmeans_n_clusters, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(X_scaled)
            if len(np.unique(kmeans_labels)) > 1:
                kmeans_score = silhouette_score(X_scaled, kmeans_labels)
                print(f"{title_prefix} - KMeans Silhouette Score (k={kmeans_n_clusters}): {kmeans_score:.3f}")
            else:
                print(f"{title_prefix} - KMeans (k={kmeans_n_clusters}): Single cluster formed, cannot compute silhouette score.")
        except Exception as e:
            print(f"❌ {title_prefix} - KMeans failed: {e}")
    else:
        print(f"⚠️ {title_prefix} - KMeans (k={kmeans_n_clusters}): Invalid number of clusters or more than samples.")

    dbscan_labels = None
    dbscan_score = np.nan
    if dbscan_eps is not None and dbscan_min_samples is not None:
        try:
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            dbscan_labels = dbscan.fit_predict(X_scaled)

            n_clusters_dbscan = len(np.unique(dbscan_labels))

            if n_clusters_dbscan > 1 and -1 not in np.unique(dbscan_labels):
                dbscan_score = silhouette_score(X_scaled, dbscan_labels)
                print(f"{title_prefix} - DBSCAN Silhouette Score (eps={dbscan_eps}, min_samples={dbscan_min_samples}): {dbscan_score:.3f}")
            elif n_clusters_dbscan > 1 and -1 in np.unique(dbscan_labels) and len(np.unique(dbscan_labels[dbscan_labels != -1])) > 1:
                dbscan_score = silhouette_score(X_scaled[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1])
                print(f"{title_prefix} - DBSCAN Silhouette Score (excluding noise, eps={dbscan_eps}, min_samples={dbscan_min_samples}): {dbscan_score:.3f}")
            else:
                print(f"{title_prefix} - DBSCAN: Single cluster or all points classified as noise, cannot compute silhouette score. (Clusters: {n_clusters_dbscan})")

        except Exception as e:
            print(f"❌ {title_prefix} - DBSCAN failed: {e}")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    fig.text(0.5, 0.98, f'{coin_name} Clustering Comparison: {title_prefix}',
             ha='center', va='top', fontsize=18, weight='bold')

    if kmeans_labels is not None:
        sns.scatterplot(x=X_2d_viz[:, 0], y=X_2d_viz[:, 1], hue=kmeans_labels, style=y_moonshot,
                        palette='viridis', legend='full', s=70, alpha=0.8, ax=axes[0])
        axes[0].set_title(f'K-Means (k={kmeans_n_clusters}, Silhouette: {kmeans_score:.3f})')
        axes[0].set_xlabel(f'PCA Component 1')
        axes[0].set_ylabel(f'PCA Component 2')
        axes[0].grid(True, linestyle='--', alpha=0.6)
    else:
        axes[0].set_title(f'K-Means - No valid clusters')
        axes[0].text(0.5, 0.5, 'Data cannot be displayed', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)

    if dbscan_labels is not None:
        sns.scatterplot(x=X_2d_viz[:, 0], y=X_2d_viz[:, 1], hue=dbscan_labels, style=y_moonshot,
                        palette='viridis', legend='full', s=70, alpha=0.8, ax=axes[1])
        axes[1].set_title(f'DBSCAN (eps={dbscan_eps}, min_samples={dbscan_min_samples}, Silhouette: {dbscan_score:.3f})')
        axes[1].set_xlabel(f'PCA Component 1')
        axes[1].set_ylabel(f'PCA Component 2')
        axes[1].grid(True, linestyle='--', alpha=0.6)
    else:
        axes[1].set_title(f'DBSCAN - No valid clusters')
        axes[1].text(0.5, 0.5, 'Data cannot be displayed', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    figure_filename_png = f"{coin_name}_{title_prefix.replace(' ', '_').replace('(', '').replace(')', '')}_Clustering.png"
    figure_filename_pdf = f"{coin_name}_{title_prefix.replace(' ', '_').replace('(', '').replace(')', '')}_Clustering.pdf"
    
    plt.savefig(os.path.join(IMAGE_SAVE_DIR, figure_filename_png), dpi=300, bbox_inches='tight')
    print(f"PNG image saved: {os.path.join(IMAGE_SAVE_DIR, figure_filename_png)}")

    plt.savefig(os.path.join(IMAGE_SAVE_DIR, figure_filename_pdf), bbox_inches='tight')
    print(f"PDF image saved: {os.path.join(IMAGE_SAVE_DIR, figure_filename_pdf)}")
    
    plt.close(fig)

for coin_symbol in COINS_TO_VISUALIZE_LIST:
    print(f"\n--- Starting visualization for {coin_symbol} ---")

    coin_results = results_df[results_df['coin'] == coin_symbol]
    kmeans_k_raw = coin_results[(coin_results['algorithm'] == 'KMeans') & (coin_results['feature_type'] == 'raw')]['best_k'].iloc[0] if not coin_results[(coin_results['algorithm'] == 'KMeans') & (coin_results['feature_type'] == 'raw')]['best_k'].empty else 3
    kmeans_k_compressed = coin_results[(coin_results['algorithm'] == 'KMeans') & (coin_results['feature_type'] == 'compressed_ae')]['best_k'].iloc[0] if not coin_results[(coin_results['algorithm'] == 'KMeans') & (coin_results['feature_type'] == 'compressed_ae')]['best_k'].empty else 3

    X_raw_scaled, X_raw_2d_viz, y_moonshot_raw = load_and_prepare_data(coin_symbol, 'raw')
    if X_raw_scaled is not None:
        perform_and_visualize_clustering(coin_symbol, X_raw_scaled, X_raw_2d_viz, y_moonshot_raw,
                                         "Raw Features", int(kmeans_k_raw),
                                         DBSCAN_EPS_RAW, DBSCAN_MIN_SAMPLES_RAW)

    X_compressed_scaled, X_compressed_2d_viz, y_moonshot_compressed = load_and_prepare_data(coin_symbol, 'compressed_ae')
    if X_compressed_scaled is not None:
        perform_and_visualize_clustering(coin_symbol, X_compressed_scaled, X_compressed_2d_viz, y_moonshot_compressed,
                                         "AE Compressed Features", int(kmeans_k_compressed),
                                         DBSCAN_EPS_COMPRESSED, DBSCAN_MIN_SAMPLES_COMPRESSED)

print(f"\n--- All coin visualizations completed. Image files saved to '{IMAGE_SAVE_DIR}' directory ---")