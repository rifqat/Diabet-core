"""
Generate all figures for the YadroSeg paper.
Output: paper/figures/*.pdf
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

warnings.filterwarnings('ignore')

FIGDIR = os.path.join('paper', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

METHODS_COLORS = {
    'DBSCAN': '#e74c3c',
    'AutoSCAN': '#3498db',
    'YadroSeg-Auto': '#2ecc71',
    'YadroSeg-Grid': '#f39c12',
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ──────────────────────────────────────────────────────────
#  Dataset loaders (from run_all_comparisons.py)
# ──────────────────────────────────────────────────────────
def get_dataset_1():
    df = pd.read_csv('datasets/dataset1_diabetes.csv')
    df.dropna(subset=['glyhb'], inplace=True)
    df['target'] = (df['glyhb'] > 7.0).astype(int)
    features = ['stab.glu', 'age', 'ratio', 'waist', 'chol', 'bp.1s', 'weight']
    df = df.dropna(subset=features)
    return df[features], df['target'], "DS$_1$ Diabetes", 5

def get_dataset_2():
    df = pd.read_csv('datasets/dataset2_hypertension.csv', sep=';')
    for col in ['bmi', 'wc', 'hc', 'whr', 'SBP', 'DBP']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    df['target'] = (df['SBP'] > 140).astype(int)
    if 'Is.Obese' in df.columns:
        df['is_obese'] = df['Is.Obese'].apply(lambda x: 1 if str(x).strip().upper() == 'YES' else 0)
    features = ['Age', 'is_obese', 'bmi', 'wc', 'hc', 'whr']
    df = df.dropna(subset=features)
    return df[features], df['target'], "DS$_2$ Hypertension", 5

def get_dataset_3():
    df = pd.read_csv('datasets/dataset3_ckd/Chronic_Kidney_Disease/ckd_clean.csv')
    df['dm'] = df['dm'].astype(str).str.replace('\t', '').str.strip()
    df['htn'] = df['htn'].astype(str).str.replace('\t', '').str.strip()
    df.replace('?', np.nan, inplace=True)
    df = df.dropna(subset=['age', 'bp', 'htn', 'dm'])
    valid_mask = df['htn'].isin(['yes', 'no']) & df['dm'].isin(['yes', 'no'])
    df = df[valid_mask]
    df['htn_num'] = (df['htn'] == 'yes').astype(float)
    df['target'] = (df['dm'] == 'yes').astype(int)
    df['age'] = df['age'].astype(float)
    df['bp'] = df['bp'].astype(float)
    features = ['age', 'bp', 'htn_num']
    return df[features], df['target'], "DS$_3$ CKD", 3

def get_dataset_4():
    df = pd.read_csv('datasets/processed_cleveland.csv', na_values='?')
    df = df.dropna()
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df['target'] = (df['target'] > 0).astype(int)
    features = [c for c in df.columns if c != 'target']
    return df[features], df['target'], "DS$_4$ Cleveland", 5

def get_dataset_5():
    df = pd.read_csv('datasets/statlog_heart.dat', sep=' ')
    df = df.dropna()
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df['target'] = (df['target'] == 2).astype(int)
    features = [c for c in df.columns if c != 'target']
    return df[features], df['target'], "DS$_5$ Statlog", 5


def select_features(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    sel = SelectFromModel(rf, prefit=True)
    mask = sel.get_support()
    cols = X.columns[mask]
    if len(cols) < 2:
        cols = X.columns
    return X[cols]


# ──────────────────────────────────────────────────────────
#  FIGURE 1: Pipeline diagram
# ──────────────────────────────────────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.0, 2.5)
    ax.axis('off')

    boxes = [
        (0.0,  'Raw Tabular\nData',        '#ecf0f1', '#2c3e50'),
        (2.0,  'Feature\nSelection',        '#dfe6e9', '#2d3436'),
        (4.0,  'YadroSeg\nOutlier Removal', '#d5f5e3', '#1e8449'),
        (6.0,  'SMOTE-ENN\nBalancing',      '#d6eaf8', '#2471a3'),
        (8.0,  'Random Forest\nClassifier',  '#fdebd0', '#ca6f1e'),
        (10.0, '10-fold CV\nEvaluation',    '#fadbd8', '#c0392b'),
    ]

    box_h, box_w = 1.4, 1.6
    y_center = 0.7
    for x, label, fc, ec in boxes:
        bx = x - box_w / 2
        by = y_center - box_h / 2
        fancy = FancyBboxPatch((bx, by), box_w, box_h,
                               boxstyle="round,pad=0.12",
                               facecolor=fc, edgecolor=ec, linewidth=1.5)
        ax.add_patch(fancy)
        ax.text(x, y_center, label, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=ec)

    for i in range(len(boxes) - 1):
        x_start = boxes[i][0] + box_w / 2 + 0.02
        x_end = boxes[i + 1][0] - box_w / 2 - 0.02
        ax.annotate('', xy=(x_end, y_center), xytext=(x_start, y_center),
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.8))

    ax.text(4.0, -0.6, r'$k$-NN graph $\to$ density variation $\to$ elbow $\to$ outlier mask',
            ha='center', va='center', fontsize=8, fontstyle='italic', color='#1e8449')

    path = os.path.join(FIGDIR, 'pipeline.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path}")


# ──────────────────────────────────────────────────────────
#  FIGURE 2: Density variation sequence + elbow
# ──────────────────────────────────────────────────────────
def compute_Rt(yadro):
    dt = np.array(yadro.Dt_sequence)
    with np.errstate(divide='ignore', invalid='ignore'):
        Rt = np.where(dt[:-1] != 0, (dt[:-1] - dt[1:]) / dt[:-1], 0)
    Rt[~np.isfinite(Rt)] = 0
    return np.append(Rt, 0.0)


def fig_density_variation():
    from segmentation_v1_2 import YadroSegmentation

    loaders = [get_dataset_1, get_dataset_3]
    labels = ['DS$_1$ (Diabetes)', 'DS$_3$ (CKD)']

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    for idx, (loader, label) in enumerate(zip(loaders, labels)):
        X, y, _, ms = loader()
        X_sel = select_features(X, y)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_sel)

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            yadro = YadroSegmentation(X_sc, metric='euclidean')
            yadro.auto_select_epsilon()
            yadro.create_graph_with_weights()
            yadro.compute_density_variation_sequence()
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        Rt = compute_Rt(yadro)
        t = np.arange(len(Rt))

        ax = axes[idx]
        ax.plot(t, Rt, color='#2c3e50', linewidth=0.6, alpha=0.85)
        ax.fill_between(t, Rt, alpha=0.15, color='#3498db')

        positive_Rt = sorted([r for r in Rt if r > 0])
        if len(positive_Rt) >= 3:
            y_pts = np.array(positive_Rt)
            x_pts = np.arange(len(y_pts))
            x_n = (x_pts - x_pts[0]) / (x_pts[-1] - x_pts[0] + 1e-10)
            y_n = (y_pts - y_pts[0]) / (y_pts[-1] - y_pts[0] + 1e-10)
            p1 = np.array([0.0, 0.0])
            line_vec = np.array([1.0, 1.0])
            vec_from_p1 = np.column_stack((x_n, y_n)) - p1
            dists = np.abs(np.cross(vec_from_p1, line_vec)) / np.linalg.norm(line_vec)
            opt_idx = np.argmax(dists)
            delta_star = opt_idx / len(positive_Rt)

            alpha_idx = int(len(positive_Rt) * delta_star)
            alpha_idx = min(alpha_idx, len(positive_Rt) - 1)
            alpha_val = positive_Rt[alpha_idx]

            ax.axhline(y=alpha_val, color='#e74c3c', linestyle='--', linewidth=1.0,
                       label=rf'$\alpha = {alpha_val:.3f}$ ($\delta^*={delta_star:.2f}$)')

        ax.set_xlabel('Removal step $t$')
        ax.set_ylabel('$R_t$')
        ax.set_title(label)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIGDIR, 'density_variation.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path}")


# ──────────────────────────────────────────────────────────
#  FIGURE 3: Geometric elbow visualization
# ──────────────────────────────────────────────────────────
def fig_elbow():
    from segmentation_v1_2 import YadroSegmentation

    X, y, _, ms = get_dataset_1()
    X_sel = select_features(X, y)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_sel)

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yadro = YadroSegmentation(X_sc, metric='euclidean')
        yadro.auto_select_epsilon()
        yadro.create_graph_with_weights()
        yadro.compute_density_variation_sequence()
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    Rt = compute_Rt(yadro)
    positive_Rt = sorted([r for r in Rt if r > 0])

    y_pts = np.array(positive_Rt)
    x_pts = np.arange(len(y_pts))

    x_n = (x_pts - x_pts[0]) / (x_pts[-1] - x_pts[0] + 1e-10)
    y_n = (y_pts - y_pts[0]) / (y_pts[-1] - y_pts[0] + 1e-10)

    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0, 1.0])
    line_vec = p2 - p1
    vec_from_p1 = np.column_stack((x_n, y_n)) - p1
    dists = np.abs(np.cross(vec_from_p1, line_vec)) / np.linalg.norm(line_vec)
    opt_idx = np.argmax(dists)
    delta_star = opt_idx / len(positive_Rt)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(x_n, y_n, 'b-', linewidth=1.2, label='Normalised sorted $R_t$')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=0.9, alpha=0.7, label='Diagonal $\\ell$')
    ax.plot(x_n[opt_idx], y_n[opt_idx], 'go', markersize=9, zorder=5,
            label=f'Elbow ($\\delta^*={delta_star:.3f}$)')

    xp, yp = x_n[opt_idx], y_n[opt_idx]
    proj_t = ((xp - 0) * 1 + (yp - 0) * 1) / 2.0
    px, py = proj_t, proj_t
    ax.plot([xp, px], [yp, py], 'g--', linewidth=0.8, alpha=0.7)

    ax.set_xlabel('Normalised index $\\tilde{x}$')
    ax.set_ylabel('Normalised $R_t$ value $\\tilde{y}$')
    ax.set_title('Geometric Elbow Method — DS$_1$ (Diabetes)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    fig.tight_layout()
    path = os.path.join(FIGDIR, 'elbow.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path}")


# ──────────────────────────────────────────────────────────
#  FIGURE 4: F1-score comparison bar chart
# ──────────────────────────────────────────────────────────
def fig_f1_comparison():
    datasets = ['DS$_1$\nDiabetes', 'DS$_2$\nHypertension', 'DS$_3$\nCKD',
                'DS$_4$\nCleveland', 'DS$_5$\nStatlog']

    f1_dbscan      = [98.22, 94.57, 98.08, 96.83, 93.18]
    f1_autoscan    = [97.73, 92.56, 98.85, 99.21, 96.70]
    f1_yadro_auto  = [98.78, 95.92, 98.87, 96.60, 94.40]
    f1_yadro_grid  = [  0.0, 88.89, 98.23, 98.48, 97.39]

    x = np.arange(len(datasets))
    width = 0.20

    fig, ax = plt.subplots(figsize=(9, 4.5))

    bars1 = ax.bar(x - 1.5 * width, f1_dbscan, width,
                   label='DBSCAN', color=METHODS_COLORS['DBSCAN'], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x - 0.5 * width, f1_autoscan, width,
                   label='AutoSCAN', color=METHODS_COLORS['AutoSCAN'], edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + 0.5 * width, f1_yadro_auto, width,
                   label='YadroSeg-Auto', color=METHODS_COLORS['YadroSeg-Auto'], edgecolor='white', linewidth=0.5)
    bars4 = ax.bar(x + 1.5 * width, f1_yadro_grid, width,
                   label='YadroSeg-Grid', color=METHODS_COLORS['YadroSeg-Grid'], edgecolor='white', linewidth=0.5)

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=6.5, rotation=90)

    ax.axhspan(0, 1, alpha=0, label='')
    ds1_note = ax.annotate('collapsed', xy=(x[0] + 1.5 * width, 1),
                           fontsize=6, ha='center', va='bottom', color='#7f8c8d', fontstyle='italic')

    ax.set_ylabel('F1-score (%)')
    ax.set_title('F1-score Comparison Across Five Chronic Disease Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylim(85, 102)
    ax.legend(loc='lower right', ncol=2, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIGDIR, 'f1_comparison.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path}")


# ──────────────────────────────────────────────────────────
#  FIGURE 5: Outlier removal before/after (PCA scatter)
# ──────────────────────────────────────────────────────────
def fig_outlier_scatter():
    from segmentation_v1_2 import YadroSegmentation

    X, y, _, ms = get_dataset_1()
    X_sel = select_features(X, y)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_sel)

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yadro = YadroSegmentation(X_sc, metric='euclidean')
        yadro.auto_select_epsilon()
        yadro.create_graph_with_weights()
        yadro.compute_density_variation_sequence()
        delta, _ = yadro.find_optimal_params_elbow(beta=ms, visualize_elbow=False)

        core_pixels = yadro.identify_core_pixels(yadro.Dt_sequence, yadro.Mt_sequence,
                                                  delta=delta, beta=ms)
        core_segments = yadro.partition_core_pixels(yadro.G, core_pixels, theta=0.1)
        segments, _ = yadro.expand_segments(yadro.G, yadro.Mt_sequence, core_segments)

        valid_idx = []
        for seg in segments:
            if len(seg) >= ms:
                valid_idx.extend(seg)
        mask = np.zeros(X_sc.shape[0], dtype=bool)
        if valid_idx:
            mask[valid_idx] = True
        else:
            mask[:] = True
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_sc)

    y_np = y.values if hasattr(y, 'values') else np.array(y)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    colors_class = {0: '#3498db', 1: '#e74c3c'}
    labels_class = {0: 'Negative', 1: 'Positive'}

    # Before
    ax = axes[0]
    for c in [0, 1]:
        m_c = y_np == c
        ax.scatter(X_2d[m_c, 0], X_2d[m_c, 1], c=colors_class[c],
                   s=15, alpha=0.6, label=labels_class[c], edgecolors='none')
    ax.set_title(f'Before YadroSeg ($N={len(X_sc)}$)')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.legend(loc='upper right', markerscale=2, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    # After
    ax = axes[1]
    outlier_mask = ~mask
    n_outliers = int(outlier_mask.sum())
    for c in [0, 1]:
        m_c = (y_np == c) & mask
        ax.scatter(X_2d[m_c, 0], X_2d[m_c, 1], c=colors_class[c],
                   s=15, alpha=0.6, label=labels_class[c], edgecolors='none')
    ax.scatter(X_2d[outlier_mask, 0], X_2d[outlier_mask, 1],
               c='none', s=25, edgecolors='#7f8c8d', linewidths=0.8,
               marker='x', alpha=0.7, label=f'Removed ({n_outliers})')
    ax.set_title(f'After YadroSeg ($N={int(mask.sum())}$, {n_outliers} removed)')
    ax.set_xlabel('PC 1')
    ax.legend(loc='upper right', markerscale=2, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    path = os.path.join(FIGDIR, 'outlier_scatter.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path}")


# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating paper figures...")
    print("=" * 50)

    print("\n[1/5] Pipeline diagram")
    fig_pipeline()

    print("\n[2/5] Density variation sequence (DS1, DS3)")
    fig_density_variation()

    print("\n[3/5] Geometric elbow visualisation (DS1)")
    fig_elbow()

    print("\n[4/5] F1-score comparison bar chart")
    fig_f1_comparison()

    print("\n[5/5] Outlier removal scatter (DS1)")
    fig_outlier_scatter()

    print("\n" + "=" * 50)
    print("All figures saved to paper/figures/")
    print("Files:")
    for f in sorted(os.listdir(FIGDIR)):
        fpath = os.path.join(FIGDIR, f)
        sz = os.path.getsize(fpath)
        print(f"  {f:30s}  {sz:>8,d} bytes")
