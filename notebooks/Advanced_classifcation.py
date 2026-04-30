#!/usr/bin/env python
# coding: utf-8

# In[92]:


# 1. Imports & Configuration
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from mlxtend.frequent_patterns import apriori, association_rules
warnings.filterwarnings("ignore")

# =====================================================================
#  GLOBAL DEEP SPACE / GALAXY THEME
# =====================================================================
# Cosmic Backgrounds and Neutrals
GALAXY_BG = '#030014'          # Extremely deep space void
GALAXY_FG = '#E2E8F0'          # Starlight white/silver
GALAXY_GRID = 'rgba(255, 255, 255, 0.05)' # Faint stardust grid lines

# Vibrant Nebula Accents (Cyber-cosmic)
GALAXY_ACCENT1 = '#FF007F'     # Nebula Pink
GALAXY_ACCENT2 = '#00F0FF'     # Cosmic Cyan
GALAXY_ACCENT3 = '#AA00FF'     # Deep Pulsar Purple
GALAXY_ACCENT4 = '#FFEA00'     # Solar Yellow
GALAXY_ACCENT5 = '#39FF14'     # Alien Plasma Green
GALAXY_ACCENT6 = '#1A73E8'     # Deep Galaxy Blue

# Standard Categorical Palette (NASA Kepler states)
GALAXY_CLASS_PALETTE = {
    'CONFIRMED': GALAXY_ACCENT2,    # Cyan
    'FALSE POSITIVE': GALAXY_ACCENT1, # Pink
    'CANDIDATE': GALAXY_ACCENT3     # Purple
}
GALAXY_PALETTE = [GALAXY_ACCENT2, GALAXY_ACCENT1, GALAXY_ACCENT3, GALAXY_ACCENT4, GALAXY_ACCENT5, GALAXY_ACCENT6]

# Continuous Colormaps
from matplotlib.colors import LinearSegmentedColormap
GALAXY_CMAP = LinearSegmentedColormap.from_list('galaxy_sequential', [GALAXY_BG, GALAXY_ACCENT3, GALAXY_ACCENT2, GALAXY_FG])
GALAXY_DIVERGING_CMAP = LinearSegmentedColormap.from_list('galaxy_diverge', [GALAXY_ACCENT1, GALAXY_BG, GALAXY_ACCENT2])

# Model Color Mapping for Consistency
GALAXY_MODEL_COLORS = [GALAXY_ACCENT1, GALAXY_ACCENT2, GALAXY_ACCENT5, GALAXY_ACCENT4, GALAXY_ACCENT3]

# =====================================================================
#  THEME APPLICATION FUNCTIONS
# =====================================================================
def apply_galaxy_figure(fig, facecolor=GALAXY_BG):
    """Applies deep space background to a Matplotlib Figure."""
    fig.patch.set_facecolor(facecolor)
    fig.patch.set_edgecolor(facecolor)

def apply_galaxy_axes(ax, grid=True, title_color=GALAXY_FG, facecolor=GALAXY_BG):
    """Applies galaxy theme to Matplotlib Axes cleanly."""
    ax.set_facecolor(facecolor)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GALAXY_FG)
    ax.spines['left'].set_color(GALAXY_FG)
    
    ax.tick_params(axis='x', colors=GALAXY_FG, labelsize=10)
    ax.tick_params(axis='y', colors=GALAXY_FG, labelsize=10)
    
    ax.xaxis.label.set_color(GALAXY_FG)
    ax.yaxis.label.set_color(GALAXY_FG)
    ax.xaxis.label.set_weight('bold')
    ax.yaxis.label.set_weight('bold')
    
    if ax.title.get_text():
        ax.title.set_color(title_color)
        ax.title.set_weight('bold')
        ax.title.set_size(13)

    if grid:
        # Use an extremely faint white string for the grid
        ax.grid(True, linestyle=':', linewidth=0.5, color='#ffffff', alpha=0.1, zorder=0)

def style_galaxy_legend(legend):
    """Applies deep space theme to a Matplotlib legend."""
    if legend:
        legend.get_frame().set_facecolor((0.01, 0.0, 0.08, 0.7)) # RGBA tuple instead of string
        legend.get_frame().set_edgecolor(GALAXY_FG)
        for text in legend.get_texts():
            text.set_color(GALAXY_FG)

def apply_galaxy_plotly(fig):
    """Applies galaxy theme to Plotly figures."""
    fig.update_layout(
        plot_bgcolor=GALAXY_BG,
        paper_bgcolor=GALAXY_BG,
        font=dict(color=GALAXY_FG),
        title_font=dict(color=GALAXY_FG, size=20, family="Arial, bold"),
        legend=dict(
            bgcolor='rgba(3, 0, 20, 0.7)', 
            bordercolor=GALAXY_FG,
            borderwidth=1,
            font=dict(color=GALAXY_FG)
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    axis_styling = dict(
        showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', gridwidth=1,
        zeroline=True, zerolinecolor=GALAXY_FG, zerolinewidth=1,
        showline=True, linecolor=GALAXY_FG, linewidth=1,
        tickfont=dict(color=GALAXY_FG)
    )
    
    if 'xaxis' in fig.layout: fig.update_xaxes(**axis_styling)
    if 'yaxis' in fig.layout: fig.update_yaxes(**axis_styling)
    if 'scene' in fig.layout:
        fig.update_layout(scene=dict(
            xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)', zerolinecolor=GALAXY_FG, backgroundcolor=GALAXY_BG, color=GALAXY_FG),
            yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)', zerolinecolor=GALAXY_FG, backgroundcolor=GALAXY_BG, color=GALAXY_FG),
            zaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)', zerolinecolor=GALAXY_FG, backgroundcolor=GALAXY_BG, color=GALAXY_FG)
        ))

# Set Seaborn global context
sns.set_theme(style='darkgrid', rc={
    'axes.facecolor': GALAXY_BG,
    'figure.facecolor': GALAXY_BG,
    'axes.edgecolor': GALAXY_FG,
    'grid.color': '#ffffff',
    'grid.linestyle': ':',
    'grid.alpha': 0.1,
    'text.color': GALAXY_FG,
    'xtick.color': GALAXY_FG,
    'ytick.color': GALAXY_FG,
    'axes.labelcolor': GALAXY_FG,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11
})

def galaxy_table_styles():
    """Returns CSS styles for pandas Styler objects."""
    return [
        {'selector': 'th', 'props': [('background-color', '#120d2b'), ('color', GALAXY_ACCENT2), ('font-weight', 'bold'), ('border', '1px solid #201a40')]},
        {'selector': 'td', 'props': [('border', '1px solid #201a40')]},
        {'selector': 'table', 'props': [('background-color', GALAXY_BG), ('color', GALAXY_FG), ('border-collapse', 'collapse')]},
        {'selector': 'tr:hover td', 'props': [('background-color', '#1a0b3b')]} # Pink/Purple glow effect on hover
    ]


# ## 2. Data Loading & Basic EDA

# In[93]:


df = pd.read_csv("../data/cumulative.csv")
display(df.head())
df.info()
display(df.describe())
display(df['koi_disposition'].value_counts())


# ## 3. Data Cleaning & Feature Selection

# In[94]:


# 1. Define columns to drop (ID and non-predictive metadata)
cols_to_drop = [
    'rowid', 'kepid', 'kepoi_name', 'kepler_name', 
    'koi_pdisposition', 'koi_score',                
    'koi_tce_delivname', 'koi_tce_plnt_num'         
]
# 2. Collect all error/uncertainty columns (containing 'err')
err_columns = [col for col in df.columns if 'err' in col]
# 3. Combine drop lists
final_drop_list = cols_to_drop + err_columns
# 4. Perform the drop to clean the dataframe
df_cleaned = df.drop(columns=final_drop_list)
# 5. Define the physical feature columns (include False Positive flags)
physical_features = [
    'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
    'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol',
    'koi_model_snr', 'koi_steff', 'koi_slogg', 'koi_srad',
    'ra', 'dec', 'koi_kepmag',
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
]

# 6. Final selection
df_final = df_cleaned[physical_features + ['koi_disposition']].copy()

# Display summary for confirmation
print(f"Final Data Shape: {df_final.shape}")
df_final.head()


# In[95]:


# Check missing values
df_final.isnull().sum()


# In[96]:


# Prepare an EDA-only copy with median-filled numeric values.
# Supervised models keep df_final unfilled and fit imputers inside train-only pipelines.
df_eda = df_final.copy()
for col in df_eda.select_dtypes(include=[np.number]).columns:
    if df_eda[col].isnull().any():
        median_val = df_eda[col].median()
        df_eda[col] = df_eda[col].fillna(median_val)

print("Missing values after EDA-only fill:")
display(df_eda.isnull().sum())


# ## 4. Exploratory Data Analysis (EDA)

# In[97]:


# Target Distribution
counts = df_eda['koi_disposition'].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))
colors = [GALAXY_CLASS_PALETTE.get(label, GALAXY_ACCENT1) for label in counts.index]
bars = ax.bar(counts.index, counts.values, color=colors, edgecolor=GALAXY_FG, width=0.5, alpha=0.9)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f'{val:,} ({val/len(df_eda)*100:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color=GALAXY_ACCENT4)

ax.set_title('Target Class Distribution - NASA Kepler KOIs', fontsize=14, fontweight='bold')
ax.set_xlabel('Disposition', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
apply_galaxy_axes(ax)
plt.tight_layout()
plt.savefig('01_target_distribution.png', dpi=150)
plt.show()


# In[98]:


# EDA - Feature Distributions (Histograms)

# 1. Define feature_cols
feature_cols = [
    'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
    'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol',
    'koi_model_snr', 'koi_steff', 'koi_slogg', 'koi_srad',
    'ra', 'dec', 'koi_kepmag'
]

skewed_features = [
    'koi_period', 'koi_impact', 'koi_duration', 'koi_depth',
    'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_srad'
]

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

# Heavy-tailed transit features are shown on log1p scale so the bulk is visible.
for i, col in enumerate(feature_cols):
    ax = axes[i]
    values = df_eda[col]
    xlabel = col
    if col in skewed_features:
        values = np.log1p(values.clip(lower=0))
        xlabel = f'log1p({col})'

    ax.hist(values, bins=40, color=GALAXY_PALETTE[i % len(GALAXY_PALETTE)], edgecolor=GALAXY_BG, alpha=0.88)
    ax.set_title(f'Dist of {xlabel}', fontsize=11, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    apply_galaxy_axes(ax)

for j in range(len(feature_cols), len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Feature Distributions (Cleaned Data, log1p for skewed features)', fontsize=18, fontweight='bold', y=1.02, color=GALAXY_ACCENT4)
apply_galaxy_figure(fig)
plt.tight_layout()
plt.savefig('02_feature_distributions.png', dpi=150)
plt.show()


# In[99]:


# Boxplots per Class
key_features = ['koi_prad', 'koi_period', 'koi_depth', 'koi_duration',
                'koi_impact', 'koi_model_snr', 'koi_teq', 'koi_insol']

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()
palette = GALAXY_CLASS_PALETTE
plot_order = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']

# Use log1p values and hide fliers so extreme KOI outliers do not flatten the boxes.
df_box = df_eda[['koi_disposition'] + key_features].copy()
for col in key_features:
    df_box[f'{col}_log1p'] = np.log1p(df_box[col].clip(lower=0))

for i, col in enumerate(key_features):
    ax = axes[i]
    plot_col = f'{col}_log1p'
    sns.boxplot(x='koi_disposition', y=plot_col, data=df_box,
                hue='koi_disposition', palette=palette, ax=ax,
                order=plot_order, legend=False, showfliers=False,
                boxprops={'edgecolor': GALAXY_FG},
                whiskerprops={'color': GALAXY_FG},
                capprops={'color': GALAXY_FG},
                medianprops={'color': GALAXY_ACCENT4})
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(f'log1p({col})')
    ax.tick_params(axis='x', rotation=15)
    apply_galaxy_axes(ax)

fig.suptitle('Key Features vs. Disposition (log1p scale, fliers hidden)', fontsize=15, fontweight='bold', color=GALAXY_ACCENT4)
plt.tight_layout()
plt.savefig('03_key_features_vs_disposition_log_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()


# In[100]:


# Correlation Heatmap
corr_features = df_eda[feature_cols].copy()
for col in skewed_features:
    corr_features[col] = np.log1p(corr_features[col].clip(lower=0))

corr = corr_features.corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=GALAXY_DIVERGING_CMAP, center=0,
            annot_kws={'size': 8, 'color': GALAXY_FG}, linewidths=0.5,
            linecolor=GALAXY_BG, cbar_kws={'label': 'Correlation'}, ax=ax)

ax.set_title('Feature Correlation Matrix (log1p for skewed features)', fontsize=14, fontweight='bold')
apply_galaxy_figure(fig)
plt.show()


# In[101]:


# Pairplot (Top 4 Features)
pair_features = ['koi_prad', 'koi_period', 'koi_depth', 'koi_impact', 'koi_disposition']
df_pair = df_eda[pair_features].copy()

for col in ['koi_prad', 'koi_period', 'koi_depth', 'koi_impact']:
    df_pair[col] = np.log1p(df_pair[col].clip(lower=0))

g = sns.pairplot(df_pair, hue='koi_disposition', palette=GALAXY_CLASS_PALETTE,
                 plot_kws={'alpha': 0.55, 's': 18, 'edgecolor': GALAXY_BG, 'linewidth': 0.2},
                 diag_kind='kde')
g.figure.suptitle('Pairplot - Key Transit Features (log1p scale)', y=1.02, fontsize=13, color=GALAXY_ACCENT4)
g.figure.patch.set_facecolor(GALAXY_BG)
for ax in g.axes.flatten():
    if ax is not None:
        apply_galaxy_axes(ax)
style_galaxy_legend(g._legend)
plt.show()


# ## 5. Feature Encoding & Scaling

# In[102]:


#  Encode Target Column 
le = LabelEncoder()
df_final['target'] = le.fit_transform(df_final['koi_disposition'])

#  Full-dataset scaled export for EDA/unsupervised artifacts only.
#  Supervised classification below uses train-only sklearn pipelines to avoid leakage.
export_preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
X_scaled_values = export_preprocessor.fit_transform(df_final[feature_cols])
X_scaled_df = pd.DataFrame(X_scaled_values, columns=feature_cols, index=df_final.index)


# In[103]:


#  Final Assembly and Save 
df_final_scaled = X_scaled_df.copy()
df_final_scaled['koi_disposition'] = df_final['koi_disposition'].values
df_final_scaled['target'] = df_final['target'].values

df_final_scaled.to_csv("cumulative_clean_scaled.csv", index=False)


# ## 6. Basic Classification
# Logistic Regression and Decision Tree models applied to the cleaned, scaled Kepler dataset.

# In[104]:


# Basic Classification Imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix


# In[105]:


# Use the cleaned dataframe from preprocessing.
# Missing values are handled inside each model pipeline after train/test split.
df_cls = df_final.copy()
display(df_cls[feature_cols + ['koi_disposition']].head())


# In[106]:


# Define Features and Target
# Remove CANDIDATE rows so training/testing only includes CONFIRMED vs FALSE POSITIVE.
df_cls = df_cls[df_cls['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()

X = df_cls[feature_cols]
y = df_cls['koi_disposition'].map({'FALSE POSITIVE': 0, 'CONFIRMED': 1}).astype(int)

class_labels = [0, 1]
class_names = ['FALSE POSITIVE', 'CONFIRMED']
positive_label = 1

class_counts = y.value_counts().sort_index().rename(index={0: 'FALSE POSITIVE', 1: 'CONFIRMED'})
display(class_counts)


# In[107]:


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[108]:


def make_scaled_preprocessor():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

# Logistic Regression with balanced class weights to address class imbalance
log_model = Pipeline([
    ('preprocess', make_scaled_preprocessor()),
    ('model', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
])
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

log_accuracy = accuracy_score(y_test, y_pred_log)
log_precision = precision_score(y_test, y_pred_log, average='weighted', zero_division=0)


# In[109]:


# Decision Tree with median imputation and balanced class weights.
tree_model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', DecisionTreeClassifier(
        criterion='entropy',
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    ))
])

tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)
tree_accuracy = accuracy_score(y_test, y_pred_tree)
tree_precision = precision_score(y_test, y_pred_tree, average='weighted', zero_division=0)


# 
# ## Basic Classification Summary
# Logistic Regression and Decision Tree are evaluated as binary classifiers for **CONFIRMED** vs **FALSE POSITIVE** only. Preprocessing is now fit inside each training pipeline to avoid test-set leakage.
# 

# ## 7. Advanced Classification
# SVM (RBF) and Naive Bayes models, compared against the basic models using ROC-AUC and multi-metric evaluation.

# In[110]:


# Advanced Classification Imports
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, recall_score
import warnings
warnings.filterwarnings('ignore')


# In[111]:


svm_model = Pipeline([
    ('preprocess', make_scaled_preprocessor()),
    ('model', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced', random_state=42))
])
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
y_pred_svm_proba = svm_model.predict_proba(X_test)

svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm, average='weighted', zero_division=0)
svm_f1 = f1_score(y_test, y_pred_svm, average='weighted', zero_division=0)
svm_recall = recall_score(y_test, y_pred_svm, average='weighted', zero_division=0)


# In[112]:


nb_model = Pipeline([
    ('preprocess', make_scaled_preprocessor()),
    ('model', GaussianNB())
])
nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)
y_pred_nb_proba = nb_model.predict_proba(X_test)

nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_precision = precision_score(y_test, y_pred_nb, average='weighted', zero_division=0)
nb_f1 = f1_score(y_test, y_pred_nb, average='weighted', zero_division=0)
nb_recall = recall_score(y_test, y_pred_nb, average='weighted', zero_division=0)


# In[113]:


# Recalculate metrics for all models (Binary Classification)
# ROC-AUC is computed with CONFIRMED as the positive class.
def get_positive_class_scores(model, X, positive_label):
    classes = getattr(model, 'classes_', None)
    if classes is None and hasattr(model, 'named_steps'):
        classes = model.named_steps[list(model.named_steps)[-1]].classes_
    positive_index = np.where(classes == positive_label)[0][0]
    return model.predict_proba(X)[:, positive_index]

# Logistic Regression metrics
log_f1 = f1_score(y_test, y_pred_log, average='weighted', zero_division=0)
log_recall = recall_score(y_test, y_pred_log, average='weighted', zero_division=0)
y_pred_log_proba = log_model.predict_proba(X_test)
y_score_log = get_positive_class_scores(log_model, X_test, positive_label)
log_auc = roc_auc_score(y_test, y_score_log)

# Decision Tree metrics
tree_f1 = f1_score(y_test, y_pred_tree, average='weighted', zero_division=0)
tree_recall = recall_score(y_test, y_pred_tree, average='weighted', zero_division=0)
y_pred_tree_proba = tree_model.predict_proba(X_test)
y_score_tree = get_positive_class_scores(tree_model, X_test, positive_label)
tree_auc = roc_auc_score(y_test, y_score_tree)

# SVM metrics
svm_f1 = f1_score(y_test, y_pred_svm, average='weighted', zero_division=0)
svm_recall = recall_score(y_test, y_pred_svm, average='weighted', zero_division=0)
y_score_svm = get_positive_class_scores(svm_model, X_test, positive_label)
svm_auc = roc_auc_score(y_test, y_score_svm)

# Naive Bayes metrics
nb_f1 = f1_score(y_test, y_pred_nb, average='weighted', zero_division=0)
nb_recall = recall_score(y_test, y_pred_nb, average='weighted', zero_division=0)
y_score_nb = get_positive_class_scores(nb_model, X_test, positive_label)
nb_auc = roc_auc_score(y_test, y_score_nb)

# Create Comparison DataFrame
comparison_results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'SVM (RBF)', 'Naive Bayes'],
    'Accuracy': [log_accuracy, tree_accuracy, svm_accuracy, nb_accuracy],
    'Precision': [log_precision, tree_precision, svm_precision, nb_precision],
    'F1-Score': [log_f1, tree_f1, svm_f1, nb_f1],
    'Recall': [log_recall, tree_recall, svm_recall, nb_recall],
    'ROC-AUC': [log_auc, tree_auc, svm_auc, nb_auc]
})

display(comparison_results)


# In[114]:


# ROC Curves for Binary Classification
# Positive class: CONFIRMED exoplanets
def compute_binary_roc(model, X, y_true, positive_label):
    y_score = get_positive_class_scores(model, X, positive_label)
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=positive_label)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

roc_models = [
    ('Logistic Regression', log_model, GALAXY_ACCENT3),
    ('Decision Tree', tree_model, GALAXY_ACCENT2),
    ('SVM (RBF)', svm_model, GALAXY_ACCENT5),
    ('Naive Bayes', nb_model, GALAXY_ACCENT6),
]

fig, ax = plt.subplots(figsize=(10, 8))

for model_name, model, color in roc_models:
    fpr, tpr, roc_auc = compute_binary_roc(model, X_test, y_test, positive_label)
    ax.plot(fpr, tpr, color=color, linewidth=2.5,
            label=f'{model_name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], linestyle='--', color=GALAXY_FG, linewidth=1.5, label='Random Classifier (AUC = 0.500)')
ax.set_title('ROC Curves - Binary Classification (Positive Class: CONFIRMED)', fontsize=14, fontweight='bold')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.legend(loc='lower right', fontsize=10)
apply_galaxy_axes(ax)
plt.tight_layout()
plt.savefig('04_roc_curves_all_models.png', dpi=300, bbox_inches='tight')
plt.show()


# In[115]:


# Visualize Model Comparison - Bar Charts
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Advanced Classification Models - Performance Comparison', fontsize=16, fontweight='bold', color=GALAXY_ACCENT4)

metrics = ['Accuracy', 'Precision', 'F1-Score', 'Recall', 'ROC-AUC']
colors_models = GALAXY_MODEL_COLORS

for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]

    bars = ax.bar(comparison_results['Model'], comparison_results[metric],
                  color=colors_models, edgecolor=GALAXY_FG, width=0.7, alpha=0.88)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color=GALAXY_ACCENT4)

    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(comparison_results['Model'])))
    ax.set_xticklabels(comparison_results['Model'], rotation=45, ha='right')
    apply_galaxy_axes(ax)

# Remove the extra subplot
fig.delaxes(axes[1, 2])

# Create a summary text box in the removed subplot space
ax_text = fig.add_subplot(2, 3, 6)
ax_text.axis('off')
ax_text.set_facecolor(GALAXY_BG)

summary_text = f"""
BEST PERFORMING MODELS:
------------------------
Accuracy:  {comparison_results.loc[comparison_results['Accuracy'].idxmax(), 'Model']}
           ({comparison_results['Accuracy'].max():.4f})

Precision: {comparison_results.loc[comparison_results['Precision'].idxmax(), 'Model']}
           ({comparison_results['Precision'].max():.4f})

F1-Score:  {comparison_results.loc[comparison_results['F1-Score'].idxmax(), 'Model']}
           ({comparison_results['F1-Score'].max():.4f})

ROC-AUC:   {comparison_results.loc[comparison_results['ROC-AUC'].idxmax(), 'Model']}
           ({comparison_results['ROC-AUC'].max():.4f})
"""

ax_text.text(0.1, 0.5, summary_text, fontsize=11, family='monospace', color=GALAXY_FG,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor=GALAXY_BG,
                                                   edgecolor=GALAXY_ACCENT1, alpha=0.95))

apply_galaxy_figure(fig)
plt.tight_layout()
plt.savefig('05_model_comparison_metrics.png', dpi=300, bbox_inches='tight')
plt.show()


# In[116]:


# Confusion Matrices for All Models
cm_log = confusion_matrix(y_test, y_pred_log, labels=class_labels)
cm_tree = confusion_matrix(y_test, y_pred_tree, labels=class_labels)
cm_svm = confusion_matrix(y_test, y_pred_svm, labels=class_labels)
cm_nb = confusion_matrix(y_test, y_pred_nb, labels=class_labels)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Confusion Matrices - All Classification Models', fontsize=16, fontweight='bold', color=GALAXY_ACCENT4)

cms = [
    (cm_log, 'Logistic Regression', GALAXY_ACCENT3),
    (cm_tree, 'Decision Tree', GALAXY_ACCENT2),
    (cm_svm, 'SVM (RBF)', GALAXY_ACCENT5),
    (cm_nb, 'Naive Bayes', GALAXY_ACCENT6),
]

for idx, (cm, title, color) in enumerate(cms):
    ax = axes[idx // 2, idx % 2]

    sns.heatmap(cm, annot=True, fmt='d', cmap=GALAXY_CMAP, ax=ax, cbar=True,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor=GALAXY_BG,
                annot_kws={'size': 12, 'weight': 'bold', 'color': GALAXY_FG})

    ax.set_title(f'{title}', fontsize=12, fontweight='bold', color=color)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.tick_params(axis='x', rotation=15)
    ax.tick_params(axis='y', rotation=0)
    apply_galaxy_axes(ax, title_color=color)

apply_galaxy_figure(fig)
plt.tight_layout()
plt.savefig('06_confusion_matrices_all_models.png', dpi=300, bbox_inches='tight')
plt.show()


# In[117]:


# Comprehensive Evaluation Report
# Render the comparison results with the galaxy theme.
display(
    comparison_results.style
    .background_gradient(cmap=GALAXY_CMAP, subset=['Accuracy', 'Precision', 'F1-Score', 'Recall', 'ROC-AUC'])
    .highlight_max(color=GALAXY_ACCENT5, subset=['Accuracy', 'Precision', 'F1-Score', 'Recall', 'ROC-AUC'])
    .set_table_styles(galaxy_table_styles())
    .set_caption('Advanced Classification Evaluation Report')
    .format(precision=4)
)

# We can still export this cleanly to CSV
comparison_results.to_csv('model_comparison_results_advanced.csv', index=False)


# In[118]:


# Additional Metric Compilation (Radar chart preparation and overall ranking)

metrics = ['Accuracy', 'ROC-AUC', 'F1-Score', 'Precision', 'Recall']
metrics_to_plot = metrics  # Keep tracking metrics list for plotting

# Let's create an overall ranking based on average of the top 3 metrics: Accuracy, ROC-AUC, F1-Score
comparison_results['Score_Avg'] = comparison_results[['Accuracy', 'ROC-AUC', 'F1-Score']].mean(axis=1)
best_model_idx = comparison_results['Score_Avg'].idxmax()

display(comparison_results.sort_values(by='Score_Avg', ascending=False))


# 
# ## Machine Learning Results Overview
# 
# The supervised task is a binary screen for **CONFIRMED** versus **FALSE POSITIVE** KOIs after removing **CANDIDATE** rows. The Decision Tree is the strongest model in this run, while Logistic Regression and SVM remain useful linear/nonlinear baselines. The next sections use unsupervised clustering and association rules as exploratory analysis rather than supervised performance evidence.
# 

# # Data Clustering
# K-Means with Elbow Method and Silhouette Score to determine optimal cluster count.

# In[119]:


# 1. Setup k-Means and calculate Elbow Method and Silhouette Score
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

cluster_features = feature_cols.copy()
X_cluster = df_final[cluster_features].copy()

# K-Means is distance-based, so reduce extreme skew before robust scaling.
for col in skewed_features:
    X_cluster[col] = np.log1p(X_cluster[col].clip(lower=0))

cluster_preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])
X_cluster_scaled = pd.DataFrame(
    cluster_preprocessor.fit_transform(X_cluster),
    columns=cluster_features,
    index=X_cluster.index
)

inertia = []
silhouette_scores = []
K_range = range(2, 11)

# Apply the algorithm for multiple possible clusters to find the best one
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_cluster_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_cluster_scaled, kmeans.labels_))


# In[120]:


# 2. Determine the optimal number of clusters using an interactive plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(K_range), y=inertia, mode='lines+markers', name='Inertia (Elbow)', yaxis='y1',
    line=dict(color=GALAXY_ACCENT3, width=3), marker=dict(color=GALAXY_ACCENT4, size=8)
))
fig.add_trace(go.Scatter(
    x=list(K_range), y=silhouette_scores, mode='lines+markers', name='Silhouette Score', yaxis='y2',
    line=dict(color=GALAXY_ACCENT5, width=3), marker=dict(color=GALAXY_ACCENT2, size=8)
))

fig.update_layout(
    title='Determining the Number of Clusters (Elbow Method & Silhouette Score)',
    xaxis=dict(title='Number of Clusters (k)', dtick=1),
    yaxis=dict(title='Inertia Value', side='left'),
    yaxis2=dict(title='Silhouette Score', side='right', overlaying='y', showgrid=False, color=GALAXY_FG),
    legend=dict(x=0.75, y=1.1),
    height=500,
)
apply_galaxy_plotly(fig)
fig.show()


# In[121]:


# 3. Apply k-Means with the optimal number of clusters and save the results
optimal_k = K_range[silhouette_scores.index(max(silhouette_scores))]
print(f"Optimal k by silhouette score: {optimal_k}")

final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
cluster_labels = final_kmeans.fit_predict(X_cluster_scaled)

df_clustered = X_cluster_scaled.copy()
df_clustered['Cluster'] = cluster_labels

# Write cluster data to a CSV file
output_cluster_file = "clustering_results_advanced.csv"
df_clustered.to_csv(output_cluster_file, index=False)


# In[122]:


# 4. Apply Dimensionality Reduction (PCA) and Plot 3D Interactive Scatter
from sklearn.decomposition import PCA

# Reduce dimensions to 3
pca = PCA(n_components=3, random_state=42)
pca_components = pca.fit_transform(X_cluster_scaled)

# Add PCA components to the clustered dataframe for visualization
df_clustered['PCA_1'] = pca_components[:, 0]
df_clustered['PCA_2'] = pca_components[:, 1]
df_clustered['PCA_3'] = pca_components[:, 2]

# Convert 'Cluster' to a string column for discrete colors
df_clustered['Cluster_str'] = 'Cluster ' + df_clustered['Cluster'].astype(str)

# Create an interactive 3D scatter plot
fig_3d = px.scatter_3d(
    df_clustered,
    x='PCA_1',
    y='PCA_2',
    z='PCA_3',
    color='Cluster_str',
    color_discrete_sequence=GALAXY_PALETTE,
    title='Interactive 3D K-Means Components Visualization (PCA on log/robust-scaled features)',
    opacity=0.72,
    labels={'Cluster_str': 'Assigned Cluster'},
    height=700,
)

# Improve layout
fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=50))
apply_galaxy_plotly(fig_3d)
fig_3d.show()


# In[123]:


explained_variance = sum(pca.explained_variance_ratio_) * 100
cluster_sizes = df_clustered['Cluster'].value_counts().sort_index()

cluster_summary = pd.DataFrame(cluster_sizes).reset_index()
cluster_summary.columns = ['Cluster', 'Count']
cluster_summary['Percentage (%)'] = (cluster_summary['Count'] / len(df_clustered) * 100).round(1)

print(f"PCA explained variance in first 3 components: {explained_variance:.1f}%")
display(cluster_summary)


# ## 9. Association Rules (Apriori)

# In[124]:


# Association rules use only physical/observational fields plus disposition.
# koi_score is excluded because it is derived from vetting and leaks disposition-like information.
df_ar = df[['koi_period', 'koi_prad', 'koi_steff', 'koi_duration', 'koi_depth', 'koi_disposition']].copy()
df_ar = df_ar.dropna()


# In[125]:


# Period
df_ar['period_cat'] = pd.qcut(df_ar['koi_period'], q=3,
                             labels=['Short Period', 'Medium Period', 'Long Period'])

# Planet Radius
df_ar['radius_cat'] = pd.qcut(df_ar['koi_prad'], q=3,
                             labels=['Small Radius', 'Medium Radius', 'Large Radius'])

# Star Temperature
df_ar['temp_cat'] = pd.qcut(df_ar['koi_steff'], q=3,
                           labels=['Cool Star', 'Warm Star', 'Hot Star'])

# Transit Duration
df_ar['duration_cat'] = pd.qcut(df_ar['koi_duration'], q=3,
                               labels=['Short Duration', 'Medium Duration', 'Long Duration'])

# Transit Depth
df_ar['depth_cat'] = pd.qcut(df_ar['koi_depth'], q=3,
                            labels=['Shallow Transit', 'Medium Transit', 'Deep Transit'])


# In[126]:


df_ready = df_ar[['period_cat', 'radius_cat', 'temp_cat', 'duration_cat', 'depth_cat', 'koi_disposition']]

df_encoded = pd.get_dummies(df_ready, dtype=bool)


# In[127]:


frequent_itemsets = apriori(df_encoded,
                           min_support=0.1,
                           use_colnames=True)

frequent_itemsets.sort_values(by='support', ascending=False).head()


# In[128]:


from mlxtend.frequent_patterns import association_rules

rules = association_rules(frequent_itemsets,
                          metric="confidence",
                          min_threshold=0.6)

rules.head()


# In[129]:


rules_strong = rules[(rules['lift'] > 1.2) & (rules['confidence'] > 0.7)]

rules_strong.sort_values(by='lift', ascending=False).head(10)


# In[130]:


top_rules_display = rules_strong.sort_values(by='lift', ascending=False).head(5)
display(top_rules_display[['antecedents', 'consequents', 'confidence', 'lift']])


# In[131]:


import matplotlib.pyplot as plt

# Top 10 itemsets
top_items = frequent_itemsets.sort_values(by='support', ascending=False).head(10).copy()
top_items['itemsets_label'] = top_items['itemsets'].apply(lambda items: ', '.join(sorted(items)))

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_items['itemsets_label'], top_items['support'],
        color=GALAXY_ACCENT3, edgecolor=GALAXY_FG, alpha=0.88)
ax.invert_yaxis()
ax.set_xlabel('Support')
ax.set_ylabel('Itemsets')
ax.set_title('Top Frequent Itemsets')
apply_galaxy_axes(ax)
plt.tight_layout()
plt.show()


# In[132]:


fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(rules['confidence'], rules['lift'],
           color=GALAXY_ACCENT2, edgecolor=GALAXY_ACCENT4, alpha=0.78, s=48)
ax.set_xlabel('Confidence')
ax.set_ylabel('Lift')
ax.set_title('Association Rules (Confidence vs Lift)')
apply_galaxy_axes(ax)
plt.tight_layout()
plt.show()


# In[133]:


import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

# Keep the strongest 5 rules so the network stays readable.
top_rules = rules_strong.sort_values(by='lift', ascending=False).head(5)

fig, ax = plt.subplots(figsize=(10, 7))

if top_rules.empty:
    ax.text(0.5, 0.5, 'No rules met the strong-rule thresholds',
            ha='center', va='center', color=GALAXY_FG, fontsize=12, fontweight='bold')
else:
    for _, row in top_rules.iterrows():
        for ant in row['antecedents']:
            for cons in row['consequents']:
                G.add_edge(ant, cons, weight=round(row['lift'], 2))

    # Stable layout for repeatable visual output.
    pos = nx.spring_layout(G, k=0.75, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=2300, node_color=GALAXY_ACCENT1,
                           edgecolors=GALAXY_ACCENT4, linewidths=1.5, alpha=0.92, ax=ax)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20,
                           edge_color=GALAXY_ACCENT3, width=1.8, alpha=0.82, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color=GALAXY_FG, ax=ax)

    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8,
                                 font_color=GALAXY_ACCENT4, ax=ax)

ax.set_title('Association Rules Network')
ax.axis('off')
apply_galaxy_axes(ax)
plt.tight_layout()
plt.show()


# 
# ## Association Rules Findings
# 
# The Apriori section now excludes `koi_score` to avoid disposition leakage. The remaining rules should be read as exploratory co-occurrence patterns among physical transit bins and disposition labels, not as supervised prediction thresholds.
# 
