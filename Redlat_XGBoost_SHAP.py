import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
#pip install shap
import shap

def train_evaluate_and_plot_roc(df, target_column, feature_sets, parameter_grid, n_splits=7):
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize plot
    plt.figure(figsize=(10, 6))
    
    metrics_list = []
    shap_values_list = []

    for feature_columns in feature_sets:
        # Initialize dictionaries to store evaluation metrics
        metrics = {
            'Feature Set': [', '.join(feature_columns)],
            'AUC': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1 Score': [],
            'Specificity': [],
            'NPV': []
        }
        
        # Extract features and target
        X = df[feature_columns]
        y = df[target_column]
        
        # Standardize features
        X_scaled = scaler.fit_transform(X)
        
        # Initialize lists to store evaluation metrics
        auc_scores = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        specificity_scores = []
        npv_scores = []
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        # Initialize SHAP explainer
        shap_explainer = None
        
        # Iterate through folds
        for train_index, test_index in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Initialize GridSearchCV
            grid_search = GridSearchCV(xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid=parameter_grid, cv=7, scoring='roc_auc')
            grid_search.fit(X_train, y_train)
            
            # Best XGBoost model
            best_xgb_model = grid_search.best_estimator_
            
            # Initialize SHAP explainer once we have the best model
            if shap_explainer is None:
                shap_explainer = shap.TreeExplainer(best_xgb_model)
            
            # Compute SHAP values for the test set
            shap_values = shap_explainer.shap_values(X_test)
            shap_values_list.append((shap_values, X_test, feature_columns))
            
            # Predict probabilities
            y_pred_prob = best_xgb_model.predict_proba(X_test)[:, 1]
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            
            # Compute AUC
            roc_auc = roc_auc_score(y_test, y_pred_prob)
            auc_scores.append(roc_auc)
            
            # Compute other evaluation metrics
            y_pred = best_xgb_model.predict(X_test)
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred))
            recall_scores.append(recall_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))
            
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp)
            npv = tn / (tn + fn)
            specificity_scores.append(specificity)
            npv_scores.append(npv)
        
        # Compute average ROC AUC
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.mean(auc_scores)
        
        # Plot ROC curve
        plt.plot(mean_fpr, mean_tpr, label=f'{", ".join(feature_columns)} (AUC = {mean_auc:.2f})')
        
        # Calculate mean scores
        metrics['AUC'].append(mean_auc)
        metrics['Accuracy'].append(np.mean(accuracy_scores))
        metrics['Precision'].append(np.mean(precision_scores))
        metrics['Recall'].append(np.mean(recall_scores))
        metrics['F1 Score'].append(np.mean(f1_scores))
        metrics['Specificity'].append(np.mean(specificity_scores))
        metrics['NPV'].append(np.mean(npv_scores))
        
        metrics_list.append(metrics)
    
    # Plot settings
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average ROC Curves for Different Feature Sets')
    plt.legend(loc='lower right')
    plt.show()
    
    # Create DataFrame from metrics dictionary
    metrics_df = pd.concat([pd.DataFrame(m) for m in metrics_list], ignore_index=True)
    
    # Plot SHAP summary for each feature set
    for shap_values, X_test, feature_columns in shap_values_list:
        shap.summary_plot(shap_values, X_test, feature_names=feature_columns)
    
    return metrics_df

#open database:
bio = pd.read_excel('Redlat_all_patients.xlsx')
lista_con_nan = list(bio.record_id)
lista_sin_nan = [x for x in lista_con_nan if isinstance(x, str)]
print(len(np.unique(lista_sin_nan)))
bio;
#fill patient REdlat ID info 
bio['record_id'] = bio.groupby('ID')['record_id'].fillna(method='ffill')
bio
#processing df:
bio['Average count'] = bio['Average count'].astype(str)
bio
# Use groupby and apply a custom function to handle duplicates
def handle_duplicates(series):
    return ','.join(series.unique())
df = bio.groupby(['record_id', 'Analyte'])['Average count'].agg(handle_duplicates).unstack('Analyte')
df = pd.merge(df, bio[['record_id', 'diagnosis']], on='record_id', how='left')
df = df.drop_duplicates()
df
# Reset the index after dropping duplicates
df.reset_index(drop=True, inplace=True)
df
# Specify the column names to convert to numeric
columns_to_convert = ['AB140P', 'AB142P', 'NFL-BL', 'pT181P']
# Convert the specified columns to numeric types
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, handle errors by setting them to NaN
#ratio ab42/40
df['AB142P/AB140P'] = df['AB142P'] / df['AB140P']
#ordering
df = df[['record_id', 'AB140P', 'AB142P', 'AB142P/AB140P', 'pT181P', 'NFL-BL', 'diagnosis']]
df

#open database demographics redlat
demog = pd.read_excel('database_SDH - Joaquin.xlsx')
demog = demog[['record_id', 'edad', 'sexo_1M_2F', 'años_educacion']]
demog
#merge with biomarkers
df = pd.merge(demog, df, on='record_id', how='inner')
df

####### APOE #############
#merge with APOE risk
apoe=pd.read_excel('ApoE Redlat.xlsx')
df = pd.merge(df, apoe, on="record_id", how="left")
df = df.drop('APOE genotype', axis=1)

###MRI data
#open fmri data
#fmri = pd.read_excel('redlat_fmri.xlsx')
#fmri['record_id'] = fmri['record_id'].str.replace('sub-', '')
#merge with biomaerkes, sex, age and sex
#df = pd.merge(fmri, df, on='record_id', how='inner')
#df

########################################
##CONTROL X AD
df_cn_vs = df[(df.diagnosis == 'CN') | (df.diagnosis == 'AD')]
df_cn_vs = df_cn_vs.copy()
df_cn_vs['diagnosis'] = df['diagnosis'].map({'CN': 0, 'AD': 1})
df_cn_vs.dropna(inplace=True)
df_cn_vs.reset_index(drop=True, inplace=True)
df_cn_vs.shape
diagnosis_counts = df_cn_vs['diagnosis'].value_counts()

#rename columns to english
df_cn_vs.rename(columns={'edad': 'age', 'sexo_1M_2F': 'sex', 'años_educacion': 'education','AB140P': 'Ab40','AB142P': 'Ab42', 'AB142P/AB140P': 'Ab ratio', 'NFL-BL': 'Nfl' }, inplace=True)


# Display the counts
print("Number of instances of CN:", diagnosis_counts.get(0, 0))
print("Number of instances of AD:", diagnosis_counts.get(1, 0))

#without APOE data:
# Number of instances of CN: 244
# Number of instances of AD: 246    

##with APOE data
# Number of instances of CN: 131
# Number of instances of AD: 160

#with MRI data
#CN = 113, AD = 133

##CONTROL X FTD
df_cn_vs = df[(df.diagnosis == 'CN') | (df.diagnosis == 'FTD')]
df_cn_vs = df_cn_vs.copy()
df_cn_vs['diagnosis'] = df['diagnosis'].map({'CN': 0, 'FTD': 1})
df_cn_vs.dropna(inplace=True)
df_cn_vs.reset_index(drop=True, inplace=True)
df_cn_vs.shape
diagnosis_counts = df_cn_vs['diagnosis'].value_counts()

# Display the counts
print("Number of instances of CN:", diagnosis_counts.get(0, 0))
print("Number of instances of FTD:", diagnosis_counts.get(1, 0))

## without APOE data:
# Number of instances of CN: 244
# Number of instances of FTD: 115

##with APOE
# Number of instances of CN: 131
# Number of instances of FTD: 72



#rename columns 
df_cn_vs.rename(columns={'edad': 'age', 'sexo_1M_2F': 'sex', 'años_educacion': 'education','AB140P': 'Ab40','AB142P': 'Ab42', 'AB142P/AB140P': 'Ab ratio', 'NFL-BL': 'Nfl', 'RIESGO':'APOE risk' }, inplace=True)



feature_sets = [
    #['Ab40'],
    #['Ab42'],
    #['Nfl'],
    #['pT181P'],
    #['Ab ratio'],
    #['APOE risk'],
    #['Ab ratio', 'Ab40', 'Ab42', 'Nfl','pT181P', 'Hippocampus R', 'Hippocampus L','Amygdala R','ParaHippocampal R','Frontal Sup Med L','Amygdala L'],#, 'APOE risk'],
    ['Ab ratio', 'Ab40', 'Ab42', 'Nfl','pT181P','APOE risk'],
    ['Ab ratio', 'Ab40', 'Ab42', 'Nfl','pT181P','APOE risk','age','sex','education']
    #['Hippocampus R', 'Hippocampus L','Amygdala R','ParaHippocampal R','Frontal Sup Med L','Amygdala L'],
    #['Hippocampus R', 'Hippocampus L','Amygdala R','ParaHippocampal R','Frontal Sup Med L','Amygdala L','Ab ratio', 'Ab40', 'Ab42', 'Nfl','pT181P', 'sex', 'education', 'age']
]

# Define the parameter grid for GridSearchCV for XGBoost
parameter_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 1, 5]
}

# Call the train_evaluate_and_plot_roc function with the feature sets
evaluation_metrics = train_evaluate_and_plot_roc(df_cn_vs, 'diagnosis', feature_sets, parameter_grid)
print(evaluation_metrics)


