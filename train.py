import pandas as pd 
import numpy as  np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib


pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

df = pd.read_csv('german_credit_data.csv')
# print("Loaded:", df.shape)
# print(df.head())
#
# print("\nAge summary:")
# print(df["Age"].describe())

#print("\nRisk distribution:")
#print(df["Risk"].value_counts())

#print("\nDataFrame Info:")
#print(df.info())

#print("\nDescribe")
#print(df.describe(include='all').T)

#print("\n Jobs")
#print(df["Job"].unique())

df = df.dropna().reset_index(drop=True)
# Drop 'Unnamed: 0' column if it exists (note: no trailing space) and avoid printing the return value of inplace
if 'Unnamed: 0' in df.columns:
	   df.drop(columns='Unnamed: 0', inplace=True)

print(df)
print(df.columns)

#df[["Age", "Credit amount", "Duration"]].hist(bins = 20, edgecolor='black')
#plt.suptitle("Distributions of Numerical Features", fontsize=14)
#plt.show()

#plt.figure(figsize=(10,6))
#for i, col in enumerate (["Age", "Credit amount", "Duration"]):
    #plt.subplot(1, 3, i+1)
    #sns.boxplot(x="Risk", y=df[col], color = "skyblue", data=df)
    #plt.title(f"Boxplot of {col} by Risk")
#plt.tight_layout()
#plt.show()


# print("\nEntries with Duration > 60:")
#print(df.query("Duration >= 60 "))


#categorical_cols = ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]
#plt.figure(figsize=(15,10))
#for i , col in enumerate(categorical_cols):
	#plt.subplot(2, 3, i+1)
	#sns.countplot(data = df, x=col, palette = "Set2", order = df[col].value_counts().index)
	#plt.title(f"Countplot of {col} ")
	#plt.xticks(rotation=45)
#plt.tight_layout()
#plt.show()


#corr = df[["Age", "Job", "Credit amount", "Duration"]].corr()
#corr


#print(df.groupby("Job")["Credit amount"].mean())
#print(df.groupby("Sex")["Credit amount"].mean())


#print(pd.pivot_table(df, values= "Credit amount", index= "Housing", columns = "Purpose" ))

#sns.scatterplot(data=df, x="Age", y="Credit amount", hue="Sex", size = "Duration", alpha=0.7, palette="Set1")
#plt.title("Credit amount vs Age colored by Sex and sized by Duration")
#plt.show()

#df["Risk"].value_counts(normalize=True) * 100

#plt.figure(figsize=(15,5))
#for i , col in enumerate(["Age", "Credit amount", "Duration"]):
	#plt.subplot(1,3, i + 1)
	#sns.boxplot(data = df, x = "Risk", y= col, palette = "Pastel2")
	#plt.title(f"{col} by Risk")

#plt.tight_layout()
#plt.show()

#df.groupby("Risk")[["Age", "Credit amount", "Duration"]].mean()
#print(df.groupby("Risk")[["Age", "Credit amount", "Duration"]].mean())

#categorical_cols = ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]
#plt.figure(figsize=(15,10))
#for i, col in enumerate(categorical_cols):
	#plt.subplot(3,3, i + 1)
	#sns.countplot(data = df, x=col, hue="Risk", palette = "Set3", order = df[col].value_counts().index)
	#plt.title(f"{col} by Risk")
	#plt.xticks(rotation=45)
#plt.tight_layout()
#plt.show()


# Feature Engineering 
features = ["Age", "Sex", "Job", "Housing", "Saving accounts", "Checking account", "Credit amount", "Duration"]
target = "Risk"

df_model = df[features + [target]].copy()
print(df_model.head())


cat_cols = df_model.select_dtypes(include=['object']).columns.drop("Risk")
le_dict = {}
for col in cat_cols: 
	le = LabelEncoder()
	df_model[col] = le.fit_transform(df_model[col])
	le_dict[col] = le
	# Save the label encoder for future use
	joblib.dump(le, f'label_encoder_{col}.pkl')

# encode target
le_target = LabelEncoder()
df_model[target] = le_target.fit_transform(df_model[target])
joblib.dump(le_target, 'target_encoder.pkl')

print(df_model.head())

# Further data processing and model training code would go here 

# Split the data
X = df_model.drop(target, axis=1)
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
X_train.shape, X_test.shape

def train_model(model, param_grid, X_train, y_train, X_test, y_test):
	grid= GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
	grid.fit(X_train, y_train)
	model.fit(X_train, y_train)
	best_model = grid.best_estimator_
	y_pred = best_model.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	print(f"Accuracy: {acc:.4f}")
	print("Classification Report:")
	print(classification_report(y_test, y_pred))
	print("Confusion Matrix:")
	print(confusion_matrix(y_test, y_pred))
	return best_model, acc, grid.best_params_

dt = DecisionTreeClassifier(random_state=1, class_weight = "balanced")
dt_param_grid = {
	'max_depth': [3, 5, 7, 10, None],
	'min_samples_split': [2, 5, 10],
	'min_samples_leaf': [1, 2, 4]
}

best_dt, dt_acc, dt_best_params = train_model(dt, dt_param_grid, X_train, y_train, X_test, y_test)

print("Decision Tree Accuracy:", dt_acc)
print("Best Decision Tree Parameters:", dt_best_params)


rf = RandomForestClassifier(random_state=1, class_weight = "balanced")
rf_param_grid = {
	'n_estimators': [100, 200],
	'max_depth': [5, 7, 10, None],
	'min_samples_split': [2, 5, 10],
	'min_samples_leaf': [1, 2, 4]
}

best_rf, rf_acc, rf_best_params = train_model(rf, rf_param_grid, X_train, y_train, X_test, y_test)
print("Random Forest Accuracy:", rf_acc)
print("Best Random Forest Parameters:", rf_best_params)

et = ExtraTreesClassifier(random_state=1, class_weight = "balanced", n_jobs = -1)
et_param_grid = {
	'n_estimators': [100, 200],
	'max_depth': [5, 7, 10, None],
	'min_samples_split': [2, 5, 10],
	'min_samples_leaf': [1, 2, 4]
} 
best_et, et_acc, et_best_params = train_model(et, et_param_grid, X_train, y_train, X_test, y_test)
print("Extra Trees Accuracy:", et_acc) 
print("Best Extra Trees Parameters:", et_best_params)

xgb = XGBClassifier(random_state=1, scale_pos_weight =(y_train == 0).sum() / (y_train == 1).sum(), use_label_encoder=False, eval_metric='logloss')
xgb_param_grid = {
	'n_estimators': [100, 200],
	'max_depth': [3, 5, 7],
	'learning_rate': [0.01, 0.1, 0.2],
	'subsample': [0.7, 1.0],
	'colsample_bytree': [0.7, 1.0]
}

best_xgb, xgb_acc, xgb_best_params = train_model(xgb, xgb_param_grid, X_train, y_train, X_test, y_test)
print("XGBoost Accuracy:", xgb_acc)
print("Best XGBoost Parameters:", xgb_best_params) 

# After you have the best models (best_dt, best_rf, best_et, best_xgb)
# choose which model to serve and save it plus the encoders for the app
final_model = best_et  # choose whichever performs best (best_xgb, best_rf, etc.)

# Save model files the app looks for and report success/failure
for fname in ("best_model.pkl", "credit_risk_model.pkl"):
    try:
        joblib.dump(final_model, fname)
        print(f"Saved model: {fname}")
    except Exception as e:
        print(f"ERROR saving {fname}: {e}")

# Ensure all categorical label encoders are saved (already stored in le_dict)
for col, le in le_dict.items():
    try:
        joblib.dump(le, f"label_encoder_{col}.pkl")
        print(f"Saved encoder: label_encoder_{col}.pkl")
    except Exception as e:
        print(f"ERROR saving encoder for {col}: {e}")

# Also save a copy of the feature list and column order (optional but useful)
try:
    joblib.dump(features, "feature_columns.pkl")
    print("Saved feature_columns.pkl")
except Exception as e:
    print("ERROR saving feature_columns.pkl:", e)

print("TRAINING COMPLETE — check .pkl files in project folder")

