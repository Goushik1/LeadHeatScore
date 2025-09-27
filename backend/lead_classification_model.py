import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_auc_score
from xgboost import XGBClassifier
import os

# Load synthetic dataset
train_path = os.path.join("data", "leads_train.csv")
val_path = os.path.join("data", "leads_val.csv")
test_path = os.path.join("data", "leads_test.csv")

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

# Categorical features
cat_features = ["source", "campaign", "region", "role", "prior_course_interest", "webinar_attended"]

# One-hot encode categorical features
train_encoded = pd.get_dummies(train_df, columns=cat_features, drop_first=True)
val_encoded = pd.get_dummies(val_df, columns=cat_features, drop_first=True)
test_encoded = pd.get_dummies(test_df, columns=cat_features, drop_first=True)

# Align columns
feature_cols = [col for col in train_encoded.columns if col not in ["lead_id", "label_conversion", "label_class"]]
train_encoded = train_encoded.reindex(columns=feature_cols + ["label_conversion"], fill_value=0)
val_encoded = val_encoded.reindex(columns=feature_cols + ["label_conversion"], fill_value=0)
test_encoded = test_encoded.reindex(columns=feature_cols + ["label_conversion"], fill_value=0)

# Features and target
X_train = train_encoded[feature_cols]
y_train = train_encoded["label_conversion"]
X_val = val_encoded[feature_cols]
y_val = val_encoded["label_conversion"]
X_test = test_encoded[feature_cols]
y_test = test_encoded["label_conversion"]

# Scale numerical features
num_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Convert to float before scaling
X_train[num_features] = X_train[num_features].astype(float)
X_val[num_features] = X_val[num_features].astype(float)
X_test[num_features] = X_test[num_features].astype(float)

scaler = StandardScaler()
X_train.loc[:, num_features] = scaler.fit_transform(X_train[num_features])
X_val.loc[:, num_features] = scaler.transform(X_val[num_features])
X_test.loc[:, num_features] = scaler.transform(X_test[num_features])

# Logistic Regression (with calibration)
lr = LogisticRegression(class_weight='balanced', max_iter=2000, solver='saga', random_state=42)
cal_lr = CalibratedClassifierCV(lr, method="sigmoid", cv=5)
cal_lr.fit(X_train, y_train)

# XGBoost Classifier
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42
)
xgb.fit(X_train, y_train)

# Lead classification thresholds
def classify_lead(prob, hot_thresh=0.7, warm_thresh=0.4):
    if prob >= hot_thresh:
        return "Hot"
    elif prob >= warm_thresh:
        return "Warm"
    else:
        return "Cold"

# Evaluate models and pick the best
models = {
    "Logistic Regression": cal_lr,
    "XGBoost": xgb
}

best_model = None
best_f1 = 0

for name, model in models.items():
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average='macro')
    print(f"\n=== {name} ===")
    print("Macro F1 Score:", f1)
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, proba))
    print(classification_report(y_test, preds))

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

print(f"\nBest model based on Macro F1: {best_model_name} ({best_f1:.2f})")

# Get top 3 leads per class
probs = best_model.predict_proba(X_test)[:, 1]
classes = [classify_lead(p) for p in probs]

top_df = test_df.copy()
top_df['prob'] = probs
top_df['class'] = classes

top_leads = pd.DataFrame()
for cls in ["Hot", "Warm", "Cold"]:
    cls_df = top_df[top_df['class'] == cls].sort_values(by='prob', ascending=False).head(3)
    top_leads = pd.concat([top_leads, cls_df])

print("\nTop 3 leads per class based on probability:")
print(top_leads[['lead_id', 'class', 'prob']])

# API-style scoring function
def score_lead(lead_json):
    lead_df = pd.DataFrame([lead_json])
    lead_df_encoded = pd.get_dummies(lead_df, columns=cat_features, drop_first=True)
    lead_df_encoded = lead_df_encoded.reindex(columns=feature_cols, fill_value=0)
    lead_df_encoded[num_features] = scaler.transform(lead_df_encoded[num_features])
    prob = best_model.predict_proba(lead_df_encoded)[:, 1][0]
    label = classify_lead(prob)
    # Feature importance for LR or XGB
    if best_model_name == "Logistic Regression":
        coef = best_model.base_estimator_.coef_[0]
        top_feats = sorted(zip(feature_cols, coef), key=lambda x: abs(x[1]), reverse=True)[:5]
    else:  # XGBoost
        coef = best_model.feature_importances_
        top_feats = sorted(zip(feature_cols, coef), key=lambda x: x[1], reverse=True)[:5]
    return {"class": label, "prob": prob, "top_features": top_feats}
