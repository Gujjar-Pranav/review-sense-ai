import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import OUTPUTS_DIR, MODEL_PATH, TEST_SIZE, RANDOM_STATE
from src.data_load import load_dataset
from src.preprocess import preprocess_text
from src.modeling_compare import compare_tfidf, compare_bert
from src.calibrate_train import train_calibrated_svm
from src.error_analysis import misclassified_report
from src.utils import save_joblib, save_json, ensure_dir

def main():
    ensure_dir(OUTPUTS_DIR)
    df = load_dataset()

    # Clean text for EDA + modeling
    df["clean_review"] = df["review"].apply(preprocess_text)

    X = df["review"].astype(str)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train_clean = X_train.apply(preprocess_text)
    X_test_clean = X_test.apply(preprocess_text)

    # 1) Compare models
    tfidf_df = compare_tfidf(X_train_clean, X_test_clean, y_train, y_test)
    bert_df = compare_bert(X_train_clean, X_test_clean, y_train, y_test)

    results_df = pd.concat([tfidf_df, bert_df], ignore_index=True)
    results_df = results_df.sort_values(by=["F1", "Accuracy"], ascending=False).reset_index(drop=True)
    results_df.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)
    print("\nSaved model comparison to outputs/reports/model_comparison.csv")
    print("\nTop models:\n", results_df.head(5))

    # 2) Train final calibrated model (your current best choice)
    calibrated_model, y_pred, y_proba, metrics = train_calibrated_svm(
        X_train_clean, y_train, X_test_clean, y_test
    )

    save_json(metrics, OUTPUTS_DIR / "calibrated_metrics.json")
    save_joblib(calibrated_model, MODEL_PATH)
    print(f"\nSaved calibrated model to: {MODEL_PATH}")

    # 3) Misclassification insights
    df_all, mis, summary = misclassified_report(X_test, X_test_clean, y_test, y_pred, y_proba)
    mis.to_csv(OUTPUTS_DIR / "misclassified.csv", index=False)
    save_json(summary, OUTPUTS_DIR / "misclassified_summary.json")

    print("\nMisclassification summary:")
    print(summary)

    print("\nSample confidence scores (P[Positive]) first 5:")
    print(y_proba[:5])

if __name__ == "__main__":
    main()
