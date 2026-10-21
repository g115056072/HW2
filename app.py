import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.ar_model import AutoReg

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
HOUSE_DATA_PATH = BASE_DIR / "data" / "modified_data.csv"
HUMAN_AI_DATA_PATH = BASE_DIR / "data" / "Human-AI Interaction.csv"

st.set_page_config(
    page_title="CRISP-DM Regression Explorer",
    page_icon="📈",
    layout="wide",
)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def detect_date_column(df: pd.DataFrame) -> str | None:
    for col in ["date", "Date", "datetime", "timestamp"]:
        if col in df.columns:
            return col
    return None


def build_regression_models(df: pd.DataFrame, target: str, dataset_name: str):
    work_df = df.copy()

    if dataset_name == "House Price Prediction":
        if "date" in work_df.columns:
            work_df["date"] = pd.to_datetime(work_df["date"], errors="coerce")
            work_df["year"] = work_df["date"].dt.year
            work_df["month"] = work_df["date"].dt.month
            work_df["day_of_month"] = work_df["date"].dt.day
        if target == "price":
            work_df = work_df.loc[work_df[target] > 0].copy()

    work_df = work_df.dropna(subset=[target]).copy()

    drop_cols = [target]
    if dataset_name == "House Price Prediction":
        for col in ["price_per_sqft", "date", "street"]:
            if col in work_df.columns:
                drop_cols.append(col)

    X = work_df.drop(columns=[c for c in drop_cols if c in work_df.columns])
    y = work_df[target]

    if y.nunique() < 2:
        raise ValueError("Target column must contain at least 2 unique values.")

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    if len(numeric_features) == 0 and len(categorical_features) == 0:
        raise ValueError("No usable features found after preprocessing.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    if len(numeric_features) == 0:
        raise ValueError("Simple linear regression requires at least one numeric predictor.")

    train_numeric = X_train[numeric_features].copy()
    train_numeric[target] = y_train.values

    single_corr = (
        train_numeric.corr(numeric_only=True)[target]
        .drop(target)
        .abs()
        .sort_values(ascending=False)
    )

    if single_corr.empty:
        raise ValueError("Unable to compute a single-feature baseline from numeric predictors.")

    best_single_feature = single_corr.index[0]

    simple_lr = LinearRegression()
    simple_lr.fit(X_train[[best_single_feature]], y_train)
    simple_pred = simple_lr.predict(X_test[[best_single_feature]])
    simple_metrics = regression_metrics(y_test, simple_pred)

    num_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    preprocessor.fit(X_train)
    transformed_train = preprocessor.transform(X_train)
    k_best = min(20, transformed_train.shape[1])

    multiple_lr = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_regression, k=k_best)),
            ("regressor", LinearRegression()),
        ]
    )

    multiple_lr.fit(X_train, y_train)
    multiple_pred = multiple_lr.predict(X_test)
    multiple_metrics = regression_metrics(y_test, multiple_pred)

    feature_names = multiple_lr.named_steps["preprocess"].get_feature_names_out()
    selector = multiple_lr.named_steps["feature_selection"]
    selected_names = feature_names[selector.get_support()]
    selected_scores = selector.scores_[selector.get_support()]

    selected_features_table = (
        pd.DataFrame({"feature": selected_names, "f_score": selected_scores})
        .sort_values("f_score", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "work_df": work_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "best_single_feature": best_single_feature,
        "simple_pred": simple_pred,
        "multiple_pred": multiple_pred,
        "simple_metrics": simple_metrics,
        "multiple_metrics": multiple_metrics,
        "selected_features_table": selected_features_table,
        "single_corr": single_corr,
    }


def run_autoregression(df: pd.DataFrame, target: str, date_col: str | None):
    if not date_col or date_col not in df.columns:
        return None

    ts_df = df[[date_col, target]].copy()
    ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
    ts_df = ts_df.dropna(subset=[date_col, target])

    if ts_df.empty:
        return None

    daily_series = (
        ts_df.groupby(date_col)[target]
        .median()
        .sort_index()
    )

    if len(daily_series) < 30:
        return None

    n = len(daily_series)
    train_end = int(n * 0.80)
    development = daily_series.iloc[:train_end]
    test_ts = daily_series.iloc[train_end:]

    val_start = int(len(development) * 0.80)
    ar_train = development.iloc[:val_start]
    ar_val = development.iloc[val_start:]

    if len(ar_train) < 8 or len(ar_val) < 5 or len(test_ts) < 5:
        return None

    lag_results = []

    for lag in range(1, 8):
        if len(ar_train) <= lag:
            continue

        model = AutoReg(ar_train, lags=lag, trend="ct", old_names=False).fit()
        forecast = model.predict(
            start=len(ar_train),
            end=len(ar_train) + len(ar_val) - 1,
            dynamic=False,
        )

        lag_results.append({"lags": lag, **regression_metrics(ar_val, forecast)})

    if len(lag_results) == 0:
        return None

    lag_results_df = pd.DataFrame(lag_results).sort_values("RMSE")
    best_lag = int(lag_results_df.iloc[0]["lags"])

    ar_model = AutoReg(
        development,
        lags=best_lag,
        trend="ct",
        old_names=False,
    ).fit()

    ar_pred = ar_model.predict(
        start=len(development),
        end=len(development) + len(test_ts) - 1,
        dynamic=False,
    )

    ar_metrics = regression_metrics(test_ts, ar_pred)

    return {
        "daily_series": daily_series,
        "development": development,
        "test_ts": test_ts,
        "ar_pred": ar_pred,
        "ar_metrics": ar_metrics,
        "lag_results": lag_results_df,
        "best_lag": best_lag,
    }


def plot_actual_vs_pred(y_true: pd.Series, y_pred: np.ndarray, title: str):
    temp = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    fig = px.scatter(temp, x="Actual", y="Predicted", opacity=0.55, title=title)

    low = float(min(temp["Actual"].min(), temp["Predicted"].min()))
    high = float(max(temp["Actual"].max(), temp["Predicted"].max()))

    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="Ideal fit",
            line={"dash": "dash"},
        )
    )

    fig.update_layout(height=500)
    return fig


def crisp_dm_tab():
    st.header("CRISP-DM Procedure")
    st.markdown(
        """
### 1) Business Understanding
- Estimate a numeric target from structured inputs.
- Compare simple linear regression, multiple linear regression, and (when time is available) autoregression.
- Use held-out testing and report MAE, RMSE, and $R^2$.

### 2) Data Understanding
- Inspect shape, dtypes, missing values, duplicates, and target distribution.
- Check categorical cardinality and numerical correlations.

### 3) Data Preparation
- Remove invalid target rows.
- Engineer time features from date fields when available.
- Prevent leakage by excluding columns derived from the target.
- Use train/test split before feature selection.

### 4) Modeling
- **Simple linear regression:** choose one numeric predictor with strongest training correlation.
- **Multiple linear regression:** impute missing values, one-hot encode categories, then apply `SelectKBest(f_regression)`.
- **Autoregression:** aggregate to daily median target and perform chronological split.

### 5) Evaluation
- Compare models by MAE, RMSE, and $R^2$.
- For autoregression, evaluate on a future-only test period.

### 6) Deployment Considerations
- Add repeatable ingestion, schema checks, model versioning, drift/error monitoring, and retraining policy.
"""
    )


def app_main():
    st.title("Interactive CRISP-DM Regression Website")
    st.caption("Switch datasets, retrain models, and inspect all plots interactively.")

    tab_analysis, tab_crisp = st.tabs(["Analysis", "CRISP-DM Overview"])

    with tab_analysis:
        st.sidebar.header("Dataset Selection")
        dataset_choice = st.sidebar.selectbox(
            "Choose dataset",
            ["House Price Prediction", "Human-AI Interaction", "Custom Dataset"],
        )

        raw_df = None

        if dataset_choice == "House Price Prediction":
            raw_df = load_csv(HOUSE_DATA_PATH)
            default_target = "price" if "price" in raw_df.columns else raw_df.columns[-1]
        elif dataset_choice == "Human-AI Interaction":
            raw_df = load_csv(HUMAN_AI_DATA_PATH)
            default_target = (
                "latent_trust_score"
                if "latent_trust_score" in raw_df.columns
                else raw_df.columns[-1]
            )
        else:
            uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
            if uploaded is None:
                st.info("Upload a CSV to start. The target defaults to the last column.")
                return
            raw_df = pd.read_csv(uploaded)
            default_target = raw_df.columns[-1]

        st.subheader(f"Dataset: {dataset_choice}")
        st.write(f"Rows: {len(raw_df):,} | Columns: {len(raw_df.columns):,}")
        st.dataframe(raw_df.head(15), use_container_width=True)

        target_col = st.sidebar.selectbox(
            "Target column",
            options=list(raw_df.columns),
            index=list(raw_df.columns).index(default_target),
        )

        candidate_date_col = detect_date_column(raw_df)
        date_col = st.sidebar.selectbox(
            "Date column for autoregression (optional)",
            options=["None"] + list(raw_df.columns),
            index=(list(raw_df.columns).index(candidate_date_col) + 1)
            if candidate_date_col in raw_df.columns
            else 0,
        )
        date_col = None if date_col == "None" else date_col

        try:
            results = build_regression_models(raw_df, target_col, dataset_choice)
        except Exception as exc:
            st.error(f"Modeling failed: {exc}")
            return

        ar_results = run_autoregression(raw_df, target_col, date_col)

        evaluation_rows = [
            {"Model": "Simple Linear Regression", **results["simple_metrics"]},
            {"Model": "Multiple Linear Regression", **results["multiple_metrics"]},
        ]

        if ar_results is not None:
            evaluation_rows.append({"Model": "Autoregression", **ar_results["ar_metrics"]})

        evaluation = pd.DataFrame(evaluation_rows).sort_values("RMSE").reset_index(drop=True)

        st.subheader("Evaluation Metrics")
        st.dataframe(evaluation, use_container_width=True)

        left, right = st.columns(2)

        with left:
            st.markdown("#### Final Plot 1 — Target Distribution")
            fig_target = px.histogram(
                results["work_df"],
                x=target_col,
                nbins=60,
                title=f"Distribution of {target_col}",
            )
            fig_target.update_layout(height=450)
            st.plotly_chart(fig_target, use_container_width=True)

        with right:
            st.markdown("#### Final Plot 5 — RMSE Comparison")
            fig_rmse = px.bar(
                evaluation,
                x="Model",
                y="RMSE",
                title="RMSE Comparison",
                text_auto=".2f",
            )
            fig_rmse.update_layout(height=450)
            st.plotly_chart(fig_rmse, use_container_width=True)

        st.markdown("#### Final Plot 2 — Simple Linear Regression: Actual vs Predicted")
        st.plotly_chart(
            plot_actual_vs_pred(
                results["y_test"],
                results["simple_pred"],
                f"Simple LR (feature: {results['best_single_feature']})",
            ),
            use_container_width=True,
        )

        st.markdown("#### Final Plot 3 — Multiple Linear Regression: Actual vs Predicted")
        st.plotly_chart(
            plot_actual_vs_pred(
                results["y_test"],
                results["multiple_pred"],
                "Multiple Linear Regression",
            ),
            use_container_width=True,
        )

        st.markdown("#### Final Plot 6 — Residual Distributions")
        simple_residuals = results["y_test"] - results["simple_pred"]
        multiple_residuals = results["y_test"] - results["multiple_pred"]

        res_df = pd.DataFrame(
            {
                "Residual": pd.concat(
                    [
                        pd.Series(simple_residuals, name="Residual"),
                        pd.Series(multiple_residuals, name="Residual"),
                    ],
                    ignore_index=True,
                ),
                "Model": (["Simple LR"] * len(simple_residuals))
                + (["Multiple LR"] * len(multiple_residuals)),
            }
        )

        fig_res = px.histogram(
            res_df,
            x="Residual",
            color="Model",
            barmode="overlay",
            opacity=0.55,
            nbins=60,
            title="Residual Distributions",
        )
        fig_res.add_vline(x=0, line_dash="dash")
        fig_res.update_layout(height=450)
        st.plotly_chart(fig_res, use_container_width=True)

        st.markdown("#### Feature Selection Outputs")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Top single-feature correlations (training only)")
            st.dataframe(results["single_corr"].head(20).to_frame("|correlation|"), use_container_width=True)
        with c2:
            st.write("Selected encoded features for multiple regression")
            st.dataframe(results["selected_features_table"], use_container_width=True)

        st.markdown("#### Final Plot 4 — Autoregression: Actual vs Predicted Over Time")
        if ar_results is None:
            st.info("Autoregression is not available for the current dataset/columns (need enough valid date-based observations).")
        else:
            ts_plot_df = pd.DataFrame(
                {
                    "Date": list(ar_results["development"].index)
                    + list(ar_results["test_ts"].index)
                    + list(ar_results["ar_pred"].index),
                    "Value": list(ar_results["development"].values)
                    + list(ar_results["test_ts"].values)
                    + list(ar_results["ar_pred"].values),
                    "Series": (["Development data"] * len(ar_results["development"]))
                    + (["Actual test"] * len(ar_results["test_ts"]))
                    + ([f"AR({ar_results['best_lag']}) forecast"] * len(ar_results["ar_pred"])),
                }
            )

            fig_ar = px.line(
                ts_plot_df,
                x="Date",
                y="Value",
                color="Series",
                title="Autoregression on Daily Median Target",
            )
            fig_ar.add_vline(x=ar_results["test_ts"].index[0], line_dash="dash")
            fig_ar.update_layout(height=500)
            st.plotly_chart(fig_ar, use_container_width=True)

            st.write("Lag-order validation table")
            st.dataframe(ar_results["lag_results"], use_container_width=True)

    with tab_crisp:
        crisp_dm_tab()


if __name__ == "__main__":
    app_main()
