from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


OPENML_DATA_ID = 42876


@st.cache_data(show_spinner=False)
def load_workers_compensation_openml() -> pd.DataFrame:
    data = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, parser="auto")
    return data.frame.copy()


def _try_parse_datetime_cols(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    dt_cols = []
    for c in out.columns:
        if any(k in str(c).lower() for k in ["date", "time", "datetime"]):
            try:
                parsed = pd.to_datetime(out[c], errors="coerce", infer_datetime_format=True)
                if parsed.notna().mean() >= 0.5:
                    out[c] = parsed
                    dt_cols.append(c)
            except Exception:
                pass
    return out, dt_cols


def _add_datetime_features(df: pd.DataFrame, dt_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in dt_cols:
        s = out[c]
        out[f"{c}_year"] = s.dt.year
        out[f"{c}_month"] = s.dt.month
        out[f"{c}_day"] = s.dt.day
        out[f"{c}_hour"] = s.dt.hour
        out[f"{c}_dayofweek"] = s.dt.dayofweek
    return out


def _pick_default_target(df: pd.DataFrame) -> str:
    preferred = [
        "UltimateIncurredClaimCost",
        "TotalClaimCost",
        "ClaimCost",
        "IncurredClaimCost",
        "Settlement",
        "target",
    ]
    for c in preferred:
        if c in df.columns:
            return c

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return numeric_cols[-1]
    return df.columns[-1]


def _prepare_xy(
    df_raw: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    df = df_raw.copy()

    df, dt_cols = _try_parse_datetime_cols(df)
    df = _add_datetime_features(df, dt_cols)
    df = df.drop(columns=dt_cols, errors="ignore")

    y = pd.to_numeric(df[target], errors="coerce")
    X = df.drop(columns=[target])

    X = X.loc[y.notna()].copy()
    y = y.loc[y.notna()].copy()

    cat_cols = [c for c in X.columns if (X[c].dtype == "object") or str(X[c].dtype).startswith("category")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    for c in num_cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    for c in cat_cols:
        X[c] = X[c].astype(str)
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].mode(dropna=True).iloc[0] if not X[c].mode(dropna=True).empty else "unknown")

    meta = {"cat_cols": cat_cols, "num_cols": num_cols}
    return X, y, meta


def _fit_encoders_and_scale(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
):
    encoders: dict[str, LabelEncoder] = {}
    Xtr = X_train.copy()
    Xte = X_test.copy()

    for c in cat_cols:
        le = LabelEncoder()
        Xtr[c] = Xtr[c].astype(str).fillna("unknown")
        Xte[c] = Xte[c].astype(str).fillna("unknown")

        le.fit(pd.concat([Xtr[c], Xte[c]], axis=0).astype(str))
        Xtr[c] = le.transform(Xtr[c].astype(str))
        Xte[c] = le.transform(Xte[c].astype(str))
        encoders[c] = le

    scaler = StandardScaler()
    Xtr_num = scaler.fit_transform(Xtr[num_cols]) if num_cols else np.empty((len(Xtr), 0))
    Xte_num = scaler.transform(Xte[num_cols]) if num_cols else np.empty((len(Xte), 0))

    Xtr_cat = Xtr[cat_cols].to_numpy() if cat_cols else np.empty((len(Xtr), 0))
    Xte_cat = Xte[cat_cols].to_numpy() if cat_cols else np.empty((len(Xte), 0))

    Xtr_final = np.hstack([Xtr_num, Xtr_cat])
    Xte_final = np.hstack([Xte_num, Xte_cat])

    feature_names = (num_cols + cat_cols)
    return Xtr_final, Xte_final, encoders, scaler, feature_names


def _make_model(name: str):
    if name == "Линейная регрессия":
        return LinearRegression()
    if name == "Ridge":
        return Ridge(alpha=1.0, random_state=42)
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError("Неизвестная модель")


def _plot_pred_vs_true(y_true: pd.Series, y_pred: np.ndarray):
    fig = plt.figure()
    plt.scatter(y_true, y_pred, s=12)
    plt.xlabel("Фактическое значение")
    plt.ylabel("Прогноз")
    plt.title("Сравнение прогноза и факта")
    st.pyplot(fig, clear_figure=True)


def _plot_residuals(y_true: pd.Series, y_pred: np.ndarray):
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig = plt.figure()
    plt.hist(resid, bins=40)
    plt.xlabel("Остаток (факт − прогноз)")
    plt.ylabel("Количество наблюдений")
    plt.title("Распределение остатков")
    st.pyplot(fig, clear_figure=True)


def analysis_and_model_page():
    st.title("Анализ и модель")

    with st.expander("Загрузка датасета", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            use_openml = st.radio(
                "Источник данных",
                ["OpenML (Workers Compensation)", "CSV"],
                horizontal=True,
                index=0,
            )
        with col2:
            if st.button("Сбросить данные/модель"):
                for k in ["df", "artifacts"]:
                    st.session_state.pop(k, None)
                st.rerun()

    df = st.session_state.get("df")
    if df is None:
        if use_openml == "OpenML (Workers Compensation)":
            with st.spinner("Загружаю датасет из OpenML..."):
                df = load_workers_compensation_openml()
            st.session_state["df"] = df
        else:
            uploaded = st.file_uploader("Загрузите CSV", type=["csv"])
            if uploaded is None:
                st.info("Загрузите CSV или выберите OpenML.")
                return
            try:
                df = pd.read_csv(uploaded)
            except Exception:
                uploaded.seek(0)
                df = pd.read_csv(uploaded, sep=";")
            st.session_state["df"] = df

    st.subheader("Просмотр данных")
    st.dataframe(df.head(30), use_container_width=True)
    st.caption(f"Строк: {len(df):,} | Столбцов: {df.shape[1]}")

    default_target = _pick_default_target(df)
    target = st.selectbox("Целевая переменная (y)", options=list(df.columns), index=list(df.columns).index(default_target))

    X, y, meta = _prepare_xy(df, target)
    if len(X) < 200:
        st.warning(f"После очистки осталось мало строк: {len(X)}")

    with st.expander("Параметры обучения", expanded=True):
        model_name = st.selectbox("Модель", ["Линейная регрессия", "Ridge", "RandomForest"], index=2)
        test_size = st.slider("Доля тестовой выборки", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("random_state", min_value=0, max_value=10_000, value=42, step=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(random_state)
    )

    if st.button("Обучить модель", type="primary"):
        with st.spinner("Обучение..."):
            Xtr, Xte, encoders, scaler, feature_names = _fit_encoders_and_scale(
                X_train, X_test, meta["cat_cols"], meta["num_cols"]
            )

            model = _make_model(model_name)
            model.fit(Xtr, y_train)
            y_pred = model.predict(Xte)

        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(root_mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        st.session_state["artifacts"] = {
            "model_name": model_name,
            "model": model,
            "encoders": encoders,
            "scaler": scaler,
            "feature_names": feature_names,
            "cat_cols": meta["cat_cols"],
            "num_cols": meta["num_cols"],
            "target": target,
        }

        st.success("Модель обучена")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{mae:,.4f}")
        c2.metric("MSE", f"{mse:,.4f}")
        c3.metric("RMSE", f"{rmse:,.4f}")
        c4.metric("R²", f"{r2:,.4f}")

        st.subheader("Графики качества")
        _plot_pred_vs_true(y_test, y_pred)
        _plot_residuals(y_test, y_pred)

        st.subheader("Важность признаков")
        try:
            def _predict_for_pi(X_df: pd.DataFrame) -> np.ndarray:
                X_df = X_df.copy()
                for c in meta["num_cols"]:
                    X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
                    X_df[c] = X_df[c].fillna(X_train[c].median() if c in X_train.columns else 0)

                for c in meta["cat_cols"]:
                    X_df[c] = X_df[c].astype(str).fillna("unknown")
                    le = encoders[c]
                    known = set(le.classes_)
                    X_df[c] = X_df[c].where(X_df[c].isin(known), other=le.classes_[0])
                    X_df[c] = le.transform(X_df[c].astype(str))

                X_num = scaler.transform(X_df[meta["num_cols"]]) if meta["num_cols"] else np.empty((len(X_df), 0))
                X_cat = X_df[meta["cat_cols"]].to_numpy() if meta["cat_cols"] else np.empty((len(X_df), 0))
                X_final = np.hstack([X_num, X_cat])
                return model.predict(X_final)

            # permutation_importance требует estimator с predict; оборачиваем через простую прокладку
            class _Wrapper:
                def __init__(self, predict_fn):
                    self._predict_fn = predict_fn
                def fit(self, X, y):  # не используется
                    return self
                def predict(self, X):
                    return self._predict_fn(X)

            wrapper = _Wrapper(_predict_for_pi)
            r = permutation_importance(
                wrapper,
                X_test,
                y_test,
                n_repeats=5,
                random_state=42,
                scoring="neg_mean_absolute_error",
            )

            imp = pd.DataFrame(
                {"Признак": meta["num_cols"] + meta["cat_cols"], "Важность": r.importances_mean}
            ).sort_values("Важность", ascending=False)

            st.dataframe(imp.head(30), use_container_width=True)

            fig = plt.figure()
            top = imp.head(20).iloc[::-1]
            plt.barh(top["Признак"], top["Важность"])
            plt.xlabel("Среднее ухудшение качества (−MAE)")
            plt.title("Топ-20 признаков по важности")
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.warning(f"Не удалось посчитать важность признаков: {e}")

    artifacts = st.session_state.get("artifacts")
    if artifacts:
        st.divider()
        st.subheader("Прогноз по одной записи")

        feat_names = artifacts["feature_names"]
        cat_cols = artifacts["cat_cols"]
        num_cols = artifacts["num_cols"]

        default_row = {}
        for c in feat_names:
            if c in X.columns and X[c].notna().any():
                default_row[c] = X[c].dropna().iloc[0]
            else:
                default_row[c] = ""

        with st.form("predict_form"):
            row = {c: st.text_input(c, value=str(default_row.get(c, ""))) for c in feat_names}
            submitted = st.form_submit_button("Спрогнозировать")

        if submitted:
            x = pd.DataFrame([row])

            for c in num_cols:
                x[c] = pd.to_numeric(x[c], errors="coerce")
                if x[c].isna().any():
                    x[c] = x[c].fillna(X[c].median() if c in X.columns else 0)

            for c in cat_cols:
                x[c] = x[c].astype(str).fillna("unknown")
                le = artifacts["encoders"][c]
                known = set(le.classes_)
                x[c] = x[c].where(x[c].isin(known), other=le.classes_[0])
                x[c] = le.transform(x[c].astype(str))

            scaler = artifacts["scaler"]
            X_num = scaler.transform(x[num_cols]) if num_cols else np.empty((len(x), 0))
            X_cat = x[cat_cols].to_numpy() if cat_cols else np.empty((len(x), 0))
            X_final = np.hstack([X_num, X_cat])

            model = artifacts["model"]
            pred = float(model.predict(X_final)[0])
            st.success(f"Прогноз: {pred:,.4f}")


analysis_and_model_page()