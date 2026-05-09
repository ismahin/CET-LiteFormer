from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from ..utils.io import ensure_dir, save_json
from ..utils.logger import print_section
from .feature_groups import build_feature_groups, save_feature_groups
from ..utils.plots import plot_feature_correlations


LABEL_CANDIDATES = [
    "Label",
    "label",
    "Class",
    "class",
    "Traffic Type",
    "traffic_type",
    "Category",
    "category",
    "Attack",
    "attack",
]


def _make_unique_columns(cols: Sequence[str]) -> List[str]:
    """
    Make column names unique for pandas DataFrames.

    - Repeated identical names (e.g. two 'Label' headers) become Label, Label__dup1, ...
    - pandas.read_csv often renames duplicate headers to 'Label', 'Label.1', 'Label.2', ...
      We treat 'Label.1' as the second occurrence of 'Label' when bare 'Label' is also
      present, so the user can select the last label via --label_col Label__dup1.
    """
    col_list = [str(c).strip() for c in cols]
    col_set = set(col_list)
    seen: Dict[str, int] = {}
    out: List[str] = []
    for raw in col_list:
        m = re.match(r"^(.+)\.(\d+)$", raw)
        base = m.group(1) if m and m.group(1) in col_set else raw
        if base not in seen:
            seen[base] = 0
            out.append(raw)
        else:
            seen[base] += 1
            out.append(f"{base}__dup{seen[base]}")
    return out


def load_csv_dataset(csv_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # low_memory=False to reduce mixed-type inference issues on large CSVs
    df = pd.read_csv(path, nrows=max_rows, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_arff_dataset(arff_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    path = Path(arff_path)
    if not path.exists():
        raise FileNotFoundError(f"ARFF not found: {arff_path}")

    try:
        from scipy.io import arff as scipy_arff  # type: ignore

        data, meta = scipy_arff.loadarff(str(path))
        df = pd.DataFrame(data)
        # decode bytes columns if any
        for c in df.columns:
            if df[c].dtype == object:
                if len(df) > 0 and isinstance(df[c].iloc[0], (bytes, bytearray)):
                    df[c] = df[c].apply(lambda x: x.decode("utf-8", errors="ignore") if isinstance(x, (bytes, bytearray)) else x)
        if max_rows is not None:
            df = df.iloc[:max_rows].copy()
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        try:
            import arff  # type: ignore

            with path.open("r", encoding="utf-8", errors="ignore") as f:
                obj = arff.load(f)
            df = pd.DataFrame(obj["data"], columns=[a[0] for a in obj["attributes"]])
            if max_rows is not None:
                df = df.iloc[:max_rows].copy()
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as e:
            raise RuntimeError(
                "Failed to load ARFF. Install `scipy` (preferred) or `liac-arff`."
            ) from e


def clean_dataframe(df: pd.DataFrame, drop_cols: Sequence[str] | None = None) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df.columns = _make_unique_columns(df.columns)

    # Replace inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if drop_cols:
        drop_set = {str(c).strip() for c in drop_cols}
        cols_to_drop = [c for c in df.columns if c in drop_set]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # Drop all-null columns
    all_null = [c for c in df.columns if df[c].isna().all()]
    if all_null:
        df.drop(columns=all_null, inplace=True)

    return df


def auto_detect_label_col(df: pd.DataFrame, label_col: str) -> str:
    if label_col:
        if label_col not in df.columns:
            # Case-insensitive resolution (common across public flow datasets)
            lc_map = {str(c).strip().lower(): c for c in df.columns}
            if str(label_col).strip().lower() in lc_map:
                resolved_ci = lc_map[str(label_col).strip().lower()]
                warnings.warn(f"Resolved label_col '{label_col}' to '{resolved_ci}' (case-insensitive match).")
                return resolved_ci

            # ISCXVPN2016 ARFFs often expose the label as 'class1'
            if str(label_col).strip().lower() == "label" and "class1" in df.columns:
                warnings.warn("Resolved label_col 'label' to 'class1' (ISCXVPN2016 naming).")
                return "class1"

            # Second label: pandas may still expose it as Label.1 before clean_dataframe runs;
            # after clean_dataframe it becomes Label__dup1.
            if label_col in ("Label__dup1", "label__dup1") and "Label.1" in df.columns:
                warnings.warn("Using column 'Label.1' as the second label (pandas duplicate header name).")
                return "Label.1"
            # allow "Label" to match the first of duplicates like Label__dup1
            matches = [c for c in df.columns if c == label_col or c.startswith(label_col + "__dup")]
            if not matches:
                raise ValueError(
                    f"Label column '{label_col}' not found. Available columns: {list(df.columns)[:50]}..."
                )
            resolved = matches[0]
            if resolved != label_col:
                warnings.warn(f"Resolved label_col '{label_col}' to '{resolved}' due to duplicate columns.")
            return resolved
        return label_col

    for cand in LABEL_CANDIDATES:
        if cand in df.columns:
            return cand
        matches = [c for c in df.columns if c == cand or c.startswith(cand + "__dup")]
        if matches:
            warnings.warn(f"Auto-detected label column '{cand}' resolved to '{matches[0]}' due to duplicates.")
            return matches[0]

    raise ValueError(
        "Could not auto-detect label column. Pass --label_col. "
        f"Available columns include: {list(df.columns)[:80]}..."
    )


def auto_detect_numeric_features(df: pd.DataFrame, label_col: str) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != label_col]
    if not numeric_cols:
        raise ValueError(
            "No numeric feature columns detected after cleaning. "
            "Check that the CSV/ARFF contains numeric flow features."
        )
    return numeric_cols


@dataclass
class ImputerState:
    strategy: str
    fill_values: np.ndarray  # [F]


def fit_imputer(X_train: np.ndarray, strategy: str = "median") -> ImputerState:
    if strategy != "median":
        raise ValueError(f"Unsupported missing_strategy: {strategy}")
    fill = np.nanmedian(X_train, axis=0)
    # if any feature is all-NaN, nanmedian returns NaN; set to 0.0
    fill = np.where(np.isfinite(fill), fill, 0.0)
    return ImputerState(strategy=strategy, fill_values=fill.astype(np.float32))


def apply_imputer(X: np.ndarray, state: ImputerState) -> np.ndarray:
    X2 = X.copy()
    mask = ~np.isfinite(X2)
    if mask.any():
        X2[mask] = np.take(state.fill_values, np.where(mask)[1])
    return X2


def save_imputer_state(state: ImputerState, path: str | Path) -> None:
    payload = {
        "strategy": state.strategy,
        "fill_values": state.fill_values.astype(float).tolist(),
    }
    save_json(payload, path)


def load_imputer_state(path: str | Path) -> ImputerState:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    fill = np.asarray(obj["fill_values"], dtype=np.float32)
    return ImputerState(strategy=str(obj["strategy"]), fill_values=fill)


class RobustLogIQRScaler:
    """
    Robust Log-IQR normalization fitted on training data only.

    For feature x_j:
      if min_train_j >= 0: x_trans = log1p(max(x,0))
      else:               x_trans = x

      x_scaled = (x_trans - median_train) / (IQR_train + eps)
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)
        self.use_log_mask: Optional[np.ndarray] = None  # [F] bool
        self.median_: Optional[np.ndarray] = None  # [F]
        self.q1_: Optional[np.ndarray] = None  # [F]
        self.q3_: Optional[np.ndarray] = None  # [F]

    def fit(self, X_train: np.ndarray) -> "RobustLogIQRScaler":
        if X_train.ndim != 2:
            raise ValueError("X_train must be 2D.")
        mins = np.nanmin(X_train, axis=0)
        self.use_log_mask = (mins >= 0.0)

        Xt = X_train.copy()
        if self.use_log_mask.any():
            idx = np.where(self.use_log_mask)[0]
            Xt[:, idx] = np.log1p(np.maximum(Xt[:, idx], 0.0))

        self.median_ = np.nanmedian(Xt, axis=0)
        self.q1_ = np.nanpercentile(Xt, 25, axis=0)
        self.q3_ = np.nanpercentile(Xt, 75, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.use_log_mask is None or self.median_ is None or self.q1_ is None or self.q3_ is None:
            raise RuntimeError("Scaler not fitted.")
        Xt = X.copy()
        if self.use_log_mask.any():
            idx = np.where(self.use_log_mask)[0]
            Xt[:, idx] = np.log1p(np.maximum(Xt[:, idx], 0.0))
        iqr = self.q3_ - self.q1_
        return (Xt - self.median_) / (iqr + self.eps)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, str(path))

    @staticmethod
    def load(path: str | Path) -> "RobustLogIQRScaler":
        return joblib.load(str(path))


@dataclass
class PreprocessArtifacts:
    feature_names: List[str]
    keep_mask: np.ndarray  # after constant removal [F]
    imputer: ImputerState
    scaler_path: str
    label_encoder_path: str
    label_mapping_path: str
    feature_metadata_path: str


def remove_constant_features_train_only(X_train: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # constant if max-min == 0
    mins = X_train.min(axis=0)
    maxs = X_train.max(axis=0)
    keep = (maxs - mins) > 0.0
    keep = keep.astype(bool)
    if keep.sum() == 0:
        raise ValueError("All features are constant after imputation; cannot train.")
    Xk = X_train[:, keep]
    names = [n for n, k in zip(feature_names, keep) if k]
    return Xk, keep, names


def encode_labels(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    # enforce no unseen labels
    def _transform_checked(y: np.ndarray) -> np.ndarray:
        unseen = set(pd.unique(y)) - set(le.classes_)
        if unseen:
            raise ValueError(f"Found unseen labels in split: {sorted(unseen)[:10]}")
        return le.transform(y)

    return y_train_enc, _transform_checked(y_val), _transform_checked(y_test), le


def compute_mi_prior(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    group_names: Optional[List[str]] = None,
    output_csv_path: Optional[str | Path] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    try:
        mi = mutual_info_classif(X_train, y_train, discrete_features=False, random_state=0)
        mi = np.asarray(mi, dtype=np.float32)
        denom = float(mi.max() + eps) if mi.size else 1.0
        mi_norm = (mi / denom).astype(np.float32)
    except Exception as e:
        warnings.warn(f"mutual_info_classif failed; using zeros MI prior. Reason: {e}")
        mi = np.zeros((X_train.shape[1],), dtype=np.float32)
        mi_norm = np.zeros_like(mi)

    if output_csv_path is not None:
        outp = Path(output_csv_path)
        ensure_dir(outp.parent)
        rows = []
        for i, name in enumerate(feature_names):
            rows.append(
                {
                    "feature_name": name,
                    "mi_score": float(mi[i]),
                    "mi_normalized": float(mi_norm[i]),
                    "group_name": (group_names[i] if group_names is not None else "unknown"),
                }
            )
        pd.DataFrame(rows).to_csv(outp, index=False)

    return mi_norm


def _spearman_corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman correlation between 1D x and y using rank transform.
    Robust to monotonic nonlinearities; handles constant columns.
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size != y.size:
        raise ValueError("x and y must have same length.")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    # rank (dense) using argsort twice; ties get arbitrary order but good enough for screening
    rx = x.argsort(kind="mergesort").argsort(kind="mergesort").astype(np.float64)
    ry = y.argsort(kind="mergesort").argsort(kind="mergesort").astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = (np.sqrt((rx * rx).sum()) * np.sqrt((ry * ry).sum())) + 1e-12
    return float((rx * ry).sum() / denom)


def compute_feature_correlations(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Train-only feature/target correlation screening.

    For multiclass, compute One-vs-Rest Spearman per class and take max abs corr.
    """
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.int64)
    n, f = X.shape
    classes = np.unique(y)

    corrs = np.zeros((f,), dtype=np.float32)
    if classes.size <= 2:
        yb = y.astype(np.float32)
        for j in range(f):
            corrs[j] = abs(_spearman_corr_1d(X[:, j], yb))
    else:
        # max over OVR correlations (more conservative screening)
        for j in range(f):
            best = 0.0
            xj = X[:, j]
            for c in classes:
                yc = (y == c).astype(np.float32)
                best = max(best, abs(_spearman_corr_1d(xj, yc)))
            corrs[j] = best

    df = pd.DataFrame({"feature_name": feature_names, "abs_corr": corrs})
    df.sort_values("abs_corr", ascending=False, inplace=True)
    return df


def select_features_by_correlation(
    corr_df: pd.DataFrame,
    feature_names: List[str],
    top_k: Optional[int],
    min_abs_corr: float = 0.0,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Returns:
      - keep_mask: [F] bool
      - selected_feature_names
      - corr_df annotated with 'selected'
    """
    f = len(feature_names)
    keep = np.ones((f,), dtype=bool)
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    selected = set()
    if top_k is not None and int(top_k) > 0:
        for n in corr_df["feature_name"].head(int(top_k)).tolist():
            selected.add(n)
        keep = np.array([n in selected for n in feature_names], dtype=bool)

    if min_abs_corr and float(min_abs_corr) > 0.0:
        allowed = set(corr_df[corr_df["abs_corr"] >= float(min_abs_corr)]["feature_name"].tolist())
        keep = keep & np.array([n in allowed for n in feature_names], dtype=bool)

    if keep.sum() == 0:
        # never allow empty selection: fall back to top 1
        best = corr_df.iloc[0]["feature_name"]
        keep = np.array([n == best for n in feature_names], dtype=bool)

    selected_names = [n for n, k in zip(feature_names, keep) if k]
    corr_df2 = corr_df.copy()
    corr_df2["selected"] = corr_df2["feature_name"].map(lambda n: bool(keep[name_to_idx[n]]))
    return keep, selected_names, corr_df2


def save_feature_metadata(
    output_dir: str | Path,
    dataset_path: str,
    label_col: str,
    drop_cols: Sequence[str],
    feature_names_original: List[str],
    feature_names_final: List[str],
    keep_mask: np.ndarray,
    corr_keep_mask: Optional[np.ndarray] = None,
) -> Path:
    out_dir = Path(output_dir)
    ensure_dir(out_dir)
    meta = {
        "dataset_path": dataset_path,
        "label_col": label_col,
        "drop_cols": list(drop_cols),
        "feature_names_original": feature_names_original,
        "feature_names_final": feature_names_final,
        "constant_keep_mask": keep_mask.astype(int).tolist(),
        "correlation_keep_mask": (corr_keep_mask.astype(int).tolist() if corr_keep_mask is not None else None),
        "num_features_original": len(feature_names_original),
        "num_features_final": len(feature_names_final),
    }
    path = out_dir / "feature_metadata.json"
    save_json(meta, path)
    return path


def build_preprocessed_splits(
    dataset_path: str,
    label_col: str,
    drop_cols: Sequence[str],
    test_size: float,
    val_size: float,
    stratify: bool,
    seed: int,
    max_rows: Optional[int],
    missing_strategy: str,
    normalize: str,
    remove_constant_features: bool,
    output_dir: str | Path,
    corr_enabled: bool = True,
    corr_top_k: Optional[int] = 256,
    corr_min_abs: float = 0.0,
) -> Dict[str, Any]:
    """
    Full pipeline:
      - load CSV/ARFF
      - clean df
      - detect label col (if empty)
      - select numeric features
      - split indices
      - fit imputer on train, apply to all
      - fit scaler on train, apply to all
      - remove constant features using train only, apply to all
      - label encode (fit on train)
      - save scaler, label mapping, feature metadata
    """
    path = Path(dataset_path)
    suffix = path.suffix.lower()
    if suffix == ".arff":
        df = load_arff_dataset(str(path), max_rows=max_rows)
    else:
        df = load_csv_dataset(str(path), max_rows=max_rows)

    df = clean_dataframe(df, drop_cols=drop_cols)

    resolved_label_col = auto_detect_label_col(df, label_col=label_col)
    if resolved_label_col != label_col and label_col:
        warnings.warn(f"Using resolved label column: {resolved_label_col}")

    feature_names_original = auto_detect_numeric_features(df, label_col=resolved_label_col)

    y_all = df[resolved_label_col].astype(str).to_numpy()
    X_all = df[feature_names_original].to_numpy(dtype=np.float32, copy=True)

    from .splits import build_train_val_test_split

    splits = build_train_val_test_split(
        n_samples=X_all.shape[0],
        y=y_all,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
        stratify=stratify,
    )

    X_train_raw = X_all[splits.train_idx]
    X_val_raw = X_all[splits.val_idx]
    X_test_raw = X_all[splits.test_idx]

    y_train_raw = y_all[splits.train_idx]
    y_val_raw = y_all[splits.val_idx]
    y_test_raw = y_all[splits.test_idx]

    # impute train-only
    imputer = fit_imputer(X_train_raw, strategy=missing_strategy)
    X_train = apply_imputer(X_train_raw, imputer)
    X_val = apply_imputer(X_val_raw, imputer)
    X_test = apply_imputer(X_test_raw, imputer)

    # scale train-only
    out_dir = Path(output_dir)
    ensure_dir(out_dir)
    scaler_path = out_dir / "scaler.joblib"

    if normalize == "log_iqr":
        scaler = RobustLogIQRScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        scaler.save(scaler_path)
    elif normalize in ("none", "", None):
        scaler = None
    else:
        raise ValueError(f"Unsupported normalize option: {normalize}")

    # constant feature removal train-only (after impute+scale)
    keep_mask = np.ones((len(feature_names_original),), dtype=bool)
    feature_names_final = list(feature_names_original)
    if remove_constant_features:
        X_train, keep_mask, feature_names_final = remove_constant_features_train_only(X_train, feature_names_original)
        X_val = X_val[:, keep_mask]
        X_test = X_test[:, keep_mask]

    # label encoding (fit on train)
    y_train, y_val, y_test, le = encode_labels(y_train_raw, y_val_raw, y_test_raw)

    # correlation analysis + optional selection (train-only, after leakage-safe transforms)
    corr_keep_mask = None
    if corr_enabled:
        corr_df = compute_feature_correlations(X_train=X_train, y_train=y_train, feature_names=feature_names_final)
        corr_keep_mask, selected_names, corr_df_annot = select_features_by_correlation(
            corr_df=corr_df,
            feature_names=feature_names_final,
            top_k=corr_top_k,
            min_abs_corr=corr_min_abs,
        )
        X_train = X_train[:, corr_keep_mask]
        X_val = X_val[:, corr_keep_mask]
        X_test = X_test[:, corr_keep_mask]
        feature_names_final = selected_names

        corr_csv_path = Path(output_dir) / "feature_correlations.csv"
        ensure_dir(corr_csv_path.parent)
        corr_df_annot.to_csv(corr_csv_path, index=False)
        # plot top correlations with selected highlighting
        try:
            plot_feature_correlations(corr_df_annot, Path(output_dir) / "feature_correlation_top.png", topk=60)
        except Exception:
            pass

    # persist label mapping
    label_mapping = {str(c): int(i) for i, c in enumerate(le.classes_)}
    label_mapping_path = out_dir / "label_mapping.json"
    save_json(label_mapping, label_mapping_path)

    # save label encoder via joblib for completeness
    label_encoder_path = out_dir / "label_encoder.joblib"
    joblib.dump(le, str(label_encoder_path))

    feature_metadata_path = save_feature_metadata(
        output_dir=out_dir,
        dataset_path=str(path),
        label_col=resolved_label_col,
        drop_cols=drop_cols,
        feature_names_original=feature_names_original,
        feature_names_final=feature_names_final,
        keep_mask=keep_mask,
        corr_keep_mask=corr_keep_mask,
    )

    # persist split indices (for leak-free reproducibility)
    splits_path = out_dir / "splits.npz"
    np.savez_compressed(
        splits_path,
        train_idx=splits.train_idx.astype(np.int64),
        val_idx=splits.val_idx.astype(np.int64),
        test_idx=splits.test_idx.astype(np.int64),
    )

    # persist imputer state (train-median fill values)
    imputer_path = out_dir / "imputer_state.json"
    save_imputer_state(imputer, imputer_path)

    # build feature groups aligned to final feature order
    group_ids, group_names, group_to_id = build_feature_groups(feature_names_final)
    feature_groups_path = save_feature_groups(
        output_dir=out_dir,
        feature_names=feature_names_final,
        group_ids=group_ids,
        group_names=group_names,
        group_to_id=group_to_id,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names_original": feature_names_original,
        "feature_names": feature_names_final,
        "keep_mask": keep_mask,
        "resolved_label_col": resolved_label_col,
        "label_mapping": label_mapping,
        "label_encoder": le,
        "imputer": imputer,
        "imputer_state_path": str(imputer_path),
        "splits_path": str(splits_path),
        "scaler_path": str(scaler_path),
        "label_mapping_path": str(label_mapping_path),
        "label_encoder_path": str(label_encoder_path),
        "feature_metadata_path": str(feature_metadata_path),
        "group_ids": group_ids,
        "group_names": group_names,
        "feature_groups_path": str(feature_groups_path),
    }

