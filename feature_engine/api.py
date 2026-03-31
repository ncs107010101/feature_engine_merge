"""
feature_engine.api
===============================
User-facing API for computing features based on data combinations.
"""

from typing import List, Optional
import pandas as pd

from .core import FeatureRegistry
from .preprocessing import preprocess


def compute_features(
    data_combination: str,
    data: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Compute features and return them as a DataFrame keyed by (StockId, Date).

    Parameters
    ----------
    data_combination : str
        The input combination type. e.g., 'single_stock_daily', 'single_stock_tick', 
        'single_stock_broker', 'cross_broker_tick'.
    data : pd.DataFrame
        Raw input data.
    feature_names : list[str] or None
        Subset of registered feature names to compute.
        If None, all registered features for this combination are computed.
    **kwargs
        Extra keyword arguments forwarded to the preprocessing layer and features.
        
    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ['StockId', 'Date', ...feature_cols].
        Each (StockId, Date) pair is unique.
    """
    if data_combination == "single_stock_broker":
        preprocessed_data, broker_day_data = preprocess(data_combination, data, return_broker_day=True)
        kwargs["_broker_day"] = broker_day_data
    elif data_combination == "cross_broker_daily":
        daily_data = kwargs.pop('daily_data', None)
        if daily_data is None:
            raise ValueError(
                "cross_broker_daily requires daily OHLC data passed as `daily_data` in kwargs."
            )
        preprocessed_data, broker_day_data = preprocess(
            data_combination, data, return_broker_day=True, daily_data=daily_data
        )
        kwargs["_broker_day"] = broker_day_data
    elif data_combination == "cross_tick_broker_daily":
        daily_data = kwargs.pop('daily_data', None)
        tick_data = kwargs.pop('tick_data', None)
        if daily_data is None:
            raise ValueError(
                "cross_tick_broker_daily requires daily OHLC data passed as `daily_data` in kwargs."
            )
        preprocessed_data, broker_day_data = preprocess(
            data_combination, data, return_broker_day=True, daily_data=daily_data, _tick_raw=tick_data
        )
        kwargs["_broker_day"] = broker_day_data
    elif data_combination == "single_stock_tick":
        kwargs["_tick_raw"] = data
        preprocessed_data = preprocess(data_combination, data, **kwargs)
    else:
        preprocessed_data = preprocess(data_combination, data, **kwargs)

    # 2. Resolve which features to compute
    all_combination_features = FeatureRegistry.get_by_combination(data_combination)
    
    if feature_names is None:
        features_to_compute = all_combination_features
    else:
        features_to_compute = {}
        for name in feature_names:
            if name not in all_combination_features:
                raise KeyError(f"Feature '{name}' is not registered under combination '{data_combination}'.")
            features_to_compute[name] = all_combination_features[name]

    if not features_to_compute:
        import warnings
        warnings.warn(
            f"No features registered for combination '{data_combination}'. Returning empty DataFrame.",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=["StockId", "Date"])

    result = None

    # 3. Compute Features
    for name, feature_cls in features_to_compute.items():
        feature = feature_cls()
        feature.validate(preprocessed_data)
        output = feature.calculate(preprocessed_data, **kwargs)

        if isinstance(output, pd.Series):
            output = output.to_frame(name=name)

        if not isinstance(output, pd.DataFrame):
            raise TypeError(
                f"Feature '{name}' returned {type(output).__name__}, "
                f"expected pd.Series or pd.DataFrame."
            )

        # Ensure (StockId, Date) uniqueness
        if "StockId" in output.columns and "Date" in output.columns:
            dups = output.duplicated(subset=["StockId", "Date"]).sum()
            if dups > 0:
                raise ValueError(
                    f"Feature '{name}' returned {dups} duplicate (StockId, Date) rows. "
                    f"Each stock-date pair must be unique."
                )

        if result is None:
            result = output
        else:
            merge_keys = ["StockId", "Date"]
            if all(k in result.columns and k in output.columns for k in merge_keys):
                result_copy = result.copy()
                output_copy = output.copy()
                if result_copy["Date"].dtype != output_copy["Date"].dtype:
                    result_copy["Date"] = result_copy["Date"].astype(str)
                    output_copy["Date"] = output_copy["Date"].astype(str)
                feature_cols = [c for c in output_copy.columns if c not in merge_keys]
                result = result_copy.merge(output_copy[merge_keys + feature_cols], on=merge_keys, how="outer")
            else:
                for col in output.columns:
                    if col not in result.columns:
                        result[col] = output[col]

    # Clean up any residual inf/nan after merge if necessary 
    # (though individual features should already be clean)
    return result

def list_features(data_combination: Optional[str] = None) -> List[str]:
    """Return registered feature names, optionally filtered by combination."""
    if data_combination:
        return list(FeatureRegistry.get_by_combination(data_combination).keys())
    return FeatureRegistry.list_names()

def describe_features() -> pd.DataFrame:
    """Return a DataFrame describing all registered features."""
    rows = []
    for name, cls in FeatureRegistry.get_all().items():
        rows.append({
            "name": name,
            "combination": cls.data_combination,
            "description": cls.description,
            "required_columns": cls.required_columns,
            "class": cls.__name__,
        })
    return pd.DataFrame(rows)
