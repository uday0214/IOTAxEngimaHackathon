"""Model factory for instantiating and loading classifier pipelines dynamically."""

import logging
from pathlib import Path
from typing import Type, Optional
from src.models.base import BaseModel
from src.models.catboost_model import CatBoostPipelineModel
from src.models.lgbm_model import LightGBMPipelineModel
from src.models.xgboost_model import XGBoostPipelineModel
from src.models.sklearn_models import RandomForestPipelineModel, LogisticRegressionPipelineModel
from src.supervisor.schemas import HyperparameterConfig

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "catboost": CatBoostPipelineModel,
    "lightgbm": LightGBMPipelineModel,
    "xgboost": XGBoostPipelineModel,
    "random_forest": RandomForestPipelineModel,
    "logistic_regression": LogisticRegressionPipelineModel,
}


def get_model_class(model_family: str) -> Type[BaseModel]:
    """Retrieves the concrete model pipeline class for a given family identifier."""
    normalized = model_family.lower().strip().replace("-", "_").replace(" ", "_")
    if normalized in MODEL_REGISTRY:
        return MODEL_REGISTRY[normalized]
    raise ValueError(f"Unknown model family '{model_family}'. Supported: {list(MODEL_REGISTRY.keys())}")


def create_model(
    config: HyperparameterConfig,
    experiment_id: int = 1,
    device: str = "auto",
) -> BaseModel:
    """Instantiates a configured classifier pipeline."""
    cls = get_model_class(config.model_family)
    return cls(config=config, experiment_id=experiment_id, device=device)


def load_champion_model(
    model_path: Path,
    model_family: Optional[str] = None,
    config: Optional[HyperparameterConfig] = None,
) -> BaseModel:
    """Loads a persisted model from disk, inferring family if needed."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # If family specified, use it directly
    if model_family:
        cls = get_model_class(model_family)
        return cls.load_model(model_path, config=config)

    # If CatBoost native .cbm extension
    if model_path.suffix == ".cbm":
        return CatBoostPipelineModel.load_model(model_path, config=config)

    # Attempt to load using joblib inspection
    import joblib
    loaded_obj = joblib.load(model_path)
    type_name = type(loaded_obj).__name__.lower()

    if "catboost" in type_name:
        inst = CatBoostPipelineModel(config=config or HyperparameterConfig(model_family="catboost"))
        inst.champion_model = loaded_obj
        return inst
    elif "lgbm" in type_name or "lightgbm" in type_name:
        inst = LightGBMPipelineModel(config=config or HyperparameterConfig(model_family="lightgbm"))
        inst.champion_model = loaded_obj
        return inst
    elif "xgb" in type_name:
        inst = XGBoostPipelineModel(config=config or HyperparameterConfig(model_family="xgboost"))
        inst.champion_model = loaded_obj
        return inst
    elif "pipeline" in type_name:
        # Check last step of sklearn pipeline
        last_step = loaded_obj.steps[-1][1]
        last_type = type(last_step).__name__.lower()
        if "forest" in last_type:
            inst = RandomForestPipelineModel(config=config or HyperparameterConfig(model_family="random_forest"))
            inst.champion_model = loaded_obj
            return inst
        elif "logistic" in last_type:
            inst = LogisticRegressionPipelineModel(config=config or HyperparameterConfig(model_family="logistic_regression"))
            inst.champion_model = loaded_obj
            return inst

    # Fallback to default CatBoost
    inst = CatBoostPipelineModel(config=config or HyperparameterConfig())
    inst.champion_model = loaded_obj
    return inst
