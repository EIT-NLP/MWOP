from __future__ import annotations
import importlib
import os
import sys
import warnings
from typing import Literal
from loguru import logger
from lmms_eval.models.registry_v2 import ModelManifest, ModelRegistryV2
logger.remove()
log_format = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'
logger.add(sys.stdout, level='WARNING', format=log_format)
AVAILABLE_SIMPLE_MODELS = {'llava_onevision': 'Llava_OneVision'}
AVAILABLE_CHAT_TEMPLATE_MODELS = {}
MODEL_ALIASES: dict[str, tuple[str, ...]] = {'dummy': ('dummy_video_reader',), 'openai': ('openai_compatible', 'openai_compatible_chat'), 'async_openai': ('async_openai_compatible_chat', 'async_openai_compatible'), 'async_hf_model': ('async_hf',), 'litellm': ('litellm_chat', 'litellm_compatible')}

def _build_class_path(model_name: str, model_type: Literal['simple', 'chat'], class_name: str) -> str:
    if '.' in class_name:
        return class_name
    return f'lmms_eval.models.{model_type}.{model_name}.{class_name}'

def _build_builtin_manifests() -> list[ModelManifest]:
    model_ids = sorted(set(AVAILABLE_SIMPLE_MODELS) | set(AVAILABLE_CHAT_TEMPLATE_MODELS))
    manifests: list[ModelManifest] = []
    for model_id in model_ids:
        simple_class = AVAILABLE_SIMPLE_MODELS.get(model_id)
        chat_class = AVAILABLE_CHAT_TEMPLATE_MODELS.get(model_id)
        aliases = MODEL_ALIASES.get(model_id, ())
        manifests.append(ModelManifest(model_id=model_id, simple_class_path=_build_class_path(model_id, 'simple', simple_class) if simple_class else None, chat_class_path=_build_class_path(model_id, 'chat', chat_class) if chat_class else None, aliases=aliases))
    return manifests

def _merge_legacy_plugin_models(registry: ModelRegistryV2) -> None:
    plugins = os.environ.get('LMMS_EVAL_PLUGINS')
    if not plugins:
        return
    warnings.warn("LMMS_EVAL_PLUGINS is deprecated. Prefer Python entry-points group 'lmms_eval.models' for plugin model registration.", DeprecationWarning, stacklevel=2)
    for plugin in plugins.split(','):
        module = importlib.import_module(f'{plugin}.models')
        for model_name, model_class in getattr(module, 'AVAILABLE_MODELS').items():
            class_path = f'{plugin}.models.{model_name}.{model_class}'
            AVAILABLE_SIMPLE_MODELS[model_name] = class_path
            registry.register_manifest(ModelManifest(model_id=model_name, simple_class_path=class_path), overwrite=True)

def _initialize_model_registry() -> ModelRegistryV2:
    registry = ModelRegistryV2()
    for manifest in _build_builtin_manifests():
        registry.register_manifest(manifest)
    _merge_legacy_plugin_models(registry)
    try:
        registry.load_entrypoint_manifests(overwrite=True)
    except Exception as exc:
        logger.warning(f'Failed to load model entry-point manifests: {exc}')
    return registry
MODEL_REGISTRY_V2 = _initialize_model_registry()

def _build_available_models_preferred() -> dict[str, str]:
    model_map: dict[str, str] = {}
    for model_id in MODEL_REGISTRY_V2.list_canonical_model_ids():
        manifest = MODEL_REGISTRY_V2.get_manifest(model_id)
        class_path = manifest.chat_class_path or manifest.simple_class_path
        if class_path:
            model_map[model_id] = class_path.rsplit('.', 1)[-1]
    return model_map
AVAILABLE_MODELS = _build_available_models_preferred()

def list_available_models(include_aliases: bool=False) -> list[str]:
    if include_aliases:
        return MODEL_REGISTRY_V2.list_model_names()
    return MODEL_REGISTRY_V2.list_canonical_model_ids()

def get_model_manifest(model_name: str) -> ModelManifest:
    return MODEL_REGISTRY_V2.get_manifest(model_name)

def get_model(model_name: str, force_simple: bool=False) -> type:
    try:
        return MODEL_REGISTRY_V2.get_model_class(model_name, force_simple=force_simple)
    except Exception as exc:
        logger.error(f"Failed to import model from '{model_name}': {exc}")
        raise
