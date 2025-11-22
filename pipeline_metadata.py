"""Central pipeline metadata shared between UI, operators, and console scripts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PipelineDescriptor:
    key: str
    label: str
    cli_value: str
    description: str
    available: bool
    status_hint: str
    icon: str
    modes: Dict[str, bool]


_PIPELINES: Dict[str, PipelineDescriptor] = {
    "TRELLIS": PipelineDescriptor(
        key="TRELLIS",
        label="TRELLIS",
        cli_value="trellis",
        description="Original TRELLIS pipeline with sparse + SLAT stages.",
        available=True,
        status_hint="Ready to generate.",
        icon='SHADERFX',
        modes={
            'IMAGE': True,
            'MULTI_IMAGE': True,
            'TEXT': True,
            'UPSCALE': False,
        },
    ),
    "DIRECT3D": PipelineDescriptor(
        key="DIRECT3D",
        label="Direct3D-S2",
        cli_value="direct3d",
        description="Direct3D-S2 sparse volumetric pipeline (image-only).",
        available=True,
        status_hint="Ready for single-image jobs (beta).",
        icon='SURFACE_DATA',
        modes={
            'IMAGE': True,
            'MULTI_IMAGE': False,
            'TEXT': False,
            'UPSCALE': True,
        },
    ),
}


PIPELINE_KEYS: List[str] = list(_PIPELINES.keys())
PIPELINE_ENUM_ITEMS = [
    (descriptor.key, descriptor.label, descriptor.description, descriptor.icon, idx)
    for idx, descriptor in enumerate(_PIPELINES.values())
]


def pipeline_cli_choices() -> List[str]:
    return [descriptor.cli_value for descriptor in _PIPELINES.values()]

def get_pipeline_descriptor(key: str) -> PipelineDescriptor:
    if key not in _PIPELINES:
        raise KeyError(f"Unknown pipeline key: {key}")
    return _PIPELINES[key]


def pipeline_cli_value(key: str) -> str:
    return get_pipeline_descriptor(key).cli_value


def pipeline_key_from_cli(cli_value: str) -> str:
    for descriptor in _PIPELINES.values():
        if descriptor.cli_value == cli_value:
            return descriptor.key
    raise KeyError(f"Unknown pipeline cli identifier: {cli_value}")


def is_pipeline_available(key: str) -> bool:
    return get_pipeline_descriptor(key).available


def is_mode_supported(key: str, mode: str) -> bool:
    descriptor = get_pipeline_descriptor(key)
    return descriptor.modes.get(mode, False)


def get_pipeline_status_info(key: str):
    descriptor = get_pipeline_descriptor(key)
    return {
        'key': descriptor.key,
        'label': descriptor.label,
        'description': descriptor.description,
        'supported': descriptor.available,
        'status_hint': descriptor.status_hint,
        'status_icon': 'CHECKMARK' if descriptor.available else 'ERROR',
    }


def ensure_pipeline_supported(pipeline_key: str, reporter=None, mode: str | None = None) -> bool:
    descriptor = get_pipeline_descriptor(pipeline_key)
    if not descriptor.available:
        message = f"{descriptor.label} pipeline integration is not available yet. Select TRELLIS to generate."
        if reporter is not None:
            reporter.report({'ERROR'}, message)
        else:
            print(f"ERROR: {message}")
        return False
    if mode and not is_mode_supported(pipeline_key, mode):
        message = f"{descriptor.label} pipeline does not support {mode.replace('_', ' ').title()} mode."
        if reporter is not None:
            reporter.report({'ERROR'}, message)
        else:
            print(f"ERROR: {message}")
        return False
    return True


def mark_pipeline_available(pipeline_key: str, *, status_hint: str | None = None):
    descriptor = get_pipeline_descriptor(pipeline_key)
    updated = PipelineDescriptor(
        key=descriptor.key,
        label=descriptor.label,
        cli_value=descriptor.cli_value,
        description=descriptor.description,
        available=True,
        status_hint=status_hint or "Ready to generate.",
        icon=descriptor.icon,
        modes=descriptor.modes,
    )
    _PIPELINES[pipeline_key] = updated
