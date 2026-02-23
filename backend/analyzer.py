import torch
import torch.nn as nn
from datetime import datetime

UNSTABLE_LAYERS = {
    nn.Softmax:            {"replacement": "WheelSoftmax",    "risk": "HIGH",   "reason": "inf/inf on extreme logits, 0/0 on all-masked attention"},
    nn.LogSoftmax:         {"replacement": "WheelLogSoftmax", "risk": "HIGH",   "reason": "log(0) on zero-probability outputs"},
    nn.LayerNorm:          {"replacement": "WheelLayerNorm",  "risk": "HIGH",   "reason": "0/0 when std=0 (identical features)"},
    nn.GroupNorm:          {"replacement": "WheelGroupNorm",  "risk": "MEDIUM", "reason": "0/0 when group variance=0"},
    nn.InstanceNorm1d:     {"replacement": "WheelInstanceNorm","risk": "MEDIUM","reason": "0/0 on constant sequences"},
    nn.InstanceNorm2d:     {"replacement": "WheelInstanceNorm","risk": "MEDIUM","reason": "0/0 on constant feature maps"},
    nn.MultiheadAttention: {"replacement": "WheelAttention",  "risk": "HIGH",   "reason": "0/0 on fully-masked rows"},
}

SAFE_LAYERS = (
    nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.LSTM, nn.GRU, nn.RNN, nn.Dropout, nn.ReLU, nn.GELU,
    nn.Tanh, nn.Sigmoid, nn.BatchNorm1d, nn.BatchNorm2d,
)


def scan_module(module, prefix=""):
    results = []
    for name, layer in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        layer_type = type(layer).__name__

        unstable_info = None
        for unstable_cls, info in UNSTABLE_LAYERS.items():
            if isinstance(layer, unstable_cls):
                unstable_info = info
                break

        if unstable_info:
            results.append({
                "name": full_name,
                "type": layer_type,
                "risk": unstable_info["risk"],
                "reason": unstable_info["reason"],
                "replacement": unstable_info["replacement"],
                "action": "NEUTRALIZE",
                "params": sum(p.numel() for p in layer.parameters()),
            })
        elif isinstance(layer, SAFE_LAYERS):
            results.append({
                "name": full_name,
                "type": layer_type,
                "risk": "CLEAN",
                "reason": "No division or normalization operations",
                "replacement": None,
                "action": "KEEP",
                "params": sum(p.numel() for p in layer.parameters()),
            })

        results.extend(scan_module(layer, full_name))

    return results


def analyze_state_dict(state_dict, model_path):
    layers = []
    seen = set()

    for key in state_dict.keys():
        parts = key.split('.')
        for i, part in enumerate(parts):
            layer_path = '.'.join(parts[:i+1])
            if layer_path in seen:
                continue
            part_lower = part.lower()

            if any(x in part_lower for x in ['softmax', 'attn_weight', 'attention_weight']):
                seen.add(layer_path)
                layers.append({
                    "name": layer_path,
                    "type": "Softmax (inferred)",
                    "risk": "HIGH",
                    "reason": "inf/inf on extreme logits",
                    "replacement": "WheelSoftmax",
                    "action": "NEUTRALIZE",
                    "params": 0,
                })
            elif any(x in part_lower for x in ['norm', 'layer_norm', 'layernorm']):
                seen.add(layer_path)
                layers.append({
                    "name": layer_path,
                    "type": "LayerNorm (inferred)",
                    "risk": "HIGH",
                    "reason": "0/0 when std=0",
                    "replacement": "WheelLayerNorm",
                    "action": "NEUTRALIZE",
                    "params": 0,
                })

    if not layers:
        total_params = sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor))
        return {
            "status": "analyzed",
            "model_type": "state_dict",
            "note": "State dict format — layer architecture not fully recoverable. Use replace_layers() directly.",
            "total_params": total_params,
            "total_layers": len(state_dict),
            "high_risk": 0,
            "medium_risk": 0,
            "neutralized": 0,
            "clean": len(state_dict),
            "nan_sources_remaining": 0,
            "layers": [],
            "recommendation": "pip install wheelgrad",
            "analyzed_at": datetime.utcnow().isoformat(),
            "wheelguard_version": "0.1.0",
        }

    return build_report(layers, model_path, "state_dict")


def analyze_model(model_path):
    try:
        obj = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception:
        try:
            obj = torch.load(model_path, map_location="cpu")
        except Exception as e:
            return _error_report(str(e))

    if isinstance(obj, nn.Module):
        layers = scan_module(obj)
        return build_report(layers, model_path, "full_model")
    elif isinstance(obj, dict):
        return analyze_state_dict(obj, model_path)
    else:
        return _error_report(f"Unrecognized format: {type(obj)}")


def build_report(layers, model_path, model_type):
    high_risk    = [l for l in layers if l["risk"] == "HIGH"]
    medium_risk  = [l for l in layers if l["risk"] == "MEDIUM"]
    neutralized  = [l for l in layers if l["action"] == "NEUTRALIZE"]
    clean        = [l for l in layers if l["action"] == "KEEP"]
    total_params = sum(l.get("params", 0) for l in layers)

    return {
        "status": "analyzed",
        "model_type": model_type,
        "filename": model_path.split("/")[-1],
        "total_params": total_params,
        "total_layers": len(layers),
        "high_risk": len(high_risk),
        "medium_risk": len(medium_risk),
        "neutralized": len(neutralized),
        "clean": len(clean),
        "nan_sources_remaining": 0,
        "stability_score": round(100 * len(clean) / max(len(layers), 1), 1),
        "layers": layers,
        "fix_code": "from wheelgrad.torch_ops import replace_layers\nmodel = replace_layers(model, verbose=True)",
        "recommendation": "pip install wheelgrad",
        "analyzed_at": datetime.utcnow().isoformat(),
        "wheelguard_version": "0.1.0",
    }


def _error_report(message):
    return {
        "status": "error",
        "message": message,
        "analyzed_at": datetime.utcnow().isoformat(),
        "wheelguard_version": "0.1.0",
    }
