"""
Utilities for extracting operator usage from a model with torch.export.

Example:
    import torch
    from src.components.models.baseline_model import BaselineModel
    from src.utils.export_ops_report import extract_and_print_model_ops

    model = BaselineModel(vocab_size=100, layers=2, d_model=64, n_heads=8, pad_id=98)
    tokens = torch.randint(0, 97, (2, 16), dtype=torch.long)
    key_padding_mask = torch.ones((2, 16), dtype=torch.bool)

    extract_and_print_model_ops(
        model,
        example_args=(tokens,),
        example_kwargs={"key_padding_mask": key_padding_mask},
    )
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn


@dataclass(slots=True)
class ModelOpsReport:
    total_graph_nodes: int
    total_op_calls: int
    unique_ops: list[str]
    op_counts: dict[str, int]
    namespace_counts: dict[str, int]


def _normalize_op_name(target: Any) -> str:
    # OpOverload string form includes namespace + overload (e.g. aten.add.Tensor).
    return str(target)


def _namespace_from_op(op_name: str) -> str:
    if op_name.startswith("<") and op_name.endswith(">"):
        return "python_builtin"
    if "." in op_name:
        return op_name.split(".", maxsplit=1)[0]
    if "::" in op_name:
        return op_name.split("::", maxsplit=1)[0]
    return "unknown"


def collect_model_ops(
    model: nn.Module,
    example_args: tuple[Any, ...],
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    strict: bool = False,
    decomp_table: dict[torch._ops.OperatorBase, Any] | None = None,
) -> ModelOpsReport:
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected `model` to be nn.Module, got: {type(model)!r}")
    if not isinstance(example_args, tuple):
        raise TypeError("`example_args` must be a tuple.")

    kwargs = dict(example_kwargs or {})
    was_training = model.training
    model.eval()
    try:
        exported = torch.export.export(
            model,
            example_args,
            kwargs=kwargs,
            strict=strict,
        )
        if decomp_table is None:
            decomposed = exported.run_decompositions()
        else:
            decomposed = exported.run_decompositions(decomp_table=decomp_table)
    finally:
        model.train(was_training)

    op_counts: Counter[str] = Counter()
    namespace_counts: Counter[str] = Counter()
    total_graph_nodes = 0
    for node in decomposed.graph_module.graph.nodes:
        total_graph_nodes += 1
        if node.op != "call_function":
            continue

        op_name = _normalize_op_name(node.target)
        op_counts[op_name] += 1
        namespace_counts[_namespace_from_op(op_name)] += 1

    unique_ops = sorted(op_counts.keys())
    return ModelOpsReport(
        total_graph_nodes=total_graph_nodes,
        total_op_calls=sum(op_counts.values()),
        unique_ops=unique_ops,
        op_counts=dict(op_counts),
        namespace_counts=dict(namespace_counts),
    )


def print_model_ops_report(report: ModelOpsReport, *, top_k: int = 20) -> None:
    print("Model Ops (after torch.export + run_decompositions)")
    print()
    print(f"Unique ops ({len(report.unique_ops)}):")
    for op_name in report.unique_ops:
        print(f"  - {op_name}")
    if not report.unique_ops:
        print("  - <none>")

    print()
    print("Summary:")
    print(f"  - Graph nodes: {report.total_graph_nodes}")
    print(f"  - Operator calls: {report.total_op_calls}")
    print(f"  - Unique operators: {len(report.unique_ops)}")

    print()
    print("Namespace breakdown:")
    for namespace, count in sorted(
        report.namespace_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  - {namespace}: {count}")
    if not report.namespace_counts:
        print("  - <none>")

    print()
    print(f"Top {max(top_k, 0)} operators by frequency:")
    if top_k > 0:
        for op_name, count in sorted(
            report.op_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]:
            print(f"  - {op_name}: {count}")
    else:
        print("  - <none requested>")


def extract_and_print_model_ops(
    model: nn.Module,
    example_args: tuple[Any, ...],
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    top_k: int = 20,
    strict: bool = False,
    decomp_table: dict[torch._ops.OperatorBase, Any] | None = None,
) -> ModelOpsReport:
    report = collect_model_ops(
        model=model,
        example_args=example_args,
        example_kwargs=example_kwargs,
        strict=strict,
        decomp_table=decomp_table,
    )
    print_model_ops_report(report, top_k=top_k)
    return report
