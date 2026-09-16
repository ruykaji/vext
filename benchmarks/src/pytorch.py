#!/usr/bin/env python3

import argparse
import gc
import json
import math
import time

import numpy as np
import torch

# Keep PyTorch's normal CPU thread policy. Its framework-level results are
# reported separately from the single-threaded Eigen kernel baseline.

SHAPES = (
    (1, 65536, 1, 1, 1),
    (1, 1048576, 1, 1, 1),
    (1, 16777216, 1, 1, 1),
    (2, 256, 256, 1, 1),
    (2, 1024, 1024, 1, 1),
    (2, 4096, 4096, 1, 1),
    (3, 4, 128, 128, 1),
    (3, 64, 128, 128, 1),
    (3, 64, 512, 512, 1),
    (4, 1, 16, 64, 64),
    (4, 16, 64, 32, 32),
    (4, 16, 256, 64, 64),
)
FAMILIES = (
    "Unary",
    "Binary",
    "Broadcast",
    "Logical",
    "Reduction",
    "AxisReduction",
    "CSRScatter",
    "Matmul",
)
OPS = {
    "Unary": (
        "ABS",
        "SIN",
        "COS",
        "TANH",
        "NEG",
        "EXP",
        "LOG",
        "SQRT",
        "SQUARE",
        "ROUND",
        "SIGMOID",
        "SOFT_RELU",
        "RELU",
        "SOFTMAX",
        "SOFTMIN",
        "LOGSOFTMAX",
        "LEAKY_RELU",
        "ELU",
        "SWISH",
        "LINEAR",
        "CLIP",
        "POW",
    ),
    "Binary": ("ADD", "SUB", "MUL", "DIV", "POW", "MIN", "MAX", "PRELU"),
    "Broadcast": ("ADD", "SUB", "MUL", "DIV", "POW", "MIN", "MAX", "PRELU"),
    "Logical": ("EQUAL", "NOT_EQUAL", "LESS", "LESS_EQUAL", "GREATER", "GREATER_EQUAL"),
    "Reduction": ("SUM", "MEAN", "MIN", "MAX", "PROD", "STD", "VAR", "L2_NORM"),
    "AxisReduction": ("SUM", "MEAN", "MIN", "MAX", "PROD", "STD", "VAR", "L2_NORM"),
    "CSRScatter": ("SUM", "MEAN"),
    "Matmul": ("Matmul",),
}

CSR_PROFILES = ((1024, 8, 16), (4096, 16, 64), (16384, 32, 128))


def prepare_case(
    device,
):
    """Release allocations from the preceding case before creating this case's inputs."""
    gc.collect()

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def tensor_values(
    shape,
    positive=False,
):
    size = math.prod(shape)
    data = np.arange(size, dtype=np.float32) % 1024 / 256 - 2
    return np.abs(data) + 0.25 if positive else data


def unary_values(
    shape,
    operation,
):
    return tensor_values(shape, operation in {"LOG", "SQRT", "POW"})


def reduction_values(
    shape,
    operation,
):
    if operation != "PROD":
        return tensor_values(shape)

    size = math.prod(shape)
    return 1.0 + ((np.arange(size, dtype=np.float32) % 17) - 8) * 0.00001


def run_torch_csr_scatter(
    device,
    operation,
    rows,
    degree,
    features,
):
    prepare_case(device)

    target = torch.device(device)
    src = torch.from_numpy(tensor_values((rows, features)).reshape(rows, features)).to(target)
    tail = torch.from_numpy(np.fromiter(((row + edge * 17) % rows for row in range(rows) for edge in range(degree)), dtype=np.int64)).to(target)
    edge_rows = torch.arange(rows, device=target, dtype=torch.long).repeat_interleave(degree)
    out = torch.empty((rows, features), device=target, dtype=torch.float32)
    inverse_degree = 1.0 / degree

    def op():
        messages = src.index_select(0, tail)

        out.zero_()
        out.index_add_(0, edge_rows, messages)

        if operation == "MEAN":
            out.mul_(inverse_degree)

        return out

    if device == "cuda":
        torch.cuda.synchronize()

        op()

        torch.cuda.synchronize()
        start, stop = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        start.record()
        result = op()
        stop.record()
        stop.synchronize()
        elapsed = int(start.elapsed_time(stop) * 1_000_000)
    else:
        op()

        start = time.perf_counter_ns()
        result = op()
        elapsed = time.perf_counter_ns() - start

    del result
    return elapsed


def run_torch(
    device,
    family,
    operation,
    shape,
):
    prepare_case(device)
    target = torch.device(device)
    tensor = lambda values, dimensions: torch.from_numpy(values.reshape(dimensions)).to(target)
    if family == "Matmul":
        rows, shared, cols = shape
        lhs, rhs = (
            tensor(tensor_values((rows, shared)), (rows, shared)),
            tensor(tensor_values((shared, cols)), (shared, cols)),
        )
        row = None
    else:
        lhs = tensor(
            unary_values(shape, operation) if family == "Unary" else tensor_values(shape, operation == "POW"),
            shape,
        )
        rhs = tensor(tensor_values(shape, True), shape)
        row = tensor(tensor_values((shape[-1],), True), (shape[-1],))

        if family in ("Reduction", "AxisReduction"):
            lhs = tensor(reduction_values(shape, operation), shape)

    def op():
        x, y = (lhs, row) if family == "Broadcast" else (lhs, rhs)

        if family == "Unary":
            operations = {
                "ABS": lambda: x.abs(),
                "SIN": lambda: x.sin(),
                "COS": lambda: x.cos(),
                "TANH": lambda: x.tanh(),
                "NEG": lambda: -x,
                "EXP": lambda: x.exp(),
                "LOG": lambda: x.log(),
                "SQRT": lambda: x.sqrt(),
                "SQUARE": lambda: x.square(),
                "ROUND": lambda: x.round(),
                "SIGMOID": lambda: x.sigmoid(),
                "SOFT_RELU": lambda: torch.nn.functional.softplus(x),
                "RELU": lambda: x.relu(),
                "SOFTMAX": lambda: x.flatten().softmax(0).reshape_as(x),
                "SOFTMIN": lambda: (-x).flatten().softmax(0).reshape_as(x),
                "LOGSOFTMAX": lambda: x.flatten().log_softmax(0).reshape_as(x),
                "LEAKY_RELU": lambda: torch.nn.functional.leaky_relu(x, 0.25),
                "ELU": lambda: torch.nn.functional.elu(x, 2.0),
                "SWISH": lambda: x * (x * 1.25).sigmoid(),
                "LINEAR": lambda: x * 1.25 + 2.0,
                "CLIP": lambda: x.clamp(1.25, 2.0),
                "POW": lambda: 1.25 * x.pow(2.0),
            }
            return operations[operation]()

        if family in ("Binary", "Broadcast"):
            operations = {
                "ADD": lambda: x + y,
                "SUB": lambda: x - y,
                "MUL": lambda: x * y,
                "DIV": lambda: x / y,
                "POW": lambda: x.pow(y),
                "MIN": lambda: torch.minimum(x, y),
                "MAX": lambda: torch.maximum(x, y),
                "PRELU": lambda: torch.maximum(x, torch.zeros_like(x)) + y * torch.minimum(x, torch.zeros_like(x)),
            }
            return operations[operation]()

        if family == "Logical":
            operations = {
                "EQUAL": lambda: x == y,
                "NOT_EQUAL": lambda: x != y,
                "LESS": lambda: x < y,
                "LESS_EQUAL": lambda: x <= y,
                "GREATER": lambda: x > y,
                "GREATER_EQUAL": lambda: x >= y,
            }
            return operations[operation]()

        dim = -1 if family == "AxisReduction" else None

        return (
            {
                "SUM": lambda: x.sum(dim),
                "MEAN": lambda: x.mean(dim),
                "MIN": lambda: x.amin(dim),
                "MAX": lambda: x.amax(dim),
                "PROD": lambda: x.prod() if dim is None else x.prod(dim),
                "STD": lambda: x.std(dim, correction=0),
                "VAR": lambda: x.var(dim, correction=0),
                "L2_NORM": lambda: torch.linalg.vector_norm(x, dim=dim),
            }[operation]()
            if family != "Matmul"
            else lhs @ rhs
        )

    if device == "cuda":
        torch.cuda.synchronize()

        op()

        torch.cuda.synchronize()
        start, stop = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        result = op()
        stop.record()
        stop.synchronize()
        elapsed = int(start.elapsed_time(stop) * 1_000_000)
    else:
        op()
        start = time.process_time_ns()
        result = op()
        elapsed = time.process_time_ns() - start

    del result
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--families", nargs="*")
    args = parser.parse_args()

    selected = args.families or FAMILIES
    runner = run_torch
    implementation, backend = "PyTorch", args.backend.upper()

    entries = []

    for family in selected:
        if family not in OPS:
            continue

        if family == "CSRScatter":
            for rows, degree, features in CSR_PROFILES:
                for operation in OPS[family]:
                    print(
                        f"PyTorch {backend} CSRScatter_{operation} {rows}/{degree}/{features}",
                        flush=True,
                    )
                    elapsed = run_torch_csr_scatter(args.backend, operation, rows, degree, features)
                    operation_name = f"CSRScatter_{operation}"
                    problem = f"{rows}/{degree}/{features}"
                    entries.append(
                        {
                            "name": f"PyTorch/{backend}/{operation_name}/{problem}/iterations:1",
                            "run_name": f"PyTorch/{backend}/{operation_name}/{problem}/iterations:1",
                            "real_time": elapsed,
                            "cpu_time": elapsed,
                            "time_unit": "ns",
                            "iterations": 1,
                        }
                    )
            continue

        shapes = (
            (
                (128, 128, 128),
                (512, 512, 512),
                (2048, 2048, 2048),
                (64, 128, 32),
                (256, 512, 128),
                (1024, 2048, 512),
            )
            if family == "Matmul"
            else SHAPES
        )

        for descriptor in shapes:
            if family == "Matmul":
                shape = descriptor
                problem = "/".join(map(str, descriptor))
            else:
                rank, *dims = descriptor
                shape = tuple(dims[:rank])
                problem = "/".join(map(str, (rank, *dims)))

            for operation in OPS[family]:
                print(f"PyTorch {backend} {family}_{operation} {problem}", flush=True)
                elapsed = runner(args.backend, family, operation, shape)
                operation_name = family if family == "Matmul" else f"{family}_{operation}"
                entries.append(
                    {
                        "name": f"{implementation}/{backend}/{operation_name}/{problem}/iterations:1",
                        "run_name": f"{implementation}/{backend}/{operation_name}/{problem}/iterations:1",
                        "real_time": elapsed,
                        "cpu_time": elapsed,
                        "time_unit": "ns",
                        "iterations": 1,
                    }
                )

    with open(args.output, "w") as output:
        json.dump(
            {"context": {"library": implementation}, "benchmarks": entries},
            output,
            indent=2,
        )


if __name__ == "__main__":
    main()
