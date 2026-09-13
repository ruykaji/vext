import argparse
import datetime
import json
import os
import pathlib
import platform
import shlex
import shutil
import subprocess
import sys

BENCHMARK_VERSIONS = {
    "Eigen": "5.0.1",
    "Google Benchmark": "1.9.5",
}

TIME_TO_NS = {
    "ns": 1.0,
    "us": 1_000.0,
    "ms": 1_000_000.0,
    "s": 1_000_000_000.0,
}


def run(
    command,
    *,
    cwd=None,
    capture=False,
    check=True,
):
    print(f"+ {shlex.join(str(part) for part in command)}", flush=True)

    result = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )

    return result.stdout.strip() if capture else ""


def ensure_tool(
    name,
):
    if shutil.which(name) is None:
        raise RuntimeError(f"Required tool '{name}' was not found in PATH.")


def ensure_conan_profile():
    profile = subprocess.run(
        ["conan", "profile", "path", "default"],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if profile.returncode != 0:
        run(["conan", "profile", "detect"])


def has_cuda_device():
    if shutil.which("nvcc") is None or shutil.which("nvidia-smi") is None:
        return False

    probe = subprocess.run(
        ["nvidia-smi", "-L"],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return probe.returncode == 0


def choose_cuda(
    mode,
):
    available = has_cuda_device()

    if mode == "on" and not available:
        raise RuntimeError(
            "CUDA benchmarking was requested, but nvcc and an accessible NVIDIA device are required."
        )

    return available if mode == "auto" else mode == "on"


def configure_and_install_vext(
    root,
    work,
    cuda_enabled,
    architecture,
    jobs,
):
    build = work / "vext"
    install = work / "install"

    run(
        [
            "cmake",
            "-S",
            root,
            "-B",
            build,
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install}",
            "-DVEXT_BUILD_TESTS=OFF",
            f"-DVEXT_ENABLE_CUDA={'ON' if cuda_enabled else 'OFF'}",
            f"-DVEXT_CUDA_ARCHITECTURES={architecture}",
        ]
    )
    run(["cmake", "--build", build, "--target", "install", "--parallel", str(jobs)])

    return install


def install_conan_dependencies(
    benchmark_root,
    work,
):
    ensure_conan_profile()

    output = work / "conan"

    run(
        [
            "conan",
            "install",
            benchmark_root,
            "--output-folder",
            output,
            "--build=missing",
            "--settings=build_type=Release",
        ]
    )

    toolchains = list(output.rglob("conan_toolchain.cmake"))

    if len(toolchains) != 1:
        raise RuntimeError(
            f"Expected one Conan CMake toolchain, found {len(toolchains)}."
        )

    return toolchains[0]


def configure_and_build_benchmarks(
    benchmark_root,
    work,
    install,
    toolchain,
    cuda_enabled,
    architecture,
    jobs,
):
    build = work / "suite"

    run(
        [
            "cmake",
            "-S",
            benchmark_root,
            "-B",
            build,
            "-G",
            "Ninja",
            f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            f"-DCMAKE_PREFIX_PATH={install}",
            f"-DVEXT_BENCHMARK_ENABLE_CUDA={'ON' if cuda_enabled else 'OFF'}",
            f"-DVEXT_CUDA_ARCHITECTURES={architecture}",
        ]
    )
    run(["cmake", "--build", build, "--parallel", str(jobs)])

    return build


def benchmark_arguments(
    output,
    minimum_time,
    repetitions,
):
    return [
        f"--benchmark_out={output}",
        "--benchmark_out_format=json",
        f"--benchmark_min_time={minimum_time}s",
        f"--benchmark_repetitions={repetitions}",
        "--benchmark_report_aggregates_only=true",
    ]


def execute_benchmarks(
    build,
    work,
    cuda_enabled,
    minimum_time,
    repetitions,
):
    raw = work / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    result_files = []
    cpu_result = raw / "compare_cpu.json"

    run(
        [
            build / "bin" / "bench_compare_cpu",
            *benchmark_arguments(cpu_result, minimum_time, repetitions),
        ]
    )
    result_files.append(cpu_result)

    if cuda_enabled:
        cuda_result = raw / "compare_cuda.json"

        run(
            [
                build / "bin" / "bench_compare_cuda",
                *benchmark_arguments(cuda_result, minimum_time, repetitions),
            ]
        )
        result_files.append(cuda_result)

    return result_files


def load_measurements(
    result_files,
):
    measurements = {}

    for result_file in result_files:
        document = json.loads(result_file.read_text())

        for entry in document.get("benchmarks", []):
            if entry.get("error_occurred"):
                continue

            aggregate = entry.get("aggregate_name")

            if aggregate not in (None, "median"):
                continue

            run_name = entry.get("run_name", entry["name"])
            parts = run_name.split("/")

            if len(parts) < 4:
                continue

            implementation, backend, operation = parts[:3]
            problem_parts = parts[3:]

            if problem_parts and problem_parts[-1] == "manual_time":
                problem_parts.pop()

            problem = "x".join(problem_parts)
            unit = entry.get("time_unit", "ns")
            measurements[(backend, operation, problem, implementation)] = (
                entry["real_time"] * TIME_TO_NS[unit]
            )

    return measurements


def cpu_name():
    cpuinfo = pathlib.Path("/proc/cpuinfo")

    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()

    return platform.processor() or "Unknown"


def gpu_name(
    cuda_enabled,
):
    if not cuda_enabled:
        return "Not benchmarked"

    return run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture=True
    ).splitlines()[0]


def cuda_toolkit(
    cuda_enabled,
):
    if not cuda_enabled:
        return "Not used"

    version = run(["nvcc", "--version"], capture=True)
    return version.splitlines()[-1]


def git_revision(
    root,
):
    revision = run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture=True)
    dirty = bool(run(["git", "status", "--porcelain"], cwd=root, capture=True))
    return f"{revision}{' (dirty)' if dirty else ''}"


def compiler_name(
    build,
):
    commands = json.loads((build / "compile_commands.json").read_text())

    if not commands:
        return "Unknown"

    compiler = shlex.split(commands[0]["command"])[0]
    version = run([compiler, "--version"], capture=True)
    return version.splitlines()[0]


def format_time(
    nanoseconds,
):
    if nanoseconds >= 1_000_000.0:
        return f"{nanoseconds / 1_000_000.0:.3f} ms"

    if nanoseconds >= 1_000.0:
        return f"{nanoseconds / 1_000.0:.3f} us"

    return f"{nanoseconds:.3f} ns"


def format_difference(ratio):
    if ratio < 1.0:
        return f"vext {(1.0 - ratio) * 100.0:.1f}% faster"

    if ratio > 1.0:
        return f"vext {(ratio - 1.0) * 100.0:.1f}% slower"

    return "equal"


def write_report(
    report,
    measurements,
    compiler,
    root,
    cuda_enabled,
    minimum_time,
    repetitions,
):
    rows = []

    vext_cases = sorted(key for key in measurements if key[3] == "Vext")

    for backend, operation, problem, _ in vext_cases:
        vext_time = measurements[(backend, operation, problem, "Vext")]
        reference_keys = sorted(
            key
            for key in measurements
            if key[:3] == (backend, operation, problem) and key[3] != "Vext"
        )

        for reference_key in reference_keys:
            reference = reference_key[3]
            reference_time = measurements[reference_key]
            ratio = vext_time / reference_time
            rows.append(
                (
                    backend,
                    operation,
                    problem,
                    reference,
                    vext_time,
                    reference_time,
                    ratio,
                )
            )

    generated = (
        datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    )

    lines = [
        "# Benchmark results",
        "",
        "Generated by `python3 benchmarks/run.py`. Lower time is better; the ratio is `vext / reference`.",
        "",
        "## Environment",
        "",
        f"- Generated: {generated}",
        f"- Revision: `{git_revision(root)}`",
        f"- System: {platform.platform()}",
        f"- CPU: {cpu_name()}",
        f"- GPU: {gpu_name(cuda_enabled)}",
        f"- CUDA toolkit: {cuda_toolkit(cuda_enabled)}",
        f"- Compiler: {compiler}",
        f"- Samples: median of {repetitions} repetitions, minimum {minimum_time} seconds each",
        "- CPU policy: Eigen internal parallelism disabled",
        f"- Dependencies: Eigen {BENCHMARK_VERSIONS['Eigen']}, Google Benchmark {BENCHMARK_VERSIONS['Google Benchmark']}",
        "- CUDA references: Thrust, cuBLAS, and cuSPARSE from the installed CUDA toolkit",
        "",
        "## Comparison",
        "",
        "| Backend | Operation | Problem size | Reference library | vext time | Reference time | Ratio | Difference |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]

    for (
        backend,
        operation,
        problem,
        reference,
        vext_time,
        reference_time,
        ratio,
    ) in rows:
        lines.append(
            f"| {backend} | {operation} | {problem} | {reference} | {format_time(vext_time)} | "
            f"{format_time(reference_time)} | {ratio:.2f}x | {format_difference(ratio)} |"
        )

    if not rows:
        lines.append("| — | — | — | — | — | — | — | No paired results were produced |")

    lines.extend(
        [
            "",
            "Results are machine-specific and should be compared only when generated on equivalent hardware with the same build configuration.",
            "",
        ]
    )

    report.write_text("\n".join(lines))
    print(f"Wrote {report}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build, run, and report vext comparison benchmarks."
    )
    parser.add_argument(
        "--cuda",
        choices=("auto", "on", "off"),
        default="auto",
        help="CUDA execution policy (default: auto).",
    )
    parser.add_argument(
        "--minimum-time",
        default="0.1",
        help="Minimum time per benchmark repetition in seconds.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of repetitions used for the median.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the selected benchmark build directory before running.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    benchmark_root = pathlib.Path(__file__).resolve().parent
    root = benchmark_root.parent
    work = (benchmark_root / "build").resolve()
    report = (benchmark_root / "RESULTS.md").resolve()
    architecture = "native"
    jobs = max(1, os.cpu_count() or 1)

    for tool in ("cmake", "conan", "ninja"):
        ensure_tool(tool)

    cuda_enabled = choose_cuda(arguments.cuda)

    if arguments.clean and work.exists():
        shutil.rmtree(work)

    work.mkdir(parents=True, exist_ok=True)

    install = configure_and_install_vext(root, work, cuda_enabled, architecture, jobs)
    toolchain = install_conan_dependencies(benchmark_root, work)
    build = configure_and_build_benchmarks(
        benchmark_root,
        work,
        install,
        toolchain,
        cuda_enabled,
        architecture,
        jobs,
    )
    result_files = execute_benchmarks(
        build,
        work,
        cuda_enabled,
        arguments.minimum_time,
        arguments.repetitions,
    )
    measurements = load_measurements(result_files)

    write_report(
        report,
        measurements,
        compiler_name(build),
        root,
        cuda_enabled,
        arguments.minimum_time,
        arguments.repetitions,
    )


if __name__ == "__main__":
    try:
        main()
    except (
        RuntimeError,
        subprocess.CalledProcessError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        print(f"benchmark run failed: {error}", file=sys.stderr)
        sys.exit(1)
