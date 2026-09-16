import argparse
import datetime
import hashlib
import json
import math
import os
import pathlib
import platform
import shlex
import shutil
import statistics
import subprocess
import sys

BENCHMARK_VERSIONS = {
    "Eigen": "3.4.1",
    "Google Benchmark": "1.9.5",
}

TIME_TO_NS = {
    "ns": 1.0,
    "us": 1_000.0,
    "ms": 1_000_000.0,
    "s": 1_000_000_000.0,
}

OPERATION_FAMILY_NAMES = {
    "AxisReduction": "Axis reduction",
    "Binary": "Binary",
    "Broadcast": "Broadcast",
    "CSRScatter": "CSR scatter",
    "CSRSpMV": "CSR SpMV",
    "Logical": "Logical",
    "Matmul": "Matrix multiplication",
    "Reduction": "Reduction",
    "Unary": "Unary",
}

OPERATION_FAMILY_OPTIONS = {
    "unary": "Unary",
    "binary": "Binary",
    "broadcast": "Broadcast",
    "logical": "Logical",
    "reduction": "Reduction",
    "axis_reduction": "AxisReduction",
    "csr_scatter": "CSRScatter",
    "csr_spmv": "CSRSpMV",
    "matmul": "Matmul",
}

ARTIFACT_REPETITIONS = 1
ARTIFACT_MINIMUM_TIME = 0
ARTIFACT_WARMUP_TIME = 0
PAPER_SAMPLES = 20


def run(
    command,
    *,
    cwd=None,
    capture=False,
    check=True,
    env=None,
):
    print(f"+ {shlex.join(str(part) for part in command)}", flush=True)

    result = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        env=env,
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


def choose_backends(
    cpu_requested,
    cuda_requested,
):
    available = has_cuda_device()

    if cuda_requested and not available:
        raise RuntimeError("CUDA benchmarking was requested, but nvcc and an accessible NVIDIA device are required.")

    if cpu_requested or cuda_requested:
        return cpu_requested, cuda_requested

    return True, available


def selected_operation_families(
    arguments,
):
    return [family for option, family in OPERATION_FAMILY_OPTIONS.items() if getattr(arguments, option)]


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
            "-s:h",
            "build_type=Release",
            "-s:b",
            "build_type=Release",
            "-s:h",
            "compiler.cppstd=17",
            "-s:b",
            "compiler.cppstd=17",
        ]
    )

    toolchains = list(output.rglob("conan_toolchain.cmake"))

    if len(toolchains) != 1:
        raise RuntimeError(f"Expected one Conan CMake toolchain, found {len(toolchains)}.")

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
    minimum_time=None,
    warmup_time=None,
    repetitions=None,
    operation_families=None,
):
    arguments = [
        f"--benchmark_out={output}",
        "--benchmark_out_format=json",
    ]

    if minimum_time is not None:
        arguments.append(f"--benchmark_min_time={minimum_time}s")
    if warmup_time is not None:
        arguments.append(f"--benchmark_min_warmup_time={warmup_time}")
    if repetitions is not None:
        arguments.extend(
            [
                f"--benchmark_repetitions={repetitions}",
                "--benchmark_report_aggregates_only=true",
                "--benchmark_display_aggregates_only=true",
            ]
        )

    if operation_families:
        family_pattern = "|".join(operation_families)
        arguments.append(f"--benchmark_filter=/({family_pattern})_")

    return arguments


def benchmark_entry_family(
    entry,
):
    run_name = entry.get("run_name", entry.get("name", ""))
    parts = run_name.split("/")

    if len(parts) < 3:
        return None

    backend = parts[1]
    family = parts[2].partition("_")[0]

    if backend not in ("CPU", "CUDA") or family not in OPERATION_FAMILY_NAMES:
        return None

    return backend, family


def cache_result_file(
    result_file,
    raw,
    *,
    overwrite,
    expected_backend=None,
    expected_families=None,
    run_policy=None,
):
    document = json.loads(result_file.read_text())
    grouped_entries = {}

    failed_entries = [entry.get("run_name", entry.get("name", "unknown benchmark")) for entry in document.get("benchmarks", []) if entry.get("error_occurred")]

    if failed_entries:
        failed_names = ", ".join(sorted(failed_entries))

        raise RuntimeError(f"Benchmark output contains failed measurements: {failed_names}.")

    new_measurements = load_measurements([result_file])
    new_variability = load_variability([result_file])

    validate_measurement_pairs(new_measurements)
    validate_variability(new_measurements, new_variability)

    for entry in document.get("benchmarks", []):
        entry_family = benchmark_entry_family(entry)

        if entry_family is None:
            continue

        backend, family = entry_family

        if expected_backend is not None and backend != expected_backend:
            continue

        grouped_entries.setdefault((backend, family), []).append(entry)

    if expected_backend is not None and not grouped_entries:
        raise RuntimeError(f"Benchmark output for {expected_backend} did not contain any recognized tables.")

    if expected_backend is not None and expected_families:
        missing_families = [family for family in expected_families if (expected_backend, family) not in grouped_entries]

        if missing_families:
            missing_names = ", ".join(OPERATION_FAMILY_NAMES[family] for family in missing_families)

            raise RuntimeError(f"Benchmark output for {expected_backend} did not contain the selected table(s): {missing_names}.")

    cached_files = []

    for (backend, family), entries in sorted(grouped_entries.items()):
        backend_cache = raw / backend
        backend_cache.mkdir(parents=True, exist_ok=True)
        cached_file = backend_cache / f"{family}.json"

        if cached_file.exists() and not overwrite:
            continue

        cached_document = dict(document)
        cached_document["benchmarks"] = entries

        if run_policy is not None:
            cached_document["vext_run_policy"] = run_policy

        temporary_file = cached_file.with_name(f".{cached_file.name}.tmp")
        temporary_file.write_text(json.dumps(cached_document, indent=2) + "\n")
        temporary_file.replace(cached_file)
        cached_files.append(cached_file)

    return cached_files


def migrate_legacy_result_cache(
    raw,
):
    for legacy_name in ("compare_cpu.json", "compare_cuda.json"):
        legacy_file = raw / legacy_name

        if legacy_file.exists():
            cache_result_file(legacy_file, raw, overwrite=False)


def cached_result_files(
    raw,
):
    return sorted(file for backend in ("CPU", "CUDA") for file in (raw / backend).glob("*.json"))


def has_cached_backend_results(
    raw,
    backend,
):
    legacy_name = "compare_cpu.json" if backend == "CPU" else "compare_cuda.json"
    return (raw / legacy_name).exists() or any((raw / backend).glob("*.json"))


def execute_benchmarks(
    build,
    work,
    cpu_enabled,
    cuda_enabled,
    minimum_time,
    warmup_time,
    repetitions,
    operation_families,
):
    raw = work / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    migrate_legacy_result_cache(raw)

    if cpu_enabled:
        cpu_result = raw / "latest_cpu.json"

        run(
            [
                build / "bin" / "bench_compare_cpu",
                *benchmark_arguments(
                    cpu_result,
                    minimum_time,
                    warmup_time,
                    repetitions,
                    operation_families,
                ),
            ]
        )
        cache_result_file(
            cpu_result,
            raw,
            overwrite=True,
            expected_backend="CPU",
            expected_families=operation_families,
            run_policy={
                "minimum_time": minimum_time,
                "warmup_time": warmup_time,
                "repetitions": repetitions,
                "execution_order": "vext_then_reference",
            },
        )

    if cuda_enabled:
        cuda_result = raw / "latest_cuda.json"

        run(
            [
                build / "bin" / "bench_compare_cuda",
                *benchmark_arguments(
                    cuda_result,
                    minimum_time,
                    warmup_time,
                    repetitions,
                    operation_families,
                ),
            ]
        )
        cache_result_file(
            cuda_result,
            raw,
            overwrite=True,
            expected_backend="CUDA",
            expected_families=operation_families,
            run_policy={
                "minimum_time": minimum_time,
                "warmup_time": warmup_time,
                "repetitions": repetitions,
                "execution_order": "vext_then_reference",
            },
        )

    return cached_result_files(raw)


def implementation_filter(implementation, backend, operation_families):
    families = "|".join(operation_families) if operation_families else ".*"
    return f"^{implementation}/{backend}/({families})_"


def run_single_results(
    build,
    work,
    cpu_enabled,
    cuda_enabled,
    operation_families,
):
    """Run each selected backend once and keep its raw JSON in the build cache."""
    raw = work / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    result_files = []

    if cpu_enabled:
        output = raw / "latest_cpu.json"
        run(
            [
                build / "bin" / "bench_compare_cpu",
                *benchmark_arguments(output, operation_families=operation_families),
            ]
        )
        result_files.append(output)

    if cuda_enabled:
        output = raw / "latest_cuda.json"
        run(
            [
                build / "bin" / "bench_compare_cuda",
                *benchmark_arguments(output, operation_families=operation_families),
            ]
        )
        result_files.append(output)

    return result_files


def write_single_run_report(report, result_files, metadata):
    measurements = load_measurements(result_files)
    validate_measurement_pairs(measurements)

    rows = []

    for backend, operation, problem, implementation in sorted(measurements):
        if implementation != "Vext":
            continue

        vext_time = measurements[(backend, operation, problem, implementation)]

        for reference_key, reference_time in sorted(measurements.items()):
            if reference_key[:3] != (backend, operation, problem) or reference_key[3] == "Vext":
                continue

            rows.append(
                {
                    "backend": backend,
                    "operation": operation,
                    "problem": problem,
                    "reference": reference_key[3],
                    "vext_ns": vext_time,
                    "reference_ns": reference_time,
                    # format_time_difference accepts a time ratio and converts
                    # it to a signed percentage exactly once.
                    "ratio": vext_time / reference_time,
                }
            )

    lines = [
        "# Benchmark results",
        "",
        "Single-run FP32 measurements. Difference is `(vext - reference) / reference`: negative values mean vext is faster; positive values mean vext is slower.",
        "",
        "## Environment",
        "",
        f"- Machine: {metadata['machine_id']}",
        f"- System: {metadata['platform']}",
        f"- CPU: {metadata['cpu']}",
        f"- GPU: {metadata.get('gpu', 'Not benchmarked')}",
        f"- Compiler: {metadata['compiler']}",
        "- Precision: FP32",
        "- Timing: one invocation per benchmark case; native C++ measurements are kernel-only.",
        "",
    ]

    for backend in ("CPU", "CUDA"):
        backend_rows = [row for row in rows if row["backend"] == backend]

        if not backend_rows:
            continue

        lines.extend([f"## {backend}", ""])
        families = sorted({split_operation(row["operation"])[0] for row in backend_rows})

        for family in families:
            family_rows = [row for row in backend_rows if split_operation(row["operation"])[0] == family]
            lines.extend([f"### {OPERATION_FAMILY_NAMES.get(family, family)}", ""])

            ranks = sorted({problem_rank(row["problem"]) for row in family_rows})

            for rank in ranks:
                rank_rows = [row for row in family_rows if problem_rank(row["problem"]) == rank]
                rank_name = f"{rank}D tensors" if rank is not None else "Specialized shapes"
                lines.extend(
                    [
                        f"#### {rank_name}",
                        "",
                        "| Operation | Size | Reference | vext time | Reference time | Difference |",
                        "| --- | --- | --- | ---: | ---: | ---: |",
                    ]
                )

                for row in rank_rows:
                    _, _, operation_name = split_operation(row["operation"])
                    lines.append(f"| {operation_name} | {format_problem_size(row['problem'])} | {row['reference']} | {format_time(row['vext_ns'])} | {format_time(row['reference_ns'])} | {format_time_difference(row['ratio'])} |")

                lines.append("")

    report.write_text("\n".join(lines).rstrip() + "\n")


def write_paper_report(report, result_files, metadata):
    """Write a paper report from independent process samples."""
    samples = {}

    for file in result_files:
        for key, elapsed in load_measurements([file]).items():
            samples.setdefault(key, []).append(elapsed)

    for key, values in samples.items():
        if len(values) != PAPER_SAMPLES:
            raise RuntimeError(f"Paper benchmark '{'/'.join(key)}' has {len(values)} samples; expected {PAPER_SAMPLES}.")

    def summary(key):
        values = sorted(samples[key])
        return statistics.median(values), values[15] - values[4]

    rows = {"Kernel benchmarks": []}

    for backend, operation, problem, implementation in sorted(samples):
        if implementation != "Vext":
            continue

        vext, vext_iqr = summary((backend, operation, problem, implementation))
        references = [key for key in samples if key[:3] == (backend, operation, problem) and key[3] != "Vext"]

        if not references:
            raise RuntimeError(f"Paper benchmark case {backend}/{operation}/{problem} has no reference.")

        for reference_key in sorted(references):
            reference, reference_iqr = summary(reference_key)
            rows["Kernel benchmarks"].append(
                (
                    backend,
                    operation,
                    problem,
                    reference_key[3],
                    vext,
                    vext_iqr,
                    reference,
                    reference_iqr,
                )
            )

    lines = [
        "# Benchmark results",
        "",
        "Paper collection: 20 independent fresh-process FP32 samples per case. Values are median (IQR). Difference is `(vext - reference) / reference`: negative is faster.",
        "",
        "## Environment",
        "",
        f"- Machine: {metadata['machine_id']}",
        f"- System: {metadata['platform']}",
        f"- CPU: {metadata['cpu']}",
        f"- GPU: {metadata.get('gpu', 'Not benchmarked')}",
        f"- Compiler: {metadata['compiler']}",
        "- Native CPU metric: Google Benchmark CPU time; native CUDA metric: CUDA event time.",
        "",
    ]
    for section, section_rows in rows.items():
        if not section_rows:
            continue

        lines.extend([f"## {section}", ""])

        for backend in ("CPU", "CUDA"):
            backend_rows = [row for row in section_rows if row[0] == backend]

            if not backend_rows:
                continue

            lines.extend([f"### {backend}", ""])

            for family in sorted({split_operation(row[1])[0] for row in backend_rows}):
                family_rows = [row for row in backend_rows if split_operation(row[1])[0] == family]
                lines.extend(
                    [
                        f"#### {OPERATION_FAMILY_NAMES.get(family, family)}",
                        "",
                        "| Operation | Size | Reference | vext median (IQR) | Reference median (IQR) | Difference |",
                        "| --- | --- | --- | ---: | ---: | ---: |",
                    ]
                )

                for (
                    _,
                    operation,
                    problem,
                    reference_name,
                    vext,
                    vext_iqr,
                    reference,
                    reference_iqr,
                ) in family_rows:
                    _, _, name = split_operation(operation)
                    difference = format_time_difference(vext / reference)
                    lines.append(f"| {name} | {format_problem_size(problem)} | {reference_name} | {format_time(vext)} ({format_time(vext_iqr)}) | {format_time(reference)} ({format_time(reference_iqr)}) | {difference} |")
                lines.append("")

    report.write_text("\n".join(lines).rstrip() + "\n")


def format_problem_size(problem):
    """Turn the rank-prefixed dense benchmark arguments into a normal tensor shape."""
    dimensions = problem.split("x")

    if len(dimensions) == 5 and dimensions[0] in {"1", "2", "3", "4"}:
        return "x".join(dimensions[1 : int(dimensions[0]) + 1])

    return problem


def problem_rank(problem):
    dimensions = problem.split("x")

    if len(dimensions) == 5 and dimensions[0] in {"1", "2", "3", "4"}:
        return int(dimensions[0])

    return None


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
            problem_parts = [part for part in parts[3:] if part != "manual_time" and not part.startswith("iterations:")]

            problem = "x".join(problem_parts)
            unit = entry.get("time_unit", "ns")
            # CPU comparisons use process CPU time; CUDA manual timings are
            # device elapsed time carried in real_time.
            elapsed_field = "cpu_time" if backend == "CPU" else "real_time"
            elapsed = entry[elapsed_field] * TIME_TO_NS[unit]
            key = (backend, operation, problem, implementation)

            if not math.isfinite(elapsed) or elapsed <= 0.0:
                raise RuntimeError(f"Benchmark '{run_name}' produced invalid elapsed time {elapsed} ns.")

            if key in measurements:
                raise RuntimeError(f"Benchmark results contain a duplicate median for '{run_name}'.")

            measurements[key] = elapsed

    return measurements


def load_variability(
    result_files,
):
    variability = {}

    for result_file in result_files:
        document = json.loads(result_file.read_text())

        for entry in document.get("benchmarks", []):
            if entry.get("error_occurred") or entry.get("aggregate_name") != "cv":
                continue

            run_name = entry.get("run_name", entry["name"])
            parts = run_name.split("/")

            if len(parts) < 4:
                continue

            implementation, backend, operation = parts[:3]
            problem_parts = [part for part in parts[3:] if part != "manual_time" and not part.startswith("iterations:")]

            problem = "x".join(problem_parts)
            coefficient = entry["real_time"]

            if not math.isfinite(coefficient) or coefficient < 0.0:
                raise RuntimeError(f"Benchmark '{run_name}' produced invalid coefficient of variation {coefficient}.")

            variability[(backend, operation, problem, implementation)] = coefficient

    return variability


def validate_measurement_pairs(
    measurements,
):
    cases = {}

    for backend, operation, problem, implementation in measurements:
        cases.setdefault((backend, operation, problem), set()).add(implementation)

    for (backend, operation, problem), implementations in sorted(cases.items()):
        references = implementations - {"Vext"}

        if "Vext" not in implementations or not references:
            available = ", ".join(sorted(implementations))
            raise RuntimeError(f"Benchmark case {backend}/{operation}/{problem} is not paired; found: {available}.")


def validate_variability(
    measurements,
    variability,
):
    missing = sorted(set(measurements) - set(variability))

    if missing:
        missing_names = ", ".join(f"{backend}/{operation}/{problem}/{implementation}" for backend, operation, problem, implementation in missing)

        raise RuntimeError(f"Benchmark results do not contain CV aggregates for: {missing_names}.")


def stability_summary(
    variability,
):
    if not variability:
        return "Unavailable for cached results."

    backend_summaries = []

    for backend in sorted({key[0] for key in variability}):
        coefficients = [coefficient for key, coefficient in variability.items() if key[0] == backend]
        maximum = max(coefficients)
        unstable = sum(coefficient > 0.05 for coefficient in coefficients)
        backend_summaries.append(f"{backend} maximum CV {maximum * 100.0:.1f}% ({unstable} measurement(s) above 5%)")

    return "; ".join(backend_summaries) + "."


def unstable_measurements(
    variability,
    threshold=0.05,
):
    return sorted(
        (
            (backend, operation, problem, implementation, coefficient)
            for (
                backend,
                operation,
                problem,
                implementation,
            ), coefficient in variability.items()
            if coefficient > threshold
        ),
        key=lambda entry: entry[4],
        reverse=True,
    )


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

    return run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture=True).splitlines()[0]


def cuda_toolkit(
    cuda_enabled,
):
    if not cuda_enabled:
        return "Not used"

    version = run(["nvcc", "--version"], capture=True)
    return version.splitlines()[-1]


def report_environment_value(
    report,
    label,
):
    if not report.exists():
        return None

    prefix = f"- {label}: "

    for line in report.read_text(errors="replace").splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :]

    return None


def git_revision(
    root,
):
    revision = run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture=True)
    dirty = bool(run(["git", "status", "--porcelain"], cwd=root, capture=True))
    return f"{revision}{' (dirty)' if dirty else ''}"


def command_output(command, *, cwd=None):
    try:
        return run(command, cwd=cwd, capture=True)
    except (OSError, subprocess.CalledProcessError):
        return "Unavailable"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paper_metadata(root, build, cuda_enabled, machine_id):
    """Capture the host and build information shown in the benchmark report."""
    metadata = {
        "schema_version": 1,
        "machine_id": machine_id,
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "cpu_governor": command_output(["bash", "-lc", "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"]),
        "lscpu": command_output(["lscpu"]),
        "memory": command_output(["free", "-b"]),
        "compiler": compiler_name(build),
        "compile_commands_sha256": sha256_file(build / "compile_commands.json"),
        "cmake_cache_sha256": sha256_file(build / "CMakeCache.txt"),
        "eigen_version": BENCHMARK_VERSIONS["Eigen"],
        "google_benchmark_version": BENCHMARK_VERSIONS["Google Benchmark"],
        "cuda_enabled": cuda_enabled,
    }

    if cuda_enabled:
        metadata.update(
            {
                "gpu": gpu_name(True),
                "cuda_toolkit": cuda_toolkit(True),
                "nvidia_smi": command_output(["nvidia-smi", "-q"]),
                "nvidia_smi_clocks": command_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=clocks.current.graphics,clocks.current.memory,power.limit,persistence_mode",
                        "--format=csv,noheader",
                    ]
                ),
            }
        )

    return metadata


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


def format_time_difference(ratio):
    difference = (ratio - 1.0) * 100.0

    if abs(difference) < 0.05:
        return "0.0%"

    return f"{difference:+.1f}%"


def split_operation(operation):
    family, separator, name = operation.partition("_")
    family_name = OPERATION_FAMILY_NAMES.get(family, family)
    operation_name = name if separator else operation
    return family, family_name, operation_name


def write_report(
    report,
    measurements,
    compiler,
    root,
    gpu,
    cuda_version,
    minimum_time,
    warmup_time,
    repetitions,
    stability,
    unstable,
):
    rows = []

    vext_cases = sorted(key for key in measurements if key[3] == "Vext")

    for backend, operation, problem, _ in vext_cases:
        vext_time = measurements[(backend, operation, problem, "Vext")]
        reference_keys = sorted(key for key in measurements if key[:3] == (backend, operation, problem) and key[3] != "Vext")

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

    generated = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    cuda_references = sorted({row[3] for row in rows if row[0] == "CUDA"})
    cuda_reference_summary = ", ".join(cuda_references) if cuda_references else "none in the currently cached results"

    lines = [
        "# Benchmark results",
        "",
        "Generated by `python3 benchmarks/run.py`. Lower time is better. The signed time difference is `(vext - reference) / reference`: negative values mean vext is faster; positive values mean vext is slower.",
        "",
        "## Environment",
        "",
        f"- Generated: {generated}",
        f"- Revision: `{git_revision(root)}`",
        f"- System: {platform.platform()}",
        f"- CPU: {cpu_name()}",
        f"- GPU: {gpu}",
        f"- CUDA toolkit: {cuda_version}",
        f"- Compiler: {compiler}",
        "- Reported statistic: median of the repetitions stored in each cached benchmark result",
        f"- Current run policy: {repetitions} repetitions, minimum {minimum_time} seconds and {warmup_time} seconds warm-up, vext benchmarks followed by reference benchmarks",
        "- Retained tables may come from earlier selective runs with different timing settings",
        f"- Stability: {stability}",
        "- CPU policy: Eigen internal parallelism disabled",
        f"- Dependencies: Eigen {BENCHMARK_VERSIONS['Eigen']}, Google Benchmark {BENCHMARK_VERSIONS['Google Benchmark']}",
        f"- CUDA references in this report: {cuda_reference_summary}",
        "",
    ]

    if unstable:
        lines.extend(
            [
                "## Reliability warnings",
                "",
                "The following measurements exceeded 5% coefficient of variation and should be rerun before drawing conclusions:",
                "",
            ]
        )

        for backend, operation, problem, implementation, coefficient in unstable:
            lines.append(f"- `{implementation}/{backend}/{operation}/{problem}`: {coefficient * 100.0:.1f}% CV")

        lines.append("")

    backends = list(dict.fromkeys(row[0] for row in rows))

    for backend in backends:
        lines.extend([f"## {backend}", ""])
        backend_rows = [row for row in rows if row[0] == backend]
        families = list(dict.fromkeys(split_operation(row[1])[0] for row in backend_rows))

        for family in families:
            family_name = OPERATION_FAMILY_NAMES.get(family, family)
            lines.extend(
                [
                    f"### {family_name}",
                    "",
                    "| Operation | Problem size | Reference library | vext time | Reference time | difference |",
                    "| --- | ---: | --- | ---: | ---: | ---: |",
                ]
            )

            for (
                row_backend,
                operation,
                problem,
                reference,
                vext_time,
                reference_time,
                ratio,
            ) in backend_rows:
                operation_family, _, operation_name = split_operation(operation)

                if operation_family != family:
                    continue

                lines.append(f"| {operation_name} | {problem} | {reference} | {format_time(vext_time)} | {format_time(reference_time)} | {format_time_difference(ratio)} |")

            lines.append("")

    if not rows:
        lines.extend(["## Results", "", "No paired results were produced.", ""])

    lines.extend(
        [
            "Results are machine-specific and should be compared only when generated on equivalent hardware with the same build configuration.",
            "",
        ]
    )

    report.write_text("\n".join(lines))
    print(f"Wrote {report}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Build, run, and report vext comparison benchmarks.")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run CPU benchmarks. When no backend is selected, CPU and available CUDA benchmarks are run.",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Run CUDA benchmarks and require an accessible NVIDIA device.",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help=f"Collect {PAPER_SAMPLES} independent fresh-process samples and write a paper report.",
    )
    operation_group = parser.add_argument_group("operation families")

    for option, family in OPERATION_FAMILY_OPTIONS.items():
        operation_group.add_argument(
            f"--{option.replace('_', '-')}",
            action="store_true",
            help=f"Run only the {OPERATION_FAMILY_NAMES[family]} benchmark table.",
        )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the selected benchmark build directory before running.",
    )
    parser.add_argument(
        "--machine-id",
        help="Stable host label recorded in the generated report (defaults to the system hostname).",
    )
    parser.add_argument(
        "--timing-scope",
        choices=("kernel", "end-to-end"),
        default="kernel",
        help="Timing scope recorded for the run; kernel measurements exclude setup and transfers.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    benchmark_root = pathlib.Path(__file__).resolve().parent
    root = benchmark_root.parent
    work = (benchmark_root / "build").resolve()
    report = (benchmark_root / ("RESULTS.md" if arguments.paper else "RESULTS.dev.md")).resolve()
    architecture = "native"
    jobs = max(1, os.cpu_count() or 1)

    for tool in ("cmake", "conan", "ninja"):
        ensure_tool(tool)

    if arguments.timing_scope != "kernel":
        raise ValueError("End-to-end timing is not implemented by the current benchmark kernels.")

    machine_id = arguments.machine_id or platform.node() or "unknown-host"

    cpu_enabled, cuda_enabled = choose_backends(arguments.cpu, arguments.cuda)
    operation_families = selected_operation_families(arguments)

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
    metadata = paper_metadata(root, build, cuda_enabled, machine_id)

    metadata.update(
        {
            "precision": "FP32",
            "timing_scope": "kernel",
            "repetitions": ARTIFACT_REPETITIONS,
            "minimum_time_seconds": ARTIFACT_MINIMUM_TIME,
            "warmup_time_seconds": ARTIFACT_WARMUP_TIME,
            "operation_families": operation_families or list(OPERATION_FAMILY_NAMES),
        }
    )

    if arguments.paper:
        stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        paper_raw = work / "raw" / "paper" / machine_id / stamp
        paper_raw.mkdir(parents=True, exist_ok=False)
        result_files = []

        for sample in range(1, PAPER_SAMPLES + 1):
            sample_files = run_single_results(build, work, cpu_enabled, cuda_enabled, operation_families)

            for source in sample_files:
                destination = paper_raw / f"sample-{sample:02d}-{source.name}"
                source.replace(destination)
                result_files.append(destination)

        write_paper_report(report, result_files, metadata)
    else:
        result_files = run_single_results(build, work, cpu_enabled, cuda_enabled, operation_families)

        write_single_run_report(report, result_files, metadata)

    print(f"Wrote {report}")


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
