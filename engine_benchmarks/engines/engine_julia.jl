#!/usr/bin/env julia
# Julia PauliPropagation.jl subprocess worker.
#
# Usage:
#     julia engine_julia.jl <config.json> <result.json> [--surrogate]
#
# Mirrors engine_qiskit.py: reads a JSON config describing the circuit,
# runs propagation with truncation, and writes a JSON result with timing,
# RAM usage, and term counts.
#
# Modes:
#   default      — per-repetition full propagation (t_eval_s), plus, when the
#                  installed PauliPropagation.jl provides `rewindgradient`
#                  (v0.8.0+), one paired forward+backward sweep per repetition
#                  timed as t_fwdbwd_s. t_fwdbwd_s is the full time to obtain
#                  expval AND the gradient in one call; it is NOT a separate
#                  backward-only time, so it is comparable to other engines'
#                  t_eval_s + t_bwd_s, not to t_bwd_s alone.
#   --surrogate  — PP.jl's compile-once/evaluate-many surrogate (NodePathProperties
#                  graph). Build the graph once and `zerofilter!` it (compile
#                  stage, as in PauliPropagation.jl v0.8.1
#                  (github.com/MSRudolph/PauliPropagation.jl): examples/PP-Surrogate.ipynb),
#                  then re-evaluate per repetition with `evaluate!` + `overlapwithzero`.
#                  The surrogate cannot truncate on numerical coefficient values, so
#                  min_abs_coeff cases return status "Unsupported". In v0.8.1 the
#                  src/Surrogate/Surrogate.jl header marks the submodule
#                  experimental/legacy; the result records that.

using PauliPropagation
using Pkg
using JSON
using Printf

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

const PAULI_SYM = Dict("X" => :X, "Y" => :Y, "Z" => :Z)

function engine_versions()
    vers = Dict{String,Any}("julia" => string(VERSION), "PauliPropagation" => nothing)
    try
        for (_, dep) in Pkg.dependencies()
            if dep.name == "PauliPropagation"
                vers["PauliPropagation"] = dep.version === nothing ? nothing : string(dep.version)
                break
            end
        end
    catch
    end
    return vers
end

function parse_pauli_string(s::AbstractString)
    out = Symbol[]
    for c in s
        push!(out, PAULI_SYM[string(c)])
    end
    return out
end

# Windowed high-water RSS: writing "5" to /proc/self/clear_refs resets the
# kernel's peak-RSS counter (VmHWM) to the current RSS, so a subsequent VmHWM
# read reports the high-water mark of only the code run since the reset.
# Without the reset, the peak is monotone over the process lifetime and
# per-step deltas after warm-up come out structurally 0.
const CLEAR_REFS_OK = Ref{Bool}(true)

function reset_peak_rss!()
    CLEAR_REFS_OK[] || return false
    try
        open("/proc/self/clear_refs", "w") do f
            write(f, "5")
        end
        return true
    catch
        CLEAR_REFS_OK[] = false
        return false
    end
end

function get_peak_rss_mb()
    try
        for line in eachline("/proc/self/status")
            if startswith(line, "VmHWM:")
                kb = parse(Int, split(line)[2])
                return kb / 1024.0
            end
        end
    catch
    end
    # Fallback: process-lifetime peak (no windowing possible)
    return Sys.maxrss() / (1024.0 ^ 2)
end

# Absolute current RSS (VmRSS), recorded alongside the windowed-HWM delta so
# "how much RAM the engine holds during the loop" is answerable (see the
# matching change in the Python workers).
function get_current_rss_mb()
    try
        for line in eachline("/proc/self/status")
            if startswith(line, "VmRSS:")
                return parse(Float64, split(line)[2]) / 1024.0
            end
        end
    catch
    end
    return Sys.maxrss() / (1024.0 ^ 2)
end

memory_kind_string() = CLEAR_REFS_OK[] ?
    "cpu_ram_highwater_delta_MB (VmHWM reset per step)" :
    "cpu_ram_peak_delta_MB (VmHWM reset unavailable; lifetime peak, steady deltas unreliable)"

# ---------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------

function build_circuit(n_qubits::Int, gate_defs::Vector, theta::Vector,
                      embedding::Union{Vector,Nothing})
    circuit = Gate[]
    for g in gate_defs
        gtype = g["type"]
        qubits = Int[q + 1 for q in g["qubits"]]  # 0-indexed → 1-indexed

        if gtype == "clifford"
            sym = g["symbol"]
            if sym == "H"
                for q in qubits
                    push!(circuit, CliffordGate(:H, q))
                end
            elseif sym == "S"
                for q in qubits
                    push!(circuit, CliffordGate(:S, q))
                end
            elseif sym == "SDG"
                for q in qubits
                    push!(circuit, CliffordGate(:Sdg, q))
                end
            elseif sym == "CX" || sym == "CNOT"
                @assert length(qubits) == 2
                push!(circuit, CliffordGate(:CNOT, (qubits[1], qubits[2])))
            elseif sym == "SWAP"
                @assert length(qubits) == 2
                push!(circuit, CliffordGate(:SWAP, (qubits[1], qubits[2])))
            else
                error("Unsupported Clifford symbol: $sym")
            end
        elseif gtype == "rotation" || gtype == "embedding"
            paulis = parse_pauli_string(g["pauli"])
            if length(paulis) == 1
                push!(circuit, PauliRotation(paulis[1], qubits[1]))
            else
                push!(circuit, PauliRotation(paulis, qubits))
            end
        else
            error("Unsupported gate type: $gtype")
        end
    end
    return circuit
end

# rewindgradient returns one component per parametrized gate, in the same order
# build_theta_vector() pushes them. Scatter those onto theta's own indexing so
# the value can be compared with the other engines (embedding angles, which
# share the gate vector, are dropped).
function scatter_grad_to_theta(gate_defs::Vector, grad, n_theta::Int)
    out = zeros(Float64, n_theta)
    pos = 0
    for g in gate_defs
        gtype = g["type"]
        if gtype == "rotation"
            pos += 1
            pos <= length(grad) || return nothing
            out[Int(g["pidx"]) + 1] = Float64(real(grad[pos]))
        elseif gtype == "embedding"
            pos += 1
        end
    end
    return out
end

function build_theta_vector(gate_defs::Vector, theta::Vector,
                             embedding::Union{Vector,Nothing})
    # Julia's propagate() takes one scalar per parametrized gate in the order
    # they were pushed into `circuit`. We replicate the same iteration order
    # used in build_circuit() above so indices line up.
    thetas = Float64[]
    for g in gate_defs
        gtype = g["type"]
        if gtype == "rotation"
            pidx = Int(g["pidx"]) + 1  # 0-indexed → 1-indexed
            push!(thetas, Float64(theta[pidx]))
        elseif gtype == "embedding"
            @assert embedding !== nothing
            eidx = Int(g["eidx"]) + 1
            push!(thetas, Float64(embedding[eidx]))
        end
        # Clifford gates carry no parameter
    end
    return thetas
end

function build_observable(n_qubits::Int, edges::Vector, fields::Vector)
    obs = PauliSum(n_qubits)
    for entry in edges
        u, v, w = Int(entry[1]) + 1, Int(entry[2]) + 1, Float64(entry[3])
        add!(obs, PauliString(n_qubits, [:Z, :Z], [u, v], w))
    end
    for entry in fields
        q, h = Int(entry[1]) + 1, Float64(entry[2])
        add!(obs, PauliString(n_qubits, :Z, q, h))
    end
    return obs
end

function is_embedding_batch(embedding)
    return embedding !== nothing && length(embedding) > 0 && embedding[1] isa AbstractVector
end

function truncation_kwargs(max_weight, min_abs_coeff)
    kw = Pair{Symbol,Any}[]
    if max_weight !== nothing && max_weight > 0
        push!(kw, :max_weight => Int(max_weight))
    end
    if min_abs_coeff > 0.0
        push!(kw, :min_abs_coeff => min_abs_coeff)
    end
    return kw
end

function propagate_with_truncation(circuit, obs, thetas, max_weight, min_abs_coeff)
    return propagate(circuit, obs, thetas; truncation_kwargs(max_weight, min_abs_coeff)...)
end

# rewindgradient (PP.jl v0.8.0+, src/gradient.jl): one paired forward+backward
# sweep returning (expval, grad). Requires all parametrized gates to be
# PauliRotations and no noise channels — both hold for this benchmark's
# circuits. Truncation kwargs pass through to applymergetruncate! exactly as
# in propagate().
grad_supported() = isdefined(PauliPropagation, :rewindgradient)

function rewind_with_truncation(circuit, obs, thetas, max_weight, min_abs_coeff)
    return rewindgradient(circuit, obs, thetas, overlapwithzero;
                          truncation_kwargs(max_weight, min_abs_coeff)...)
end

function parse_common(config::Dict)
    problem   = config["problem"]
    gate_defs = config["gate_defs"]
    theta     = config["theta"]
    embedding = get(config, "embedding", nothing)
    n_reps    = Int(get(config, "n_reps", 5))
    n_qubits  = Int(problem["n_qubits"])
    edges     = problem["edges"]
    fields    = problem["fields"]
    raw_max_weight = get(config, "max_weight", nothing)
    max_weight = (raw_max_weight === nothing) ? nothing : Int(raw_max_weight)
    raw_mac = get(config, "min_abs_coeff", nothing)
    min_abs_coeff = (raw_mac === nothing) ? 0.0 : Float64(raw_mac)
    return (; gate_defs, theta, embedding, n_reps, n_qubits, edges, fields,
            max_weight, min_abs_coeff)
end

function summarize(history::Vector; batch_size::Union{Int,Nothing}=nothing)
    first = history[1]
    steady = length(history) > 1 ? history[2:end] : history
    avg(key) = sum(x[key] for x in steady) / length(steady)
    has(key) = haskey(first, key)
    summary = Dict{String,Any}(
        "t_eval_s_first"     => first["t_eval_s"],
        "t_eval_s_steady"    => avg("t_eval_s"),
        "step_ram_MB_first"  => first["step_ram_MB"],
        "step_ram_MB_steady" => avg("step_ram_MB"),
        # Mean occupancy over ALL reps — matches the Python summarize_history.
        "ram_used_end_MB_mean" => (all(haskey(x, "ram_used_end_MB") for x in history) ?
            sum(Float64(x["ram_used_end_MB"]) for x in history) / length(history) : nothing),
        "final_n_terms"      => Int(history[end]["n_terms"]),
        "final_expval"       => Float64(history[end]["expval"]),
    )
    if has("t_fwdbwd_s")
        summary["t_fwdbwd_s_first"]  = first["t_fwdbwd_s"]
        summary["t_fwdbwd_s_steady"] = avg("t_fwdbwd_s")
    end
    if batch_size !== nothing
        summary["batch_size"] = batch_size
        summary["batch_throughput_evals_s_first"]  = first["throughput_evals_s"]
        summary["batch_throughput_evals_s_steady"] = avg("throughput_evals_s")
        if has("throughput_train_s")
            summary["batch_throughput_train_s_first"]  = first["throughput_train_s"]
            summary["batch_throughput_train_s_steady"] = avg("throughput_train_s")
        end
        summary["native_batch"] = false
    end
    return summary
end

# ---------------------------------------------------------------------
# Default mode: full propagation per repetition (+ rewindgradient fwd+bwd)
# ---------------------------------------------------------------------

function run_embedding_batch(config::Dict)
    c = parse_common(config)
    embedding = c.embedding
    batch_size = length(embedding)

    GC.gc()
    reset_peak_rss!()
    rss_start = get_peak_rss_mb()
    t0 = time_ns()
    circuit = build_circuit(c.n_qubits, c.gate_defs, c.theta, embedding[1])
    obs = build_observable(c.n_qubits, c.edges, c.fields)
    compile_time = (time_ns() - t0) / 1e9
    compile_mem = max(0.0, get_peak_rss_mb() - rss_start)

    row_thetas(i) = build_theta_vector(c.gate_defs, c.theta, embedding[i])
    call_one(i) = propagate_with_truncation(circuit, obs, row_thetas(i),
                                            c.max_weight, c.min_abs_coeff)
    with_grad = grad_supported()
    rewind_one(i) = rewind_with_truncation(circuit, obs, row_thetas(i),
                                           c.max_weight, c.min_abs_coeff)

    # Warmup (Julia JIT compile) for both code paths
    GC.gc()
    try
        _ = call_one(1)
        if with_grad
            _ = rewind_one(1)
        end
    catch e
        return Dict(
            "engine" => "julia",
            "versions" => engine_versions(),
            "status" => "Error",
            "error"  => "batch warmup failed: $(sprint(showerror, e))",
            "compile_time_s" => compile_time,
            "compile_ram_MB" => compile_mem,
            "history" => [],
        )
    end

    history = []
    grad_row0 = nothing
    for step in 0:(c.n_reps - 1)
        GC.gc()
        reset_peak_rss!()
        mem_baseline = get_peak_rss_mb()
        rss_start = get_current_rss_mb()
        expval = NaN
        n_terms = 0
        t0 = time_ns()
        for i in 1:batch_size
            result = call_one(i)
            ev = overlapwithzero(result)
            if i == 1
                expval = real(ev)
            end
            n_terms = length(result)
        end
        t_eval = (time_ns() - t0) / 1e9

        rec = Dict{String,Any}(
            "step"        => step,
            "t_eval_s"    => t_eval,
            "n_terms"     => n_terms,
            "expval"      => expval,
            "batch_size"  => batch_size,
            "throughput_evals_s" => batch_size / t_eval,
            "native_batch" => false,
        )

        if with_grad
            t0 = time_ns()
            for i in 1:batch_size
                _ = rewind_one(i)
            end
            t_fwdbwd = (time_ns() - t0) / 1e9
            rec["t_fwdbwd_s"] = t_fwdbwd
            rec["throughput_train_s"] = batch_size / t_fwdbwd
            # Untimed, once: keep row 1's gradient so the paired sweep above is
            # verifiable against the other engines instead of only timed.
            if step == 0 && grad_row0 === nothing
                try
                    _, g = rewind_one(1)
                    grad_row0 = scatter_grad_to_theta(c.gate_defs, g, length(c.theta))
                catch
                    grad_row0 = nothing
                end
            end
        end

        rec["step_ram_MB"] = max(0.0, get_peak_rss_mb() - mem_baseline)
        rec["ram_used_start_MB"] = rss_start
        rec["ram_used_end_MB"] = get_current_rss_mb()
        push!(history, rec)
    end

    return Dict(
        "engine"          => "julia",
        "versions"        => engine_versions(),
        "status"          => "Success",
        # PP.jl propagation has no zero filter: the count is everything propagated.
        "n_terms_kind"    => "propagated",
        "bwd_supported"   => true,
        "grad_row0"       => grad_row0,
        "grad_kind"       => "d(expval of batch row 1)/d theta (rewindgradient)",
        "compile_time_s"  => compile_time,
        "compile_ram_MB"  => compile_mem,
        "history"         => history,
        "summary"         => summarize(history; batch_size=batch_size),
        "config_echo"     => Dict(
            "max_weight_used"     => c.max_weight,
            "min_abs_coeff_used"  => c.min_abs_coeff,
            "n_reps"              => c.n_reps,
            "device"              => "cpu",
            "memory_kind"         => memory_kind_string(),
            "setup_kind"          => "build Julia circuit/observable; per-sample theta vector in batch loop",
            "batch_support"       => "single-sample loop",
            "gradient_kind"       => with_grad ?
                "rewindgradient (native, v0.8.0+): t_fwdbwd_s is expval+grad in one paired sweep" :
                "unavailable (PauliPropagation.jl < 0.8.0 has no built-in gradient entry point)",
        ),
    )
end

function run_benchmark(config::Dict)
    c = parse_common(config)
    if is_embedding_batch(c.embedding)
        return run_embedding_batch(config)
    end

    GC.gc()
    reset_peak_rss!()
    rss_start = get_peak_rss_mb()

    # Compile stage: build circuit + observable + theta vector
    t0 = time_ns()
    circuit = build_circuit(c.n_qubits, c.gate_defs, c.theta, c.embedding)
    obs = build_observable(c.n_qubits, c.edges, c.fields)
    thetas = build_theta_vector(c.gate_defs, c.theta, c.embedding)
    compile_time = (time_ns() - t0) / 1e9
    compile_mem = max(0.0, get_peak_rss_mb() - rss_start)

    call_propagate() = propagate_with_truncation(circuit, obs, thetas,
                                                 c.max_weight, c.min_abs_coeff)
    with_grad = grad_supported()
    call_rewind() = rewind_with_truncation(circuit, obs, thetas,
                                           c.max_weight, c.min_abs_coeff)

    # Warmup (Julia JIT compile) for both code paths
    GC.gc()
    try
        _ = call_propagate()
        if with_grad
            _ = call_rewind()
        end
    catch e
        return Dict(
            "engine" => "julia",
            "versions" => engine_versions(),
            "status" => "Error",
            "error"  => "warmup failed: $(sprint(showerror, e))",
            "compile_time_s" => compile_time,
            "compile_ram_MB" => compile_mem,
            "history" => [],
        )
    end

    history = []
    grad_row0 = nothing
    for step in 0:(c.n_reps - 1)
        GC.gc()
        reset_peak_rss!()
        mem_baseline = get_peak_rss_mb()
        rss_start = get_current_rss_mb()

        t0 = time_ns()
        result = call_propagate()
        ev = overlapwithzero(result)
        t_eval = (time_ns() - t0) / 1e9

        rec = Dict{String,Any}(
            "step"        => step,
            "t_eval_s"    => t_eval,
            "n_terms"     => length(result),
            "expval"      => real(ev),
        )

        if with_grad
            t0 = time_ns()
            expec, grad = call_rewind()
            t_fwdbwd = (time_ns() - t0) / 1e9
            rec["t_fwdbwd_s"] = t_fwdbwd
            rec["expval_fwdbwd"] = real(expec)
            rec["grad_l2"] = sqrt(sum(abs2, grad))
            if step == 0 && grad_row0 === nothing
                grad_row0 = scatter_grad_to_theta(c.gate_defs, grad, length(c.theta))
            end
        end

        rec["step_ram_MB"] = max(0.0, get_peak_rss_mb() - mem_baseline)
        rec["ram_used_start_MB"] = rss_start
        rec["ram_used_end_MB"] = get_current_rss_mb()
        push!(history, rec)
    end

    return Dict(
        "engine"          => "julia",
        "versions"        => engine_versions(),
        "status"          => "Success",
        # PP.jl propagation has no zero filter: the count is everything propagated.
        "n_terms_kind"    => "propagated",
        "bwd_supported"   => true,
        "grad_row0"       => grad_row0,
        "grad_kind"       => "d(expval)/d theta (rewindgradient)",
        "compile_time_s"  => compile_time,
        "compile_ram_MB"  => compile_mem,
        "history"         => history,
        "summary"         => summarize(history),
        "config_echo"     => Dict(
            "max_weight_used"     => c.max_weight,
            "min_abs_coeff_used"  => c.min_abs_coeff,
            "n_reps"              => c.n_reps,
            "device"              => "cpu",
            "memory_kind"         => memory_kind_string(),
            "setup_kind"          => "build Julia circuit, observable, and theta vector",
            "gradient_kind"       => with_grad ?
                "rewindgradient (native, v0.8.0+): t_fwdbwd_s is expval+grad in one paired sweep" :
                "unavailable (PauliPropagation.jl < 0.8.0 has no built-in gradient entry point)",
        ),
    )
end

# ---------------------------------------------------------------------
# Surrogate mode: build the NodePathProperties graph once, evaluate many
# ---------------------------------------------------------------------

function run_surrogate(config::Dict)
    c = parse_common(config)

    if c.min_abs_coeff > 0.0
        return Dict(
            "engine"   => "julia_surrogate",
            "versions" => engine_versions(),
            "status"   => "Unsupported",
            "error"    => "PP.jl surrogate cannot truncate on numerical coefficient values (min_abs_coeff)",
            "history"  => [],
        )
    end

    is_batch = is_embedding_batch(c.embedding)
    embed0 = is_batch ? c.embedding[1] : c.embedding

    GC.gc()
    reset_peak_rss!()
    rss_start = get_peak_rss_mb()

    # Compile stage: build the surrogate graph (circuit structure only; the
    # parameter values come later, at evaluate! time), then `zerofilter!` it so
    # `evaluate!` touches only |0>-contributing paths. This is the step the
    # official surrogate example applies before repeated evaluation
    # (examples/PP-Surrogate.ipynb) and the counterpart of PADO-Pauli's zero
    # filtering, so it belongs inside the timed compile stage.
    t0 = time_ns()
    circuit = build_circuit(c.n_qubits, c.gate_defs, c.theta, embed0)
    obs = build_observable(c.n_qubits, c.edges, c.fields)
    wrapped = wrapcoefficients(obs, NodePathProperties)
    surrogate = if c.max_weight !== nothing && c.max_weight > 0
        propagate(circuit, wrapped; max_weight=Int(c.max_weight))
    else
        propagate(circuit, wrapped)
    end
    n_terms_propagated = length(surrogate)
    zerofilter!(surrogate)
    compile_time = (time_ns() - t0) / 1e9
    compile_mem = max(0.0, get_peak_rss_mb() - rss_start)

    row_thetas(i) = build_theta_vector(c.gate_defs, c.theta, c.embedding[i])
    thetas0 = build_theta_vector(c.gate_defs, c.theta, embed0)

    function eval_surrogate(thetas)
        evaluate!(surrogate, thetas)
        return real(overlapwithzero(surrogate))
    end

    # Warmup (Julia JIT compile)
    GC.gc()
    try
        _ = eval_surrogate(thetas0)
    catch e
        return Dict(
            "engine" => "julia_surrogate",
            "versions" => engine_versions(),
            "status" => "Error",
            "error"  => "warmup failed: $(sprint(showerror, e))",
            "compile_time_s" => compile_time,
            "compile_ram_MB" => compile_mem,
            "n_terms_propagated" => n_terms_propagated,
            "history" => [],
        )
    end

    batch_size = is_batch ? length(c.embedding) : nothing
    n_terms = length(surrogate)

    history = []
    for step in 0:(c.n_reps - 1)
        GC.gc()
        reset_peak_rss!()
        mem_baseline = get_peak_rss_mb()
        rss_start = get_current_rss_mb()
        expval = NaN
        t0 = time_ns()
        if is_batch
            for i in 1:length(c.embedding)
                ev = eval_surrogate(row_thetas(i))
                if i == 1
                    expval = ev
                end
            end
        else
            expval = eval_surrogate(thetas0)
        end
        t_eval = (time_ns() - t0) / 1e9

        rec = Dict{String,Any}(
            "step"        => step,
            "t_eval_s"    => t_eval,
            "n_terms"     => n_terms,
            "expval"      => expval,
        )
        if is_batch
            rec["batch_size"] = batch_size
            rec["throughput_evals_s"] = batch_size / t_eval
            rec["native_batch"] = false
        end
        rec["step_ram_MB"] = max(0.0, get_peak_rss_mb() - mem_baseline)
        rec["ram_used_start_MB"] = rss_start
        rec["ram_used_end_MB"] = get_current_rss_mb()
        push!(history, rec)
    end

    return Dict(
        "engine"          => "julia_surrogate",
        "versions"        => engine_versions(),
        "status"          => "Success",
        # The evaluated set is post-zerofilter!, like PADO's; the pre-filter
        # graph size is reported separately as n_terms_propagated.
        "n_terms_kind"    => "post_zero_filter",
        "bwd_supported"   => false,
        "compile_time_s"  => compile_time,
        "compile_ram_MB"  => compile_mem,
        "n_terms_propagated" => n_terms_propagated,
        "history"         => history,
        "summary"         => summarize(history; batch_size=batch_size),
        "config_echo"     => Dict(
            "max_weight_used"     => c.max_weight,
            "min_abs_coeff_used"  => c.min_abs_coeff,
            "n_reps"              => c.n_reps,
            "device"              => "cpu",
            "memory_kind"         => memory_kind_string(),
            "setup_kind"          => "PP.jl surrogate: propagate NodePathProperties graph once + zerofilter! (compile), evaluate! per query",
            "batch_support"       => is_batch ? "single-sample loop over evaluate!" : "single sample",
            "surrogate_note"      => "upstream marks src/Surrogate as experimental/legacy; CliffordGate+PauliRotation circuits only; no coefficient-value truncation; no gradient",
        ),
    )
end

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

function main(args)
    if length(args) < 2
        println(stderr, "usage: julia engine_julia.jl <config.json> <result.json> [--surrogate]")
        exit(2)
    end
    config_path = args[1]
    result_path = args[2]
    surrogate_mode = "--surrogate" in args[3:end]
    engine_name = surrogate_mode ? "julia_surrogate" : "julia"
    config = JSON.parsefile(config_path)
    result = nothing
    try
        result = surrogate_mode ? run_surrogate(config) : run_benchmark(config)
    catch e
        result = Dict(
            "engine"   => engine_name,
            "status"   => "Error",
            "error"    => sprint(showerror, e),
            "history"  => [],
        )
    end
    result["test_id"] = get(config, "test_id", nothing)
    mkpath(dirname(result_path))
    open(result_path, "w") do f
        JSON.print(f, result, 2)
    end
    if get(result, "status", "") in ("Success", "Unsupported")
        exit(0)
    else
        exit(1)
    end
end

main(ARGS)
