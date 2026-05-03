# vLLM v0.16.0 Code Dive — Mixed Prefill + Decode in One Forward Pass

## Scope
This note documents how vLLM `v0.16.0` executes a batch containing both prefill and decode requests.

Tag inspected:
- `v0.16.0` (`89a77b108`)

---

## Key conclusion

In mixed batches, vLLM does **not** use one monolithic kernel to jointly handle prefill and decode.

Instead it:
1. reorders the batch so decode tokens are first,
2. computes split metadata (`num_decode_tokens`, `num_prefill_tokens`),
3. runs decode and prefill attention through separate code paths/kernels inside one forward invocation.

---

## Evidence in v0.16.0 code

## 1) Reorder + split
- `vllm/v1/attention/backends/utils.py`
  - `split_decodes_and_prefills(...)`
  - `reorder_batch_to_split_decodes_and_prefills(...)`

This logic computes boundaries and arranges requests in decode→prefill order.

## 2) Runner calls reorder
- `vllm/v1/worker/gpu_model_runner.py` imports and calls reorder path.

## 3) Backend metadata carries both segments
- `vllm/v1/attention/backends/flashinfer.py`
  - metadata fields include `num_decode_tokens`, `num_prefill_tokens`
  - comments and slicing show decode at front, prefill at back.

## 4) Forward path is explicitly split
In `FlashInferImpl.forward(...)`:
- prefill uses slices like `query[num_decode_tokens:]` and writes to `output[num_decode_tokens:]`
- decode uses slices like `query[:num_decode_tokens]` and writes to `output[:num_decode_tokens]`

Equivalent split behavior appears in other attention backends (CPU/ROCm/Linear/GDN variants).

---

## Operational interpretation

A mixed scheduler step is “single forward” at orchestration level, but attention compute is still dispatched through specialized per-subtype kernels/pathways.

Why:
- decode attention has very different shape and KV access pattern (often short query length, cache-centric),
- prefill attention has larger query lengths and different parallelism profile.

---

## Practical takeaway

When profiling mixed prefill+decode workloads in vLLM v0.16.0, treat performance as a composition of:
- decode attention path,
- prefill attention path,
- scheduling/reordering overhead,
- KV cache movement behavior.

Do not assume one unified kernel dominates both phases.

---

## Related notes
- [[deepseek_v3_rocm_attention_backend_kernel_map]]
- `ai/llm/compilers/kernel_dtype_specialization_and_dispatch.md`
