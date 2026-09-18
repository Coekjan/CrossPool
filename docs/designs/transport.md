# Transport

Transport owns the rank-local Instance-to-AtnAgent mailbox and its CUDA IPC
arena. See [Fabric](fabric.md) for the distributed AtnAgent-to-FfnAgent
invocation that follows a published request.

Each Instance rank and its AtnAgent share one CUDA IPC Transport arena. The
mailbox is single-producer/single-consumer and permits one live FFN request.
The normal ownership cycle moves from idle, through request staging and
publication, to result availability and acknowledgement. Closed is terminal.

The Instance validates tensor device, dtype, rank, shape, contiguity, row
capacity, and optional physical DP-row metadata before launch. The request
kernel copies the live input prefix, release-publishes the request, waits for a
result or canonical failure, copies the output into caller-owned storage, and
acknowledges reuse. Optional int32 or int64 DP-row counts are converted into the
arena's uint32 representation by one block-wide cooperative transform; every
participant observes completion before request publication.

Decode and Prefill use the same Transport protocol. They differ only in live
row count and the smallest configured capacity that contains it. SGLang owns
the enclosing Graph mode; Transport records carry mailbox state only.

The AtnAgent Transport Resident is one cooperative grid with exactly one
256-thread block owning each Transport arena. Transport-to-Fabric input copy,
Fabric-to-Transport output copy, and NVSHMEM publication use the complete block.
Block-shared typed state communicates leader observations to the block. Fabric
publication uses block-collective NVSHMEM operations and preserves the required
payload-before-release and remote-completion-before-signal ordering. The block
size is a production constant rather than configuration or an alternate path.
