"""PPU tests for `LowerTileOp` copy annotations affecting cp.async sync (path 2)."""

import tilelang.language as T
import tilelang.ppu.transform
import tilelang.testing
from tilelang import tvm
from tvm.tirx.stmt_functor import post_order_visit


def _count_calls(func: tvm.tirx.PrimFunc):
    counts = {}

    def _visit(node):
        if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op):
            name = str(node.op.name)
            counts[name] = counts.get(name, 0) + 1

    post_order_visit(func.body, _visit)
    return counts


def _vectorized_cp_async_element_counts(func: tvm.tirx.PrimFunc):
    mod = tilelang.transform.VectorizeLoop()(tvm.IRModule.from_expr(func))
    element_counts = []

    def _visit(node):
        if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op) and str(node.op.name) == "tl.ptx_cp_async":
            assert isinstance(node.args[2], tvm.tirx.IntImm)
            element_counts.append(int(node.args[2].value))

    post_order_visit(mod["main"].body, _visit)
    return element_counts


def _shared_sync_stores(func: tvm.tirx.PrimFunc):
    stores = []

    def _visit(node):
        if isinstance(node, tvm.tirx.BufferStore) and node.buffer.scope().startswith("shared"):
            stores.append(node)

    post_order_visit(func.body, _visit)
    return stores


def _lower_pipeline_managed_subword_copy(
    extent: int,
    *,
    dtype="uint16",
    src_offset: int = 0,
    dst_offset: int = 0,
    src_elem_offset: int = 0,
    dst_elem_offset: int = 0,
    coalesced_width=None,
):
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def before(
        A_handle: T.handle,
        B_handle: T.handle,
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        A = T.match_buffer(
            A_handle,
            (extent + src_offset,),
            dtype=dtype,
            elem_offset=src_elem_offset,
            align=128,
            offset_factor=1,
        )
        B = T.match_buffer(B_handle, (extent,), dtype=dtype, align=128)
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        S = T.sblock_alloc_buffer(
            (extent + dst_offset,),
            dtype=dtype,
            scope="shared",
            elem_offset=dst_elem_offset,
            align=128,
            offset_factor=1,
        )
        if coalesced_width is None:
            T.copy(
                A[src_offset : src_offset + extent],
                S[dst_offset : dst_offset + extent],
                annotations={"no_implicit_async_commit_wait": T.int32(1)},
            )
        else:
            T.copy(
                A[src_offset : src_offset + extent],
                S[dst_offset : dst_offset + extent],
                coalesced_width=coalesced_width,
                annotations={"no_implicit_async_commit_wait": T.int32(1)},
            )
        if tx < extent:
            B[tx] = S[dst_offset + tx]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        return tilelang.ppu.transform.LowerTileOp()(mod)["main"]


def _lower_pipeline_managed_padded_2d_copy():
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def before(
        A: T.Tensor((2, 5), "uint8"),
        B: T.Tensor((2, 4), "uint8"),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        S = T.sblock_alloc_buffer((2, 5), dtype="uint8", scope="shared", align=128)
        T.copy(
            A[0:2, 0:4],
            S[0:2, 0:4],
            annotations={"no_implicit_async_commit_wait": T.int32(1)},
        )
        if tx < 8:
            B[tx // 4, tx % 4] = S[tx // 4, tx % 4]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        return tilelang.ppu.transform.LowerTileOp()(mod)["main"]


def _lower_pipeline_managed_padded_1d_layout_copy(extent=64):
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def before(
        A: T.Tensor((extent,), "uint8"),
        B: T.Tensor((extent,), "uint8"),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        S = T.sblock_alloc_buffer((extent,), dtype="uint8", scope="shared", align=128)
        with T.sblock("root"):
            T.annotate_layout({S: T.Layout((extent,), lambda i: (i // 4) * 5 + i % 4)})
            T.copy(
                A[0:extent],
                S[0:extent],
                annotations={"no_implicit_async_commit_wait": T.int32(1)},
            )
            if tx < extent:
                B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        return tilelang.ppu.transform.LowerTileOp()(mod)["main"]


def _lower_pipeline_managed_chunk_permuted_1d_layout_copy():
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def before(
        A: T.Tensor((64,), "uint8"),
        B: T.Tensor((64,), "uint8"),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        S = T.sblock_alloc_buffer((64,), dtype="uint8", scope="shared", align=128)
        with T.sblock("root"):
            T.annotate_layout(
                {
                    S: T.Layout(
                        (64,),
                        lambda i: (i // 8) * 8 + (1 - (i // 4) % 2) * 4 + i % 4,
                    )
                }
            )
            T.copy(
                A[0:64],
                S[0:64],
                annotations={"no_implicit_async_commit_wait": T.int32(1)},
            )
            if tx < 64:
                B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        return tilelang.ppu.transform.LowerTileOp()(mod)["main"]


def test_ppu_lower_tile_op_respects_copy_annotation_for_pipeline_managed_cp_async():
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float32),
        B: T.Tensor((16,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 16)
        S = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        T.copy(
            A[0:16],
            S,
            annotations={"no_implicit_async_commit_wait": T.int32(1)},
        )
        B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tilelang.ppu.transform.LowerTileOp()(mod)
    calls = _count_calls(mod["main"])

    assert calls.get("tl.ptx_cp_async", 0) > 0
    assert calls.get("tirx.ptx_commit_group", 0) == 0
    assert calls.get("tirx.ptx_wait_group", 0) == 0


def test_ppu_lower_tile_op_respects_copy_annotation_for_explicit_async_copy():
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float32),
        B: T.Tensor((16,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 16)
        S = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        T.async_copy(
            A[0:16],
            S,
            annotations={"no_implicit_async_commit_wait": T.int32(1)},
        )
        B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tilelang.ppu.transform.LowerTileOp()(mod)
    calls = _count_calls(mod["main"])

    assert calls.get("tl.ptx_cp_async", 0) > 0
    assert calls.get("tirx.ptx_commit_group", 0) == 0
    assert calls.get("tirx.ptx_wait_group", 0) == 0


def test_ppu_lower_tile_op_respects_parallel_loop_async_annotation_without_pipeline_context():
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float32),
        B: T.Tensor((16,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 16)
        S = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        for i in T.parallel(
            16,
            annotations={"parallel_async_without_async_commit_wait": T.bool(True)},
        ):
            S[i] = A[i]
        B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        mod = tilelang.ppu.transform.LowerTileOp()(mod)
    calls = _count_calls(mod["main"])

    assert calls.get("tl.ptx_cp_async", 0) > 0
    assert calls.get("tirx.ptx_commit_group", 0) == 0
    assert calls.get("tirx.ptx_wait_group", 0) == 0


def test_ppu_pipeline_managed_subword_copy_uses_minimum_legal_async_width():
    dtype_widths = {
        T.float4_e2m1fn: 8,
        "uint8": 4,
        "int8": 4,
        "float8_e4m3": 4,
        "float8_e4m3fn": 4,
        "float8_e5m2": 4,
        "float8_e8m0fnu": 4,
        "uint16": 2,
        "int16": 2,
        "float16": 2,
        "bfloat16": 2,
    }
    for dtype, expected_elements in dtype_widths.items():
        for extent in (64, 128):
            lowered = _lower_pipeline_managed_subword_copy(extent, dtype=dtype)
            calls = _count_calls(lowered)
            assert calls.get("tl.ptx_cp_async", 0) > 0, (dtype, extent)
            assert _vectorized_cp_async_element_counts(lowered) == [expected_elements], (dtype, extent)
            assert not _shared_sync_stores(lowered), (dtype, extent)
            assert calls.get("tirx.ptx_commit_group", 0) == 0
            assert calls.get("tirx.ptx_wait_group", 0) == 0

    # A naturally wider copy remains wider; the new policy is a floor and must
    # not cap an existing legal width.
    naturally_wider = _lower_pipeline_managed_subword_copy(512)
    calls = _count_calls(naturally_wider)
    assert calls.get("tl.ptx_cp_async", 0) > 0
    assert _vectorized_cp_async_element_counts(naturally_wider) == [4]

    # Copies outside the one-wave promotion limit still pass through the
    # alignment gate before preserving the common vectorizer's natural width.
    for side in ("src_offset", "dst_offset"):
        kwargs = {side: 2}
        naturally_wider = _lower_pipeline_managed_subword_copy(512, **kwargs)
        assert _count_calls(naturally_wider).get("tl.ptx_cp_async", 0) > 0, kwargs

    # Aligned region minima remain eligible.  Buffer-level elem_offset is
    # intentionally handled more conservatively below because LowerAccessPtr
    # does not currently fold it into the final pointer.
    aligned = (
        {"dtype": "uint8", "src_offset": 4},
        {"dtype": "uint8", "dst_offset": 4},
        {"dtype": "uint16", "src_offset": 2},
        {"dtype": "uint16", "dst_offset": 2},
        {"dtype": T.float4_e2m1fn, "src_offset": 8},
        {"dtype": T.float4_e2m1fn, "dst_offset": 8},
    )
    for kwargs in aligned:
        lowered = _lower_pipeline_managed_subword_copy(64, **kwargs)
        assert _count_calls(lowered).get("tl.ptx_cp_async", 0) > 0, kwargs

    # A layout may permute complete four-byte chunks without making the copy
    # unsafe.  Admit it as long as every final transaction start is aligned;
    # the common vectorizer separately proves unit stride inside each chunk.
    lowered = _lower_pipeline_managed_chunk_permuted_1d_layout_copy()
    assert _count_calls(lowered).get("tl.ptx_cp_async", 0) > 0
    assert _vectorized_cp_async_element_counts(lowered) == [4]
    assert not _shared_sync_stores(lowered)


def test_ppu_pipeline_managed_subword_copy_rejects_ragged_length():
    cases = tuple(
        {"extent": 64 + remainder, "dtype": dtype}
        for dtype, width in (
            (T.float4_e2m1fn, 8),
            ("uint8", 4),
            ("uint16", 2),
        )
        for remainder in range(1, width)
    )
    for kwargs in cases:
        lowered = _lower_pipeline_managed_subword_copy(**kwargs)
        calls = _count_calls(lowered)
        assert calls.get("tl.ptx_cp_async", 0) == 0, kwargs
        assert _shared_sync_stores(lowered), kwargs

    # A copy larger than one wave remains outside width promotion, but a known
    # incomplete four-byte transaction is still a safety failure rather than
    # an instruction to leave natural vectorization unchanged.
    lowered = _lower_pipeline_managed_subword_copy(129, dtype="uint16")
    assert _count_calls(lowered).get("tl.ptx_cp_async", 0) == 0
    assert _shared_sync_stores(lowered)


def test_ppu_pipeline_managed_subword_copy_rejects_misaligned_start():
    cases = tuple(
        {"extent": 64, "dtype": dtype, side: offset}
        for dtype, width in (
            (T.float4_e2m1fn, 8),
            ("uint8", 4),
            ("uint16", 2),
        )
        for side in ("src_offset", "dst_offset")
        for offset in range(1, width)
    )
    for kwargs in cases:
        lowered = _lower_pipeline_managed_subword_copy(**kwargs)
        calls = _count_calls(lowered)
        assert calls.get("tl.ptx_cp_async", 0) == 0, kwargs
        assert _shared_sync_stores(lowered), kwargs


def test_ppu_pipeline_managed_subword_copy_rejects_unqualified_cases():
    cases = (
        {"extent": 64, "dtype": "uint8", "src_elem_offset": 4},
        {"extent": 64, "dtype": "uint8", "dst_elem_offset": 4},
        {"extent": 64, "dtype": "uint16", "src_elem_offset": 2},
        {"extent": 64, "dtype": "uint16", "dst_elem_offset": 2},
        {"extent": 64, "dtype": T.float4_e2m1fn, "src_elem_offset": 8},
        {"extent": 64, "dtype": T.float4_e2m1fn, "dst_elem_offset": 8},
        {"extent": 64, "coalesced_width": 1},
        {"extent": 384},
        {"extent": 64, "dtype": T.float4_e2m1_unpacked},
    )
    for kwargs in cases:
        lowered = _lower_pipeline_managed_subword_copy(**kwargs)
        calls = _count_calls(lowered)
        assert calls.get("tl.ptx_cp_async", 0) == 0, kwargs
        assert _shared_sync_stores(lowered), kwargs

    # The first address and total byte count are both 4-byte aligned, but the
    # padded row stride makes the second transaction start at byte offset 5.
    # V1 therefore admits only a single active region axis.
    lowered = _lower_pipeline_managed_padded_2d_copy()
    assert _count_calls(lowered).get("tl.ptx_cp_async", 0) == 0
    assert _shared_sync_stores(lowered)

    # A single logical axis is not sufficient: this explicit layout is
    # contiguous only inside each group of four, with physical starts at
    # 0, 5, 10, ... .  Every transaction start must remain four-byte aligned.
    lowered = _lower_pipeline_managed_padded_1d_layout_copy()
    assert _count_calls(lowered).get("tl.ptx_cp_async", 0) == 0
    assert _shared_sync_stores(lowered)

    # The same final-layout check also guards copies outside the one-wave
    # auto-promotion limit, where natural vectorization remains possible.
    lowered = _lower_pipeline_managed_padded_1d_layout_copy(512)
    assert _count_calls(lowered).get("tl.ptx_cp_async", 0) == 0
    assert _shared_sync_stores(lowered)


def test_ppu_pipeline_managed_subword_copy_rejects_unsafe_offset_when_naturally_wide():
    # Without an explicit safety-failure result from the width planner, the
    # common vectorizer can still select a naturally wide cp.async even though
    # LowerAccessPtr does not fold Buffer::elem_offset into the final pointer.
    for side in ("src_elem_offset", "dst_elem_offset"):
        lowered = _lower_pipeline_managed_subword_copy(512, dtype="uint16", **{side: 2})
        assert _count_calls(lowered).get("tl.ptx_cp_async", 0) == 0, side
        assert _shared_sync_stores(lowered), side

    # A larger copy is not auto-promoted, but it can still become a naturally
    # wide cp.async.  Reject a two-byte region start before that vectorization.
    for side in ("src_offset", "dst_offset"):
        lowered = _lower_pipeline_managed_subword_copy(512, dtype="uint16", **{side: 1})
        assert _count_calls(lowered).get("tl.ptx_cp_async", 0) == 0, side
        assert _shared_sync_stores(lowered), side


def test_ppu_pipeline_subword_copy_is_annotated_and_lowered_end_to_end():
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    def make_pipeline(dtype, extent, alignment_elements):
        storage_extent = ((extent + alignment_elements - 1) // alignment_elements) * alignment_elements

        @T.prim_func
        def before(
            A: T.Tensor((4, storage_extent), dtype),
            B: T.Tensor((4, extent), dtype),
        ):
            T.func_attr({"global_symbol": "main", "target": target})
            T.launch_thread("blockIdx.x", 1)
            tx = T.launch_thread("threadIdx.x", 128)
            S = T.alloc_buffer((storage_extent,), dtype=dtype, scope="shared")
            for ko in T.Pipelined(4, num_stages=2):
                T.copy(A[ko, 0:extent], S[0:extent])
                if tx < extent:
                    B[ko, tx] = S[tx]

        return before

    for dtype, extent, expected_elements, expect_async in (
        (T.float4_e2m1fn, 64, 8, True),
        ("uint8", 64, 4, True),
        ("uint16", 64, 2, True),
        (T.float4_e2m1fn, 67, 8, False),
        ("uint8", 67, 4, False),
        ("uint16", 65, 2, False),
    ):
        before = make_pipeline(dtype, extent, expected_elements)
        mod = tvm.IRModule.from_expr(before)
        with target:
            mod = tilelang.transform.IfStmtBinding()(mod)
            mod = tilelang.transform.PipelinePlanning()(mod)
            mod = tilelang.transform.InjectSoftwarePipeline()(mod)
            mod = tilelang.transform.Simplify()(mod)
            mod = tilelang.ppu.transform.LayoutInference()(mod)
            mod = tilelang.ppu.transform.LowerTileOp()(mod)

        calls = _count_calls(mod["main"])
        if expect_async:
            assert calls.get("tl.ptx_cp_async", 0) > 0, dtype
            element_counts = _vectorized_cp_async_element_counts(mod["main"])
            assert element_counts, dtype
            assert all(count == expected_elements for count in element_counts), (
                dtype,
                element_counts,
            )
            assert not _shared_sync_stores(mod["main"]), dtype
        else:
            assert calls.get("tl.ptx_cp_async", 0) == 0, dtype
            assert _shared_sync_stores(mod["main"]), dtype


if __name__ == "__main__":
    tilelang.testing.main()
