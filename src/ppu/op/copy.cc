/*!
 * \file tl/ppu/op/copy.cc
 * \brief PPU implementation for tl.copy lowering.
 */

#include "op/copy.h"
#include "support/check.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>

#include "backend/common/target_utils.h"
#include "cuda/transform/ptx_async_copy_injector.h"
#include "layout/layout.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "ppu/op/copy.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"

#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

PrimExpr AiuBytesFromElements(PrimExpr elements, DataType dtype) {
  PrimExpr elements_i64 = cast(DataType::Int(64), elements);
  int bits = dtype.bits();
  if (bits % 8 == 0) {
    return elements_i64 * IntImm(DataType::Int(64), bits / 8);
  }
  return FloorDiv(elements_i64 * IntImm(DataType::Int(64), bits) +
                      IntImm(DataType::Int(64), 7),
                  IntImm(DataType::Int(64), 8));
}

int64_t AiuBytesFromElements(int64_t elements, DataType dtype) {
  ICHECK_EQ((elements * dtype.bits()) % 8, 0)
      << elements << " elements of " << dtype
      << " cannot be represented as whole bytes";
  return elements * dtype.bits() / 8;
}

int64_t AiuElementsForBytes(int64_t bytes, DataType dtype) {
  ICHECK_EQ((bytes * 8) % dtype.bits(), 0)
      << bytes << " bytes cannot be represented as whole elements of " << dtype;
  return bytes * 8 / dtype.bits();
}

bool GetBoolAnnotation(const CopyNode &op, const char *key) {
  if (auto val = op.annotations.Get(key)) {
    if (auto int_val = val->as<IntImmNode>()) {
      return int_val->value != 0;
    }
  }
  return false;
}

bool GetIsAsyncCopy(const CopyNode &op) {
  if (GetBoolAnnotation(op, "is_async_copy")) {
    return true;
  }
  // Backward-compatibility with historical annotation key.
  return GetBoolAnnotation(op, "force_cp_async");
}

bool GetNoImplicitAsyncCommitWait(const CopyNode &op) {
  return GetBoolAnnotation(op, attr::kAsyncCopyNoImplicitCommitWait);
}

constexpr int kPPUAsyncCopyMinTransferBits = 32;

bool HasSafeFourByteAlignedBase(const Buffer &buffer,
                                arith::Analyzer *analyzer) {
  // LowerAccessPtr currently forms the pointer from load indices only and does
  // not fold Buffer::elem_offset into tl.access_ptr.  Until that common pass is
  // fixed, only a zero buffer-level offset is safe here.  Region minima are
  // checked separately against the final physical layout before lowering.
  return buffer->data_alignment >= 4 && buffer->data_alignment % 4 == 0 &&
         analyzer->CanProve(buffer->elem_offset == 0);
}

bool IsSupportedPPUSubwordAsyncCopyType(DataType dtype) {
  if (!dtype.is_scalar()) {
    return false;
  }
  if (dtype.is_float4_e2m1fn()) {
    // FP4 remains a logical four-bit dtype here.  Eight adjacent logical
    // elements form the minimum four-byte PPU async-copy transaction.
    return true;
  }
  if (dtype.bits() == 8) {
    // Only admit scalar types the PPU code generator can represent.  In
    // particular, float4_e2m1_unpacked also occupies eight storage bits but is
    // an internal shared-memory storage tag, not an FP8 register type.
    return dtype.is_int() || dtype.is_uint() || dtype.is_float8_e4m3() ||
           dtype.is_float8_e4m3fn() || dtype.is_float8_e5m2() ||
           dtype.is_float8_e8m0fnu();
  }
  if (dtype.bits() == 16) {
    return dtype.is_int() || dtype.is_uint() || dtype.is_float16() ||
           dtype.is_bfloat16();
  }
  return false;
}

bool IsPPUSubwordAsyncCopyAutoWidthCandidate(const CopyNode &op,
                                             ppu::CopyInst copy_inst) {
  return copy_inst == ppu::CopyInst::kCPAsync &&
         GetNoImplicitAsyncCommitWait(op) && !GetIsAsyncCopy(op) &&
         !op.annotations.count(attr::kCoalescedWidth) &&
         !op.annotations.count(attr::kParallelLoopLayout) &&
         IsGlobalBuffer(op.src) && IsSharedBuffer(op.dst) &&
         op.src->dtype == op.dst->dtype &&
         IsSupportedPPUSubwordAsyncCopyType(op.src->dtype);
}

enum class PPUAsyncCopyWidthAction {
  kNotApplicable,
  kValidateNaturalWidth,
  kPromote,
  kForceSynchronous,
};

struct PPUAsyncCopyWidthPlan {
  PPUAsyncCopyWidthAction action;
  int min_width;
};

PPUAsyncCopyWidthPlan PlanPPUAsyncCopyWidth(
    const CopyNode &op, ppu::CopyInst copy_inst, Range thread_bounds,
    const Map<Buffer, Buffer> &buffer_remap, arith::Analyzer *analyzer) {
  int scalar_bits = op.src->dtype.bits();

  if (!IsPPUSubwordAsyncCopyAutoWidthCandidate(op, copy_inst)) {
    return {PPUAsyncCopyWidthAction::kNotApplicable, 1};
  }
  int min_elements = kPPUAsyncCopyMinTransferBits / scalar_bits;

  // Automatic width planning is intentionally limited to statically sized,
  // non-empty copies.  Preserve the established lowering for symbolic sizes
  // instead of turning an inability to prove the new policy into a
  // synchronous-copy requirement.
  PrimExpr total_elements = IntImm(DataType::Int(64), 1);
  for (const IterVar &iv : op.MakeIterVars()) {
    total_elements = total_elements * cast(DataType::Int(64), iv->dom->extent);
  }
  PrimExpr simplified_total = analyzer->Simplify(total_elements);
  const int64_t *total = as_const_int(simplified_total);
  if (total == nullptr || *total <= 0) {
    return {PPUAsyncCopyWidthAction::kNotApplicable, min_elements};
  }

  Buffer actual_src =
      buffer_remap.count(op.src) ? buffer_remap[op.src] : op.src;
  Buffer actual_dst =
      buffer_remap.count(op.dst) ? buffer_remap[op.dst] : op.dst;
  if (!HasSafeFourByteAlignedBase(actual_src, analyzer) ||
      !HasSafeFourByteAlignedBase(actual_dst, analyzer)) {
    // This is a safety failure, not merely an inapplicable width promotion.
    // Leaving the loop unchanged could still let the common vectorizer choose
    // a naturally wide cp.async from a base whose alignment is unproven.
    return {PPUAsyncCopyWidthAction::kForceSynchronous, min_elements};
  }

  // Keep v1 within the measured one-wave scale-slab case.  Larger copies can
  // make the padding heuristic choose widths above the requested floor and
  // need separate performance qualification.
  if (*total % min_elements != 0) {
    return {PPUAsyncCopyWidthAction::kForceSynchronous, min_elements};
  }
  PrimExpr simplified_threads = analyzer->Simplify(thread_bounds->extent);
  const int64_t *threads = as_const_int(simplified_threads);
  if (threads == nullptr || *threads <= 0 || *total > *threads) {
    // Do not add a minimum-width hint outside the measured one-wave case, but
    // still validate it: the common vectorizer may independently select a
    // naturally wide cp.async.
    return {PPUAsyncCopyWidthAction::kValidateNaturalWidth, min_elements};
  }
  return {PPUAsyncCopyWidthAction::kPromote, min_elements};
}

std::optional<PrimExpr>
LinearOffsetOfIndices(const Buffer &logical_buffer, Array<PrimExpr> indices,
                      const Map<Buffer, Buffer> &buffer_remap,
                      const LayoutMap &layout_map) {
  if (indices.size() != logical_buffer->shape.size()) {
    return std::nullopt;
  }

  Buffer physical_buffer = logical_buffer;
  if (buffer_remap.count(logical_buffer)) {
    if (!layout_map.count(logical_buffer)) {
      return std::nullopt;
    }
    indices = layout_map[logical_buffer]->Forward(indices);
    physical_buffer = buffer_remap[logical_buffer];
  }
  if (indices.size() != physical_buffer->shape.size()) {
    return std::nullopt;
  }

  Array<PrimExpr> physical = physical_buffer.OffsetOf(indices);
  Buffer flattened = physical_buffer.GetFlattenedBuffer();
  if (physical.empty() || physical.size() != flattened->shape.size()) {
    return std::nullopt;
  }
  PrimExpr linear = physical[0];
  for (size_t axis = 1; axis < physical.size(); ++axis) {
    linear = linear * flattened->shape[axis] + physical[axis];
  }
  return linear;
}

std::optional<size_t> GetSingleActiveAxis(const Array<Range> &ranges,
                                          arith::Analyzer *analyzer) {
  std::optional<size_t> active_axis;
  for (size_t axis = 0; axis < ranges.size(); ++axis) {
    if (analyzer->CanProveEqual(ranges[axis]->extent, 1)) {
      continue;
    }
    if (active_axis.has_value()) {
      return std::nullopt;
    }
    active_axis = axis;
  }
  return active_axis;
}

bool HasAlignedPPUAsyncCopyTransactions(const Buffer &logical_buffer,
                                        const Array<Range> &ranges,
                                        int transaction_elements,
                                        const Map<Buffer, Buffer> &buffer_remap,
                                        const LayoutMap &layout_map,
                                        arith::Analyzer *analyzer) {
  auto active_axis = GetSingleActiveAxis(ranges, analyzer);
  if (!active_axis.has_value()) {
    return false;
  }

  PrimExpr simplified_extent =
      analyzer->Simplify(ranges[active_axis.value()]->extent);
  const int64_t *extent = as_const_int(simplified_extent);
  if (extent == nullptr || *extent <= 0 ||
      *extent % transaction_elements != 0) {
    return false;
  }

  Buffer physical_buffer = buffer_remap.count(logical_buffer)
                               ? buffer_remap[logical_buffer]
                               : logical_buffer;
  int64_t physical_element_bits =
      static_cast<int64_t>(physical_buffer->dtype.bits()) *
      physical_buffer->dtype.lanes();

  Array<PrimExpr> indices;
  indices.reserve(ranges.size());
  for (const Range &range : ranges) {
    indices.push_back(range->min);
  }

  // The common vectorizer proves unit stride inside each vector transaction.
  // Here, prove the complementary condition that every transaction begins on
  // a four-byte boundary after the final layout remap.
  const Range &active_range = ranges[active_axis.value()];
  auto transaction_is_aligned = [&](PrimExpr delta) {
    indices.Set(active_axis.value(), active_range->min + delta);
    auto physical_offset = LinearOffsetOfIndices(logical_buffer, indices,
                                                 buffer_remap, layout_map);
    if (!physical_offset.has_value()) {
      return false;
    }
    PrimExpr bit_offset = cast(DataType::Int(64), physical_offset.value()) *
                          IntImm(DataType::Int(64), physical_element_bits);
    return analyzer->CanProveEqual(
        FloorMod(bit_offset, IntImm(DataType::Int(64), 32)),
        IntImm(DataType::Int(64), 0));
  };

  // First try to prove all transaction starts at once.  This keeps validation
  // of naturally wide copies outside the one-wave promotion limit O(1).
  Var transaction("ppu_async_copy_transaction", active_range->min.dtype());
  PrimExpr symbolic_delta =
      transaction * make_const(active_range->min.dtype(), transaction_elements);
  if (transaction_is_aligned(symbolic_delta)) {
    return true;
  }

  // Non-affine layouts may be provable only after substituting concrete
  // indices.  Bound that fallback to the one-wave-scale cases this policy was
  // measured for; larger unproven layouts conservatively use synchronous copy.
  constexpr int64_t kMaxEnumeratedTransactions = 128;
  int64_t transaction_count = *extent / transaction_elements;
  if (transaction_count > kMaxEnumeratedTransactions) {
    return false;
  }
  for (int64_t delta = 0; delta < *extent; delta += transaction_elements) {
    if (!transaction_is_aligned(make_const(active_range->min.dtype(), delta))) {
      return false;
    }
  }
  return true;
}

bool RequiresSynchronousPPUSubwordCopy(const CopyNode &op,
                                       ppu::CopyInst copy_inst,
                                       Range thread_bounds,
                                       const Map<Buffer, Buffer> &buffer_remap,
                                       const LayoutMap &layout_map,
                                       arith::Analyzer *analyzer) {
  PPUAsyncCopyWidthPlan plan = PlanPPUAsyncCopyWidth(
      op, copy_inst, thread_bounds, buffer_remap, analyzer);
  if (plan.action == PPUAsyncCopyWidthAction::kNotApplicable) {
    return false;
  }
  if (plan.action == PPUAsyncCopyWidthAction::kForceSynchronous) {
    return true;
  }

  // Auto-promote only complete, aligned four-byte transactions.  A mixed
  // async-main/synchronous-tail lowering regressed all measured ragged cases,
  // so an unaligned or non-divisible copy keeps the established synchronous
  // lowering instead of being peeled.
  if (!HasAlignedPPUAsyncCopyTransactions(op.src, op.src_range, plan.min_width,
                                          buffer_remap, layout_map, analyzer) ||
      !HasAlignedPPUAsyncCopyTransactions(op.dst, op.dst_range, plan.min_width,
                                          buffer_remap, layout_map, analyzer)) {
    return true;
  }
  return false;
}

For MakePPUSIMTLoop(const CopyNode &op, ppu::CopyInst copy_inst,
                    Range thread_bounds,
                    const Map<Buffer, Buffer> &buffer_remap,
                    arith::Analyzer *analyzer) {
  For loop = op.MakeSIMTLoop(analyzer);
  PPUAsyncCopyWidthPlan plan = PlanPPUAsyncCopyWidth(
      op, copy_inst, thread_bounds, buffer_remap, analyzer);
  if (plan.action == PPUAsyncCopyWidthAction::kPromote) {
    // This is a floor, not a forced width.  The common vectorizer still proves
    // the final loop extent, remapped offsets, predicates, and unit stride.  If
    // the required adjacent subword elements are not legal, the floor is
    // ignored and the established synchronous fallback remains in effect.
    loop.CopyOnWrite()->annotations.Set(
        attr::kMinCoalescedWidth, IntImm(DataType::Int(32), plan.min_width));
  }
  return loop;
}

} // namespace

namespace ppu {

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level);

  static Stmt Lower(const CopyNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer);

private:
  static Layout ComputeLinearLayout(const Buffer &shared_tensor);

  static CopyInst SelectInst(const CopyNode &op, Target target,
                             const LayoutMap &layout_map,
                             arith::Analyzer *analyzer, bool buffer_oob);

  static void CheckParallelLoopLayout(const CopyNode &op, CopyInst copy_inst);

  static Stmt LowerNormal(const CopyNode &op, const LowerArgs &lower_args,
                          arith::Analyzer *analyzer);

  static Stmt LowerCPAsync(const CopyNode &op, const LowerArgs &lower_args,
                           arith::Analyzer *analyzer);

  static Stmt LowerLDSM(const CopyNode &op, const LowerArgs &lower_args,
                        arith::Analyzer *analyzer, CopyInst copy_inst);

  static Stmt LowerAiu(const CopyNode &op, const LowerArgs &lower_args,
                       arith::Analyzer *analyzer);
};

Layout Copy::ComputeLinearLayout(const Buffer &shared_tensor) {
  Array<PrimExpr> input_size = shared_tensor->shape;
  Array<PrimExpr> forward_vars;
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_vars.push_back(InputPlaceholder(i));
  }

  Array<PrimExpr> forward_index;
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_index.push_back(FloorDiv(forward_vars[i], 256));
  }
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_index.push_back(FloorMod(forward_vars[i], 256));
  }
  return Layout(input_size, forward_index);
}

LayoutMap Copy::InferLayout(const CopyNode &op,
                            const LayoutInferArgs &layout_args,
                            InferLevel level) {
  CopyInst copy_inst =
      SelectInst(op, layout_args.target, layout_args.layout_map,
                 layout_args.analyzer, layout_args.buffer_oob);
  CheckParallelLoopLayout(op, copy_inst);

  if (!op.par_op_.defined()) {
    // MakeSIMTLoop binds its fresh iteration variables.  Preserve outer-loop
    // facts needed by the width policy without mutating the pass-owned
    // analyzer, matching the isolation of CopyNode::InferSIMTLayout.
    auto analyzer = layout_args.analyzer->Clone();
    op.par_op_ =
        ParallelOp(MakePPUSIMTLoop(op, copy_inst, layout_args.thread_bounds,
                                   layout_args.buffer_remap, analyzer.get()));
  }
  return op.par_op_->InferLayout(layout_args, level);
}

void Copy::CheckParallelLoopLayout(const CopyNode &op, CopyInst copy_inst) {
  if (!op.annotations.count(attr::kParallelLoopLayout)) {
    return;
  }
  if (copy_inst == CopyInst::kNormal || copy_inst == CopyInst::kCPAsync) {
    return;
  }

  std::ostringstream oss;
  oss << "T.copy loop layout annotation requires SIMT copy; got "
      << CopyInstToString(copy_inst) << " for src=" << op.src->name
      << ", dst=" << op.dst->name
      << ". Remove loop_layout or change copy pattern.";
  LOG(FATAL) << oss.str();
}

CopyInst Copy::SelectInst(const CopyNode &op, Target target,
                          const LayoutMap &layout_map,
                          arith::Analyzer *analyzer, bool buffer_oob) {
  CopyAnalysisContext ctx;
  ctx.target = target;
  ctx.layout_map = &layout_map;
  ctx.analyzer = analyzer;
  ctx.buffer_oob = buffer_oob;
  ctx.emit_diagnostics = true;
  auto result = SelectCopyInstForLowering(op, ctx);
  ICHECK(result.supported) << result.reason;
  return result.inst;
}

Stmt Copy::Lower(const CopyNode &op, const LowerArgs &lower_args,
                 arith::Analyzer *analyzer) {
  auto copy_inst = SelectInst(op, lower_args.target, lower_args.layout_map,
                              analyzer, /*buffer_oob=*/false);
  if (op.dst_block.defined()) {
    LOG(FATAL) << "T.copy with dst_block requires ppu0015+ cluster-copy/TMA, "
               << "but PPU only supports ppu0010/ppu0015. Got target="
               << lower_args.target;
  }
  if (copy_inst == CopyInst::kLDSM) {
    auto ldsm_copy = LowerLDSM(op, lower_args, analyzer, copy_inst);
    ICHECK(ldsm_copy.defined()) << "Failed to lower tix matrix copy";
    return ldsm_copy;
  } else if (copy_inst == CopyInst::kCPAsync) {
    auto cp_async_copy = LowerCPAsync(op, lower_args, analyzer);
    ICHECK(cp_async_copy.defined()) << "Failed to lower cp.async copy";
    return cp_async_copy;
  } else if (copy_inst == CopyInst::kAiuLoad) {
    auto aiu_copy = LowerAiu(op, lower_args, analyzer);
    ICHECK(aiu_copy.defined()) << "Failed to lower PPU AIU copy";
    return aiu_copy;
  } else if (copy_inst == CopyInst::kNormal) {
    return LowerNormal(op, lower_args, analyzer);
  } else {
    LOG(FATAL) << "Unsupported copy inst " << static_cast<int>(copy_inst);
  }
}

Stmt Copy::LowerCPAsync(const CopyNode &op, const LowerArgs &lower_args,
                        arith::Analyzer *analyzer) {
  using namespace tvm::transform;

  PassContext pass_ctx = PassContext::Current();
  bool enable_async_copy =
      pass_ctx->GetConfig<Bool>(kEnableAsyncCopy, Bool(true)).value();
  bool no_implicit_commit_wait = GetNoImplicitAsyncCommitWait(op);
  bool explicit_async_semantics = no_implicit_commit_wait || GetIsAsyncCopy(op);
  if (!enable_async_copy && !explicit_async_semantics) {
    return LowerNormal(op, lower_args, analyzer);
  }

  if (RequiresSynchronousPPUSubwordCopy(
          op, CopyInst::kCPAsync, lower_args.thread_bounds,
          lower_args.buffer_remap, lower_args.layout_map, analyzer)) {
    return LowerNormal(op, lower_args, analyzer);
  }

  auto simt_loop =
      MakePPUSIMTLoop(op, CopyInst::kCPAsync, lower_args.thread_bounds,
                      lower_args.buffer_remap, analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
  auto par_op = ParallelOp(fused_loop);

  std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                    InferLevel::kFree};
  for (auto level : levels) {
    par_op->InferLayout({lower_args.target,
                         lower_args.thread_bounds,
                         lower_args.layout_map,
                         analyzer,
                         false,
                         lower_args.buffer_remap,
                         {}},
                        level);
  }
  auto loop_layout = par_op->GetLoopLayout();
  Stmt lowered_loop = LowerParallelLoop(
      par_op->GetRoot(), loop_layout, lower_args.thread_var, analyzer,
      lower_args.layout_map, par_op->GetPredicate(lower_args.thread_var),
      /*parallel_loop=*/true, /*should_vectorize=*/true,
      par_op->LoopLayoutRequiresPaddingGuard());

  auto inject_result =
      InjectPTXAsyncCopy(lowered_loop,
                         /*async_without_async_commit_wait=*/
                         no_implicit_commit_wait || GetIsAsyncCopy(op));
  Stmt cp_async_loop = inject_result.stmt;
  if (!inject_result.injected_ptx_async_copy) {
    DLOG(WARNING) << "cp.async rewrite miss for copy src=" << op.src->name
                  << " (scope=" << op.src.scope() << ", dtype=" << op.src->dtype
                  << "), dst=" << op.dst->name << " (scope=" << op.dst.scope()
                  << ", dtype=" << op.dst->dtype
                  << "), no_implicit_async_commit_wait="
                  << no_implicit_commit_wait
                  << ", is_async_copy=" << GetIsAsyncCopy(op);
    if (no_implicit_commit_wait) {
      DLOG(WARNING)
          << "Pipeline-managed async copy fallback to normal copy because "
             "cp.async rewrite found no eligible global->shared store.";
      return lowered_loop;
    }
    if (explicit_async_semantics) {
      LOG(FATAL) << "Explicit async copy semantics require cp.async lowering, "
                    "but no eligible global->shared store was rewritten.";
    }
    DLOG(WARNING) << "Fallback to normal copy because cp.async rewrite found "
                     "no eligible global->shared store.";
    return LowerNormal(op, lower_args, analyzer);
  }
  if (no_implicit_commit_wait) {
    return cp_async_loop;
  }
  if (GetIsAsyncCopy(op)) {
    Stmt commit_group =
        Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
    return SeqStmt({cp_async_loop, commit_group});
  }
  return cp_async_loop;
}

Stmt Copy::LowerNormal(const CopyNode &op, const LowerArgs &lower_args,
                       arith::Analyzer *analyzer) {
  return tl::LowerNormalCopy(op, lower_args, analyzer);
}

Stmt Copy::LowerLDSM(const CopyNode &op, const LowerArgs &lower_args,
                     arith::Analyzer *analyzer, CopyInst copy_inst) {
  const Buffer &src = op.src;
  const Buffer &dst = op.dst;
  const Array<Range> &src_range = op.src_range;
  const Array<Range> &dst_range = op.dst_range;

  ICHECK(copy_inst == CopyInst::kLDSM)
      << "Invalid copy inst " << static_cast<int>(copy_inst);

  Array<IterVar> loop_vars = op.MakeIterVars();
  if (loop_vars.size() < 2) {
    return LowerNormal(op, lower_args, analyzer);
  }
  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);
  PrimExpr src_predicate = op.MakePredicate(analyzer, loop_vars, src->shape, 0);
  PrimExpr dst_predicate = op.MakePredicate(analyzer, loop_vars, dst->shape, 1);
  if (src_predicate.defined() || dst_predicate.defined()) {
    return LowerNormal(op, lower_args, analyzer);
  }

  Buffer shared_tensor = src;
  Buffer local_tensor = dst;
  Array<Range> local_region = src_range;
  bool is_full_range = true;
  for (size_t i = 0; i < local_region.size(); i++) {
    if (!analyzer->CanProveEqual(local_region[i]->extent,
                                 local_tensor->shape[i])) {
      is_full_range = false;
      break;
    }
  }
  if (!is_full_range) {
    return LowerNormal(op, lower_args, analyzer);
  }

  Array<PrimExpr> local_indices = op.MakeIndices(loop_vars, 1);
  Fragment local_layout =
      Downcast<Fragment>(lower_args.layout_map[local_tensor]);
  Array<PrimExpr> local_indices_transformed =
      local_layout->Forward(local_indices);
  local_tensor = lower_args.buffer_remap[local_tensor];
  if (local_layout->OutputDim() != 1) {
    return LowerNormal(op, lower_args, analyzer);
  }

  Array<PrimExpr> shared_indices = op.MakeIndices(loop_vars, 0);
  bool is_transposed;
  IterVar col_var = loop_vars[loop_vars.size() - 1];
  IterVar row_var = loop_vars[loop_vars.size() - 2];
  PrimExpr local_layout_thread_map =
      FloorMod(local_layout->ForwardThread(local_indices, std::nullopt), 32);
  PrimExpr matrix_8x8_thread_map = MakeGemmFragment8x8()->ForwardThread(
      {FloorMod(row_var, 8), FloorMod(col_var, 8)}, std::nullopt);
  PrimExpr matrix_8x8_thread_map_trans =
      MakeGemmFragment8x8Transposed()->ForwardThread(
          {FloorMod(row_var, 8), FloorMod(col_var, 8)}, std::nullopt);
  PrimExpr local_indices_flattened =
      local_tensor.OffsetOf(local_indices_transformed).back();
  if (analyzer->CanProveEqual(matrix_8x8_thread_map, local_layout_thread_map) &&
      IndicesCanVectorize(local_indices_flattened, col_var->var,
                          col_var->dom->extent, 2, analyzer)) {
    is_transposed = false;
  } else if (analyzer->CanProveEqual(matrix_8x8_thread_map_trans,
                                     local_layout_thread_map) &&
             IndicesCanVectorize(local_indices_flattened, row_var->var,
                                 row_var->dom->extent, 2, analyzer)) {
    is_transposed = true;
  } else {
    return LowerNormal(op, lower_args, analyzer);
  }
  if (shared_tensor->dtype.bytes() != 2) {
    return LowerNormal(op, lower_args, analyzer);
  }
  PrimExpr flattened_indice = shared_tensor.OffsetOf(shared_indices).back();
  if (!IndicesCanVectorize(flattened_indice, loop_vars.back()->var,
                           loop_vars.back()->dom->extent, 8, analyzer)) {
    return LowerNormal(op, lower_args, analyzer);
  }

  for (size_t i = 0; i < dst_range.size(); i++) {
    if (!is_zero(dst_range[i]->min) ||
        !analyzer->CanProveEqual(dst_range[i]->extent, dst->shape[i]))
      return LowerNormal(op, lower_args, analyzer);
  }

  PrimExpr extent = local_tensor->shape[0];
  int num = 1;
  if (analyzer->CanProveEqual(FloorMod(extent, 8), 0))
    num = 4;
  else if (analyzer->CanProveEqual(FloorMod(extent, 4), 0))
    num = 2;

  Array<PrimExpr> args;
  const Op &copy_op = tl::ptx_ldmatrix();
  args.push_back(static_cast<int>(is_transposed));
  args.push_back(num);

  Var local_iter("i");
  Layout inv = local_layout->Inverse();
  Array<PrimExpr> shared_coords;
  PrimExpr warp = FloorDiv(lower_args.thread_var, 32) * 32;
  if (!is_transposed) {
    auto local_index = analyzer->Simplify(
        local_iter * 2 * num +
        2 * FloorMod(FloorDiv(lower_args.thread_var, 8), num));
    auto thread_index =
        analyzer->Simplify(warp + FloorMod(lower_args.thread_var, 8) * 4);
    shared_coords = inv->Forward({local_index, thread_index});
  } else {
    auto local_index = analyzer->Simplify(
        local_iter * 2 * num +
        2 * FloorMod(FloorDiv(lower_args.thread_var, 8), num) +
        FloorMod(lower_args.thread_var, 2));
    auto thread_index = analyzer->Simplify(
        warp + FloorDiv(FloorMod(lower_args.thread_var, 8), 2));
    shared_coords = inv->Forward({local_index, thread_index});
  }
  shared_coords.pop_back();
  PrimExpr shared_addr =
      Call(DataType::Handle(), tl::access_ptr(),
           {BufferLoad(shared_tensor, shared_coords), PrimExpr(2 * num),
            make_const(DataType::Int(32), 1)});
  args.push_back(shared_addr);

  if (local_tensor->dtype != shared_tensor->dtype) {
    return LowerNormal(op, lower_args, analyzer);
  }
  PrimExpr local_addr =
      Call(DataType::Handle(), tl::access_ptr(),
           {BufferLoad(local_tensor, {local_iter * 2 * num}), PrimExpr(2 * num),
            make_const(DataType::Int(32), 2)});
  args.push_back(local_addr);

  auto body = Evaluate(Call(DataType::Handle(), copy_op, args));
  For for_node =
      For(local_iter, 0, FloorDiv(extent, 2 * num), ForKind::kSerial, body);
  for_node = PragmaUnrollLoop(for_node);
  auto range = lower_args.thread_bounds;
  if (range.defined()) {
    auto thread_var = lower_args.thread_var;
    auto thread_var_with_offset = thread_var - range->min;
    for_node.CopyOnWrite()->body =
        Substitute(for_node->body, {{thread_var, thread_var_with_offset}});
  }
  return for_node;
}

static void RequireAIUSmemAlignment(const LowerArgs &lower_args,
                                    const Buffer &shared_tensor) {
  if (!lower_args.require_smem_alignment)
    return;
  lower_args.require_smem_alignment(shared_tensor->data, 128);
}

Stmt Copy::LowerAiu(const CopyNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
  Buffer global_tensor = op.src;
  Buffer shared_tensor = op.dst;
  Buffer shared_tensor_unmapped = shared_tensor;
  Array<Range> global_range = op.src_range;
  Array<Range> shared_range = op.dst_range;

  bool explicit_aiu = false;
  if (auto prefer = op.annotations.Get("prefer_instruction")) {
    if (auto str = prefer->as<StringImmNode>()) {
      explicit_aiu = str->value == "aiu";
    }
  }
  auto fallback_to_normal = [&](const std::string &reason) {
    if (explicit_aiu) {
      LOG(FATAL) << "T.copy prefer_instruction=\"aiu\" could not be honored: "
                 << reason << ", src=" << op.src->name
                 << ", dst=" << op.dst->name;
    }
    DLOG(WARNING) << "PPU AIU copy fallback to normal copy: " << reason
                  << ", src=" << op.src->name << ", dst=" << op.dst->name;
    return LowerNormal(op, lower_args, analyzer);
  };

  if (!TargetIsPPU(lower_args.target) || !TargetHasAiuCopy(lower_args.target)) {
    return fallback_to_normal("target has no PPU AIU copy support");
  }
  if (lower_args.layout_map.count(global_tensor)) {
    return fallback_to_normal("global tensor has a non-linear layout");
  }
  if (global_tensor->dtype != shared_tensor->dtype) {
    ICHECK_EQ(global_tensor->dtype, shared_tensor->dtype)
        << "Copy between buffer " << global_tensor->name << " and "
        << shared_tensor->name << " with different data type "
        << global_tensor->dtype << " and " << shared_tensor->dtype;
  }
  if (!global_tensor->dtype.is_float16() &&
      !global_tensor->dtype.is_bfloat16() &&
      !global_tensor->dtype.is_float8_e4m3fn() &&
      !global_tensor->dtype.is_float8_e5m2() &&
      !global_tensor->dtype.is_float4_e2m1fn()) {
    return fallback_to_normal(
        "AIU copy only supports fp16/bf16/fp8/fp4 payloads");
  }

  auto rank = global_tensor->shape.size();
  if (rank < 2 || rank > 5) {
    return fallback_to_normal("global tensor rank is outside [2, 5]");
  }

  Array<PrimExpr> shared_indices;
  for (auto r : shared_range) {
    shared_indices.push_back(r->min);
  }
  std::vector<PrimExpr> shared_strides;
  PrimExpr shared_stride = 1;
  for (size_t i = 0; i < shared_tensor->shape.size(); i++) {
    auto s = shared_tensor->shape[shared_tensor->shape.size() - i - 1];
    shared_strides.insert(shared_strides.begin(), shared_stride);
    shared_stride *= s;
  }
  ICHECK_EQ(shared_strides.size(), shared_indices.size())
      << "shared_strides.size() != shared_indices.size()";
  PrimExpr shared_offset = 0;
  for (size_t i = 0; i < shared_indices.size(); i++) {
    shared_offset += shared_indices[i] * shared_strides[i];
  }

  auto global_addr = global_tensor->data;
  auto global_shape = ReverseArray(global_tensor->shape);
  Array<PrimExpr> global_stride;
  Array<PrimExpr> global_coords =
      ReverseArray(global_range.Map([](Range r) { return r->min; }));
  if (!global_tensor->strides.empty()) {
    global_stride = ReverseArray(global_tensor->strides);
  } else {
    PrimExpr stride = 1;
    global_stride.reserve(rank);
    for (size_t i = 0; i < rank; i++) {
      global_stride.push_back(stride);
      stride *= global_shape[i];
    }
  }
  ICHECK(is_one(global_stride[0])) << global_stride;
  global_stride = global_stride.Map([&](PrimExpr e) {
    return AiuBytesFromElements(e, global_tensor->dtype);
  });
  for (size_t i = 1; i < global_stride.size(); i++) {
    if (auto stride = global_stride[i].as<IntImmNode>()) {
      if (stride->value % 16 != 0 || stride->value >= (1ULL << 63)) {
        return fallback_to_normal("unsupported global stride");
      }
    }
  }

  size_t shared_range_idx = 0;
  for (size_t i = 0; i < global_range.size(); i++) {
    auto g_range = global_range[i];
    if (is_one(g_range->extent)) {
      continue;
    }
    while (shared_range_idx < shared_range.size() &&
           is_one(shared_range[shared_range_idx]->extent)) {
      shared_range_idx++;
    }
    if (shared_range_idx >= shared_range.size()) {
      return fallback_to_normal(
          "global and shared ranges have incompatible ranks");
    }
    auto s_range = shared_range[shared_range_idx++];
    ICHECK(StructuralEqual()(g_range->extent, s_range->extent))
        << global_tensor->name << "[" << i << "] is illegal, "
        << global_tensor->name << "[" << i << "] = " << g_range->extent << ", "
        << shared_tensor->name << "[" << shared_range_idx
        << "] = " << s_range->extent;
  }

  Array<PrimExpr> smem_box =
      ReverseArray(global_range.Map([](Range r) { return r->extent; }));
  std::vector<size_t> cube_layout_pos;
  for (size_t i = 0; i < smem_box.size(); i++) {
    if (!is_one(smem_box[i])) {
      cube_layout_pos.push_back(i);
    }
  }
  if (cube_layout_pos.size() != 2) {
    return fallback_to_normal("AIU global cube shape is not 2D");
  }

  Layout shared_layout;
  if (lower_args.layout_map.count(shared_tensor)) {
    shared_layout = lower_args.layout_map.at(shared_tensor);
    ICHECK(lower_args.buffer_remap.count(shared_tensor))
        << "shared_tensor: " << shared_tensor->name
        << " not found in buffer_remap";
    shared_tensor = lower_args.buffer_remap.at(shared_tensor);
  }
  if (!shared_layout.defined()) {
    return fallback_to_normal("shared tensor has no swizzled layout");
  }
  if (StructuralEqual()(shared_layout,
                        Copy::ComputeLinearLayout(shared_tensor_unmapped))) {
    return fallback_to_normal("shared tensor uses linear layout");
  }

  int swizzle = -1;
  SwizzleMode swizzle_mode =
      DetectSwizzleMode(shared_layout, shared_tensor_unmapped);
  if (swizzle_mode == SwizzleMode::Swizzle64B()) {
    swizzle = 1;
  } else if (swizzle_mode == SwizzleMode::Swizzle128B()) {
    swizzle = 0;
  } else {
    return fallback_to_normal("shared layout is not 64B/128B swizzled");
  }

  RequireAIUSmemAlignment(lower_args, shared_tensor_unmapped);

  auto inner_box_dim = as_const_int(smem_box[0]);
  auto outer_box_dim = as_const_int(smem_box[cube_layout_pos[1]]);
  auto thread_extent = as_const_int(lower_args.thread_bounds->extent);
  if (inner_box_dim == nullptr || outer_box_dim == nullptr ||
      thread_extent == nullptr) {
    return fallback_to_normal("AIU split dimensions must be static integers");
  }

  int inner_box_dim_value = static_cast<int>(*inner_box_dim);
  int outer_box_dim_value = static_cast<int>(*outer_box_dim);
  int thread_extent_value = static_cast<int>(*thread_extent);

  // FP4 sub-byte: AIU .b8 requires row width >= 64B (minimum swizzle
  // granularity). FP4 with block_K < 128 -> row_bytes < 64 -> fallback to SIMT
  // copy.
  int dtype_bits = global_tensor->dtype.bits();
  bool is_sub_byte = (dtype_bits < 8);
  if (is_sub_byte) {
    int64_t row_bytes =
        static_cast<int64_t>(inner_box_dim_value) * dtype_bits / 8;
    if (row_bytes < 64) {
      return fallback_to_normal("AIU sub-byte dtype row width " +
                                std::to_string(row_bytes) +
                                "B < 64B minimum swizzle granularity");
    }
  }
  // Sub-byte: shape_0 and coord_0 are FloorDiv'd by packing_factor; reject
  // values that are not exactly divisible to avoid silent truncation.
  if (is_sub_byte) {
    int pf = 8 / dtype_bits;
    if (auto *shape_imm = global_shape[0].as<IntImmNode>()) {
      if (shape_imm->value % pf != 0) {
        return fallback_to_normal(
            "sub-byte inner extent not divisible by packing factor");
      }
    } else if (!analyzer->CanProveEqual(FloorMod(global_shape[0], pf), 0)) {
      return fallback_to_normal(
          "sub-byte inner extent not divisible by packing factor");
    }
    if (auto *coord_imm = global_coords[0].as<IntImmNode>()) {
      if (coord_imm->value % pf != 0) {
        return fallback_to_normal(
            "sub-byte inner offset not divisible by packing factor");
      }
    } else if (!analyzer->CanProveEqual(FloorMod(global_coords[0], pf), 0)) {
      return fallback_to_normal(
          "sub-byte inner offset not divisible by packing factor");
    }
  }

  // instruction_dim_elems: always in elements (for IR-level addressing and
  // splits). For sub-byte types, hardware .b8 args are in bytes; convert when
  // pushing args.
  int instruction_dim_elems = inner_box_dim_value;
  if (swizzle_mode == SwizzleMode::Swizzle64B()) {
    instruction_dim_elems = AiuElementsForBytes(64, shared_tensor->dtype);
  } else if (swizzle_mode == SwizzleMode::Swizzle128B()) {
    instruction_dim_elems = AiuElementsForBytes(128, shared_tensor->dtype);
  }
  if (instruction_dim_elems > 256) {
    ICHECK(inner_box_dim_value % 256 == 0)
        << "inner_box_dim: " << inner_box_dim_value
        << " is not divisible by 256";
    instruction_dim_elems = 256;
  }
  ICHECK(inner_box_dim_value % instruction_dim_elems == 0)
      << "inner_box_dim: " << inner_box_dim_value
      << " is not divisible by instruction_dim: " << instruction_dim_elems;

  int64_t inner_box_bytes =
      AiuBytesFromElements(instruction_dim_elems, shared_tensor->dtype);
  int max_swizzle_bytes = swizzle_mode == SwizzleMode::Swizzle64B() ? 64 : 128;
  if (inner_box_bytes > max_swizzle_bytes) {
    return fallback_to_normal("AIU inner box exceeds swizzle byte width");
  }

  int inner_splits = inner_box_dim_value / instruction_dim_elems;
  int num_warps = std::max(1, thread_extent_value / 32);
  int target_outer_splits = std::max(1, num_warps / inner_splits);
  int outer_splits = 1;
  for (int factor = 1;
       factor <= std::min(outer_box_dim_value, target_outer_splits); ++factor) {
    if (outer_box_dim_value % factor == 0) {
      outer_splits = factor;
    }
  }
  int outer_per_warp = outer_box_dim_value / outer_splits;
  int participating_warps = inner_splits * outer_splits;

  // Ensure outer_per_warp >= swizzle row period (8 for both 128B and 64B
  // layouts). If the warp split is too fine, reduce outer_splits to meet the
  // constraint.
  constexpr int kSwizzleRowPeriod = 8;
  if (outer_per_warp < kSwizzleRowPeriod) {
    if (outer_box_dim_value < kSwizzleRowPeriod) {
      return LowerNormal(op, lower_args, analyzer);
    }
    // Find the largest factor of outer_box_dim_value that yields outer_per_warp
    // >= kSwizzleRowPeriod
    int max_outer_splits = outer_box_dim_value / kSwizzleRowPeriod;
    for (int s = max_outer_splits; s >= 1; s--) {
      if (outer_box_dim_value % s == 0) {
        outer_splits = s;
        break;
      }
    }
    outer_per_warp = outer_box_dim_value / outer_splits;
    participating_warps = inner_splits * outer_splits;
  }

  smem_box.Set(0, PrimExpr(instruction_dim_elems));
  smem_box.Set(cube_layout_pos[1], PrimExpr(outer_per_warp));

  // Compute shape_0 (inner dim, in elements) and shape_1 (outer dim, in rows).
  PrimExpr shape_0_elements;
  PrimExpr shape_1_rows;
  {
    PrimExpr acc = 1;
    for (size_t i = 0; i < rank; ++i) {
      acc = acc * global_shape[i];
      if (i == cube_layout_pos[1] - 1) {
        shape_0_elements = acc;
        acc = 1;
      } else if (i == rank - 1) {
        shape_1_rows = acc;
      }
    }
  }

  PrimExpr total_elements = 1;
  for (auto e : smem_box) {
    total_elements *= e;
  }

  // Warp partitioning (unchanged).
  PrimExpr warp_id =
      FloorDiv(lower_args.thread_var, IntImm(DataType::Int(32), 32));
  PrimExpr warp_inner_idx =
      FloorMod(warp_id, IntImm(DataType::Int(32), inner_splits));
  PrimExpr warp_outer_idx =
      FloorDiv(warp_id, IntImm(DataType::Int(32), inner_splits));
  PrimExpr shared_addr = shared_tensor.access_ptr(
      2, DataType::Handle(), 1,
      shared_offset +
          warp_inner_idx * (instruction_dim_elems * outer_box_dim_value) +
          warp_outer_idx * (instruction_dim_elems * outer_per_warp),
      total_elements);

  // Coordinate computation (in elements).
  global_coords.Set(0,
                    global_coords[0] + instruction_dim_elems * warp_inner_idx);
  global_coords.Set(cube_layout_pos[1], global_coords[cube_layout_pos[1]] +
                                            outer_per_warp * warp_outer_idx);
  auto compute_coord = [&](size_t begin, size_t end) {
    PrimExpr result = 0;
    PrimExpr stride = 1;
    for (size_t i = begin; i <= end; ++i) {
      result += global_coords[i] * stride;
      stride *= global_shape[i];
    }
    return result;
  };
  PrimExpr coord_0_elements = compute_coord(0, cube_layout_pos[1] - 1);
  PrimExpr coord_1_rows = compute_coord(cube_layout_pos[1], rank - 1);

  // Build 10-parameter byte-mode args for ppu_aiu_load.
  // C-dimension params (dim_c, cube_c, start_c) are in bytes;
  // W-dimension params (dim_w, cube_w, start_w) are in rows.
  DataType dtype = global_tensor->dtype;
  Array<PrimExpr> args;
  args.reserve(10);
  args.push_back(shared_addr); // [0] smem_ptr
  args.push_back(global_addr); // [1] gmem_ptr
  args.push_back(
      AiuBytesFromElements(shape_0_elements, dtype)); // [2] dim_c (bytes)
  args.push_back(shape_1_rows);                       // [3] dim_w (rows)
  args.push_back(AiuBytesFromElements(PrimExpr(instruction_dim_elems),
                                      dtype));       // [4] cube_c (bytes)
  args.push_back(smem_box[cube_layout_pos[1]]);      // [5] cube_w (rows)
  args.push_back(global_stride[cube_layout_pos[1]]); // [6] stride_w_bytes
  args.push_back(
      AiuBytesFromElements(coord_0_elements, dtype)); // [7] start_c (bytes)
  args.push_back(coord_1_rows);                       // [8] start_w (rows)
  args.push_back(swizzle);                            // [9] swzl_mode

  Stmt aiu_copy = Evaluate(Call(DataType::Handle(), ppu_aiu_load(), args));
  return IfThenElse(LT(warp_id, IntImm(DataType::Int(32), participating_warps)),
                    aiu_copy);
}

} // namespace ppu

namespace {

bool MatchPpuCopyTarget(Target target) { return TargetIsPPU(target); }

bool RegisterPpuCopy() {
  RegisterCopyImpl(CopyImpl{
      "ppu.Copy",
      MatchPpuCopyTarget,
      200,
      ppu::Copy::InferLayout,
      ppu::Copy::Lower,
  });
  return true;
}

const bool ppu_copy_registered = RegisterPpuCopy();

} // namespace

} // namespace tl
} // namespace tvm
