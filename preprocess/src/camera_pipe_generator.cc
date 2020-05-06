#include "Halide.h"
#include <cstdint>

namespace {

using std::vector;

using namespace Halide;
using namespace Halide::ConciseCasts;

// Shared variables
Var x, y, c, yi, yo, yii, xi;

Expr protune(Func input) {
  Func im;
  return log(112.f * input(x, y, c) + 1.f) / log(113.f);
}

class Demosaic : public Halide::Generator<Demosaic> {
 public:
  GeneratorParam<LoopLevel> output_compute_at{"output_compute_at", LoopLevel::inlined()};

  // Inputs and outputs
  Input<Func> deinterleaved{"deinterleaved", Float(32), 3};
  Input<Buffer<int>> cfa{"cfa", 2};
  Output<Func> output{"output", Float(32), 3};

  // Defines outputs using inputs
  void generate() {
    // Give more convenient names to the four channels we know
    Func r_r, g_gr, g_gb, b_b;

    g_gr(x, y) = deinterleaved(x, y, cfa(0, 1));
    r_r(x, y) = deinterleaved(x, y, cfa(0, 0));
    b_b(x, y) = deinterleaved(x, y, cfa(1, 1));
    g_gb(x, y) = deinterleaved(x, y, cfa(1, 0));

    // Resulting color channels
    Func r, g, b;

    // Interleave the resulting channels
    r(x, y) = (r_r(2 * x, 2 * y) + r_r(2 * x + 1, 2 * y) +
        r_r(2 * x, 2 * y + 1) + r_r(2 * x + 1, 2 * y + 1)) * 0.25f;
    g(x, y) = (g_gr(2 * x, 2 * y) + g_gr(2 * x + 1, 2 * y) +
        g_gr(2 * x, 2 * y + 1) + g_gr(2 * x + 1, 2 * y + 1) +
        g_gb(2 * x, 2 * y) + g_gb(2 * x + 1, 2 * y) +
        g_gb(2 * x, 2 * y + 1) + g_gb(2 * x + 1, 2 * y + 1)) * 0.125f;
    b(x, y) = (b_b(2 * x, 2 * y) + b_b(2 * x + 1, 2 * y) +
        b_b(2 * x, 2 * y + 1) + b_b(2 * x + 1, 2 * y + 1)) * 0.25f;

    output(x, y, c) = mux(c, {r(x, y), g(x, y), b(x, y)});
  }

  void schedule() {
    Pipeline p(output);

    if (auto_schedule) {
      // blank
    } else if (get_target().has_gpu_feature()) {
      output.compute_at(output_compute_at)
          .gpu_threads(x, y)
          .reorder(c, x, y);
    } else {
      int vec = get_target().natural_vector_size(Float(32));
      output.compute_at(output_compute_at)
          .vectorize(x)
          .reorder(c, x, y);
    }
  }
};

class CameraPipe : public Halide::Generator<CameraPipe> {
 public:
  GeneratorParam<Type> result_type{"result_type", UInt(8)};

  Input<Buffer<float>> input{"input", 2};
  Input<Buffer<int>> cfa{"cfa", 2};
  Input<int> blackLevel{"blackLevel"};
  Input<int> whiteLevel{"whiteLevel"};
  Output<Buffer<uint8_t>> processed{"processed", 3};

  void generate();

 private:
  Func scale(Func input);
  Func deinterleave(Func raw);
  Func apply_curve(Func input);
};

Func CameraPipe::scale(Func input) {
  Func scaled("scaled");
  Expr diff = 1.f / (whiteLevel - blackLevel);
  scaled(x, y) = clamp((input(x, y) - blackLevel) * diff, 0.f, 1.f);
  return scaled;
}

Func CameraPipe::deinterleave(Func raw) {
  // Deinterleave the color channels
  Func deinterleaved("deinterleaved");

  deinterleaved(x, y, c) = mux(c,
                               {raw(2 * x, 2 * y),
                                raw(2 * x + 1, 2 * y),
                                raw(2 * x, 2 * y + 1),
                                raw(2 * x + 1, 2 * y + 1)});
  return deinterleaved;
}

Func CameraPipe::apply_curve(Func input) {
  Func curved("curved");
  curved(x, y, c) = cast<uint8_t>(clamp(protune(input) * 255, 0, 255));
  return curved;
}

void CameraPipe::generate() {
  Func extended = BoundaryConditions::repeat_edge(input);
  Func scaled = scale(extended);

  Func deinterleaved = deinterleave(scaled);

  auto demosaiced = create<Demosaic>();
  demosaiced->apply(deinterleaved, cfa);

  Func curved = apply_curve(demosaiced->output);

  processed(x, y, c) = curved(x, y, c);

  /* ESTIMATES */
  // (This can be useful in conjunction with RunGen and benchmarks as well
  // as auto-schedule, so we do it in all cases.)
  input.set_estimates({{0, 4096}, {0, 3072}});
  cfa.set_estimates({{0, 3}, {0, 3}});
  blackLevel.set_estimate(0);
  whiteLevel.set_estimate(1);
  processed.set_estimates({{0, 1024}, {0, 768}, {0, 3}});

  // Schedule
  if (auto_schedule) {
    // nothing
  } else if (get_target().has_gpu_feature()) {
    // We can generate slightly better code if we know the output is even-sized
    if (!auto_schedule) {
      // TODO: The autoscheduler really ought to be able to
      // accommodate bounds on the output Func.
      Expr out_width = processed.width();
      Expr out_height = processed.height();
      processed.bound(c, 0, 3)
          .bound(x, 0, out_width)
          .bound(y, 0, out_height);
    }

    Var xi, yi, xii, xio;

    /* 1391us on a gtx 980. */
    processed.compute_root()
        .reorder(c, x, y)
        .unroll(x, 2)
        .gpu_tile(x, y, xi, yi, 28, 12);

    curved.compute_at(processed, x)
        .unroll(x, 2)
        .gpu_threads(x, y);

    demosaiced->output_compute_at.set({processed, x});

    deinterleaved.compute_at(processed, x)
        .unroll(x, 2)
        .gpu_threads(x, y)
        .reorder(c, x, y)
        .bound(c, 0, 4)
        .unroll(c);

    scaled.compute_at(processed, x)
        .unroll(x, 2)
        .gpu_threads(x, y);

  } else {
    Expr out_width = processed.width();
    Expr out_height = processed.height();

    Expr strip_size = 32;
    strip_size = (strip_size / 2) * 2;

    int vec = get_target().natural_vector_size(Float(32));

    processed
        .compute_root()
        .reorder(c, x, y)
        .split(y, yi, yii, 2, TailStrategy::RoundUp)
        .split(yi, yo, yi, strip_size / 2)
        .vectorize(x, vec, TailStrategy::RoundUp)
        .unroll(c)
        .parallel(yo);

    deinterleaved
        .compute_at(processed, yi)
        .store_at(processed, yo)
        .fold_storage(y, 8)
        .reorder(c, x, y)
        .vectorize(x, vec, TailStrategy::RoundUp);

    curved
        .compute_at(processed, yi)
        .store_at(processed, yo)
        .reorder(c, x, y)
        .tile(x, y, x, y, xi, yi, vec, 2, TailStrategy::RoundUp)
        .vectorize(xi)
        .unroll(yi)
        .unroll(c);

    demosaiced->output_compute_at.set({curved, x});

    // We can generate slightly better code if we know the splits divide the extent.
    processed
        .bound(c, 0, 3)
        .bound(x, 0, out_width)
        .bound(y, 0, out_height);
  }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(CameraPipe, camera_pipe)
