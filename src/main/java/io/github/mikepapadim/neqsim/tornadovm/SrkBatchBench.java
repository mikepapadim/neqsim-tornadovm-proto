package io.github.mikepapadim.neqsim.tornadovm;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Validates the batched SRK kernel against NeqSim and times it.
 *
 * <p>Usage: {@code SrkBatchBench validate <states>} or {@code SrkBatchBench bench <states...>}.
 * The device is chosen with {@code -Dtornado.backend.device=<backend>:<device>} (default 0:0).
 */
public final class SrkBatchBench {

  private SrkBatchBench() {}

  /** Random states around the feed: T 200-500 K, P 1-250 bara, perturbed composition. */
  static void states(int count, int n, long seed, DoubleArray tp, DoubleArray x) {
    Random rnd = new Random(seed);
    for (int s = 0; s < count; s++) {
      tp.set(2 * s, 200.0 + 300.0 * rnd.nextDouble());
      tp.set(2 * s + 1, 1.0 + 249.0 * rnd.nextDouble());
      double sum = 0.0;
      double[] v = new double[n];
      for (int i = 0; i < n; i++) {
        v[i] = NeqSimSrk.FEED[i] * Math.exp(0.5 * rnd.nextGaussian());
        sum += v[i];
      }
      for (int i = 0; i < n; i++) {
        x.set(s * n + i, v[i] / sum);
      }
    }
  }

  static TornadoDevice device() {
    String spec = System.getProperty("tornado.backend.device", "0:0");
    String[] parts = spec.split(":");
    return TornadoRuntimeProvider.getTornadoRuntime()
        .getBackend(Integer.parseInt(parts[0]))
        .getDevice(Integer.parseInt(parts[1]));
  }

  /** Builds a plan for the batch; inputs are copied on every execution, the model once. */
  static TornadoExecutionPlan plan(
      DoubleArray model,
      DoubleArray tp,
      DoubleArray x,
      DoubleArray out,
      DoubleArray work,
      IntArray dims,
      int count) {
    KernelContext context = new KernelContext();
    TaskGraph graph =
        new TaskGraph("srk")
            .transferToDevice(DataTransferMode.FIRST_EXECUTION, model, dims, work)
            .transferToDevice(DataTransferMode.EVERY_EXECUTION, tp, x)
            .task("lnphi", SrkBatchKernels::srkKernel, context, model, tp, x, out, work, dims)
            .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
    ImmutableTaskGraph itg = graph.snapshot();
    WorkerGrid1D worker = new WorkerGrid1D(((count + 127) / 128) * 128);
    worker.setLocalWork(128, 1, 1);
    GridScheduler grid = new GridScheduler("srk.lnphi", worker);
    TornadoExecutionPlan plan = new TornadoExecutionPlan(itg);
    plan.withGridScheduler(grid).withDevice(device());
    return plan;
  }

  static void cpu(
      boolean parallel,
      int count,
      int n,
      DoubleArray model,
      DoubleArray tp,
      DoubleArray x,
      DoubleArray out,
      DoubleArray work) {
    IntStream range = IntStream.range(0, count);
    (parallel ? range.parallel() : range)
        .forEach(s -> SrkBatchKernels.state(s, n, model, tp, x, out, work));
  }

  public static void main(String[] args) throws Exception {
    String mode = args.length > 0 ? args[0] : "validate";
    if (mode.equals("validate")) {
      validate(args.length > 1 ? Integer.parseInt(args[1]) : 2000);
    } else {
      int[] sizes =
          args.length > 1
              ? Arrays.stream(args, 1, args.length).mapToInt(Integer::parseInt).toArray()
              : new int[] {1_000, 10_000, 100_000, 1_000_000};
      for (int size : sizes) {
        bench(size);
      }
    }
  }

  static void validate(int count) {
    NeqSimSrk neqsim = new NeqSimSrk();
    int n = neqsim.components();
    DoubleArray model = neqsim.packModel();
    DoubleArray tp = new DoubleArray(2 * count);
    DoubleArray x = new DoubleArray(count * n);
    states(count, n, 42L, tp, x);
    DoubleArray outCpu = new DoubleArray(count * SrkBatchKernels.outStride(n));
    DoubleArray outGpu = new DoubleArray(count * SrkBatchKernels.outStride(n));
    DoubleArray work = new DoubleArray(count * n);
    IntArray dims = IntArray.fromElements(n, count);

    cpu(false, count, n, model, tp, x, outCpu, work);
    try (TornadoExecutionPlan plan = plan(model, tp, x, outGpu, new DoubleArray(count * n), dims, count)) {
      plan.execute();
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }

    double[] ref = new double[2 * n + 2];
    double[] xs = new double[n];
    double maxRefLnphi = 0;
    double maxRefZ = 0;
    double maxGpuCpu = 0;
    int threeRoot = 0;
    int worst = -1;
    int stride = SrkBatchKernels.outStride(n);
    for (int s = 0; s < count; s++) {
      for (int i = 0; i < n; i++) {
        xs[i] = x.get(s * n + i);
      }
      neqsim.reference(tp.get(2 * s), tp.get(2 * s + 1), xs, ref);
      int o = s * stride;
      if (Math.abs(outGpu.get(o) - outGpu.get(o + 1)) > 1e-9) {
        threeRoot++;
      }
      for (int k = 0; k < stride; k++) {
        maxGpuCpu = Math.max(maxGpuCpu, Math.abs(outGpu.get(o + k) - outCpu.get(o + k)));
      }
      maxRefZ = Math.max(maxRefZ, Math.abs(outGpu.get(o) - ref[2 * n]));
      maxRefZ = Math.max(maxRefZ, Math.abs(outGpu.get(o + 1) - ref[2 * n + 1]));
      for (int i = 0; i < n; i++) {
        double d =
            Math.max(
                Math.abs(outGpu.get(o + 2 + i) - ref[i]),
                Math.abs(outGpu.get(o + 2 + n + i) - ref[n + i]));
        if (d > maxRefLnphi) {
          maxRefLnphi = d;
          worst = s;
        }
      }
    }
    System.out.printf(
        "validate states=%d device=%s three-root=%d%n", count, device().getDeviceName(), threeRoot);
    System.out.printf("  max |lnphi GPU - NeqSim| = %.3e (worst state %d)%n", maxRefLnphi, worst);
    System.out.printf("  max |Z GPU - NeqSim|     = %.3e%n", maxRefZ);
    System.out.printf("  max |GPU - CPU kernel|   = %.3e%n", maxGpuCpu);
    if (worst >= 0) {
      int o = worst * stride;
      for (int i = 0; i < n; i++) {
        xs[i] = x.get(worst * n + i);
      }
      neqsim.reference(tp.get(2 * worst), tp.get(2 * worst + 1), xs, ref);
      System.out.printf(
          "  worst: T=%.2f P=%.2f Z gpu=(%.6f,%.6f) neqsim=(%.6f,%.6f)%n",
          tp.get(2 * worst), tp.get(2 * worst + 1), outGpu.get(o), outGpu.get(o + 1),
          ref[2 * n], ref[2 * n + 1]);
    }
  }

  static double median(double[] v) {
    double[] c = v.clone();
    Arrays.sort(c);
    return c[c.length / 2];
  }

  static void bench(int count) throws Exception {
    NeqSimSrk neqsim = new NeqSimSrk();
    int n = neqsim.components();
    int reps = Integer.getInteger("reps", 9);
    DoubleArray model = neqsim.packModel();
    DoubleArray tp = new DoubleArray(2 * count);
    DoubleArray x = new DoubleArray(count * n);
    states(count, n, 7L, tp, x);
    DoubleArray out = new DoubleArray(count * SrkBatchKernels.outStride(n));
    DoubleArray work = new DoubleArray(count * n);
    IntArray dims = IntArray.fromElements(n, count);

    double[] t1 = new double[reps];
    double[] tN = new double[reps];
    double[] tGpu = new double[reps];
    double[] tKernel = new double[reps];
    for (int w = 0; w < 3; w++) {
      cpu(false, count, n, model, tp, x, out, work);
      cpu(true, count, n, model, tp, x, out, work);
    }
    for (int r = 0; r < reps; r++) {
      long a = System.nanoTime();
      cpu(false, count, n, model, tp, x, out, work);
      long b = System.nanoTime();
      cpu(true, count, n, model, tp, x, out, work);
      long c = System.nanoTime();
      t1[r] = (b - a) / 1e6;
      tN[r] = (c - b) / 1e6;
    }

    // NeqSim object path: one system, per-state setTemperature/setPressure/composition/init(1).
    int neqCount = Math.min(count, Integer.getInteger("neqsimStates", 20_000));
    double[] ref = new double[2 * n + 2];
    double[] xs = new double[n];
    double neqMs = Double.NaN;
    for (int pass = 0; pass < 2; pass++) {
      long a = System.nanoTime();
      for (int s = 0; s < neqCount; s++) {
        for (int i = 0; i < n; i++) {
          xs[i] = x.get(s * n + i);
        }
        neqsim.reference(tp.get(2 * s), tp.get(2 * s + 1), xs, ref);
      }
      neqMs = (System.nanoTime() - a) / 1e6 * count / neqCount;
    }

    try (TornadoExecutionPlan plan = plan(model, tp, x, out, work, dims, count)) {
      plan.withProfiler(ProfilerMode.SILENT);
      long c0 = System.nanoTime();
      plan.execute();
      double first = (System.nanoTime() - c0) / 1e6;
      for (int w = 0; w < 3; w++) {
        plan.execute();
      }
      for (int r = 0; r < reps; r++) {
        long a = System.nanoTime();
        TornadoExecutionResult res = plan.execute();
        tGpu[r] = (System.nanoTime() - a) / 1e6;
        tKernel[r] = res.getProfilerResult().getDeviceKernelTime() / 1e6;
      }
      System.out.printf(
          "states=%d n=%d | NeqSim objects 1T %.2f ms (extrapolated from %d) | arrays 1T %.2f ms |"
              + " arrays %dT %.2f ms | TornadoVM %s: first %.1f ms, warm %.2f ms (kernel %.2f ms)%n",
          count, n, neqMs, neqCount, median(t1),
          Runtime.getRuntime().availableProcessors(), median(tN), device().getDeviceName(),
          first, median(tGpu), median(tKernel));
    }
  }
}
