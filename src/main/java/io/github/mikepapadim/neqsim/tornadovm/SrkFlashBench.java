package io.github.mikepapadim.neqsim.tornadovm;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.IntStream;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Validates the batched flash against NeqSim's TPflash and times it.
 *
 * <p>Usage: {@code SrkFlashBench validate <states>} or {@code SrkFlashBench bench <states...>}.
 */
public final class SrkFlashBench {

  private SrkFlashBench() {}

  static final int MAX_IT = Integer.getInteger("maxIt", 500);

  /** Layout of the per-state arrays on the device; the CPU always uses state-major. */
  static final boolean GPU_COMPONENT_MAJOR =
      !"state".equals(System.getProperty("gpuLayout", "component"));

  /** States in the two-phase region and around it: T 220-400 K, P 5-150 bara. */
  static void states(
      int count, int n, long seed, DoubleArray tp, DoubleArray z, boolean componentMajor) {
    Random rnd = new Random(seed);
    for (int s = 0; s < count; s++) {
      tp.set(2 * s, 220.0 + 180.0 * rnd.nextDouble());
      tp.set(2 * s + 1, 5.0 + 145.0 * rnd.nextDouble());
      double sum = 0.0;
      double[] v = new double[n];
      for (int i = 0; i < n; i++) {
        v[i] = NeqSimSrk.FEED[i] * Math.exp(0.5 * rnd.nextGaussian());
        sum += v[i];
      }
      for (int i = 0; i < n; i++) {
        z.set(SrkFlashKernels.index(s, i, n, count, componentMajor), v[i] / sum);
      }
    }
  }

  static TornadoExecutionPlan plan(
      DoubleArray model,
      DoubleArray crit,
      DoubleArray tp,
      DoubleArray z,
      DoubleArray out,
      DoubleArray work,
      IntArray dims,
      int count) {
    KernelContext context = new KernelContext();
    TaskGraph graph =
        new TaskGraph("flash")
            .transferToDevice(DataTransferMode.FIRST_EXECUTION, model, crit, dims, work)
            .transferToDevice(DataTransferMode.EVERY_EXECUTION, tp, z)
            .task(
                "tp",
                SrkFlashKernels::flashKernel,
                context,
                model,
                crit,
                tp,
                z,
                out,
                work,
                dims)
            .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
    int local = Integer.getInteger("local", 64);
    WorkerGrid1D worker = new WorkerGrid1D(((count + local - 1) / local) * local);
    worker.setLocalWork(local, 1, 1);
    TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot());
    plan.withGridScheduler(new GridScheduler("flash.tp", worker))
        .withDevice(SrkBatchBench.device());
    return plan;
  }

  static void cpu(
      boolean parallel,
      int count,
      int n,
      DoubleArray model,
      DoubleArray crit,
      DoubleArray tp,
      DoubleArray z,
      DoubleArray out,
      DoubleArray work) {
    IntStream range = IntStream.range(0, count);
    (parallel ? range.parallel() : range)
        .forEach(
            s -> SrkFlashKernels.flash(s, n, MAX_IT, count, false, model, crit, tp, z, out, work));
  }

  public static void main(String[] args) throws Exception {
    if (args[0].equals("validate")) {
      validate(Integer.parseInt(args[1]));
    } else {
      for (int i = 1; i < args.length; i++) {
        bench(Integer.parseInt(args[i]));
      }
    }
  }

  static void validate(int count) throws Exception {
    NeqSimSrk neqsim = new NeqSimSrk();
    int n = neqsim.components();
    DoubleArray model = neqsim.packModel();
    DoubleArray crit = neqsim.packCrit();
    DoubleArray tp = new DoubleArray(2 * count);
    DoubleArray z = new DoubleArray(count * n);
    DoubleArray zGpu = new DoubleArray(count * n);
    states(count, n, 42L, tp, z, false);
    states(count, n, 42L, new DoubleArray(2 * count), zGpu, GPU_COMPONENT_MAJOR);
    int stride = SrkFlashKernels.outStride(n);
    DoubleArray outCpu = new DoubleArray(count * stride);
    DoubleArray outGpu = new DoubleArray(count * stride);
    IntArray dims = IntArray.fromElements(n, count, MAX_IT, GPU_COMPONENT_MAJOR ? 1 : 0);
    cpu(
        false, count, n, model, crit, tp, z, outCpu,
        new DoubleArray(count * SrkFlashKernels.WORK_PER_COMPONENT * n));
    try (TornadoExecutionPlan plan =
        plan(model, crit, tp, zGpu, outGpu,
            new DoubleArray(count * SrkFlashKernels.WORK_PER_COMPONENT * n), dims, count)) {
      plan.execute();
    }

    double[] ref = new double[2 + 2 * n];
    double[] zs = new double[n];
    int bothTwo = 0;
    int bothOne = 0;
    int neqTwoOnly = 0;
    int gpuTwoOnly = 0;
    int notConverged = 0;
    int off = 0;
    int swapped = 0;
    double maxBeta = 0;
    double maxX = 0;
    double maxGpuCpu = 0;
    long iterations = 0;
    for (int s = 0; s < count; s++) {
      for (int i = 0; i < n; i++) {
        zs[i] = z.get(s * n + i);
      }
      int phases = neqsim.referenceFlash(tp.get(2 * s), tp.get(2 * s + 1), zs, ref);
      int o = s * stride;
      for (int k = 0; k < stride; k++) {
        maxGpuCpu = Math.max(maxGpuCpu, Math.abs(gpu(outGpu, s, k, n, count) - outCpu.get(o + k)));
      }
      double status = gpu(outGpu, s, 2, n, count);
      iterations += (long) gpu(outGpu, s, 1, n, count);
      double betaGpu = gpu(outGpu, s, 0, n, count);
      if (status < 0) {
        notConverged++;
      }
      boolean gpuTwo = status > 0.5;
      if (phases == 2 && gpuTwo) {
        bothTwo++;
        double db = Math.abs(betaGpu - ref[0]);
        if (db > 1e-6) {
          off++;
        }
        if (db > 1e-6 && Math.abs(betaGpu - (1.0 - ref[0])) < 1e-6) {
          swapped++;
        } else {
          maxBeta = Math.max(maxBeta, db);
          for (int i = 0; i < 2 * n; i++) {
            maxX = Math.max(maxX, Math.abs(gpu(outGpu, s, 3 + i, n, count) - ref[2 + i]));
          }
        }
      } else if (phases == 1 && !gpuTwo && status >= 0) {
        bothOne++;
      } else if (phases == 2) {
        neqTwoOnly++;
      } else if (gpuTwo) {
        gpuTwoOnly++;
      }
    }
    System.out.printf(
        "flash validate states=%d device=%s maxIt=%d mean iterations=%.1f%n",
        count, SrkBatchBench.device().getDeviceName(), MAX_IT, (double) iterations / count);
    System.out.printf(
        "  both two-phase %d, both single-phase %d, NeqSim two-phase only %d, kernel two-phase only"
            + " %d, not converged %d%n",
        bothTwo, bothOne, neqTwoOnly, gpuTwoOnly, notConverged);
    System.out.printf(
        "  two-phase states with |beta diff| > 1e-6: %d (of which phase labels swapped: %d)%n  two-phase agreement: max |beta| %.3e, max |x,y| %.3e; max |GPU - CPU kernel| %.3e%n",
        off, swapped, maxBeta, maxX, maxGpuCpu);
  }

  /** Value k of state s in the device output. */
  static double gpu(DoubleArray out, int s, int k, int n, int count) {
    return out.get(
        SrkFlashKernels.index(s, k, SrkFlashKernels.outStride(n), count, GPU_COMPONENT_MAJOR));
  }

  static void bench(int count) throws Exception {
    NeqSimSrk neqsim = new NeqSimSrk();
    int n = neqsim.components();
    int reps = Integer.getInteger("reps", 7);
    DoubleArray model = neqsim.packModel();
    DoubleArray crit = neqsim.packCrit();
    DoubleArray tp = new DoubleArray(2 * count);
    DoubleArray z = new DoubleArray(count * n);
    DoubleArray zGpu = new DoubleArray(count * n);
    states(count, n, 7L, tp, z, false);
    states(count, n, 7L, new DoubleArray(2 * count), zGpu, GPU_COMPONENT_MAJOR);
    DoubleArray out = new DoubleArray(count * SrkFlashKernels.outStride(n));
    DoubleArray work = new DoubleArray(count * SrkFlashKernels.WORK_PER_COMPONENT * n);
    IntArray dims = IntArray.fromElements(n, count, MAX_IT, GPU_COMPONENT_MAJOR ? 1 : 0);

    double[] t1 = new double[reps];
    double[] tN = new double[reps];
    for (int w = 0; w < 2; w++) {
      cpu(true, count, n, model, crit, tp, z, out, work);
    }
    for (int r = 0; r < reps; r++) {
      long a = System.nanoTime();
      if (count <= 100_000) {
        cpu(false, count, n, model, crit, tp, z, out, work);
      }
      long b = System.nanoTime();
      cpu(true, count, n, model, crit, tp, z, out, work);
      long c = System.nanoTime();
      t1[r] = (count <= 100_000) ? (b - a) / 1e6 : Double.NaN;
      tN[r] = (c - b) / 1e6;
    }

    // NeqSim TPflash, one isolated system per worker thread (the reuse pattern from #4587).
    int threads = Runtime.getRuntime().availableProcessors();
    int neqCount = Math.min(count, Integer.getInteger("neqsimStates", 64_000));
    ExecutorService pool = Executors.newFixedThreadPool(threads);
    double neqMs = Double.NaN;
    for (int pass = 0; pass < 2; pass++) {
      long a = System.nanoTime();
      Future<?>[] fs = new Future<?>[threads];
      for (int w = 0; w < threads; w++) {
        final int worker = w;
        fs[w] =
            pool.submit(
                () -> {
                  NeqSimSrk local = new NeqSimSrk();
                  double[] ref = new double[2 + 2 * n];
                  double[] zs = new double[n];
                  for (int s = worker; s < neqCount; s += threads) {
                    for (int i = 0; i < n; i++) {
                      zs[i] = z.get(s * n + i);
                    }
                    local.referenceFlash(tp.get(2 * s), tp.get(2 * s + 1), zs, ref);
                  }
                });
      }
      for (Future<?> f : fs) {
        f.get();
      }
      neqMs = (System.nanoTime() - a) / 1e6 * count / neqCount;
    }
    pool.shutdown();

    double[] tGpu = new double[reps];
    double[] tKernel = new double[reps];
    try (TornadoExecutionPlan plan = plan(model, crit, tp, zGpu, out, work, dims, count)) {
      plan.withProfiler(ProfilerMode.SILENT);
      long c0 = System.nanoTime();
      plan.execute();
      double first = (System.nanoTime() - c0) / 1e6;
      for (int w = 0; w < 2; w++) {
        plan.execute();
      }
      for (int r = 0; r < reps; r++) {
        long a = System.nanoTime();
        TornadoExecutionResult res = plan.execute();
        tGpu[r] = (System.nanoTime() - a) / 1e6;
        tKernel[r] = res.getProfilerResult().getDeviceKernelTime() / 1e6;
      }
      System.out.printf(
          "flash states=%d maxIt=%d gpuLayout=%s | NeqSim TPflash %dT %.1f ms (from %d) | kernel on CPU 1T %.1f ms, %dT"
              + " %.1f ms | TornadoVM %s: first %.1f ms, warm %.1f ms (kernel %.1f ms)%n",
          count, MAX_IT, GPU_COMPONENT_MAJOR ? "component" : "state", threads, neqMs, neqCount, SrkBatchBench.median(t1), threads,
          SrkBatchBench.median(tN), SrkBatchBench.device().getDeviceName(), first,
          SrkBatchBench.median(tGpu), SrkBatchBench.median(tKernel));
    }
  }
}
