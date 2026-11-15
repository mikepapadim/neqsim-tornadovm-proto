package io.github.mikepapadim.neqsim.tornadovm;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Batched two-phase SRK TP flash, in FP64, one state per GPU thread: Wilson K-values, then
 * successive substitution with a Rachford-Rice solve per iteration.
 *
 * <p>This is a throughput experiment, not a complete flash: there is no stability analysis and no
 * acceleration (DEM/Newton) as in NeqSim's TPflash. A state whose Rachford-Rice equation has no root
 * in (0, 1) is reported single phase.
 *
 * <p>Layouts, with {@code n} components ({@code model} as in {@link SrkBatchKernels}):
 *
 * <ul>
 *   <li>{@code crit}: {@code Pc[n] (bar), omega[n]}
 *   <li>{@code tp}: {@code T, P} per state; {@code z}: feed, {@code n} per state
 *   <li>{@code out}: {@code beta, iterations, status, x[n], y[n]} per state; status 1 two-phase,
 *       0 single phase, -1 not converged
 *   <li>{@code work}: {@link #WORK_PER_COMPONENT} times {@code n} doubles per state
 *   <li>{@code dims}: {@code n, states, maxIterations, componentMajor (0 or 1)}
 *   <li>Per-state arrays ({@code z}, {@code out}, {@code work}) are state-major (value k of state s
 *       at {@code s * width + k}) or component-major (at {@code k * states + s})
 * </ul>
 */
public final class SrkFlashKernels {

  /** Scratch doubles per component and state: K, x, y, lnphiL, lnphiV, sqrt(aT). */
  public static final int WORK_PER_COMPONENT = 6;

  private static final double R = SrkBatchKernels.R;

  private SrkFlashKernels() {}

  /** Doubles per state in {@code out}. */
  public static int outStride(int n) {
    return 3 + 2 * n;
  }

  /** TornadoVM entry point: one thread per state. */
  public static void flashKernel(
      KernelContext context,
      DoubleArray model,
      DoubleArray crit,
      DoubleArray tp,
      DoubleArray z,
      DoubleArray out,
      DoubleArray work,
      IntArray dims) {
    int s = context.globalIdx;
    if (s < dims.get(1)) {
      flash(
          s, dims.get(0), dims.get(2), dims.get(1), dims.get(3) == 1, model, crit, tp, z, out,
          work);
    }
  }

  /**
   * Log fugacity coefficients of composition {@code comp[co..co+n)} at the liquid (smallest above
   * B) or vapour (largest) root, written to {@code dst[dO..dO+n)}. {@code sa[so..so+n)} holds
   * sqrt(a_i(T)).
   */
  static void lnphi(
      int n,
      int st,
      DoubleArray model,
      double t,
      double p,
      DoubleArray comp,
      int co,
      boolean liquid,
      DoubleArray dst,
      int dO,
      DoubleArray sa,
      int so) {
    double amix = 0.0;
    double bmix = 0.0;
    for (int i = 0; i < n; i++) {
      double si = sa.get(so + i * st);
      double sum = 0.0;
      for (int j = 0; j < n; j++) {
        sum +=
            comp.get(co + j * st) * si * sa.get(so + j * st) * (1.0 - model.get(4 * n + i * n + j));
      }
      dst.set(dO + i * st, sum);
      double ci = comp.get(co + i * st);
      amix += ci * sum;
      bmix += ci * model.get(n + i);
    }
    double rt = R * t;
    double bigA = amix * p / (rt * rt);
    double bigB = bmix * p / rt;
    double c1 = bigA - bigB - bigB * bigB;
    double c0 = -bigA * bigB;
    double pp = c1 - 1.0 / 3.0;
    double qq = -2.0 / 27.0 + c1 / 3.0 + c0;
    double disc = 0.25 * qq * qq + pp * pp * pp / 27.0;
    double zr;
    if (disc > 0.0) {
      double sd = TornadoMath.sqrt(disc);
      zr = polish(cbrt(-0.5 * qq + sd) + cbrt(-0.5 * qq - sd) + 1.0 / 3.0, c1, c0);
    } else {
      double r = TornadoMath.sqrt(-pp / 3.0);
      double arg = -qq / (2.0 * r * r * r);
      arg = (arg > 1.0) ? 1.0 : ((arg < -1.0) ? -1.0 : arg);
      double phi = TornadoMath.acos(arg);
      double twoPi = 6.283185307179586;
      double z0 = polish(2.0 * r * TornadoMath.cos(phi / 3.0) + 1.0 / 3.0, c1, c0);
      if (liquid) {
        double z1 = polish(2.0 * r * TornadoMath.cos((phi + twoPi) / 3.0) + 1.0 / 3.0, c1, c0);
        double z2 =
            polish(2.0 * r * TornadoMath.cos((phi + 2.0 * twoPi) / 3.0) + 1.0 / 3.0, c1, c0);
        zr = (z1 > bigB) ? z1 : ((z2 > bigB) ? z2 : z0);
      } else {
        zr = z0;
      }
    }
    double lnZb = TornadoMath.log(zr - bigB);
    double lnAtt = TornadoMath.log(1.0 + bigB / zr);
    double aOverB = bigA / bigB;
    for (int i = 0; i < n; i++) {
      double biOverB = model.get(n + i) / bmix;
      double coef = aOverB * (2.0 * dst.get(dO + i * st) / amix - biOverB);
      dst.set(dO + i * st, biOverB * (zr - 1.0) - lnZb - coef * lnAtt);
    }
  }

  private static double polish(double zz, double c1, double c0) {
    double z = zz;
    for (int it = 0; it < 4; it++) {
      double f = ((z - 1.0) * z + c1) * z + c0;
      double df = (3.0 * z - 2.0) * z + c1;
      if (df != 0.0) {
        z = z - f / df;
      }
    }
    return z;
  }

  private static double cbrt(double v) {
    double r = TornadoMath.pow(TornadoMath.abs(v), 1.0 / 3.0);
    return (v < 0.0) ? -r : r;
  }

  /** Rachford-Rice function at beta. */
  private static double rr(
      int n, int st, DoubleArray z, int zo, DoubleArray w, int ko, double beta) {
    double f = 0.0;
    for (int i = 0; i < n; i++) {
      double km1 = w.get(ko + i * st) - 1.0;
      f += z.get(zo + i * st) * km1 / (1.0 + beta * km1);
    }
    return f;
  }

  /** Flashes one state. */
  public static void flash(
      int s,
      int n,
      int maxIterations,
      int states,
      boolean componentMajor,
      DoubleArray model,
      DoubleArray crit,
      DoubleArray tp,
      DoubleArray z,
      DoubleArray out,
      DoubleArray work) {
    double t = tp.get(2 * s);
    double p = tp.get(2 * s + 1);
    // State-major: each state's values are contiguous (best for CPU caches). Component-major:
    // value k of every state is contiguous, so neighbouring GPU threads read neighbouring doubles.
    int st = componentMajor ? states : 1;
    int zo = componentMajor ? s : s * n;
    int wo = componentMajor ? s : s * WORK_PER_COMPONENT * n;
    int oo = componentMajor ? s : s * (3 + 2 * n);
    int ko = wo;
    int xo = wo + n * st;
    int yo = wo + 2 * n * st;
    int lio = wo + 3 * n * st;
    int vo = wo + 4 * n * st;
    int so = wo + 5 * n * st;

    for (int i = 0; i < n; i++) {
      double m = model.get(2 * n + i);
      double tc = model.get(3 * n + i);
      double f = 1.0 + m * (1.0 - TornadoMath.sqrt(t / tc));
      work.set(so + i * st, TornadoMath.sqrt(model.get(i)) * f);
      double pc = crit.get(i);
      double omega = crit.get(n + i);
      work.set(ko + i * st, pc / p * TornadoMath.exp(5.373 * (1.0 + omega) * (1.0 - tc / t)));
    }

    double beta = 0.5;
    double status = -1.0;
    int it = 0;
    while (it < maxIterations && status < 0.0) {
      it++;
      double f0 = rr(n, st, z, zo, work, ko, 0.0);
      double f1 = rr(n, st, z, zo, work, ko, 1.0);
      if (f0 <= 0.0 || f1 >= 0.0) {
        beta = (f0 <= 0.0) ? 0.0 : 1.0;
        status = 0.0;
      } else {
        // Newton on beta, kept inside the bracket [lo, hi] by bisection.
        double lo = 0.0;
        double hi = 1.0;
        for (int k = 0; k < 60; k++) {
          double f = 0.0;
          double df = 0.0;
          for (int i = 0; i < n; i++) {
            double km1 = work.get(ko + i * st) - 1.0;
            double d = 1.0 + beta * km1;
            f += z.get(zo + i * st) * km1 / d;
            df -= z.get(zo + i * st) * km1 * km1 / (d * d);
          }
          if (f > 0.0) {
            lo = beta;
          } else {
            hi = beta;
          }
          double next = beta - f / df;
          if (next <= lo || next >= hi) {
            next = 0.5 * (lo + hi);
          }
          double step = TornadoMath.abs(next - beta);
          beta = next;
          if (step < 1e-15) {
            k = 60;
          }
        }
        for (int i = 0; i < n; i++) {
          double xi = z.get(zo + i * st) / (1.0 + beta * (work.get(ko + i * st) - 1.0));
          work.set(xo + i * st, xi);
          work.set(yo + i * st, work.get(ko + i * st) * xi);
        }
        lnphi(n, st, model, t, p, work, xo, true, work, lio, work, so);
        lnphi(n, st, model, t, p, work, yo, false, work, vo, work, so);
        double change = 0.0;
        for (int i = 0; i < n; i++) {
          double lnK = work.get(lio + i * st) - work.get(vo + i * st);
          double d = TornadoMath.abs(lnK - TornadoMath.log(work.get(ko + i * st)));
          change = (d > change) ? d : change;
          work.set(ko + i * st, TornadoMath.exp(lnK));
        }
        if (change < 1e-10) {
          status = 1.0;
        }
      }
    }
    out.set(oo, beta);
    out.set(oo + st, it);
    out.set(oo + 2 * st, status);
    for (int i = 0; i < n; i++) {
      out.set(oo + (3 + i) * st, work.get(xo + i * st));
      out.set(oo + (3 + n + i) * st, work.get(yo + i * st));
    }
  }

  /** Index of value k of state s in a per-state array of the given width. */
  public static int index(int s, int k, int width, int states, boolean componentMajor) {
    return componentMajor ? k * states + s : s * width + k;
  }
}
