package io.github.mikepapadim.neqsim.tornadovm;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Batched SRK (classical van der Waals mixing) compressibility and log fugacity coefficients, in
 * FP64, for many independent states at once.
 *
 * <p>The per-state body {@link #state} is plain Java: the CPU reference runs it directly, and the
 * TornadoVM kernel {@link #srkKernel} calls it with one GPU thread per state.
 *
 * <p>Layouts, with {@code n} components:
 *
 * <ul>
 *   <li>{@code model}: {@code a[n], b[n], m[n], Tc[n], kij[n*n]} (NeqSim units: bar, K, R in
 *       J/mol/K)
 *   <li>{@code tp}: {@code T, P} per state
 *   <li>{@code x}: {@code n} mole fractions per state (normalised by the caller)
 *   <li>{@code out}: {@code Zliq, Zvap, lnphiLiq[n], lnphiVap[n]} per state
 *   <li>{@code work}: {@code n} scratch doubles per state (sqrt of a_i(T), then sum_j x_j a_ij)
 *   <li>{@code dims}: {@code n, states}
 * </ul>
 */
public final class SrkBatchKernels {

  /** Gas constant used by NeqSim. */
  public static final double R = 8.3144621;

  private SrkBatchKernels() {}

  /** Doubles per state in {@code out}. */
  public static int outStride(int n) {
    return 2 + 2 * n;
  }

  /** Doubles in the model buffer. */
  public static int modelSize(int n) {
    return 4 * n + n * n;
  }

  /** TornadoVM entry point: one thread per state. */
  public static void srkKernel(
      KernelContext context,
      DoubleArray model,
      DoubleArray tp,
      DoubleArray x,
      DoubleArray out,
      DoubleArray work,
      IntArray dims) {
    int s = context.globalIdx;
    if (s < dims.get(1)) {
      state(s, dims.get(0), model, tp, x, out, work);
    }
  }

  /** Newton refinement of a root of z^3 - z^2 + c1 z + c0. */
  private static double polish(double z, double c1, double c0) {
    for (int it = 0; it < 4; it++) {
      double f = ((z - 1.0) * z + c1) * z + c0;
      double df = (3.0 * z - 2.0) * z + c1;
      if (df != 0.0) {
        z = z - f / df;
      }
    }
    return z;
  }

  /** Real cube root, keeping the sign. */
  private static double cbrt(double v) {
    double r = TornadoMath.pow(TornadoMath.abs(v), 1.0 / 3.0);
    return (v < 0.0) ? -r : r;
  }

  /** Computes one state: both cubic roots and the log fugacity coefficients at each. */
  public static void state(
      int s,
      int n,
      DoubleArray model,
      DoubleArray tp,
      DoubleArray x,
      DoubleArray out,
      DoubleArray work) {
    double t = tp.get(2 * s);
    double p = tp.get(2 * s + 1);
    int xo = s * n;
    int oo = s * (2 + 2 * n);

    // work[i] = sqrt(a_i(T)) with the Soave alpha function.
    for (int i = 0; i < n; i++) {
      double m = model.get(2 * n + i);
      double f = 1.0 + m * (1.0 - TornadoMath.sqrt(t / model.get(3 * n + i)));
      work.set(xo + i, TornadoMath.sqrt(model.get(i)) * f);
    }
    // amix = sum_i x_i S_i with S_i = sum_j x_j a_ij; S_i is kept in out (liquid slots) for now.
    double amix = 0.0;
    double bmix = 0.0;
    for (int i = 0; i < n; i++) {
      double si = work.get(xo + i);
      double sum = 0.0;
      for (int j = 0; j < n; j++) {
        sum += x.get(xo + j) * si * work.get(xo + j) * (1.0 - model.get(4 * n + i * n + j));
      }
      out.set(oo + 2 + i, sum);
      double xi = x.get(xo + i);
      amix += xi * sum;
      bmix += xi * model.get(n + i);
    }
    double rt = R * t;
    double bigA = amix * p / (rt * rt);
    double bigB = bmix * p / rt;

    // z^3 - z^2 + (A - B - B^2) z - A B = 0, solved with Cardano and polished with Newton.
    double c1 = bigA - bigB - bigB * bigB;
    double c0 = -bigA * bigB;
    double pp = c1 - 1.0 / 3.0;
    double qq = -2.0 / 27.0 + c1 / 3.0 + c0;
    double disc = 0.25 * qq * qq + pp * pp * pp / 27.0;
    double zLiq;
    double zVap;
    if (disc > 0.0) {
      double sd = TornadoMath.sqrt(disc);
      double z = cbrt(-0.5 * qq + sd) + cbrt(-0.5 * qq - sd) + 1.0 / 3.0;
      z = polish(z, c1, c0);
      zLiq = z;
      zVap = z;
    } else {
      double r = TornadoMath.sqrt(-pp / 3.0);
      double arg = -qq / (2.0 * r * r * r);
      arg = (arg > 1.0) ? 1.0 : ((arg < -1.0) ? -1.0 : arg);
      double phi = TornadoMath.acos(arg);
      double twoPi = 6.283185307179586;
      double z0 = polish(2.0 * r * TornadoMath.cos(phi / 3.0) + 1.0 / 3.0, c1, c0);
      double z1 = polish(2.0 * r * TornadoMath.cos((phi + twoPi) / 3.0) + 1.0 / 3.0, c1, c0);
      double z2 = polish(2.0 * r * TornadoMath.cos((phi + 2.0 * twoPi) / 3.0) + 1.0 / 3.0, c1, c0);
      zVap = z0;
      // Smallest root above B is the liquid root; z1 is the smallest of the three.
      zLiq = (z1 > bigB) ? z1 : ((z2 > bigB) ? z2 : z0);
    }
    out.set(oo, zLiq);
    out.set(oo + 1, zVap);

    double lnZbL = TornadoMath.log(zLiq - bigB);
    double lnZbV = TornadoMath.log(zVap - bigB);
    double lnAttL = TornadoMath.log(1.0 + bigB / zLiq);
    double lnAttV = TornadoMath.log(1.0 + bigB / zVap);
    double aOverB = bigA / bigB;
    for (int i = 0; i < n; i++) {
      double biOverB = model.get(n + i) / bmix;
      double coef = aOverB * (2.0 * out.get(oo + 2 + i) / amix - biOverB);
      out.set(oo + 2 + i, biOverB * (zLiq - 1.0) - lnZbL - coef * lnAttL);
      out.set(oo + 2 + n + i, biOverB * (zVap - 1.0) - lnZbV - coef * lnAttV);
    }
  }
}
