package io.github.mikepapadim.neqsim.tornadovm;

import neqsim.thermo.component.ComponentEos;
import neqsim.thermo.phase.PhaseEos;
import neqsim.thermo.phase.PhaseInterface;
import neqsim.thermo.phase.PhaseType;
import neqsim.thermo.system.SystemInterface;
import neqsim.thermo.system.SystemSrkEos;
import neqsim.thermodynamicoperations.ThermodynamicOperations;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;

/**
 * NeqSim side of the prototype: builds the SRK system, packs its parameters for the kernels, and
 * computes reference log fugacity coefficients with NeqSim's own object-based code.
 */
public final class NeqSimSrk {

  /** Component order used throughout (the synthetic gas from equinor/neqsim#4587). */
  public static final String[] COMPONENTS = {
    "nitrogen", "CO2", "methane", "ethane", "propane", "i-butane",
    "n-butane", "i-pentane", "n-pentane", "n-hexane", "n-heptane", "n-octane"
  };

  /** Feed mole fractions from equinor/neqsim#4587. */
  public static final double[] FEED = {
    0.01, 0.02, 0.65, 0.10, 0.06, 0.02, 0.03, 0.015, 0.015, 0.03, 0.03, 0.02
  };

  private final SystemInterface system;

  /** A two-phase SRK system with the classic mixing rule. */
  public NeqSimSrk() {
    system = new SystemSrkEos(300.0, 50.0);
    for (int i = 0; i < COMPONENTS.length; i++) {
      system.addComponent(COMPONENTS[i], FEED[i]);
    }
    system.setMixingRule("classic");
    system.init(0);
  }

  public int components() {
    return COMPONENTS.length;
  }

  /** Packs a, b, m, Tc and kij as read from NeqSim (see {@link SrkBatchKernels}). */
  public DoubleArray packModel() {
    int n = components();
    PhaseInterface phase = system.getPhase(0);
    DoubleArray model = new DoubleArray(SrkBatchKernels.modelSize(n));
    for (int i = 0; i < n; i++) {
      ComponentEos c = (ComponentEos) phase.getComponent(i);
      model.set(i, c.geta());
      model.set(n + i, c.getb());
      model.set(2 * n + i, c.getAttractiveParameter().getm());
      model.set(3 * n + i, c.getTC());
    }
    PhaseEos eos = (PhaseEos) phase;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        model.set(4 * n + i * n + j, eos.getMixingRule().getBinaryInteractionParameter(i, j));
      }
    }
    return model;
  }

  /**
   * NeqSim's log fugacity coefficients at T [K], P [bara] and composition x, for the liquid root
   * (out[0..n)) and the gas root (out[n..2n)), plus Z of each in out[2n], out[2n+1].
   */
  public void reference(double t, double p, double[] x, double[] out) {
    int n = components();
    system.setTemperature(t);
    system.setPressure(p);
    system.setMolarComposition(x);
    system.init(0);
    system.setPhaseType(0, PhaseType.LIQUID);
    system.setPhaseType(1, PhaseType.GAS);
    system.init(1);
    for (int i = 0; i < n; i++) {
      out[i] = system.getPhase(0).getComponent(i).getLogFugacityCoefficient();
      out[n + i] = system.getPhase(1).getComponent(i).getLogFugacityCoefficient();
    }
    out[2 * n] = system.getPhase(0).getZ();
    out[2 * n + 1] = system.getPhase(1).getZ();
  }

  /** Packs Pc [bar] and the acentric factor (see {@link SrkFlashKernels}). */
  public DoubleArray packCrit() {
    int n = components();
    DoubleArray crit = new DoubleArray(2 * n);
    for (int i = 0; i < n; i++) {
      crit.set(i, system.getPhase(0).getComponent(i).getPC());
      crit.set(n + i, system.getPhase(0).getComponent(i).getAcentricFactor());
    }
    return crit;
  }

  /**
   * NeqSim TPflash at T [K], P [bara] and feed z. Writes {@code beta, phases, x[n], y[n]} to out,
   * with beta the vapour mole fraction; returns the number of phases.
   */
  public int referenceFlash(double t, double p, double[] z, double[] out) {
    int n = components();
    system.setTemperature(t);
    system.setPressure(p);
    system.setMolarComposition(z);
    system.init(0);
    new ThermodynamicOperations(system).TPflash();
    int phases = system.getNumberOfPhases();
    out[1] = phases;
    if (phases == 2) {
      int gas = system.getPhase(0).getType() == PhaseType.GAS ? 0 : 1;
      int liq = 1 - gas;
      out[0] = system.getBeta(gas);
      for (int i = 0; i < n; i++) {
        out[2 + i] = system.getPhase(liq).getComponent(i).getx();
        out[2 + n + i] = system.getPhase(gas).getComponent(i).getx();
      }
    } else {
      out[0] = system.getPhase(0).getType() == PhaseType.GAS ? 1.0 : 0.0;
    }
    return phases;
  }
}
