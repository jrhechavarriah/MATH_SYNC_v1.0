import numpy as np
from mathsync.g12_convergence_scaling import solve_huber_diagnostic
from mathsync.legacy_v1_synthetic import SCENARIOS,SENSORS,simulate_anchor_observations,solve_joint_huber

def test_diagnostic_solver_matches_legacy_at_1e10_tolerance():
    cfg=dict(SCENARIOS['nominal']); seed=24260821
    tau,observations,_=simulate_anchor_observations(seed=seed,**cfg)
    z_legacy=solve_joint_huber(observations,len(tau),len(SENSORS),tolerance=1e-10)
    fit=solve_huber_diagnostic(observations,len(tau),len(SENSORS),tolerance=1e-10)
    assert np.allclose(fit.z,z_legacy,rtol=0.0,atol=1e-12)
