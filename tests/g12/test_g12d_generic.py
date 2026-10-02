import numpy as np
from mathsync.g12_convergence_scaling import build_design_matrix_generic,run_generic_trial,simulate_generic_problem

def test_generic_problem_dimensions():
    tau,obs,_,_=simulate_generic_problem(seed=25260821,n_streams=10,n_anchors=16)
    A,b=build_design_matrix_generic(obs,len(tau),10)
    assert A.shape[0]==len(obs); assert A.shape[1]==16+2*9; assert len(b)==len(obs)

def test_generic_trial_is_deterministic_except_runtime():
    a=run_generic_trial(seed=25260822,n_streams=5,n_anchors=16,tolerance=1e-10)
    b=run_generic_trial(seed=25260822,n_streams=5,n_anchors=16,tolerance=1e-10)
    keys=['n_observations','n_parameters','rank_A','identifiable','iterations','converged','status',
          'final_relative_parameter_change','final_huber_loss','rmse_ms','mae_ms']
    for key in keys:
        if isinstance(a[key],float): assert np.isclose(a[key],b[key],equal_nan=True)
        else: assert a[key]==b[key]

def test_stream_scaling_parameter_count_formula():
    for streams in [5,10,20,40]:
        anchors=8; _,obs,_,_=simulate_generic_problem(seed=25260823,n_streams=streams,n_anchors=anchors)
        A,_=build_design_matrix_generic(obs,anchors,streams)
        assert A.shape[1]==anchors+2*(streams-1)
