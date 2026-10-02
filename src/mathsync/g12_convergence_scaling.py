from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import time
import numpy as np
import pandas as pd

HUBER_C = 1.345
MIN_SCALE = 1e-6
MAX_ITER = 30

@dataclass
class HuberDiagnostics:
    z: np.ndarray
    weights: np.ndarray
    iterations: int
    converged: bool
    status: str
    final_relative_parameter_change: float
    final_scale: float
    final_huber_loss: float
    runtime_s: float

def build_design_matrix_generic(observations: pd.DataFrame, n_events: int, n_sensors: int):
    n_parameters = n_events + 2 * (n_sensors - 1)
    A = np.zeros((len(observations), n_parameters), dtype=float)
    b = np.zeros(len(observations), dtype=float)
    for row_idx, row in enumerate(observations.itertuples(index=False)):
        event_idx = int(row.event); sensor_idx = int(row.sensor_idx); t_obs = float(row.t_obs)
        if sensor_idx == 0:
            A[row_idx, event_idx] = 1.0; b[row_idx] = t_obs
        else:
            A[row_idx, event_idx] = 1.0
            offset = n_events + 2 * (sensor_idx - 1)
            A[row_idx, offset] = -t_obs; A[row_idx, offset + 1] = -1.0
    return A, b

def _scale_and_weights(residual, huber_c=HUBER_C):
    median_r = np.median(residual)
    mad = np.median(np.abs(residual - median_r))
    scale = max(1.4826 * mad, MIN_SCALE)
    standardized = np.abs(residual) / (huber_c * scale)
    weights = np.ones_like(standardized)
    mask = standardized > 1.0
    weights[mask] = 1.0 / standardized[mask]
    return scale, weights

def _huber_rho_sum(residual, scale, huber_c=HUBER_C):
    u = residual / scale; a = np.abs(u)
    rho = np.where(a <= huber_c, 0.5 * u**2, huber_c * a - 0.5 * huber_c**2)
    return float(np.sum(rho))

def solve_huber_diagnostic(observations, n_events, n_sensors, *, tolerance, max_iter=MAX_ITER, huber_c=HUBER_C):
    A, b = build_design_matrix_generic(observations, n_events, n_sensors)
    if np.linalg.matrix_rank(A) < A.shape[1]:
        return HuberDiagnostics(np.full(A.shape[1], np.nan), np.full(A.shape[0], np.nan), 0, False,
            'rank_deficient_design', np.nan, np.nan, np.nan, 0.0)
    start=time.perf_counter(); z,*_=np.linalg.lstsq(A,b,rcond=None)
    converged=False; final_rel=np.nan; iterations=0
    for iteration in range(1,max_iter+1):
        residual=A@z-b; scale,weights=_scale_and_weights(residual,huber_c)
        sw=np.sqrt(weights); z_new,*_=np.linalg.lstsq(A*sw[:,None],b*sw,rcond=None)
        final_rel=float(np.linalg.norm(z_new-z)/(1.0+np.linalg.norm(z)))
        z=z_new; iterations=iteration
        if final_rel <= tolerance:
            converged=True; break
    runtime_s=time.perf_counter()-start
    residual=A@z-b; scale,weights=_scale_and_weights(residual,huber_c)
    objective=_huber_rho_sum(residual,scale,huber_c)
    return HuberDiagnostics(z,weights,iterations,converged,'ok' if converged else 'max_iterations_reached',
        final_rel,float(scale),objective,float(runtime_s))

def structural_diagnostics(A, weights=None):
    rank=int(np.linalg.matrix_rank(A)); identifiable=rank==A.shape[1]
    sigma_min=np.nan; condition=np.inf
    if identifiable and weights is not None:
        sw=np.sqrt(np.asarray(weights,float)); s=np.linalg.svd(A*sw[:,None],compute_uv=False)
        sigma_min=float(s[-1]); condition=float(s[0]/s[-1])
    return {'n_observations':int(A.shape[0]),'n_parameters':int(A.shape[1]),'rank_A':rank,
        'identifiable':bool(identifiable),'sigma_min_weighted_A':sigma_min,'condition_number_weighted_A':condition}

def simulate_generic_problem(*, seed, n_streams, n_anchors, duration=120.0, drift_ppm=100.0, offset_s=0.25,
                             jitter_ms=1.0, missing_prob=0.10, outlier_frac=0.05, outlier_ms=60.0):
    rng=np.random.default_rng(seed)
    base=np.linspace(3.0,duration-3.0,n_anchors)
    tau=np.sort(np.clip(base+rng.uniform(-0.8,0.8,n_anchors),1.0,duration-1.0))
    rows=[]; true_clock: Dict[str,Tuple[float,float]]={}
    sensor_names=['REF']+[f'S{i:03d}' for i in range(1,n_streams)]
    for sensor_idx,name in enumerate(sensor_names):
        if sensor_idx==0: a_i,b_i=1.0,0.0
        else:
            a_i=1.0+rng.uniform(-drift_ppm,drift_ppm)*1e-6; b_i=rng.uniform(-offset_s,offset_s)
        true_clock[name]=(a_i,b_i)
        observed=a_i*tau+b_i+rng.normal(0.0,jitter_ms/1000.0,n_anchors)
        if sensor_idx==0:
            keep=np.ones(n_anchors,dtype=bool); is_outlier=np.zeros(n_anchors,dtype=bool)
        else:
            keep=rng.random(n_anchors)>=missing_prob
            if keep.sum()<min(5,n_anchors):
                forced=rng.choice(n_anchors,min(5,n_anchors),replace=False); keep[forced]=True
            is_outlier=np.zeros(n_anchors,dtype=bool); candidates=np.where(keep)[0]
            n_out=int(round(outlier_frac*len(candidates)))
            if n_out>0:
                idx=rng.choice(candidates,n_out,replace=False); is_outlier[idx]=True
                signs=rng.choice([-1.0,1.0],size=n_out)
                mags=rng.uniform(0.5*outlier_ms,1.5*outlier_ms,size=n_out)/1000.0
                observed[idx]+=signs*mags
        for event_idx in np.where(keep)[0]:
            rows.append((sensor_idx,name,int(event_idx),float(tau[event_idx]),float(observed[event_idx]),bool(is_outlier[event_idx])))
    obs=pd.DataFrame(rows,columns=['sensor_idx','sensor','event','tau_true','t_obs','is_outlier'])
    return tau,obs,true_clock,sensor_names

def extract_params_generic(z,n_events,sensor_names):
    params={sensor_names[0]:(1.0,0.0)}
    for sensor_idx in range(1,len(sensor_names)):
        off=n_events+2*(sensor_idx-1); params[sensor_names[sensor_idx]]=(float(z[off]),float(z[off+1]))
    return params

def evaluate_generic(estimated_params,true_clock,sensor_names,*,seed,duration,jitter_ms,n_eval_samples=300):
    rng=np.random.default_rng(seed); errors_ms=[]
    for name in sensor_names[1:]:
        tau_true=rng.uniform(0.0,duration,n_eval_samples); a_i,b_i=true_clock[name]
        local=a_i*tau_true+b_i+rng.normal(0.0,jitter_ms/1000.0,n_eval_samples)
        gamma,kappa=estimated_params[name]; tau_hat=gamma*local+kappa
        errors_ms.extend((tau_hat-tau_true)*1000.0)
    e=np.asarray(errors_ms,float)
    return {'rmse_ms':float(np.sqrt(np.mean(e**2))),'mae_ms':float(np.mean(np.abs(e)))}

def run_generic_trial(*, seed, n_streams, n_anchors, tolerance):
    tau,obs,true_clock,names=simulate_generic_problem(seed=seed,n_streams=n_streams,n_anchors=n_anchors)
    n_events=len(tau); fit=solve_huber_diagnostic(obs,n_events,n_streams,tolerance=tolerance)
    A,_=build_design_matrix_generic(obs,n_events,n_streams)
    diag=structural_diagnostics(A,fit.weights if fit.converged else None)
    row={**diag,'iterations':fit.iterations,'converged':fit.converged,'status':fit.status,
         'solver_failure':not fit.converged,'final_relative_parameter_change':fit.final_relative_parameter_change,
         'final_huber_loss':fit.final_huber_loss,'runtime_s':fit.runtime_s,'rmse_ms':np.nan,'mae_ms':np.nan}
    if fit.converged:
        params=extract_params_generic(fit.z,n_events,names)
        row.update(evaluate_generic(params,true_clock,names,seed=seed+55555,duration=120.0,jitter_ms=1.0,n_eval_samples=300))
    return row
