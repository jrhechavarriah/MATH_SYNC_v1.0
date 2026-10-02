from __future__ import annotations
import json,time
from pathlib import Path
import pandas as pd
from mathsync.g12_convergence_scaling import run_generic_trial
ROOT=Path(__file__).resolve().parents[2]
SPEC=ROOT/'config'/'G12D_CONVERGENCE_SCALING_SPEC.json'
OUT=ROOT/'results'/'g12'/'g12d'; REPORT=ROOT/'reports'/'g12'

def main():
    cfg=json.loads(SPEC.read_text(encoding='utf-8-sig'))
    if cfg['status']!='FROZEN_BEFORE_G12D_OUTCOME_INSPECTION': raise RuntimeError('G12-D specification is not frozen.')
    OUT.mkdir(parents=True,exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True)
    d2_rows=[]; d3_rows=[]; start_total=time.perf_counter()
    d2=cfg['D2_tolerance_sensitivity']; d2_seed_base=int(cfg['seed_policy']['D2_seed_base'])
    for rep in range(int(d2['replicates'])):
        seed=d2_seed_base+rep
        for tol in d2['tolerances']:
            row=run_generic_trial(seed=seed,n_streams=int(d2['stream_count']),n_anchors=int(d2['anchor_count']),tolerance=float(tol))
            row.update({'experiment':'D2_tolerance','replicate':rep,'seed':seed,'tolerance':float(tol),
                        'n_streams':int(d2['stream_count']),'n_anchors':int(d2['anchor_count'])})
            d2_rows.append(row)
    d3=cfg['D3_scaling']; d3_seed_base=int(cfg['seed_policy']['D3_seed_base'])
    total_d3=len(d3['stream_counts'])*len(d3['anchor_counts'])*int(d3['replicates']); done=0
    for rep in range(int(d3['replicates'])):
        seed=d3_seed_base+rep
        for streams in d3['stream_counts']:
            for anchors in d3['anchor_counts']:
                row=run_generic_trial(seed=seed,n_streams=int(streams),n_anchors=int(anchors),tolerance=float(d3['tolerance']))
                row.update({'experiment':'D3_scaling','replicate':rep,'seed':seed,'tolerance':float(d3['tolerance']),
                            'n_streams':int(streams),'n_anchors':int(anchors)})
                d3_rows.append(row); done+=1
                if done%100==0 or done==total_d3:
                    print(f'G12-D D3 progress: {done}/{total_d3} ({100.0*done/total_d3:.1f}%)',flush=True)
    d2_df=pd.DataFrame(d2_rows); d3_df=pd.DataFrame(d3_rows)
    if len(d2_df)!=int(d2['expected_trials']): raise RuntimeError('Unexpected D2 row count.')
    if len(d3_df)!=int(d3['expected_trials']): raise RuntimeError('Unexpected D3 row count.')
    d2_path=OUT/'G12D_D2_tolerance_results.csv'; d3_path=OUT/'G12D_D3_scaling_results.csv'
    d2_df.to_csv(d2_path,index=False,float_format='%.15g'); d3_df.to_csv(d3_path,index=False,float_format='%.15g')
    elapsed=time.perf_counter()-start_total
    audit={'gate':'G12D_EXECUTION_AUDIT','D2_rows':int(len(d2_df)),'D3_rows':int(len(d3_df)),
           'total_rows':int(len(d2_df)+len(d3_df)),'expected_total_rows':int(cfg['trial_totals']['total']),
           'D2_converged':int(d2_df['converged'].sum()),'D2_failures':int(d2_df['solver_failure'].sum()),
           'D3_converged':int(d3_df['converged'].sum()),'D3_failures':int(d3_df['solver_failure'].sum()),
           'elapsed_seconds':float(elapsed),'public_release_modified':False}
    (REPORT/'G12D_execution_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))
if __name__=='__main__': main()
