import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SPEC=ROOT/'config'/'G12D_CONVERGENCE_SCALING_SPEC.json'
def test_g12d_frozen_spec():
    cfg=json.loads(SPEC.read_text(encoding='utf-8-sig'))
    assert cfg['status']=='FROZEN_BEFORE_G12D_OUTCOME_INSPECTION'
    assert cfg['D2_tolerance_sensitivity']['tolerances']==[1e-4,1e-6,1e-8]
    assert cfg['D3_scaling']['stream_counts']==[5,10,20,40]
    assert cfg['D3_scaling']['anchor_counts']==[8,16,32,64,128]
    assert cfg['trial_totals']['total']==1150
    assert cfg['public_release_modified'] is False
    assert cfg['legacy_core_modified'] is False
