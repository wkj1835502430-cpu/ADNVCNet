"""Check the supplied official DINOv2 source and optional feature API.

File fingerprints identify the actual source snapshot used for validation.
They are not presented as an upstream git commit when that commit is unknown.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--fingerprint', type=Path, default=Path(__file__).resolve().parents[1]/'configs/dinov2_source_fingerprint.json')
    parser.add_argument('--strict', action='store_true', help='Fail if source files differ from the validated snapshot')
    parser.add_argument('--feature-check', action='store_true', help='Load supplied weights and test the last-four-layer API without dataset images')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    expected = json.loads(args.fingerprint.read_text('utf-8'))
    missing, different = [], []
    for row in expected['files']:
        path = args.repo/row['path']
        if not path.is_file():
            missing.append(row['path'])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            different.append(row['path'])
    report = {'upstream_repository':expected['upstream_repository'], 'upstream_git_commit':expected['upstream_git_commit'],
              'fingerprint_match':not missing and not different, 'missing_files':missing, 'different_files':different,
              'feature_check_run':False}
    if args.strict and (missing or different):
        print(json.dumps(report,indent=2))
        raise SystemExit('DINOv2 source differs from the recorded snapshot; inspect the source version before reproducing numerical results')
    if args.feature_check:
        if not args.weights:
            parser.error('--feature-check requires --weights')
        import numpy as np
        import torch
        from models.public_inference import _LocalDino
        from models.dinov2_features import _extract_multilayer_patch_and_global
        model = _LocalDino(args.repo,args.weights,args.device)
        patch,grid,cls,source = _extract_multilayer_patch_and_global(model,np.zeros((512,512,3),np.float32))
        if tuple(patch.shape)!=(1369,384) or tuple(cls.shape)!=(384,) or tuple(grid)!=(37,37):
            raise RuntimeError('DINOv2 representation shape mismatch')
        if source!='dinov2_last4_normalized_equal_mean_cls' or not bool(torch.isfinite(patch).all()) or not bool(torch.isfinite(cls).all()):
            raise RuntimeError('DINOv2 representation identity or finite-value check failed')
        report.update(feature_check_run=True,patch_shape=list(patch.shape),cls_shape=list(cls.shape),grid=list(grid),
                      representation=source,feature_api_compatible=True,
                      full_numerical_equivalence_claimed=False)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
