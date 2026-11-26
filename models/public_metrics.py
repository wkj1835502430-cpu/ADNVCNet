"""Memory-bounded exact ranking and the current 200-threshold AUPRO protocol."""
from pathlib import Path
import gc
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score


def disk_pixel_auc(score_maps,gts,path):
    path=Path(path);assert not path.exists()
    assert len(score_maps)==len(gts)>0
    negative_count=sum(int(np.count_nonzero(np.asarray(g)<=.5)) for g in gts)
    positive_count=sum(int(np.count_nonzero(np.asarray(g)>.5)) for g in gts)
    assert negative_count>0 and positive_count>0
    assert 2*negative_count*positive_count<2**63
    negatives=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=(negative_count,))
    offset=0
    for scores,gt in zip(score_maps,gts):
        scores=np.asarray(scores,np.float64);mask=np.asarray(gt)>.5
        assert scores.shape==mask.shape and np.isfinite(scores).all()
        values=scores[~mask];negatives[offset:offset+len(values)]=values;offset+=len(values)
    assert offset==negative_count
    del values,scores,mask;gc.collect()
    negatives.flush();negatives.sort(kind='quicksort');negatives.flush()
    twice=0;counted=0
    for scores,gt in zip(score_maps,gts):
        positives=np.asarray(scores,np.float64)[np.asarray(gt)>.5]
        left=np.searchsorted(negatives,positives,side='left')
        right=np.searchsorted(negatives,positives,side='right')
        twice+=int(np.sum(left+right,dtype=np.int64));counted+=len(positives)
    assert counted==positive_count
    value=twice/(2*negative_count*positive_count)
    assert 0<=value<=1
    negatives._mmap.close();del negatives;gc.collect()
    return value,dict(positive_pixels=positive_count,negative_pixels=negative_count,
        twice_strict_win_plus_tie_count=twice,denominator=2*negative_count*positive_count,
        exact_float64_scores=True,score_quantization=False,pixel_sampling=False,
        original_metric_definition='P(positive score > negative score) + 0.5*P(tie)')


def threshold_aupro(score_maps,gts,max_fpr=.3,thresholds=200):
    """Same 200 thresholds, connectivity, ties, integration as adnvc_metrics.

    Sorted per-image/component scores replace repeated threshold scans. Integer
    counts, component order, averaging order, duplicate-FPR handling and endpoint
    extension are retained. This changes computational cost, not the metric.
    """
    assert len(score_maps)==len(gts) and score_maps
    hi=max(float(np.max(a)) for a in score_maps)
    lo=min(float(np.min(a)) for a in score_maps)
    normal_pixels=sum(int(np.count_nonzero(np.asarray(g)<=.5)) for g in gts)
    if not normal_pixels:return float('nan')
    ts=np.linspace(hi+1e-7,lo-1e-7,int(thresholds))
    fp=np.zeros(len(ts),np.int64);overlaps=[]
    for a,g in zip(score_maps,gts):
        a=np.asarray(a,np.float64);mask=np.asarray(g)>.5
        normal=np.sort(a[~mask])
        fp+=normal.size-np.searchsorted(normal,ts,side='left')
        count,labels=cv2.connectedComponents(mask.astype(np.uint8),connectivity=8)
        for i in range(1,count):
            values=np.sort(a[labels==i])
            overlaps.append((values.size-np.searchsorted(values,ts,side='left'))/float(values.size))
    if not overlaps:return float('nan')
    if hi<=lo+1e-12:return 0.
    points=[(0.,0.)]
    for j in range(len(ts)):
        fpr=int(fp[j])/float(normal_pixels)
        if fpr<=max_fpr+1e-12:
            points.append((fpr,float(np.mean([x[j] for x in overlaps]))))
    points.sort(key=lambda p:p[0]);xs=[];ys=[]
    for x,y in points:
        if xs and abs(x-xs[-1])<1e-12:ys[-1]=max(ys[-1],y)
        else:xs.append(x);ys.append(y)
    if xs[-1]<max_fpr:xs.append(max_fpr);ys.append(ys[-1])
    xs=np.asarray(xs,np.float64);ys=np.asarray(ys,np.float64);keep=xs<=max_fpr+1e-12
    return float(np.trapz(ys[keep],xs[keep]))/float(max_fpr)
