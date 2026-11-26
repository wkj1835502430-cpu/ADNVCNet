"""Frozen CPU score operator; normal-derived monotone evidence, not a p-value."""
import numpy as np
import cv2

COMPONENTS=('cnn','global','spatial_r0','spatial_r1')
RAW_KEYS={'global':'raw_global','spatial_r0':'raw_spatial_r0','spatial_r1':'raw_spatial_r1'}

def sample(a):
    a=np.asarray(a,dtype=np.float64).reshape(-1)
    assert a.size and np.isfinite(a).all()
    if a.size<=4096:
        return a.copy()
    return a[(np.arange(4096,dtype=np.int64)*(a.size-1))//4095]

def fit(values):
    a=np.asarray(values,dtype=np.float64).reshape(-1)
    assert a.size and np.isfinite(a).all()
    u,counts=np.unique(a,return_counts=True)
    ge=a.size-np.concatenate(([0],np.cumsum(counts[:-1])))
    evidence=-np.log((1.0+ge)/(a.size+1.0))
    q50,q95=np.quantile(a,[.5,.95])
    scale=max(float(q95-q50),1e-12)
    return dict(knots=u,evidence=evidence,count=np.asarray(a.size,np.int64),
                q50=np.asarray(q50),q95=np.asarray(q95),scale=np.asarray(scale))

def transform(values,cal):
    a=np.asarray(values,dtype=np.float64)
    assert np.isfinite(a).all()
    u=cal['knots'];e=cal['evidence'];scale=float(cal['scale'])
    out=np.interp(a.reshape(-1),u,e,left=0.0,right=float(e[-1])).reshape(a.shape)
    out+=np.maximum(a-float(u[-1]),0.0)/scale
    assert np.isfinite(out).all() and (out>=0).all()
    return out

def top_fraction(a):
    a=np.asarray(a,dtype=np.float64).reshape(-1)
    k=max(1,int(np.ceil(a.size*.01)))
    return float(np.mean(np.partition(a,a.size-k)[-k:]))

def pixel(raw,source,cal):
    ec=transform(raw,cal['cnn'])
    assert ec.shape==(512,512)
    out={'e_cnn':ec}
    patches=[]
    for name in COMPONENTS[1:]:
        patch=transform(source[RAW_KEYS[name]],cal[name])
        assert patch.shape==(37,37)
        out['e_'+name+'_patch']=patch
        patches.append(cv2.resize(patch,(512,512),interpolation=cv2.INTER_LINEAR))
    out['pixel_score']=.5*ec+.5*(patches[0]+patches[1]+patches[2])/3.0
    out['pixel_top1']=np.asarray(top_fraction(out['pixel_score']))
    return out

def predict(raw,source,cal):
    out=pixel(raw,source,cal)
    out['image_top_evidence']=np.asarray(transform(out['pixel_top1'],cal['top']))
    out['image_cls_evidence']=np.asarray(transform(source['image_raw_distance'],cal['cls']))
    out['image_score']=.5*out['image_top_evidence']+.5*out['image_cls_evidence']
    return out

def save_calibration(path,cal):
    np.savez_compressed(path,**{name+'__'+k:v for name,part in cal.items() for k,v in part.items()})

def load_calibration(path):
    with np.load(path,allow_pickle=False) as z:
        return {name:{k.split('__')[1]:z[k].copy() for k in z.files if k.startswith(name+'__')}
                for name in COMPONENTS+('top','cls')}

def distribution(values):
    a=np.asarray(values,dtype=np.float64).reshape(-1)
    assert a.size and np.isfinite(a).all()
    return dict(n=int(a.size),mean=float(a.mean()),std=float(a.std()),minimum=float(a.min()),
                q50=float(np.quantile(a,.5)),q95=float(np.quantile(a,.95)),
                q99=float(np.quantile(a,.99)),maximum=float(a.max()))
