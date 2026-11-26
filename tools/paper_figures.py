"""Regenerate empirical manuscript panels from recorded numerical plot inputs.

No model is instantiated and no experimental value is fitted by this exporter.
Use --data-dir to render another run using the documented CSV/NPZ schemas.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tools.public_utils import relative_file, sha256

ROOT=Path(__file__).resolve().parents[1]
EMPIRICAL=(1,3,5,6,7,8,9,10,11)
BLUE='#30658C';ORANGE='#BF7B49';GRAY='#7A8790';TEAL='#348780';PURPLE='#68528F'
METRICS=('image_auroc','pixel_auroc','aupro')
LABELS=('I-AUROC','P-AUROC','AUPRO')

def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))

def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,
        'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none',
        'svg.hashsalt':'ADNVCNet-paper-figures','savefig.facecolor':'white'})
    return plt

class Exporter:
    def __init__(self,args):
        self.args=args;self.data=args.data_dir.resolve();self.output=args.output.resolve()
        self.output.mkdir(parents=True,exist_ok=False)
        self.plt=plotting();self.receipts=[]

    def save(self,fig,number,sources,values=None,suffix=''):
        stem=f'figure_{number:02d}'+suffix
        for ext in ('png','svg'):
            fig.savefig(self.output/(stem+'.'+ext),dpi=600,bbox_inches='tight',pad_inches=.05,
                        metadata={'Date':None} if ext=='svg' else None)
        self.plt.close(fig)
        self.receipts.append({'figure':number,'name':stem,'source_files':[
            {'path':str(p.relative_to(self.data)),'sha256':sha256(p)} for p in sources],
            'png_sha256':sha256(self.output/(stem+'.png')),
            'svg_sha256':sha256(self.output/(stem+'.svg')),'plot_values':values})
        print('EXPORTED '+stem,flush=True)

    def numeric(self,n):
        if n==3:return self.calibration()
        if n==5:return self.ablation()
        if n==6:return self.order()
        if n==7:return self.spatial()
        if n==8:return self.intervals()
        if n==11:return self.efficiency()
        return self.qualitative(n)

    def ablation(self):
        source=self.data/'figure05_ablation.csv';rows=csv_rows(source)
        assert len(rows)==42
        fig,axes=self.plt.subplots(2,3,figsize=(11.4,6.7),layout='constrained')
        variants=['A1','A2','A3','A4','A5','A6','T1']
        for y,ds in enumerate(('mvtec','visa')):
            for x,metric in enumerate(METRICS):
                axis=axes[y,x];axis.axvline(0,color=GRAY,lw=.8)
                for pos,v in enumerate(variants):
                    item=next(r for r in rows if r['dataset']==ds and r['variant']==v and r['metric']==metric)
                    value=float(item['A0_minus_control_pp']);shared=item['inherited_identical_pixel_map'].lower()=='true'
                    if shared and value!=0:raise ValueError('Shared pixel metrics differ')
                    color=GRAY if shared else BLUE if value>=0 else ORANGE
                    axis.plot([0,value],[pos,pos],color=color,lw=1.2);axis.plot(value,pos,'o',color=color,ms=3.5)
                    label='0*' if shared else f'{value:+.4f}' if 0<abs(value)<.001 else f'{value:+.3f}'
                    axis.annotate(label,(value,pos),xytext=(4,5),textcoords='offset points',fontsize=8,color=color)
                axis.set(yticks=range(7),yticklabels=variants,ylim=(6.7,-.8),
                         xlim=[(-1.8,7.8),(-.65,5.0),(-1.,6.8)][x])
                if y==0:axis.set_title(LABELS[x],fontweight='bold')
                if x==0:axis.set_ylabel('MVTec AD' if ds=='mvtec' else 'VisA',fontweight='bold')
                if y==1:axis.set_xlabel('Delta = A0 - control (pp)')
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([],[],color=c,marker='o',ls='',label=t) for c,t in
            [(BLUE,'A0 higher'),(ORANGE,'Control higher'),(GRAY,'Shared map 0*')]],
            loc='outside lower center',ncol=3,frameon=False)
        self.save(fig,5,[source],rows)

    def order(self):
        source=self.data/'figure06_fusion_order.csv';rows=csv_rows(source)
        assert len(rows)==29
        fig,axes=self.plt.subplots(1,2,figsize=(10.5,7),layout='constrained')
        for axis,ds in zip(axes,('visa','mvtec')):
            group=[r for r in rows if r['dataset']==ds];axis.axvline(0,color=GRAY,lw=.8)
            labels=[]
            for pos,r in enumerate(group):
                macro=r['category'] in ('宏平均','Macro mean');labels.append('Macro mean' if macro else r['category'])
                v=float(r['joint_minus_separate_pp']);c=PURPLE if macro else BLUE if v>0 else ORANGE if v<0 else GRAY
                axis.plot([0,v],[pos,pos],color=c);axis.plot(v,pos,'o',color=c,ms=3.5)
                axis.annotate(f'{v:+.3f}',(v,pos),xytext=(4,4),textcoords='offset points',fontsize=8,color=c)
            axis.set(yticks=range(len(group)),yticklabels=labels,ylim=(len(group)-.4,-.7),
                     xlabel='Delta I-AUROC (pp)',title='VisA' if ds=='visa' else 'MVTec AD')
            axis.margins(x=.23)
        self.save(fig,6,[source],rows)

    def spatial(self):
        source=self.data/'figure07_spatial_macro.csv';rows=csv_rows(source);values=[]
        fig,axes=self.plt.subplots(2,3,figsize=(8.4,4.3),layout='constrained')
        for i,ds in enumerate(('mvtec','visa')):
            baseline=next(r for r in rows if r['dataset']==ds and r['variant']=='A0')
            for j,m in enumerate(METRICS):
                ax=axes[i,j];ax.axvline(0,color=GRAY,lw=.7);ax.grid(axis='x',color='#eeeeee');ax.set_axisbelow(True)
                for y,(v,color) in enumerate(zip(('A0','S1_DOUBLE','S0_S8'),(BLUE,ORANGE,TEAL))):
                    row=next(r for r in rows if r['dataset']==ds and r['variant']==v)
                    delta=float(row[m])-float(baseline[m]);values.append(dict(dataset=ds,variant=v,metric=m,delta_pp=delta))
                    ax.plot([0,delta],[y,y],color=color,lw=1.3);ax.plot(delta,y,'o',color=color,ms=5)
                ax.set(yticks=range(3),yticklabels=['Same + 3x3','3x3 only','Same + neighbors'] if j==0 else [],
                       ylim=(2.1,-.1),xlabel='Delta '+LABELS[j]+' (pp)')
                ax.margins(x=.13)
                if j==0:ax.set_title('MVTec AD' if ds=='mvtec' else 'VisA',loc='left',fontweight='bold')
        self.save(fig,7,[source],values)

    def intervals(self):
        source=self.data/'figure08_intervals.csv';rows=csv_rows(source)
        fig,axes=self.plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
        from matplotlib.lines import Line2D
        for axis,comp,title in zip(axes,('A0-O1','A0-A4'),('Fuse first: A0-O1','Spatial refs.: A0-A4')):
            axis.axvline(0,color=GRAY,lw=.8);axis.grid(axis='x',color='#e5e8eb');axis.set_axisbelow(True)
            for y,ds in enumerate(('visa','mvtec')):
                row=next(r for r in rows if r['dataset']==ds and r['comparison']==comp)
                v=float(row['delta_I_AUROC_pp']);lo=float(row['ci95_low_pp']);hi=float(row['ci95_high_pp'])
                blo=float(row['familywise95_bonferroni_low_pp']);bhi=float(row['familywise95_bonferroni_high_pp'])
                if not blo<=lo<=hi<=bhi:raise ValueError('CI nesting is inconsistent')
                axis.plot([blo,bhi],[y,y],color='#98b8cf',lw=1.2,marker='|',ms=7)
                axis.plot([lo,hi],[y,y],color=BLUE,lw=4,solid_capstyle='round');axis.plot(v,y,'o',color=BLUE,ms=5)
                axis.annotate(f'{v:+.3f}',(v,y),xytext=(0,12),textcoords='offset points',ha='center',color=BLUE)
            axis.set(yticks=[0,1],yticklabels=['VisA','MVTec AD'],ylim=(1.5,-.55),
                     xlabel='Delta I-AUROC (pp)',title=title);axis.margins(x=.10)
        fig.legend(handles=[Line2D([],[],color=BLUE,lw=4,label='95% CI'),
            Line2D([],[],color='#98b8cf',lw=1.2,marker='|',label='Adjusted 98.75% CI')],
            loc='outside lower center',ncol=2,frameon=False)
        self.save(fig,8,[source],rows)

    def calibration(self):
        source=self.data/'figure03_calibration.npz'
        with np.load(source,allow_pickle=False) as saved:cal={k:saved[k].copy() for k in saved.files}
        x=cal['knots'];y=cal['evidence'];fig,ax=self.plt.subplots(figsize=(5.4,3.3),layout='constrained')
        ax.plot(np.r_[0,x[0],x],np.r_[0,0,y],color=BLUE,lw=1.2,label='Empirical mapping')
        end=float(x[-1]+.15*(x[-1]-x[0]));ax.axvline(x[-1],color=GRAY,ls='--',lw=.6)
        ax.plot([x[-1],end],[y[-1],y[-1]+(end-x[-1])/float(cal['scale'])],color=ORANGE,ls='--',label='Linear extrapolation')
        for key,label,offset in [('q50',r'$q_{50}$',(-20,10)),('q95',r'$q_{95}$',(6,-14))]:
            value=float(cal[key]);e=float(np.interp(value,x,y));ax.plot([value,value],[0,e],color=GRAY,ls='--',lw=.6)
            ax.plot(value,e,'o',color=BLUE,ms=3);ax.annotate(label,(value,e),xytext=offset,textcoords='offset points')
        ax.set(xlabel='Raw CNN response a',ylabel=r'$E_C(a)$',xlim=(0,.15),ylim=(0,12.5),title='Cable normal calibration responses')
        ax.legend(frameon=False,fontsize=8)
        self.save(fig,3,[source],{'knots':len(x),'q50':float(cal['q50']),'q95':float(cal['q95']),'scale':float(cal['scale'])},'_calibration')

    def sample(self,row):
        data_root=self.args.mvtec_root if row['dataset']=='mvtec' else self.args.visa_root
        if data_root is None:raise ValueError('Qualitative figures require --mvtec-root / --visa-root')
        path=relative_file(data_root,row['input'])
        if sha256(path)!=row['input_sha256']:raise ValueError('Dataset image bytes differ from the recorded plot input')
        with Image.open(path) as im:rgb=im.convert('RGB').resize((512,512),Image.Resampling.BICUBIC)
        response=relative_file(self.data,row['response_file'])
        if sha256(response)!=row['response_sha256']:raise ValueError('Response archive changed')
        with np.load(response,allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
        for key,array in arrays.items():
            if hashlib.sha256(array.tobytes()).hexdigest()!=row['array_sha256'][key]:raise ValueError('Response array changed')
        return rgb,arrays,data_root,response

    def qualitative(self,number):
        source=self.data/'qualitative_selection.json';selection=json.loads(source.read_text('utf-8'));rows=selection['records']
        if number==1:
            import cv2
            rows=[r for r in rows if r['dataset']=='mvtec' and r['category']=='cable'];sources=[source]
            normal=next(r for r in rows if r['condition']=='normal');abnormal=next(r for r in rows if r['condition']=='anomaly')
            nr,na,_,npz=self.sample(normal);ar,aa,_,apz=self.sample(abnormal);sources += [npz,apz]
            fig=self.plt.figure(figsize=(11.5,5.5));fig.text(.035,.97,'(a) Normal and anomalous responses',weight='bold')
            fig.text(.51,.97,'(b) High-response locations',weight='bold')
            for i,(rgb,a,label) in enumerate(((nr,na,'Normal'),(ar,aa,'Anomalous'))):
                for j,(value,title) in enumerate(((rgb,'Input image'),(a['e_cnn'],'CNN response'))):
                    ax=fig.add_axes([.04+j*.23,.54-i*.43,.205,.365]);self.tile(ax,value,heat='magma' if j==1 else False)
                    if i==0:ax.set_title(title)
                    if j==0:ax.set_ylabel(label)
            aligned=cv2.resize(aa['e_global_patch'],(512,512),interpolation=cv2.INTER_LINEAR)
            for j,(a,title) in enumerate(((aa['e_cnn'],'CNN response'),(aligned,'Global map'))):
                ax=fig.add_axes([.51+j*.245,.27,.225,.54]);self.tile(ax,a,heat='magma');flat=a.reshape(-1)
                mask=np.zeros(flat.size,bool);mask[np.argsort(flat,kind='stable')[-int(np.ceil(.01*flat.size)):]]=True
                ax.contour(mask.reshape(a.shape),levels=[.5],colors=['#BDE8EC'],linewidths=.7);ax.set_title(title)
                ax.set_xlabel('Top-1% locations in the same image')
            from matplotlib.colors import ListedColormap, Normalize
            from matplotlib.cm import ScalarMappable
            lut=cv2.cvtColor(cv2.applyColorMap(np.arange(256,dtype=np.uint8)[None,:],cv2.COLORMAP_MAGMA),cv2.COLOR_BGR2RGB)[0]/255.
            fig.colorbar(ScalarMappable(norm=Normalize(0,10),cmap=ListedColormap(lut)),
                         cax=fig.add_axes([.55,.105,.40,.025]),orientation='horizontal',ticks=[0,5,10],label='Calibrated response')
            self.save(fig,1,sources,{'selection':[r['input'] for r in rows]});return
        dataset='mvtec' if number==9 else 'visa';rows=[r for r in rows if r['dataset']==dataset];sources=[source]
        cats=selection['datasets'][dataset];fig,axes=self.plt.subplots(5,5,figsize=(9,9))
        for ax,t in zip(axes[0],('Normal image','Normal map','Anomalous image','Anomalous map','Ground truth')):ax.set_title(t,fontsize=9)
        for i,cat in enumerate(cats):
            normal=next(r for r in rows if r['category']==cat and r['condition']=='normal')
            abnormal=next(r for r in rows if r['category']==cat and r['condition']=='anomaly')
            nr,na,root,npz=self.sample(normal);ar,aa,root,apz=self.sample(abnormal);sources += [npz,apz]
            mask_path=relative_file(root,abnormal['mask'])
            if sha256(mask_path)!=abnormal['mask_sha256']:raise ValueError('GT bytes differ from recorded display input')
            with Image.open(mask_path) as im:mask=im.convert('L').resize((512,512),Image.Resampling.NEAREST)
            for j,value in enumerate((nr,na['pixel_score'],ar,aa['pixel_score'],mask)):
                artist=self.tile(axes[i,j],value,heat=j in (1,3),mask=j==4)
                if j==3:heat=artist
            axes[i,0].set_ylabel(cat,fontsize=9)
        fig.subplots_adjust(left=.08,right=.91,top=.95,bottom=.02,wspace=.06,hspace=.08)
        fig.colorbar(heat,cax=fig.add_axes([.94,.20,.012,.60]),ticks=[0,2,4,6,8,10])
        self.save(fig,number,sources,{'selection':[r['input'] for r in rows],'GT_display_only':True})

    @staticmethod
    def tile(ax,value,heat=False,mask=False):
        if heat=='magma':
            import cv2
            indices=(np.clip(np.asarray(value)/10,0,1)*255).astype(np.uint8)
            value=cv2.cvtColor(cv2.applyColorMap(indices,cv2.COLORMAP_MAGMA),cv2.COLOR_BGR2RGB)
            heat=False
        artist=ax.imshow(value,cmap='viridis' if heat else 'gray' if mask else None,
                         vmin=0 if heat or mask else None,vmax=10 if heat else 255 if mask else None,
                         interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
        return artist

    def efficiency(self):
        paths=[self.data/'figure11_efficiency_classes.csv',self.data/'figure11_efficiency_summary.json']
        rows=csv_rows(paths[0]);summary=json.loads(paths[1].read_text('utf-8-sig'));assert len(rows)==27
        fig,axes=self.plt.subplots(2,2,figsize=(11.5,7.3),layout='constrained')
        keys=['preprocess','host_to_device','cnn_feature_response','dino_feature','matching_raw','tail','control_and_sync']
        labels=['Validation / preprocessing','Host-to-device transfer','CNN features / responses','DINO features',
                'Normal memory matching','Scoring / config loading','Control / synchronization']
        colors=['#9FB1BE','#C2CED5','#769DBD','#4F86AB','#72A69D','#D3A176','#A7A1B8']
        for y,ds in enumerate(('mvtec','visa')):
            start=0
            for key,label,color in zip(keys,labels,colors):
                value=float(summary['datasets'][ds]['stage_mean_ms'][key]);axes[0,0].barh(y,value,left=start,color=color,label=label if y==0 else None);start+=value
            if abs(start-float(summary['datasets'][ds]['latency_ms']['mean']))>1e-8:raise ValueError('Stage times do not sum to total')
            axes[0,0].text(start+5,y,f'{start:.1f}',va='center')
        axes[0,0].set(yticks=[0,1],yticklabels=['MVTec AD','VisA'],xlabel='Mean class latency (ms)',title='(a) Synced call latency')
        axes[0,0].invert_yaxis()
        for x,ds,color in ((0,'mvtec',BLUE),(1,'visa',TEAL)):
            group=[r for r in rows if r['dataset']==ds];positions=x+np.linspace(-.2,.2,len(group))
            axes[0,1].scatter(positions,[float(r['total_mean_ms']) for r in group],s=18,color=color)
            axes[0,1].plot([x-.25,x+.25],[float(summary['datasets'][ds]['latency_ms']['mean'])]*2,color=ORANGE)
            for offset,key,label,c in ((-.16,'peak_allocated_GiB','Allocated',BLUE),(.16,'peak_reserved_GiB','Reserved','#9FB1BE')):
                value=float(summary['datasets'][ds][key]);axes[1,0].bar(x+offset,value,width=.30,color=c,label=label if x==0 else None)
                axes[1,0].annotate(f'{value:.2f}',(x+offset,value),xytext=(0,4),textcoords='offset points',ha='center')
            axes[1,1].scatter(positions,[float(r['loading_validation_calibration_ms'])/1000 for r in group],s=18,color=color)
            axes[1,1].plot([x-.25,x+.25],[float(summary['datasets'][ds]['loading_validation_calibration_ms']['median'])/1000]*2,color=ORANGE)
        axes[0,1].set(ylabel='Per-class mean latency (ms)',title='(b) Per-class latency')
        axes[1,0].set(ylabel='Peak GPU memory (GiB)',title='(c) Peak GPU memory');axes[1,0].legend(frameon=False)
        axes[1,0].set_ylim(0,max(float(summary['datasets'][d]['peak_reserved_GiB']) for d in ('mvtec','visa'))*1.18)
        axes[1,1].set(ylabel='Load + full check (s)',title='(d) Initialization cost')
        for ax in (axes[0,1],axes[1,0],axes[1,1]):ax.set(xticks=[0,1],xticklabels=['MVTec AD','VisA'],xlim=(-.5,1.5))
        handles,legend=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,legend,loc='outside lower center',ncol=3,frameon=False,fontsize=8)
        self.save(fig,11,paths,{'class_count':len(rows),'datasets':summary['datasets']})

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'paper/plot_data')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--figures',nargs='+',default=['3','5','6','7','8','11'])
    parser.add_argument('--mvtec-root',type=Path)
    parser.add_argument('--visa-root',type=Path)
    args=parser.parse_args(argv)
    selected=EMPIRICAL if args.figures==['all'] else tuple(int(n) for n in args.figures)
    if len(selected)!=len(set(selected)) or any(n not in EMPIRICAL for n in selected):parser.error('Use empirical figures 1,3,5,6,7,8,9,10,11 or all')
    exporter=Exporter(args)
    for n in selected:exporter.numeric(n)
    (exporter.output/'plot_provenance.json').write_text(json.dumps({'schema':'adnvcnet_paper_figures_v1',
        'new_training':0,'new_inference':0,'new_metric_calculations':0,'new_bootstrap':False,
        'display_clipping_only':[0,10],'figures':exporter.receipts},indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':main()
