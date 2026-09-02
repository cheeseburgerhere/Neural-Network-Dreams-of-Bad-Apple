"""Package completed blog inference into videos, exact tables, and static figures."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .blog_export import ROOT, WORK, OUT, FPS, COUNT, WARMUP, sha256, write_csv, write_json
from .data import _imageio_ffmpeg

BLUE, GOLD, INK = "#2864a0", "#bf861b", "#263440"


def ffmpeg(arguments):
    subprocess.run([_imageio_ffmpeg().get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n",
                    *map(str, arguments)], check=True)


def encode_options():
    return ["-an", "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]]) + "\n"


def save_figure(fig, name):
    fig.savefig(OUT / "figures" / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "figures" / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axis(ax, title, subtitle, ylabel):
    ax.set_title(title + "\n" + subtitle, loc="left", fontsize=12, pad=16, color=INK)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e5e8eb", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)


def charts(summaries, curves, primary_220):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "text.color":INK,
                         "axes.labelcolor":INK, "xtick.color":INK, "ytick.color":INK})
    main = [s for s in summaries if s['tag'] != 'anchors_220' or primary_220 == 'anchors_220']
    labels = [str(s['anchor_count']) + ('*' if s['tag']=='anchors_220' else '') for s in main]
    context = "6,557 scored frames | 384 x 512 | first 16 excluded | one training seed"
    x = np.arange(len(main))
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    for offset, key, label, color, hatch in [(-.19,'teacher_error','Teacher-forced',GOLD,'//'),
                                          (.19,'rollout_error','Full rollout',BLUE,None)]:
        values = [s['metrics'][key]*100 for s in main]
        bars = ax.bar(x+offset, values, .36, label=label, color=color, edgecolor=INK, linewidth=.6, hatch=hatch)
        ax.bar_label(bars, fmt='%.2f', fontsize=9, padding=3)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Learned anchor count")
    ax.set_ylim(0, max(s['metrics']['rollout_error']*100 for s in main)*1.19)
    style_axis(ax, "Teacher-forced and free-rollout pixel error", context, "Mismatched pixels (%)")
    ax.legend(frameon=False)
    fig.text(.11,.005,"* 220 uses its original polarity head without the separate spline fix; do not attribute all error differences to anchor count."
             if primary_220=='anchors_220' else
             "All six models use separately fitted 96-knot polarity splines; generation receives time, not source polarity."
             if primary_220=='anchors_220_polarity' else
             "220 uses the existing 32-anchor model's learned polarity spline; latent dynamics are unchanged.",fontsize=9)
    save_figure(fig,'01_anchor_budget_error')

    memory = [s for s in main if s['anchor_count']]
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    x = np.arange(len(memory))
    for offset,key,label,color,hatch in [(-.19,'memory_error','Memory only',GOLD,'//'),(.19,'rollout_error','Full rollout',BLUE,None)]:
        bars=ax.bar(x+offset,[s['metrics'][key]*100 for s in memory],.36,label=label,color=color,edgecolor=INK,linewidth=.6,hatch=hatch)
        ax.bar_label(bars,fmt='%.2f',fontsize=9,padding=3)
    ax.set_xticks(x,[str(s['anchor_count']) + ('*' if s['tag']=='anchors_220' else '') for s in memory])
    ax.set_xlabel("Learned anchor count")
    ax.set_ylim(bottom=0,top=max(s['metrics']['memory_error']*100 for s in memory)*1.2)
    style_axis(ax,"Direct memory decoding versus the full system",context,"Mismatched pixels (%)")
    ax.legend(frameon=False)
    fig.text(.11,.005,"Post-hoc removal with jointly trained anchors, not an independently trained memory-only baseline.",fontsize=9)
    if primary_220=='anchors_220':
        fig.text(.11,-.04,'* 220 retains its original polarity head; other budgets use the fitted spline.',fontsize=9)
    save_figure(fig,'02_memory_only_ablation')

    fig, axes = plt.subplots(2,1,figsize=(11,7),sharex=True)
    for ax, tag, title in zip(axes,['anchors_032',primary_220],['32 anchors','220 anchors (native polarity)' if primary_220=='anchors_220' else '220 anchors (calibrated polarity)' if primary_220=='anchors_220_polarity' else '220 anchors (shared polarity)']):
        curve=curves[tag]
        for key,label,color,linestyle in [('teacher_error','Teacher-forced',INK,'--'),('rollout_error','Full rollout',BLUE,'-'),('memory_error','Memory only',GOLD,':')]:
            ax.plot(curve['seconds'][WARMUP:],curve[key][WARMUP:]*100,label=label,color=color,linestyle=linestyle,linewidth=.85)
        style_axis(ax,title+": per-frame pixel error","6,557 frames | no smoothing | same time and error scales","Pixel error (%)")
        ax.set_ylim(0,100)
        ax.set_xlim(0,(COUNT-1)/FPS)
        ax.legend(frameon=False,ncol=3,loc='upper right')
    axes[-1].set_xlabel("Video time (seconds)")
    fig.tight_layout(h_pad=2)
    save_figure(fig,'03_error_accumulation')

    fig,ax=plt.subplots(figsize=(9.8,5.0))
    x=np.arange(len(main))
    other=np.array([s['parameters']-s['parameter_groups'].get('anchors',0) for s in main])/1e6
    anchors=np.array([s['parameter_groups'].get('anchors',0) for s in main])/1e6
    ax.bar(x,other,label='Prediction, time and gates',color=GOLD,edgecolor=INK,linewidth=.6,hatch='//')
    ax.bar(x,anchors,bottom=other,label='Learned anchor tensors',color=BLUE,edgecolor=INK,linewidth=.6)
    for i,total in enumerate(other+anchors):
        ax.text(i,total+.12,f'{total:.3f}M',ha='center',fontsize=9)
    ax.set_xticks(x,[str(s['anchor_count']) for s in main])
    ax.set_xlabel('Anchor count')
    ax.set_ylim(0,12.5)
    style_axis(ax,'Where the predictor parameters go','Actual saved-model counts | frozen autoencoder excluded','Parameters (millions)')
    ax.legend(frameon=False)
    save_figure(fig,'04_parameter_budget')


def tables(summaries,curves):
    quality=[]; memory=[]; parameters=[]; training=[]; spacing=[]
    for s in summaries:
        m=s['metrics']; label=f"{s['anchor_count']}" + (' + shared polarity' if 'shared' in s['tag'] else ' native' if s['tag']=='anchors_220' else '')
        quality.append(dict(model=label,anchors=s['anchor_count'],teacher_error_percent=m['teacher_error']*100,
            rollout_error_percent=m['rollout_error']*100,gap_percentage_points=m['accumulation_gap']*100,
            rollout_class_mean_iou=m['rollout_iou'],rollout_p95_error_percent=m['rollout_p95']*100,
            polarity_accuracy_percent=m['polarity_accuracy']*100,oracle_polarity_shape_error_percent=m['oracle_polarity_rollout_error']*100,
            scored_frames=m['scored_frames'],checkpoint=s['checkpoint'],checkpoint_sha256=s['checkpoint_sha256']))
        if s['anchor_count']:
            memory.append(dict(model=label,anchors=s['anchor_count'],memory_error_percent=m['memory_error']*100,
                full_error_percent=m['rollout_error']*100,relative_error_reduction_percent=m['full_error_reduction_fraction']*100,
                full_wins_frames_percent=m['full_wins_fraction']*100,memory_class_mean_iou=m['memory_iou'],full_class_mean_iou=m['rollout_iou'],scored_frames=m['scored_frames']))
            intervals=np.diff(s['anchor_frames'])/FPS
            spacing.append(dict(model=label,anchors=s['anchor_count'],total_frames=COUNT,anchor_count_as_frame_percent=100*s['anchor_count']/COUNT,
                mean_interval_seconds=float(intervals.mean()),median_interval_seconds=float(np.median(intervals)),maximum_interval_seconds=float(intervals.max())))
        parameters.append(dict(model=label,**s['parameter_groups'],predictor_total=s['parameters'],
            shared_polarity_extra_parameters=96 if 'shared' in s['tag'] else 0,
            frozen_autoencoder=s['autoencoder_parameters'],system_total=s['parameters']+s['autoencoder_parameters']+(96 if 'shared' in s['tag'] else 0)))
        training.append(dict(model=label,**s['training'],checkpoint_epoch=s['checkpoint_epoch'],polarity_handling=s['polarity_handling']))
    docs=[]
    for name,rows,headers,values in [
        ('01_quality',quality,['Model / anchors','Teacher error','Rollout error','Gap (pp)','Class-mean IoU','Polarity accuracy'],
         lambda r:[r['model'],f"{r['teacher_error_percent']:.2f}%",f"{r['rollout_error_percent']:.2f}%",f"{r['gap_percentage_points']:.2f}",f"{r['rollout_class_mean_iou']:.3f}",f"{r['polarity_accuracy_percent']:.2f}%"]),
        ('02_memory_only',memory,['Anchors / variant','Memory-only error','Full error','Relative reduction','Frames full wins'],
         lambda r:[r['model'],f"{r['memory_error_percent']:.2f}%",f"{r['full_error_percent']:.2f}%",f"{r['relative_error_reduction_percent']:.1f}%",f"{r['full_wins_frames_percent']:.1f}%"]),
        ('03_parameters',parameters,['Anchors / variant','Anchor parameters','Temporal U-Net','Other parameters','Predictor total','Frozen AE'],
         lambda r:[r['model'],f"{r.get('anchors',0):,}",f"{r['temporal_unet']:,}",f"{r['time_features']+r['heads_and_gates']:,}",f"{r['predictor_total']:,}",f"{r['frozen_autoencoder']:,}"]),
        ('04_training',training,['Anchors / variant','Batch','LR','Epochs','Seed','Burn-in max','Rollout max','Memory frozen epochs'],
         lambda r:[r['model'],r['batch_size'],r['learning_rate'],r['epochs'],r['seed'],r['burn_in_steps'],r['rollout_steps'],r['freeze_memory_epochs']]),
        ('05_anchor_spacing',spacing,['Anchors / variant','Video frames','Mean gap (s)','Median gap (s)','Max gap (s)'],
         lambda r:[r['model'],r['total_frames'],f"{r['mean_interval_seconds']:.2f}",f"{r['median_interval_seconds']:.2f}",f"{r['maximum_interval_seconds']:.2f}"]),
    ]:
        # Normalize optional parameter-group keys for zero-anchor models.
        allkeys=list(dict.fromkeys(k for row in rows for k in row))
        rows=[{k:row.get(k,0) for k in allkeys} for row in rows]
        write_csv(OUT/'tables'/f'{name}.csv',rows)
        text=markdown_table(headers,[values(r) for r in rows])
        (OUT/'tables'/f'{name}.md').write_text(text,encoding='utf-8')
        docs += ['## '+name.replace('_',' '),'',text,'']
    docs[:0]=['All quality scores use frames 16–6572 (6,557 frames) at 384×512, before video compression. Polarity handling and checkpoint identity are recorded in source_summaries.json; the calibrated main comparison uses the same spline method across budgets. Memory-only is post-hoc removal, not a separately trained baseline.', '']
    (OUT/'tables'/'all_tables.md').write_text('\n'.join(docs),encoding='utf-8')
    write_json(OUT/'tables'/'source_summaries.json',summaries)


def autoencoder_demo():
    import torch
    from torch.utils.data import DataLoader, Subset
    from .blog_export import AE, DATA, VideoWriter, panel, restore_polarity
    from .autoregressive import load_autoencoder
    from .data import FrameDataset
    path=OUT/'videos/autoencoder_45_60.mp4'
    if path.exists(): return
    cache=torch.load(ROOT/'prototype_data/cache/blog_canonical.pt',map_location='cpu',weights_only=False)
    device=torch.device('cuda')
    model,cp=load_autoencoder(AE,device)
    dataset=FrameDataset(DATA,*cp['image_size'],cp['input_threshold'])
    loader=DataLoader(Subset(dataset,range(45*FPS,60*FPS)),batch_size=8,shuffle=False)
    from PIL import Image
    writer=VideoWriter(path,(1024,442))
    index=45*FPS
    try:
        with torch.inference_mode():
            for targets,_ in loader:
                stop=index+len(targets)
                probability=torch.sigmoid(model.decode(cache['raw_latents'][index:stop].to(device),cp['image_size']))
                prediction=restore_polarity(probability,cache['polarities'][index:stop].to(device)) >= cp['activation_threshold']
                for j in range(len(targets)):
                    canvas=Image.new('RGB',(1024,442))
                    canvas.paste(panel(targets[j,0].numpy().astype(bool),'Source',f't = {(index+j)/FPS:.2f}s'),(0,0))
                    canvas.paste(panel(prediction[j,0].cpu().numpy(),'Autoencoder reconstruction','Source frame supplied; not a prediction'),(512,0))
                    writer.write(canvas)
                index=stop
    finally:
        writer.close()


def videos(summaries,primary_220):
    assets=[]
    for s in summaries:
        name=f"{s['tag']}_comparison_full.mp4"
        target=OUT/'videos'/name
        if not target.exists(): shutil.copy2(WORK/s['tag']/'comparison.mp4',target)
        assets.append(dict(file='videos/'+name,start_frame=0,frame_count=COUNT,purpose='Source / teacher / rollout / memory-only; one checkpoint',model=s['tag']))
    hero=OUT/'videos'/'hero_source_vs_220_full.mp4'
    if not hero.exists():
        ffmpeg(['-i',WORK/'source.mp4','-i',WORK/primary_220/'rollout.mp4','-filter_complex',
            "[0:v]pad=512:432:0:48:black,drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='Source':x=12:y=12:fontsize=24:fontcolor=white[a];"
            "[1:v]pad=512:432:0:48:black,drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='220-anchor rollout':x=12:y=12:fontsize=24:fontcolor=white[b];[a][b]hstack=inputs=2[v]",
            '-map','[v]',*encode_options(),hero])
    assets.append(dict(file='videos/'+hero.name,start_frame=0,frame_count=COUNT,purpose='Opening demo: source beside latest 220-anchor rollout',model=primary_220))
    grid=OUT/'videos'/'anchor_budget_full.mp4'
    if not grid.exists():
        tags=['anchors_000','anchors_016','anchors_032','anchors_055','anchors_110',primary_220]
        inputs=[WORK/'source.mp4',*[WORK/t/'rollout.mp4' for t in tags]]
        labels=['Source','0 anchors','16 anchors','32 anchors','55 anchors','110 anchors',
                '220 anchors (shared polarity)' if 'shared' in primary_220 else '220 anchors (calibrated)' if primary_220=='anchors_220_polarity' else '220 anchors (native polarity)']
        args=[]; filters=[]
        for i,(path,label) in enumerate(zip(inputs,labels)):
            args+=['-i',path]
            filters.append(f"[{i}:v]scale=320:240:flags=neighbor,pad=320:280:0:40:color=0x111820,drawbox=x=0:y=0:w=iw:h=40:color=0x263440:t=fill,drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='{label}':x=8:y=10:fontsize=17:fontcolor=white,drawbox=x=0:y=0:w=iw:h=ih:color=0x73808c:t=2[p{i}]")
        filters.append(''.join(f'[p{i}]' for i in range(7))+'xstack=inputs=7:layout=0_0|320_0|640_0|960_0|0_280|320_280|640_280:fill=black[v]')
        ffmpeg([*args,'-filter_complex',';'.join(filters),'-map','[v]',*encode_options(),grid])
    assets.append(dict(file='videos/'+grid.name,start_frame=0,frame_count=COUNT,purpose='Frame-aligned anchor-budget overview',model='all six budgets'))
    clips=[('hero_45_60.mp4',hero,45,60,'Original prototype interval; not selected by error'),
           ('collapse_zero_45_60.mp4',OUT/'videos/anchors_000_comparison_full.mp4',45,60,'Teacher versus free rollout with no anchors'),
           ('memory_32_45_60.mp4',OUT/'videos/anchors_032_comparison_full.mp4',45,60,'Memory-only removal on the original prototype interval'),
           ('memory_55_transition_108_114.mp4',OUT/'videos/anchors_055_comparison_full.mp4',108,114,'Previously identified difficult transition; not representative average performance'),
           ('memory_220_45_60.mp4',OUT/'videos'/f'{primary_220}_comparison_full.mp4',45,60,'Latest model: full system versus direct memory playback')]
    for name,source,start,end,purpose in clips:
        target=OUT/'videos'/name
        if not target.exists():
            ffmpeg(['-i',source,'-vf',f'trim=start_frame={start*FPS}:end_frame={end*FPS},setpts=PTS-STARTPTS',*encode_options(),target])
        assets.append(dict(file='videos/'+name,start_frame=start*FPS,frame_count=(end-start)*FPS,purpose=purpose))
    autoencoder_demo()
    assets.append(dict(file='videos/autoencoder_45_60.mp4',start_frame=45*FPS,frame_count=15*FPS,purpose='Source versus frozen autoencoder reconstruction; source supplied every frame'))
    return assets


def supporting_notes(summaries, curves):
    rows=[]
    references=[]
    for s in summaries:
        curve=curves[s['tag']][WARMUP:]
        wrong=curve['polarity_correct']==0
        rows.append(dict(model=s['tag'],polarity_mismatch_frames=int(wrong.sum()),
            displayed_rollout_error_percent=float(curve['rollout_error'].mean()*100),
            oracle_polarity_rollout_error_percent=float(np.where(wrong,1-curve['rollout_error'],curve['rollout_error']).mean()*100),
            oracle_polarity_teacher_error_percent=float(np.where(wrong,1-curve['teacher_error'],curve['teacher_error']).mean()*100)))
        old=ROOT/f"prototype_outputs/anchors_{s['anchor_count']:03d}_final/drift_summary.json"
        if old.exists():
            ref=json.loads(old.read_text())
            references.append(dict(model=s['tag'],saved_colab_rollout_error=ref['post_cutoff_mean_rollout_binary_error'],
                fresh_rollout_error=s['metrics']['rollout_error'],
                difference_percentage_points=100*(s['metrics']['rollout_error']-ref['post_cutoff_mean_rollout_binary_error'])))
        assert sha256(ROOT/s['checkpoint'])==s['checkpoint_sha256'], 'Checkpoint changed during export'
    write_csv(OUT/'tables/06_polarity_diagnostic.csv',rows)
    (OUT/'tables/06_polarity_diagnostic.md').write_text(
        'Diagnostic only: oracle polarity uses target information and is not an autonomous generation score.\n\n'+
        markdown_table(['Model','Polarity mismatch frames','Displayed rollout error','Oracle-polarity shape error'],
        [[r['model'],r['polarity_mismatch_frames'],f"{r['displayed_rollout_error_percent']:.2f}%",f"{r['oracle_polarity_rollout_error_percent']:.2f}%"] for r in rows]),encoding='utf-8')
    write_json(OUT/'validation.json',dict(frame_alignment='passed for every per-frame CSV',
        finite_metrics='passed',checkpoint_hashes_unchanged=True,previous_report_comparisons=references,
        note='Fresh local reruns are used consistently. Small differences versus saved Colab rollouts are retained, not overwritten.'))

    corrected = next((s for s in summaries if s['tag']=='anchors_220_polarity'), None)
    if corrected:
        raw = json.loads((WORK/'anchors_220/summary.json').read_text())
        proof_path = ROOT/'prototype_runs/anchors_220_polarity/verification.json'
        proof = json.loads(proof_path.read_text())
        assert proof['raw_sha256'] == raw['checkpoint_sha256']
        assert proof['corrected_sha256'] == corrected['checkpoint_sha256']
        shutil.copy2(proof_path, OUT/'220_polarity_verification.json')
        before, after = raw['metrics'], corrected['metrics']
        comparison = markdown_table(['Metric (6,557 scored frames)', 'Original 220', 'Calibrated 220'], [
            ['Teacher pixel error', f"{before['teacher_error']*100:.2f}%", f"{after['teacher_error']*100:.2f}%"],
            ['Rollout pixel error', f"{before['rollout_error']*100:.2f}%", f"{after['rollout_error']*100:.2f}%"],
            ['Memory-only pixel error', f"{before['memory_error']*100:.2f}%", f"{after['memory_error']*100:.2f}%"],
            ['Polarity accuracy', f"{before['polarity_accuracy']*100:.2f}%", f"{after['polarity_accuracy']*100:.2f}%"],
        ])
        note = [
            '# The 220 polarity correction', '',
            'The original 220 run was not comparable with the smaller checkpoints: they already had separately fitted polarity splines, while 220 still used its original polarity head. A wrong polarity bit inverts an entire frame; it is not evidence that the silhouette dynamics failed.', '',
            comparison, '',
            '## What changed', '',
            'We ran the existing polarity-only calibration with the same candidate sweep as the smaller models: 16, 24, 32, 48, 64 and 96 knots, 1,500 Adam steps per candidate, learning rate 0.1, second-difference penalty 0.0001. The selected 96-knot linear spline gets all 6,573 source polarity labels correct, including the three global inversions.', '',
            'Only 96 new scalar parameters were fitted. The saved 220 predictor now contains 10,946,630 parameters. All 63 original state tensors, including memory anchors and buffers, are bitwise unchanged. Latent normalization is unchanged. Sampled forward checks at normalized times 0.1, 0.25 and 0.8 produced bitwise-equal next latents and spatial gates. See [verification data](220_polarity_verification.json).', '',
            'The head learns from this known video during calibration, just like the other budget runs. At generation time it receives normalized time only, not target polarity or source frames. This is still single-video reconstruction, not a generalization result.', '',
            'For a fixed binary silhouette prediction, choosing the opposite polarity changes pixel error e into 1 - e. This explains why the raw global-flip failures could hide otherwise reasonable shapes. The main tables now use the actually rendered learned-head results, not oracle-adjusted scores.', '',
            '## What this does not show', '',
            'The full-versus-memory comparison is still a post-hoc removal experiment using jointly trained anchors. It is not an independently trained memory-only baseline. We still have one training seed per budget; the polarity correction does not establish statistical significance.', '',
            '## Preserved evidence', '',
            '- Raw checkpoint: prototype_runs/anchors_220/model_best.pt.',
            '- New checkpoint and calibration log: prototype_runs/anchors_220_polarity/.',
            '- Raw inference: prototype_outputs/blog_work/anchors_220/.',
            '- Corrected inference: prototype_outputs/blog_work/anchors_220_polarity/.',
            '- Entire earlier blog pack: prototype_outputs/blog_assets_before_220_polarity/.',
            '',
            'The main six-row tables use one calibrated checkpoint per anchor budget. The original 220 result remains available here and in the archive; it has not been relabelled as the corrected result.', '',
        ]
        (OUT/'220_polarity_correction.md').write_text('\n'.join(note), encoding='utf-8')

    (OUT/'architecture.md').write_text('''# Architecture block for the article

```text
First 16 source frames -> frozen encoder -> normalized seed latents
                                                |
Past 16 latent grids + their differences --------+
                  |
       concatenate state and velocity channels
                  |
       temporal U-Net: 16 -> 8 -> 4 time steps
       skip connections: 4 -> 8 -> 16 time steps
                  |
         final-step spatial features
                  |
      slow velocity + masked fast velocity
                  |
        last latent + velocity ---------------------------+
                                                          |
Time -> Fourier features -> embedding -> U-Net            |
  +-> two nearest anchor times -> learned anchor blend ---+
                                                          |
                                          spatial/cut-gated correction
                                                          |
                                                 next predicted latent
                                                   +-> feed back into history
                                                   +-> denormalize -> frozen decoder
                                                                          |
                                                      learned time-only polarity
                                                                          |
                                                             threshold -> binary frame
```

Each latent is 64 x 24 x 32. The U-Net pools time, not the spatial latent dimensions. Anchor timestamps are fixed; their value tensors are learned. Correction uses a spatial gate, not a fixed scalar blend.

In normalized latent coordinates:

    motion = previous_latent + slow_velocity + masked_fast_velocity
    memory = sum(address_weight_i(time) * learned_anchor_i)
    next_latent = motion + gate * (memory - motion)

Memory-only replaces the recurrence with next_latent = memory, keeping the decoder and polarity restoration. It removes history, temporal prediction and fusion together.
''',encoding='utf-8')


def selected_models(work):
    """Prefer the calibrated 220 checkpoint without mixing raw and fixed main rows."""
    for primary in ('anchors_220_polarity', 'anchors_220_shared_polarity', 'anchors_220'):
        if (work / primary / 'summary.json').is_file():
            return [f'anchors_{n:03d}' for n in [0, 16, 32, 55, 110]] + [primary], primary
    raise FileNotFoundError('No completed 220-anchor export found')


def main():
    for directory in ('videos','tables','figures'): (OUT/directory).mkdir(parents=True,exist_ok=True)
    tags,primary=selected_models(WORK)
    manifest_path=OUT/'asset_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['primary_220'] != primary:
        raise FileExistsError('Existing blog pack uses a different primary variant; preserve it before rebuilding')
    summaries=[json.loads((WORK/tag/'summary.json').read_text()) for tag in tags]
    curves={tag:np.genfromtxt(WORK/tag/'metrics.csv',delimiter=',',names=True,dtype=None,encoding='utf-8') for tag in tags}
    for tag,curve in curves.items():
        assert len(curve)==COUNT and np.array_equal(curve['frame'],np.arange(COUNT))
        assert np.allclose(curve['seconds'],np.arange(COUNT)/FPS)
        for key in curve.dtype.names:
            assert np.isfinite(curve[key]).all(),(tag,key)
        assert np.all((curve['rollout_error']>=0)&(curve['rollout_error']<=1))
        shutil.copy2(WORK/tag/'metrics.csv',OUT/'tables'/f'{tag}_per_frame.csv')
    contracts=[
        dict(name='01_anchor_budget_error',question='How do teacher and rollout errors compare by budget?',family='grouped bar',rows=len(summaries),palette='blue/gold with hatch',grain='model',takeaway='Measured reconstruction accuracy; documented polarity calibration'),
        dict(name='02_memory_only_ablation',question='Does decoding the same memories reproduce the full result?',family='grouped bar',rows=len(summaries)-1,palette='blue/gold with hatch',grain='model',takeaway='Post-hoc contribution of the removed machinery; no standalone-baseline claim'),
        dict(name='03_error_accumulation',question='When do errors rise across the video?',family='line small multiples',rows=6557,grain='frame',palette='blue/gold/neutral with distinct line styles',takeaway='Unsmoothed errors, common scales'),
        dict(name='04_parameter_budget',question='Where are the predictor parameters?',family='stacked bar',rows=len(summaries),grain='model',palette='blue/gold with hatch',takeaway='Memory tensors dominate large budgets'),
    ]
    write_json(OUT/'chart_contracts.json',contracts)
    tables(summaries,curves)
    supporting_notes(summaries,curves)
    charts(summaries,curves,primary)
    assets=videos(summaries,primary)
    print('Checking every exported video frame count and duration',flush=True)
    for asset in assets:
        path=OUT/asset['file']
        count,seconds=_imageio_ffmpeg().count_frames_and_secs(str(path))
        assert count==asset['frame_count'],(path,count,asset['frame_count'])
        assert abs(seconds-count/FPS)<.04,(path,seconds)
        asset.update(bytes=path.stat().st_size,duration_seconds=seconds,sha256=sha256(path))
        print(f"Verified {path.name}: {count} frames, {seconds:.2f}s",flush=True)
    write_json(OUT/'asset_manifest.json',{'primary_220':primary,'videos':assets,'frame_count':COUNT,'fps':FPS,
        'scored_frames':COUNT-WARMUP,'scores_computed_before_video_compression':True})
    lines=['# Blog asset pack','','## Start here','',
        '- Compression stage: [autoencoder reconstruction](videos/autoencoder_45_60.mp4).',
        '- Opening demo: [45–60 second clip](videos/hero_45_60.mp4).',
        '- Main experiment: [32-anchor memory-only comparison](videos/memory_32_45_60.mp4).',
        '- Failure mechanism: [zero-anchor teacher versus rollout](videos/collapse_zero_45_60.mp4).',
        '- All budgets: [full comparison montage](videos/anchor_budget_full.mp4).',
        '- Copy-ready [Markdown tables](tables/all_tables.md); exact CSVs and per-frame data are alongside them.',
        '- Copy-ready [architecture pipeline](architecture.md) and [polarity diagnostic](tables/06_polarity_diagnostic.md).',
        '- Figures are supplied as PNG for embedding and SVG for editable vector output.','',
        '## Captions and figure placement','',
        '1. `01_anchor_budget_error`: Teacher-forced and free-rollout pixel error across anchor budgets. The model receives only 16 source frames in rollout. Scores cover frames 16–6572 at 384×512; one seed per budget.',
        '2. `02_memory_only_ablation`: Same learned anchors and decoder, with or without the temporal/history/fusion machinery. The anchors were jointly trained, so this is a post-hoc removal diagnostic, not a standalone memory-only training comparison.',
        '3. `03_error_accumulation`: Unsmoothed per-frame errors for 32 and 220 anchors. All curves use the same axes; no difficult frames are removed.',
        '4. `04_parameter_budget`: Actual saved predictor parameters. Frozen autoencoder parameters are excluded here and listed separately in the parameter table. Anchor reference buffers are not trainable parameters.',
        '', '## Figures', '',
        '![Teacher versus rollout by budget](figures/01_anchor_budget_error.png)', '',
        '![Memory-only removal comparison](figures/02_memory_only_ablation.png)', '',
        '![Per-frame error curves](figures/03_error_accumulation.png)', '',
        '![Parameter budget](figures/04_parameter_budget.png)',
        '', '## Video reading guide','',
        'Four-panel videos: source upper left; teacher-forced upper right; full rollout lower left; direct memory decoding lower right. Teacher sees true history every step. Rollout sees source frames 0–15 only (last source timestamp 0.5 s; first generated frame at 0.5333 s). Memory-only receives no source history. Warmup frames are excluded from scores for all methods.',
        'The 45–60 second clips reuse the original prototype interval. The 108–114 second clip is a deliberately selected known difficult transition, not an average example. Full-length videos are included to make this selection inspectable. All videos are silent.',
        '', '## Comparability and limits','',
        '- All six budget runs specify batch 16, learning rate 0.0003, 12 epochs, seed 7, maximum burn-in 128, maximum supervised rollout 32, and six memory-frozen epochs. This is still one run per budget, not a replicated statistical result.',
        ('- All six main checkpoints use separately fitted 96-knot time-only polarity splines. The 220 correction fits only those 96 scalars; temporal dynamics, memory and gates are unchanged. The raw pack is preserved at prototype_outputs/blog_assets_before_220_polarity. See [correction report](220_polarity_correction.md).'
         if primary=='anchors_220_polarity' else '- This pack uses the explicitly labelled 220 polarity variant; inspect source_summaries.json before comparing it with the smaller spline-calibrated checkpoints.'),
        '- `oracle_polarity_shape_error_percent` uses target polarity to separate silhouette error from global black/white inversion. It is a diagnostic, NOT an autonomous generation score. Main scores and videos use the learned head predictions, never oracle polarity. When the fitted head is correct, the diagnostic and main score coincide.',
        '- The memory baseline removes history, temporal prediction and fusion gates together. It does not isolate the U-Net alone and was not independently trained.',
        '- This is reconstruction of one known video with explicit time input, not evidence of general video generation.',
        '- Pixel error is the fraction of mismatched binary pixels. IoU is the framewise mean across black and white classes, with absent classes handled by the existing evaluator; it is not foreground-only IoU.',
        '- Reported means exclude the first 16 frames. Relative error reduction is 1 − full_error / memory_error; the accumulation gap is rollout_error − teacher_error.',
        '- Scores are fresh local reruns; small differences from saved Colab scores are recorded in validation.json. Do not mix old and new paired comparisons.',
        '- The old small-batch 220-anchor run is not part of the main matched-training table.',
        '', '## Publication','',
        'Keep Markdown tables, PNG/SVG figures, captions and source metadata in the public blog repository. Link or upload the chosen MP4 clips separately as appropriate; these local files have not been published. The private model repository and weights need not be shared. Credit the original Bad Apple video and the earlier GPT-2 attention projects in the article.',
        '', '## Reproduction','',
        'From the private project root, with the torch-gpu environment:',
        '```powershell','python -m neural_bad_apple.blog_export --anchors 220 32 55 110 16 0','python -m neural_bad_apple.blog_package','```','',
        'The exporter never trains or changes checkpoints. The separate polarity-calibration step fits only the time-based head and saves a new checkpoint; it does not retrain motion or memory. Existing finished exports are reused only when checkpoint identities match; interrupted videos are not silently overwritten. Inference reports are kept in `prototype_outputs/blog_work/<model>/report.md`.','',
        '## Exported videos','',markdown_table(['File','Purpose','Frames'],[[f"[{Path(a['file']).name}]({a['file']})",a['purpose'],a['frame_count']] for a in assets])]
    if primary=='anchors_220_polarity':
        lines.insert(lines.index('From the private project root, with the torch-gpu environment:'),
            'The one-time 220 head calibration was run before export (do not rerun over the saved corrected checkpoint):\n\n'
            '```powershell\npython prototype.py fix-polarity --checkpoint prototype_runs/anchors_220/model_best.pt --target-csv prototype_outputs/blog_work/anchors_220/metrics.csv --run-dir prototype_runs/anchors_220_polarity --device cuda\n```\n')
    if primary=='anchors_220_shared_polarity':
        lines.insert(lines.index('## Comparability and limits')+2,
            'The main 220 comparison uses the **already-trained 32-anchor polarity spline**, reused without fitting. Native 220 results remain available separately. The spline changes displayed black/white polarity only, not latent prediction, anchors or gates. Add 96 spline parameters to the 220 native predictor count for this variant.\n')
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'report.md').write_text('# Blog media export report\n\nInference from six checkpoints; 220 polarity was calibrated separately before export, without motion or memory retraining. '
        'Every video was decoded to verify frame count and duration. Tables use fresh full-resolution inference, not compressed previews. '
        'See README.md for captions, score definitions, polarity handling, limitations and reproduction.\n',encoding='utf-8')
    for directory, explanation in [
        ('videos','Silent, frame-aligned H.264 exports. See ../asset_manifest.json for frame counts, durations, timestamps and source model. All files were decoded for frame-count verification. The anchor-budget montage uses attached header bars and cell borders so labels stay visibly grouped with their panels.'),
        ('tables','Exact per-frame measurements and copy-ready Markdown summaries. All main scores exclude frames 0–15; polarity diagnostics are explicitly separate. Source checkpoint identities are in source_summaries.json.'),
        ('figures','PNG and SVG exports from reviewed tables. Chart contracts are in ../chart_contracts.json. Models share score denominators; polarity handling is documented.'),
    ]:
        (OUT/directory/'report.md').write_text('# Blog '+directory+' export\n\n'+explanation+'\n',encoding='utf-8')
    (WORK/'report.md').write_text('# Blog inference working files\n\nFresh inference from six existing checkpoints. No optimization is performed by the exporter. The separate 220 polarity calibration is documented in prototype_runs/anchors_220_polarity/report.md; raw outputs are retained. Each model folder contains its metrics, parameter counts and checkpoint hashes. Source-only and raw model streams are intermediates; publish selected comparisons from blog_assets instead.\n',encoding='utf-8')
    print(f'Blog asset pack ready at {OUT}',flush=True)


if __name__=='__main__': main()






