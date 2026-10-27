"""BDAPP presentation figures from verified data. Maral Farhatnejad, 23012856.

Run from any directory: python source/generate_figures.py
Outputs: PNG (3840x2160), SVG, a vector PDF overview, captions and numeric sources.
Schematic drawings are explicitly labeled; no artificial neural observations are used.
"""
from pathlib import Path
import json
import textwrap
import hashlib
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, MaxNLocator
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
for folder in ['PNG', 'SVG', 'preview']:
    (ROOT / folder).mkdir(exist_ok=True)

INK = '#172C3A'
MUTED = '#5B6D79'
TEAL = '#18747B'
PURPLE = '#8061A8'
GOLD = '#BE8530'
CORAL = '#BA5A58'
GRAY = '#87949E'
PALE = '#E9EFF1'
GRID = '#E2E8EB'
WHITE = '#FFFFFF'
BLUE = '#427DA4'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 14, 'text.color': INK,
    'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#B9C4CA', 'axes.labelsize': 15,
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'pdf.fonttype': 42, 'svg.fonttype': 'path',
    'savefig.facecolor': WHITE, 'figure.facecolor': WHITE,
})
CATALOG, LAYOUT = [], []
BOOK = PdfPages(ROOT / 'BDAPP_Figure_Overview.pdf')


def wrap(s, width=110):
    return '\n'.join(textwrap.wrap(s, width=width, break_long_words=False))


def base(title, subtitle, source):
    fig = plt.figure(figsize=(16, 9))
    fig.text(.065, .943, title, fontsize=27, fontweight='bold', va='top')
    fig.text(.065, .875, subtitle, fontsize=16, color=MUTED, va='top', linespacing=1.35)
    fig.text(.065, .065, wrap(source, 156), fontsize=10.3, color=MUTED, va='top', linespacing=1.35)
    return fig


def flat(fig):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return ax


def clean(ax, grid='y'):
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=.8)
    ax.tick_params(length=0, pad=9)
    return ax


def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=1.7):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops={'arrowstyle': '-|>', 'color': color, 'lw': lw, 'mutation_scale': 15})


def text(ax, x, y, s, size=16, color=INK, bold=False, ha='left', va='top', **kw):
    return ax.text(x, y, s, fontsize=size, color=color,
                   fontweight='bold' if bold else 'normal', ha=ha, va=va, linespacing=1.4, **kw)


def save(fig, slug, section, caption, sources, recommended=False):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    clipped = []
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bbox = artist.get_window_extent(renderer)
        if bbox.x0 < -2 or bbox.y0 < -2 or bbox.x1 > bounds.x1 + 2 or bbox.y1 > bounds.y1 + 2:
            clipped.append(artist.get_text())
    LAYOUT.append({'figure': slug, 'text_outside_canvas': clipped})
    png_buffer = io.BytesIO()
    fig.savefig(png_buffer, format='png', dpi=240)
    (ROOT / 'PNG' / (slug + '.png')).write_bytes(png_buffer.getvalue())
    fig.savefig(ROOT / 'SVG' / (slug + '.svg'))
    BOOK.savefig(fig)
    CATALOG.append(dict(file=slug, section=section, caption=caption, sources=sources,
                        recommended_for_main_talk=recommended))
    plt.close(fig)
    print('Created', slug, flush=True)


FS = {e: pd.read_csv(DATA / f'{e}_hourly_features.csv') for e in ['FS369', 'FS437']}
for ex, d in FS.items():
    d['window_start'] = pd.to_datetime(d.window_start, utc=True)
    d['assessment_utc'] = pd.to_datetime(d.assessment_utc, utc=True)
SCORES = pd.read_csv(DATA / 'finalspark_primary_scores.csv')
BENCH = pd.read_csv(DATA / 'culture_age_pooled_scores.csv')
FOLDS = pd.read_csv(DATA / 'culture_age_fold_scores.csv')
PRED = pd.read_csv(DATA / 'culture_age_primary_forest_predictions.csv')
INV = pd.read_csv(DATA / 'culture_age_inventory.csv')
DANDI = pd.read_csv(DATA / 'dandi_descriptive.csv')
ZEN = pd.read_csv(DATA / 'zenodo_descriptive.csv')

# 01. Conceptual motivation. Connectivity is illustrative and labeled as a schematic.
fig = base('Can electrical activity help estimate culture age?',
           'Original research aim: estimate age and investigate remaining functional life.',
           'Conceptual schematic. Tissue shape, electrode contacts and connections are illustrative. Results and available labels determine the supported claims.')
ax = flat(fig)
rng = np.random.default_rng(11)
theta = np.linspace(0, 2*np.pi, 240)
r = .105 * (1 + .07*np.sin(5*theta) + .035*np.cos(9*theta))
ax.fill(.21 + r*np.cos(theta), .53 + 1.36*r*np.sin(theta), color='#F2E8EC', ec='#BA94A5', lw=2)
for x in [.15, .19, .23, .27]:
    for y in [.37, .40]:
        ax.add_patch(patches.Circle((x, y), .006, facecolor=GOLD, edgecolor='none'))
points = np.array([[.17,.53],[.21,.58],[.25,.54],[.21,.47],[.16,.61],[.26,.61],[.27,.47]])
for a,b in [(0,1),(1,2),(0,3),(2,3),(1,4),(1,5),(2,6),(3,6)]:
    ax.plot(points[[a,b],0], points[[a,b],1], color='#A77A95', lw=1.3)
ax.scatter(points[:,0],points[:,1],s=40,c='#A77A95',zorder=3)
text(ax,.21,.29,'Living neural culture',18,bold=True,ha='center')
text(ax,.21,.235,'MEA event recordings',15,color=MUTED,ha='center')
arrow(ax,.34,.53,.415,.53)
text(ax,.45,.64,'Describe the activity',21,bold=True)
text(ax,.45,.565,'Counts and channel activity\nTiming and bursts\nEnvironmental measurements',17)
text(ax,.45,.34,'Evaluate on held-out groups',16,color=TEAL,bold=True)
arrow(ax,.69,.53,.75,.53)
text(ax,.79,.655,'Age',22,color=TEAL,bold=True)
text(ax,.79,.59,'Requires a defined\nrecording-age label',16)
ax.plot([.79,.94],[.49,.49],color=GRID,lw=1.5)
text(ax,.79,.445,'Functional lifespan',20,color=PURPLE,bold=True)
text(ax,.79,.38,'Requires follow-up\nand endpoint labels',16)
save(fig,'01_research_goal','Motivation',
     'The project tests whether recorded electrical activity contains information about culture age. Remaining functional lifespan requires suitable longitudinal endpoint labels.', [], True)

# 02. Four tracks, deliberately not pooled.
fig = base('Four data tracks answer different questions',
           'Recording rows, clock windows and cultures represent different units of evidence.',
           'Sources: user-derived FinalSpark, DANDI 002020 and Zenodo 22755004 tables; KCL dataset DOI 10.18742/30394168. Event totals are detector-specific.')
ax = flat(fig)
xs=[.07,.25,.48,.72]
for x,s in zip(xs,['Data track','Material / format','Available data','Analysis role']):text(ax,x,.755,s,15,color=MUTED,bold=True)
rows=[
 ('FinalSpark','Aggregate human cultures\nHDF5 event exports','FS369: 94.15 million events\nFS437: 2.64 million events','Exploratory age and\nrecorded-endpoint models',TEAL),
 ('DANDI 002020','Dissociated rat cultures\nNWB detected-spike tables','129 sessions\n7 cultures','Descriptive activity\nacross recording sessions',BLUE),
 ('Zenodo 22755004','Human forebrain organoids\nMAT individual recordings','45 recordings\n16 experiment prefixes','Descriptive activity\nwith file-level quality checks',PURPLE),
 ('KCL culture time','Organoid-derived 2D cultures\nPublished well-average CSV','920 recordings / 216 wells\n6 lines / 15 line-batches','Days after dissociation\nGrouped age benchmark',GOLD),
]
for i,(name,material,amount,role,col) in enumerate(rows):
    y=.68-i*.135
    ax.plot([.065,.94],[y+.035,y+.035],color=GRID,lw=1)
    text(ax,xs[0],y,name,17,color=col,bold=True)
    text(ax,xs[1],y,material,15)
    text(ax,xs[2],y,amount,15)
    text(ax,xs[3],y,role,15)
text(ax,.07,.165,'The added age benchmark uses explicit DPD labels. The external descriptive tracks do not validate lifespan.',15,color=MUTED)
save(fig,'02_data_tracks','Data',
     'Four analysis tracks remain separate because their biological materials, recording units and available target labels differ. KCL DPD means days after dissociation.',
     ['FS369_hourly_features.csv','FS437_hourly_features.csv','dandi_descriptive.csv','zenodo_descriptive.csv','culture_age_inventory.csv'],True)

# 03. Two actual analysis paths.
fig=base('Two modeling paths share a reproducible workflow',
         'FinalSpark features come from event exports. The culture-age extension uses published five-minute well summaries.',
         'Methods: BDAPP hourly extractor and models; age extension analysis plan. Imputation and scaling fit only on training data.')
ax=flat(fig)
columns=[.07,.30,.53,.76]
for y,label,col,items in [
 (.66,'FinalSpark',TEAL,[('HDF5 event tables','Read-only, 100,000 rows\nper input chunk'),('Hourly features','Partition by clock hour\nSort within each channel'),('Training pipeline','Activity / environment\nMedian, Ridge, forest'),('Transfer evaluation','Train FS369\nEvaluate FS437')]),
 (.37,'Culture-age benchmark',PURPLE,[('Published source CSV','Join 920 record IDs\nto verified metadata'),('Six activity features','Keep measured summaries\nExclude age and identifiers'),('Training pipeline','Fold-specific preprocessing\nFixed model settings'),('Grouped evaluation','Hold out plates / batches\nThen donor cell lines')])]:
    text(ax,.07,y+.13,label,17,color=col,bold=True)
    for i,(heading,body) in enumerate(items):
        x=columns[i]
        ax.plot([x,x+.175],[y+.035,y+.035],color=col,lw=3)
        text(ax,x,y,heading,17,bold=True)
        text(ax,x,y-.063,body,14.5,color=MUTED)
        if i<3:arrow(ax,x+.185,y-.025,x+.215,y-.025,col)
text(ax,.07,.165,'Each run saves source audits, split memberships, predictions, metrics and software provenance.',16,color=INK)
save(fig,'03_processing_workflow','Methods',
     'The FinalSpark path extracts hourly event features, while the separate age extension uses author-computed five-minute well averages. Both use train-only preprocessing and explicit evaluation splits.',[],True)

# 04. Arithmetic labels versus biological endpoints.
fig=base('The FinalSpark targets follow the recorded experiment clock',
         'Calendar batch age and time to the recorded endpoint share the same assessment timestamp.',
         'Source: FinalSpark metadata. Endpoint causes: "Almost no activity" and "No activity". This schematic does not show independently verified individual deaths.')
ax=flat(fig)
ticks=[.12,.40,.64,.89]
ax.plot([ticks[0],ticks[-1]],[.48,.48],color=INK,lw=2)
for x,label in zip(ticks,['Batch creation','Recording start','Assessment time','Recorded endpoint']):
    ax.plot([x,x],[.46,.50],color=INK,lw=2)
    text(ax,x,.41,label,15,bold=True,ha='center')
arrow(ax,.12,.68,.64,.68,TEAL,2.5)
text(ax,.38,.745,'Batch calendar age',19,color=TEAL,bold=True,ha='center')
arrow(ax,.64,.59,.89,.59,PURPLE,2.5)
text(ax,.765,.655,'Time to recorded endpoint',17,color=PURPLE,bold=True,ha='center')
text(ax,.10,.28,'Age = assessment time − reported batch creation',17,color=TEAL)
text(ax,.10,.225,'Remaining recorded time = endpoint − assessment time',17,color=PURPLE)
save(fig,'04_target_definitions','Methods',
     'The FinalSpark labels are arithmetic experiment-level quantities. They sum to a constant within an experiment and do not provide independent confirmation of biological maturation or individual survival.',
     ['FS369_metadata.json','FS437_metadata.json'],True)

# 05. Actual feature definitions, without fabricated importance values.
fig=base('Activity features summarize several aspects of the recording',
         'These are measured event summaries. Their predictive value comes from held-out evaluation.',
         'FinalSpark hourly features: counts, channel distribution, amplitude, timing and fixed-ISI bursts. Environmental columns remain source-reported values.')
ax=flat(fig)
groups=[('01','Counts','Event count\nActive observed channels',TEAL),('02','Channel distribution','Entropy of channel shares\nLargest channel event share',BLUE),('03','Amplitude','Absolute median\nAbsolute interquartile range',PURPLE),
        ('04','Timing','Within-channel intervals\nCV, CV2 and short intervals',TEAL),('05','Channel bursts','At least 5 linked events\nIntervals at most 100 ms',BLUE),('06','Environment','Temperature, CO₂ and O₂\nExported stimulation logs',PURPLE)]
for i,(num,head,body,c) in enumerate(groups):
    x=.08+(i%3)*.305;y=.74-(i//3)*.285
    text(ax,x,y,num,15,color=c,bold=True)
    ax.plot([x+.045,x+.255],[y-.012,y-.012],color=GRID,lw=1)
    text(ax,x,y-.06,head,20,bold=True)
    text(ax,x,y-.125,body,16,color=MUTED)
save(fig,'05_feature_families','Methods',
     'Feature families describe event activity without identifying individual neurons. Fixed-ISI channel bursts are not a measure of network synchrony. No feature-importance ranking is implied.',[],False)

# 06. Actual public event times, never a fabricated raster.
events=pd.read_csv(DATA/'zenodo_fs363_org0_events.csv')
fig=base('A real event raster shows the input to activity analysis',
         'Zenodo fs363-org0, first 60 seconds. Each tick is one detected event on a source channel.',
         'Source: Zenodo 22755004, fs363-org0.mat. Actual spike times, no waveform data. The event-count panel uses one-second bins.')
ax=clean(fig.add_axes([.09,.42,.84,.34]),grid=None)
shown=events[events.time_s.lt(60)]
for c,g in shown.groupby('source_channel'):
    ax.vlines(g.time_s,c-.28,c+.28,color=TEAL,lw=.9)
ax.set_xlim(0,60);ax.set_ylim(-.7,7.7);ax.set_yticks(range(8));ax.set_ylabel('Source channel');ax.set_xticklabels([])
ax2=clean(fig.add_axes([.09,.20,.84,.16]))
counts,bins=np.histogram(shown.time_s,np.arange(61))
ax2.bar(bins[:-1],counts,width=1,align='edge',color=PURPLE,alpha=.85)
ax2.set_xlim(0,60);ax2.set_xlabel('Time within source recording (seconds)');ax2.set_ylabel('Events / bin')
save(fig,'06_real_event_raster','Methods',
     'A 60-second excerpt of actual detected events from one Zenodo organoid recording illustrates channel/time event data. One-second bin counts below are descriptive event counts, not identified-neuron firing rates.',
     ['zenodo_fs363_org0_events.csv'],False)

# 07. Missing records are observation uncertainty.
fig=base('Event-free export windows have unknown observation status',
         'The hourly grids contain 510 clock hours for FS369 and 138 for FS437.',
         'Source: FinalSpark hourly tables. Exported-event presence does not establish continuous recording exposure. Missing event records do not establish physiological silence.')
ax=clean(fig.add_axes([.20,.28,.63,.38]),grid='x')
for i,(ex,d) in enumerate(FS.items()):
    n=len(d);present=d.detected_events.notna().sum();fraction=present/n
    ax.barh(i,fraction,color=TEAL,height=.35)
    ax.barh(i,1-fraction,left=fraction,color='#D9DFE3',height=.35)
    ax.text(.025,i,f'{present} event-containing hours',va='center',color=WHITE,fontsize=18,fontweight='bold')
    ax.text(1.025,i,f'{n-present} unknown\nhour' + ('s' if n-present!=1 else ''),va='center',fontsize=15,color=MUTED)
ax.set_yticks([0,1],['FS369','FS437']);ax.invert_yaxis();ax.set_xlim(0,1)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:.0%}'))
ax.set_xlabel('Share of the clock-hour grid')
ax.legend(handles=[patches.Patch(color=TEAL,label='Exported events present'),patches.Patch(color='#D9DFE3',label='No exported events; exposure unknown')],
          loc='upper left',bbox_to_anchor=(0,1.30),frameon=False,ncol=1,fontsize=13)
save(fig,'07_event_record_missingness','Data quality',
     'FS369 has 509 event-containing hours and one hour without exported events. FS437 has 115 event-containing hours and 23 without. Unknown exposure prevents interpreting those gaps as silence.',
     ['FS369_hourly_features.csv','FS437_hourly_features.csv'],True)

# 08. Actual trajectories and measured environmental context.
fig=base('Activity and source-reported temperature differ between experiments',
         'Hourly exports retain missing values. Counts are not normalized by verified recording exposure.',
         'Source: user-derived FinalSpark hourly features. Gray shading marks absent exported values in the plotted quantity. Environmental differences do not establish causal effects.')
for j,(ex,d) in enumerate(FS.items()):
    x=(d.window_start-d.window_start.min()).dt.total_seconds()/86400
    c=TEAL if j==0 else PURPLE
    for k,(col,ylabel) in enumerate([('detected_events','Detected events / clock hour'),('temperature_mean','Temperature (source value)')]):
        ax=clean(fig.add_axes([.09+j*.46,.49-k*.29,.37,.23]))
        missing=d[col].isna()
        for t in x[missing]:ax.axvspan(t,t+1/24,color='#DCE2E5',zorder=0)
        ax.plot(x,d[col],color=c,lw=1.2)
        ax.set_xlim(0,float(x.max()+1/24))
        if k==0:
            ax.set_yscale('log');ax.set_ylim(10,1.2e6);ax.set_title(ex,loc='left',fontsize=17,color=c,fontweight='bold',pad=10)
        else:
            ax.set_ylim(33,38.5);ax.set_xlabel('Days since first exported event hour')
        if j==0:ax.set_ylabel(ylabel,fontsize=13)
        else:ax.tick_params(labelleft=False)
save(fig,'08_activity_and_temperature','Results: FinalSpark',
     'Actual hourly event counts and environmental measurements show markedly different experiment contexts. Gray bands identify missing values for each plotted quantity; they are not inactivity periods.',
     ['FS369_hourly_features.csv','FS437_hourly_features.csv'],False)

# 09. Non-overlapping age ranges and temperature distributions.
fig=base('FS437 falls outside the training age range',
         'The transfer test combines age extrapolation with a different acquisition and environmental context.',
         'Source: model-eligible FinalSpark hours, assessed at window end. Boxplots show medians, quartiles and full observed ranges. Temperature retains source-reported units.')
for j,(col,xlabel) in enumerate([('batch_calendar_age_days','Batch calendar age at assessment (days)'),('temperature_mean','Temperature (source value)')]):
    ax=clean(fig.add_axes([.16+j*.43,.31,.34,.35]),grid='x')
    for i,(ex,d) in enumerate(FS.items()):
        eligible=d[d.eligible_for_exploratory_model.astype(str).str.lower().eq('true')]
        a=eligible[col].dropna()
        b=ax.boxplot(a,positions=[i],vert=False,widths=.35,patch_artist=True,whis=(0,100),showfliers=False,
            boxprops={'facecolor':[TEAL,PURPLE][i],'alpha':.65,'edgecolor':[TEAL,PURPLE][i]},
            medianprops={'color':INK,'linewidth':2},whiskerprops={'color':[TEAL,PURPLE][i]},capprops={'color':[TEAL,PURPLE][i]})
        ax.text(float(a.median()),i-.34,f'{a.min():.1f} to {a.max():.1f}',ha='center',fontsize=13,color=[TEAL,PURPLE][i])
    ax.set_yticks([0,1],['FS369 train','FS437 test']);ax.invert_yaxis();ax.set_ylim(1.6,-.6);ax.set_xlabel(xlabel,fontsize=13)
    if j==0:ax.set_xlim(90,230)
    else:ax.set_xlim(32.5,39)
save(fig,'09_finalspark_dataset_shift','Results: FinalSpark',
     'The calendar-age ranges do not overlap between training and transfer evaluation. Temperature measurements also differ. These measured shifts are plausible contributors to poor transfer, but the plot does not establish their causal contribution.',
     ['FS369_hourly_features.csv','FS437_hourly_features.csv'],True)

# 10. All eight primary model configurations, local Windows scores.
names={'training_median':'Training median','count_only_ridge':'Count-only Ridge','rich_neural_ridge':'Activity Ridge','rich_neural_forest':'Activity forest',
       'environment_only_ridge':'Environment Ridge','neural_environment_ridge':'Activity + environment Ridge',
       'neural_environment_forest':'Activity + environment forest','neural_environment_stimulation_ridge':'Activity + environment + logs Ridge'}
fig=base('FinalSpark transfer remains poor across the primary models',
         'Train on 509 FS369 hours. Evaluate on 115 FS437 hours. Every primary transfer R² is negative.',
         'Source: final successful Windows run reported by the user, 27 October 2026. Two different experiment-level targets; neither measures individual biological lifespan.')
for j,(target,label) in enumerate([('batch_calendar_age_days','Batch calendar age'),('days_until_recorded_experiment_endpoint','Time to recorded experiment endpoint')]):
    d=SCORES[SCORES.target.eq(target)]
    ax=clean(fig.add_axes([.30+j*.35,.25,.29,.48]),grid='x')
    colors=[GRAY if m=='training_median' else (GOLD if m=='environment_only_ridge' else (TEAL if m in ['count_only_ridge','rich_neural_ridge','rich_neural_forest'] else PURPLE)) for m in d.model]
    ax.barh(np.arange(len(d)),d.MAE_days,color=colors,height=.58)
    ax.set_yticks(np.arange(len(d)),[names[m] for m in d.model] if j==0 else ['']*len(d))
    ax.invert_yaxis();ax.set_xlim(0,float(d.MAE_days.max())*1.18)
    for y,value in enumerate(d.MAE_days):ax.text(value+float(d.MAE_days.max())*.025,y,f'{value:.2f}',va='center',fontsize=12.5)
    ax.set_title(label,fontsize=14.5,fontweight='bold',pad=16)
    ax.set_xlabel('Mean absolute error (days)',fontsize=13)
save(fig,'10_finalspark_primary_model_results','Results: FinalSpark',
     'All eight primary hourly model configurations are shown. Some improve MAE relative to the training median, but all have negative transfer R². Remaining recorded experiment time is not validated individual survival.',
     ['finalspark_primary_scores.csv'],True)

# 11. Distribution of the labeled extension sample.
fig=base('The age benchmark has explicit labels at five culture times',
         '920 five-minute recordings from 216 repeated wells, six donor lines and 15 line-batch combinations.',
         'Source: KCL DOI 10.18742/30394168, development time course with MEA-NAP features and joined metadata. DPD = days post dissociation.')
ax=clean(fig.add_axes([.10,.25,.79,.49]))
ages=[6,15,22,29,37]
untreated=INV[INV.condition.isna()].groupby('age_DPD').size().reindex(ages,fill_value=0)
vehicle=INV[INV.condition.notna()].groupby('age_DPD').size().reindex(ages,fill_value=0)
pos=np.arange(5)
ax.bar(pos,untreated,width=.57,color=TEAL,label='No treatment recorded')
ax.bar(pos,vehicle,bottom=untreated,width=.57,color=GOLD,label='Vehicle control')
for i,(u,v) in enumerate(zip(untreated,vehicle)):ax.text(i,u+v+7,str(u+v),ha='center',fontsize=18,fontweight='bold')
ax.set_xticks(pos,[f'{a} DPD' for a in ages]);ax.set_ylim(0,265);ax.set_ylabel('Recording / well rows')
ax.legend(frameon=False,ncol=2,loc='upper right',fontsize=13)
save(fig,'11_labeled_age_dataset','Data: age extension',
     'The labeled extension samples five post-dissociation ages. Late observations are fewer and comprise vehicle controls, so row count and treatment composition change with age. Rows are repeated well observations.',
     ['culture_age_inventory.csv'],True)

# 12. Actual connected validation groups.
fig=base('Whole plate and batch groups stay outside training',
         'Eight connected groups prevent shared wells, plates and differentiation batches across each split.',
         'Source: age benchmark fold membership. Plates 41 and 42 share a line-batch and remain one group. The secondary donor-line test also excludes linked plates from training.')
groups=sorted(INV.validation_group.unique(),key=int)
ax=fig.add_axes([.25,.22,.44,.53])
matrix=np.zeros((8,8));np.fill_diagonal(matrix,1)
from matplotlib.colors import ListedColormap
ax.imshow(matrix,cmap=ListedColormap(['#B8DADB',PURPLE]),vmin=0,vmax=1,aspect='auto')
labels=[]
for group in groups:
    plates=sorted(INV.loc[INV.validation_group.eq(group),'plate_number'].unique())
    labels.append('Plate '+(' + '.join(map(str,plates))))
ax.set_yticks(range(8),labels);ax.set_xticks(range(8),[str(i+1) for i in range(8)])
ax.set_xlabel('Evaluation fold');ax.set_ylabel('Source plate group')
ax.set_xticks(np.arange(-.5,8,1),minor=True);ax.set_yticks(np.arange(-.5,8,1),minor=True)
ax.grid(which='minor',color=WHITE,lw=3);ax.tick_params(which='both',length=0,pad=10)
for spine in ax.spines.values():spine.set_visible(False)
legend=flat(fig)
legend.add_patch(patches.Rectangle((.75,.66),.023,.032,facecolor=PURPLE,edgecolor='none'))
text(legend,.79,.69,'Held-out group',16)
legend.add_patch(patches.Rectangle((.75,.57),.023,.032,facecolor='#B8DADB',edgecolor='none'))
text(legend,.79,.60,'Training groups',16)
text(legend,.75,.43,'216 repeated wells',18,bold=True)
text(legend,.75,.36,'15 line-batches',18,bold=True)
text(legend,.75,.29,'9 physical plates',18,bold=True)
save(fig,'12_grouped_validation','Methods: age extension',
     'Each fold excludes one connected plate/batch component. Plates 41 and 42 stay together. The matrix represents actual validation groups, not 920 independent biological replicates.',
     ['culture_age_inventory.csv'],True)

# 13. Main positive but modest benchmark result.
models=['training_median','firing_rate_ridge','activity_ridge','activity_forest']
model_labels=['Training median','Firing-rate Ridge','Six-feature Ridge','Six-feature forest']
fig=base('Activity features provide a modest culture-time prediction signal',
         'Primary evaluation: all 920 recording rows, with whole plate/batch groups held out.',
         'Source: verified culture-age benchmark. Target = days after dissociation from five-minute well summaries. Pooled cross-validation scores; no prospective external validation.')
d=BENCH[BENCH.protocol.eq('plate_and_batch_group_holdout')].set_index('model').loc[models]
for j,(col,title,limits) in enumerate([('MAE_days','Mean absolute error (days)',(0,9.6)),('R2','Pooled R²',(-.12,.4))]):
    ax=clean(fig.add_axes([.25+j*.36,.26,.29,.45]),grid='x')
    vals=d[col].to_numpy();colors=[GRAY,BLUE,PURPLE,TEAL]
    ax.barh(range(4),vals,color=colors,height=.55)
    ax.set_yticks(range(4),model_labels if j==0 else ['']*4);ax.invert_yaxis();ax.set_xlim(*limits)
    if col=='R2':ax.axvline(0,color=GRAY,lw=1)
    for y,value in enumerate(vals):
        if col=='MAE_days':ax.text(value+.14,y,f'{value:.2f}',va='center',fontsize=17,fontweight='bold' if y==3 else 'normal')
        else:ax.text(value+.012 if value>=0 else .015,y,f'{value:.3f}',va='center',fontsize=15)
    ax.set_xlabel(title)
fig.text(.25,.18,'Forest MAE is 18.7% lower than the training-median baseline.',fontsize=17,color=TEAL,fontweight='bold')
save(fig,'13_age_benchmark_main_result','Results: age extension',
     'The six-feature forest has pooled MAE 6.596 days and R² 0.302, versus median-baseline MAE 8.113 days. The improvement is modest and specific to days after dissociation in this dataset.',
     ['culture_age_pooled_scores.csv'],True)

# 14. All predeclared sensitivity protocols with honest populations.
fig=base('The age signal persists under donor-line exclusion',
         'The untreated-only check uses a narrower population and age range.',
         'Source: all three predeclared culture-age protocols. Dots show pooled MAE, not confidence intervals. Untreated-only errors are not directly comparable to the full-age test.')
ax=clean(fig.add_axes([.38,.26,.46,.45]),grid='x')
protocols=['plate_and_batch_group_holdout','unseen_cell_line_excluding_linked_plates','untreated_only_group_holdout']
labels=['Plate / batch holdout\n920 recordings, 6–37 DPD','Unseen donor line\n920 recordings, 6–37 DPD','Untreated-only holdout\n840 recordings, 6–29 DPD']
for i,p in enumerate(protocols):
    q=BENCH[BENCH.protocol.eq(p)].set_index('model')
    baseline=q.loc['training_median','MAE_days'];forest=q.loc['activity_forest','MAE_days']
    ax.plot([forest,baseline],[i,i],color='#CBD5DA',lw=3,zorder=1)
    ax.scatter([baseline],[i],s=120,color=GRAY,zorder=3)
    ax.scatter([forest],[i],s=140,color=TEAL,zorder=3)
    ax.text(forest-.10,i-.18,f'{forest:.2f}',fontsize=16,color=TEAL,fontweight='bold',ha='center')
    ax.text(baseline+.10,i+.26,f'{baseline:.2f}',fontsize=14,color=MUTED,ha='center')
ax.set_yticks(range(3),labels);ax.set_ylim(2.65,-.65);ax.set_xlim(5.3,8.8);ax.set_xlabel('Pooled mean absolute error (days after dissociation)')
ax.legend(handles=[plt.Line2D([],[],marker='o',color=GRAY,lw=0,label='Training median',markersize=8),plt.Line2D([],[],marker='o',color=TEAL,lw=0,label='Six-feature forest',markersize=8)],
          frameon=False,loc='upper center',bbox_to_anchor=(.5,1.26),ncol=2,fontsize=13)
save(fig,'14_age_benchmark_robustness','Results: age extension',
     'The forest MAE is 6.727 days under donor-line exclusion and 6.003 days on the untreated-only subset. The latter excludes all 37-DPD rows and therefore evaluates a different age range.',
     ['culture_age_pooled_scores.csv'],False)

# 15. Raw held-out prediction scatter, no artificial x jitter.
fig=base('Held-out predictions recover some culture-time structure',
         'Each point is a recording row. Repeated wells contribute observations at several sampled ages.',
         'Source: primary grouped forest, 920 out-of-fold predictions. Diamonds mark mean predictions at each age. The dashed line marks perfect prediction.')
ax=clean(fig.add_axes([.09,.20,.59,.57]),grid='both')
for age,group in PRED.groupby('age_DPD'):
    ax.scatter(group.age_DPD,group.predicted_age_DPD,s=20,color=TEAL,alpha=.20,edgecolors='none')
    ax.scatter([age],[group.predicted_age_DPD.mean()],s=95,marker='D',color=GOLD,edgecolors=WHITE,lw=1,zorder=4)
ax.plot([0,42],[0,42],color=INK,lw=1.4,ls='--');ax.set_xlim(0,42);ax.set_ylim(0,42)
ax.set_xticks([6,15,22,29,37]);ax.set_yticks([0,10,20,30,40])
ax.set_xlabel('Actual days after dissociation');ax.set_ylabel('Predicted days after dissociation')
a=flat(fig)
text(a,.75,.73,'6.60 days',29,color=TEAL,bold=True)
text(a,.75,.665,'Mean absolute error',16,color=MUTED)
text(a,.75,.55,'R² = 0.302',25,bold=True)
text(a,.75,.48,'Whole plate/batch\ngroups held out',16,color=MUTED)
text(a,.75,.32,'Accuracy remains limited\nat the youngest and\noldest sampled ages.',16,color=CORAL)
save(fig,'15_age_actual_vs_predicted','Results: age extension',
     'Actual out-of-fold forest estimates show a modest association with culture time and marked regression toward intermediate ages. The scatter preserves true discrete x values and does not jitter or clip estimates.',
     ['culture_age_primary_forest_predictions.csv'],True)

# 16. Calibration failures are not hidden behind the pooled score.
age_errors=pd.read_csv(DATA/'culture_age_age_specific_errors.csv')
d=age_errors[(age_errors.protocol=='plate_and_batch_group_holdout')&(age_errors.model=='activity_forest')].sort_values('age_DPD')
fig=base('Early ages are overestimated and late ages underestimated',
         'Mean signed error = predicted age minus actual age, using the primary grouped forest.',
         'Source: all primary forest predictions, grouped by actual DPD. This plot shows average bias, not confidence intervals. Pooled MAE alone does not describe calibration.')
ax=clean(fig.add_axes([.19,.25,.67,.48]),grid='x')
ax.axvline(0,color=GRAY,lw=1.5)
for i,row in enumerate(d.itertuples(index=False)):
    value=row.mean_signed_error_days;c=CORAL if value>0 else TEAL
    ax.barh(i,value,color=c,height=.5)
    ax.text(value+(.35 if value>0 else -.35),i,f'{value:+.2f} d',va='center',ha='left' if value>0 else 'right',fontsize=16,fontweight='bold')
ax.set_yticks(range(len(d)),[f'{a} DPD  (n={n})' for a,n in zip(d.age_DPD,d.recordings)]);ax.invert_yaxis();ax.set_xlim(-15,14)
ax.set_xlabel('Mean signed prediction error (days)')
save(fig,'16_age_prediction_bias','Limitations',
     'The mean signed error is +9.90 days at 6 DPD and -11.75 days at 37 DPD. This substantial early/late bias limits practical use despite positive pooled R².',
     ['culture_age_age_specific_errors.csv'],True)

# 17. DANDI stays a separate descriptive rat-culture track.
fig=base('DANDI adds longitudinal descriptive evidence from rat cultures',
         '129 recording sessions from seven cultures. Each point marks an available session.',
         'Source: DANDI 002020, user-processed NWB detected-spike tables. Rates use converter-reported nominal duration. Elapsed recording time is not culture age.')
ax=clean(fig.add_axes([.10,.23,.78,.49]),grid='both')
palette=[TEAL,PURPLE,GOLD,BLUE,CORAL,'#4A8E73','#7A6955']
for (culture,g),c in zip(DANDI.groupby('culture_id'),palette):
    g=g.sort_values('recording_elapsed_days')
    ax.plot(g.recording_elapsed_days,g.events_per_nominal_recording_second,'o-',color=c,lw=1.3,ms=4,label=str(culture))
ax.set_yscale('log');ax.set_xlabel('Days since each culture’s first processed recording')
ax.set_ylim(3,12000);ax.set_yticks([10,100,1000,10000])
ax.set_ylabel('Events per nominal recording second')
ax.legend(title='Rat culture ID',frameon=False,ncol=4,loc='upper center',bbox_to_anchor=(.5,1.18),fontsize=11,title_fontsize=12)
save(fig,'17_dandi_descriptive_activity','External descriptive data',
     'Rat-culture event activity varies across recording sessions. Nominal rates use source duration metadata, and days since the first processed session do not supply biological culture age or lifespan labels.',
     ['dandi_descriptive.csv'],False)

# 18. Zenodo summaries, using file-specific durations.
fig=base('Zenodo summarizes activity in 45 human organoid recordings',
         'Sixteen experiment prefixes may contain related recordings. The individual age/death labels remain unavailable.',
         'Source: Zenodo 22755004. Rates use file-declared durations and channel counts. fs417-org0 retains 33,096 event times while its mismatched amplitudes remain missing.')
for j,(col,xlabel) in enumerate([('mean_events_per_declared_channel_second','Events per declared channel-second'),('fraction_events_in_channel_bursts','Fraction of events in fixed-ISI channel bursts')]):
    ax=clean(fig.add_axes([.10+j*.46,.25,.36,.48]))
    vals=np.sort(ZEN[col].to_numpy());cdf=np.arange(1,len(vals)+1)/len(vals)
    ax.step(vals,cdf,where='post',color=TEAL if j==0 else PURPLE,lw=2)
    ax.scatter(vals,cdf,s=20,color=TEAL if j==0 else PURPLE,alpha=.7)
    ax.set_ylim(0,1.05);ax.set_xlabel(xlabel,fontsize=13)
    if j==0:ax.set_ylabel('Cumulative fraction of recordings')
    else:ax.set_yticklabels([])
save(fig,'18_zenodo_descriptive_activity','External descriptive data',
     'Empirical distributions summarize activity and channel bursts for 45 human organoid recordings. File-declared duration supports descriptive rate calculations but does not verify uninterrupted exposure or provide individual age/death labels.',
     ['zenodo_descriptive.csv'],False)

# 19. Honest project conclusion, all scientific scopes explicit.
fig=base('The evidence supports a modest culture-time benchmark',
         'The project combines reproducible data engineering with explicit limits on prediction claims.',
         'Source: verified BDAPP extraction/model checks and separate KCL culture-time benchmark. Neither study validates individual remaining functional lifespan.')
ax=flat(fig)
entries=[('Data pipeline','Large event exports processed and audited',TEAL),
         ('FinalSpark transfer','Primary models have negative R² on FS437',CORAL),
         ('Culture-time benchmark','Grouped forest MAE 6.60 DPD, pooled R² 0.302',TEAL),
         ('Remaining functional life','Needs suitable individual follow-up and endpoint labels',PURPLE)]
for i,(label,result,c) in enumerate(entries):
    y=.72-i*.13
    ax.add_patch(patches.Circle((.085,y-.017),.007,facecolor=c,edgecolor='none'))
    text(ax,.115,y,label,18,bold=True)
    text(ax,.41,y,result,17,color=c)
    ax.plot([.115,.93],[y-.075,y-.075],color=GRID,lw=1)
save(fig,'19_supported_conclusions','Conclusion',
     'A reproducible pipeline and modest within-study culture-time prediction are supported. Poor FinalSpark transfer, age-specific bias and missing individual functional endpoints limit broader claims.',[],True)

# 20. Supplementary feature ablation, not a causal importance chart.
ab=pd.read_csv(DATA/'neural_feature_ablations.csv')
fig=base('Feature ablations do not resolve the FinalSpark transfer problem',
         'Change in MAE after removing one feature family from the full activity Ridge model.',
         'Source: prespecified one-hour Ridge ablations, reference computation. Negative change means lower error on this inspected transfer set. This is not causal feature importance.')
for j,(target,label) in enumerate([('batch_calendar_age_days','Batch calendar age'),('days_until_recorded_experiment_endpoint','Time to recorded endpoint')]):
    d=ab[ab.target.eq(target)]
    values=d.MAE_increase_vs_full_neural_ridge_days
    ax=clean(fig.add_axes([.23+j*.38,.25,.30,.46]),grid='x')
    ax.barh(range(len(d)),values,color=[TEAL if v<0 else CORAL for v in values],height=.5)
    ax.axvline(0,color=GRAY,lw=1)
    labels=[m.replace('ablation_without_','').replace('distribution','channel distribution').capitalize() for m in d.model]
    ax.set_yticks(range(len(d)),labels if j==0 else ['']*len(d));ax.invert_yaxis()
    ax.set_xlabel('Change in MAE (days)',fontsize=13);ax.set_title(label,fontsize=15,pad=18)
    ax.margins(x=.25)
    for y,v in enumerate(values):ax.text(v+(.15 if v>=0 else -.15),y,f'{v:+.2f}',ha='left' if v>=0 else 'right',va='center',fontsize=12)
save(fig,'20_supplement_finalspark_ablations','Appendix',
     'Removing some families changes transfer MAE, sometimes reducing it. All compared transfer configurations still fail to establish reliable prediction, and the ablations do not measure causal biological importance.',
     ['neural_feature_ablations.csv'],False)

# 21. Fold variability, no invented confidence interval.
fig=base('Age-benchmark performance varies across held-out groups',
         'Each row is a full primary validation group. Every forest test-fold R² is positive in this benchmark.',
         'Source: eight primary plate/batch folds. These are fold-level errors, not independent well-level confidence intervals. Plates 41 and 42 form one group.')
ax=clean(fig.add_axes([.25,.23,.59,.53]),grid='x')
q=FOLDS[FOLDS.protocol.eq('plate_and_batch_group_holdout')].copy()
groups=sorted(q.fold.unique(),key=lambda s:int(s))
for i,g in enumerate(groups):
    d=q[q.fold.eq(g)].set_index('model')
    a,b=d.loc['activity_forest','MAE_days'],d.loc['training_median','MAE_days']
    ax.plot([a,b],[i,i],color='#CBD5DA',lw=2.5)
    ax.scatter([b],[i],s=65,color=GRAY,zorder=3)
    ax.scatter([a],[i],s=85,color=TEAL,zorder=3)
    ax.text(a-.12,i,f'{a:.2f}',ha='right',va='center',fontsize=12,color=TEAL)
ax.set_yticks(range(len(groups)),['Plates 41 + 42' if str(g)=='41' else 'Plate '+str(g) for g in groups]);ax.invert_yaxis()
ax.set_xlim(4.8,9);ax.set_xlabel('Mean absolute error (days after dissociation)')
ax.legend(handles=[plt.Line2D([],[],marker='o',color=GRAY,lw=0,label='Training median',markersize=7),plt.Line2D([],[],marker='o',color=TEAL,lw=0,label='Six-feature forest',markersize=7)],
          frameon=False,loc='upper center',bbox_to_anchor=(.5,1.16),ncol=2,fontsize=12)
save(fig,'21_supplement_age_fold_variability','Appendix',
     'Primary forest MAE ranges from about 5.51 to 7.55 DPD across held-out groups. The plot shows every fold and the corresponding median baseline, without treating repeated recordings as independent biological replicates.',
     ['culture_age_fold_scores.csv'],False)

# 22. Range shifts, directly computed from prepared model inputs.
drift=pd.read_csv(DATA/'feature_range_shift.csv')
pretty={'log_event_count':'Event count (log)','active_observed_channels':'Active channels','channel_count_entropy':'Channel-share entropy',
        'largest_channel_event_share':'Largest channel share','amplitude_abs_median_uv':'Median absolute amplitude','amplitude_abs_iqr_uv':'Amplitude IQR',
        'channel_interval_median_s':'Median channel interval','channel_interval_cv':'Channel interval CV','channel_interval_cv2':'Channel interval CV2',
        'fraction_channel_intervals_under_10ms':'Short channel intervals','log_burst_count':'Channel bursts (log)',
        'fraction_events_in_channel_bursts':'Events in channel bursts','temperature_mean':'Temperature','co2_mean':'CO₂','o2_mean':'O₂'}
fig=base('Many FS437 feature values lie outside the FS369 range',
         'Fraction of nonmissing test observations below the training minimum or above the training maximum.',
         'Source: hourly feature-range audit. An out-of-range fraction describes observed dataset shift; it does not identify its cause or establish biological deterioration.')
d=drift.sort_values('test_fraction_outside_train_range',ascending=True)
ax=clean(fig.add_axes([.30,.20,.55,.57]),grid='x')
values=d.test_fraction_outside_train_range.to_numpy()
ax.barh(range(len(d)),values,color=[GOLD if c in ['temperature_mean','co2_mean','o2_mean'] else TEAL for c in d.feature],height=.6)
ax.set_yticks(range(len(d)),[pretty[c] for c in d.feature],fontsize=11.5)
ax.set_xlim(0,1.10);ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:.0%}'))
ax.set_xlabel('Share of observed FS437 values outside the FS369 range',fontsize=13)
for y,v in enumerate(values):ax.text(v+.015,y,f'{v:.0%}',va='center',fontsize=10.5)
save(fig,'22_supplement_feature_range_shift','Appendix',
     'Observed FS437 values frequently exceed the training range for channel, amplitude and environmental features. Temperature is outside the source range for all nonmissing test observations.',
     ['feature_range_shift.csv'],False)

BOOK.close()
(ROOT/'figure_catalog.json').write_text(json.dumps(CATALOG,indent=2),encoding='utf-8')
(ROOT/'source/layout_check.json').write_text(json.dumps(LAYOUT,indent=2),encoding='utf-8')

# Overview images for visual QA and quick user browsing.
thumb_w,thumb_h=640,360
try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
except OSError:font=ImageFont.load_default()
for page in range(4):
    selected=CATALOG[page*6:(page+1)*6]
    canvas=Image.new('RGB',(thumb_w*2, (thumb_h+30)*3),WHITE)
    draw=ImageDraw.Draw(canvas)
    for i,item in enumerate(selected):
        im=Image.open(ROOT/'PNG'/(item['file']+'.png')).convert('RGB').resize((thumb_w,thumb_h),Image.Resampling.LANCZOS)
        x=(i%2)*thumb_w;y=(i//2)*(thumb_h+30)
        canvas.paste(im,(x,y))
        draw.text((x+15,y+thumb_h+3),item['file'].replace('_',' '),fill=INK,font=font)
    preview_buffer=io.BytesIO()
    canvas.save(preview_buffer,format='PNG')
    (ROOT/'preview'/f'overview_{page+1:02d}.png').write_bytes(preview_buffer.getvalue())
print('Completed',len(CATALOG),'figures. Text boundary warnings:',sum(bool(d['text_outside_canvas']) for d in LAYOUT),flush=True)
