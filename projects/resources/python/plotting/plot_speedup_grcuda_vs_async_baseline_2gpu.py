import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns              
import matplotlib.ticker as tkr
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import os
import matplotlib.lines as lines
import matplotlib.ticker as ticker
from plot_utils import *


# INPUT_DATE = "2020_09_19_grcuda"
OUTPUT_DATE = "2020_10_14"
PLOT_DIR = "../../../../grcuda-data/plots"

BENCHMARK_NAMES = {"b1": "Vector Squares", "b5": "B&S", "b8": "Images", "b6": "ML Ensemble", "b7": "HITS", "b10": "DL"}

PALETTE_GW = [COLORS[r] for r in ["b1","b2","b3", "b4", "b5"]]
HATCHES = ['', '/'*4, '\\'*4, '++++', '**']

if __name__ == "__main__":

    data = pd.read_csv("2GPU_allParents_vs_1GPU_Async.csv", sep=';')
    singleGPU = data['number_GPU']==1
    data.loc[singleGPU,'parent_stream_policy'] = ['Baseline Async 1 GPU']*len(data[singleGPU])

    # sns.set_theme(style="whitegrid")
    sns.set_style("whitegrid", {"ytick.left": True})
    plt.rcParams["font.family"] = ["serif"]
    plt.rcParams["font.size"] = 12
    plt.rcParams['hatch.linewidth'] = 0.6
    plt.rcParams['axes.labelpad'] = 5 
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    conf = data['parent_stream_policy'].unique()
    ylabels = {'computation_sec':"Computation Time [s]"}
    ylims = {}

    for var in ["computation_sec"]: 
        g = sns.catplot(data=data, kind='bar', ci=99, x="size",
        y=var, hue='parent_stream_policy', row='benchmark',
        alpha=1, palette=PALETTE_GW, height=2.5, aspect=5, legend_out=False, 
        sharey=False, sharex=False, margin_titles=True)
        g.set_axis_labels("Input Size", ylabels[var])
        #g.despine(left=True)
        #  .set_titles("{col_name}")
        # .set_yticklabels(list(range(100,1000,100))+list(range(1000,10000,1000))+list(range(10000, 75000, 10000)))

        for i,axes in enumerate(g.axes):  
            for ii,ax in enumerate(axes):
        #         ax.set(ylim=ylims[var][i])
        #         plt.sca(axes[ii])
        #     #if ii!=0:
        #         a = list(range(0, ylims[var][i][1]+1,(ylims[var][i][1]+1)//4))
        #         print(a)
        #         plt.yticks(a)
        #         if ii!=0:
        #             plt.yticks(a, ['']*5)
        #         g.set_axis_labels("Input Size", ylabels[var])
                for j, bar in enumerate(ax.patches):
                    bar.set_hatch(HATCHES[(j//5)])
                    # bar.set_edgecolor('k')
                    
                # for j, bar in enumerate([p for p in ax.patches if not pd.isna(p)]):
                #     bar.set_hatch(HATCHES[j // len(axes)])
        #     ax.yaxis.set_minor_locator(tkr.LogLocator(base=10, subs='all'))
        #     ax.yaxis.set_minor_formatter(tkr.NullFormatter())
        # if var == 'Wall_train':
        #     g.set(yscale='log')
            
        # Add legend;
        g.legend.remove()  # Remove the existing legend again;
        custom_lines = [Patch(facecolor=PALETTE_GW[0], hatch=HATCHES[0], edgecolor="w", label=conf[0]),
                        Patch(facecolor=PALETTE_GW[1], hatch=HATCHES[1], edgecolor="w", label=conf[1]),
                        Patch(facecolor=PALETTE_GW[2], hatch=HATCHES[2], edgecolor="w", label=conf[2]),
                        Patch(facecolor=PALETTE_GW[3], hatch=HATCHES[3], edgecolor="w", label=conf[3]), 
                        Patch(facecolor=PALETTE_GW[4], hatch=HATCHES[4], edgecolor="w", label=conf[4])] 
                        
        legend_data = {a:b for a,b in zip(conf,custom_lines)}
        g.add_legend(legend_data, loc="center left", bbox_to_anchor=(0., 0.6), fontsize=11, ncol=1, handletextpad=0.2, columnspacing=0.4, fancybox=True)
        g.legend.set_title("Parent Stream Policy")
        #g._legend_box.align = "left"
        g.figure.suptitle("2 GPU All Parent Policies vs 1 GPU Async Baseline")
        g.legend.get_frame().set_facecolor('white')
        plt.subplots_adjust(left=0.07, bottom=0.065, right=0.98, top=0.96, hspace=0.2, wspace=0.14)
        # plt.savefig(f"Total{var}.pdf")
        plt.show()

