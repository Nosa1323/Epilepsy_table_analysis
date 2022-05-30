import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
import seaborn as sns
from statannot import add_stat_annotation


def histplot_fine_view(data, x, 
                        binwidth, binrange, 
                        ytick, xtick, 
                        ylim, xlim, 
                        ylabel, xlabel,
                        figname):
                        
    plt.figure(figsize=(7, 4))
    sns.set_theme(font_scale=2, style='ticks',context='notebook')
    fg = sns.histplot(data=data, x=x, hue = 'exp_group',stat='probability',
                        binwidth= binwidth, kde = True, binrange = binrange, 
                        line_kws=dict(linewidth=3))
    sns.despine()
    fg.spines['left'].set_linewidth(2)
    fg.spines['bottom'].set_linewidth(2)
    fg.yaxis.set_major_locator(ticker.MultipleLocator(ytick))
    fg.xaxis.set_major_locator(ticker.MultipleLocator(xtick))
    plt.ylim(ylim)
    plt.xlim(xlim)

    fg.set_ylabel(ylabel, fontsize=20)
    fg.set_xlabel(xlabel, fontsize=20)
    

    plt.legend(['ЭС','Контроль'], bbox_to_anchor=(1.1, 1.2))
    plt.tight_layout() 

    plt.savefig(f'figs/{figname}.tiff', dpi=600, transparent= True, bbox_inches='tight')     


def boxplot_fine_view(data, box_pairs, 
                        ylabel, ylim,
                        ytick, figname):
    plt.figure(figsize=(3, 5))
    sns.set_theme(font_scale=1.5, style="ticks",context='notebook')
    plot = sns.boxplot(data = data, 
                    palette="vlag",  linewidth = 3)
    
    add_stat_annotation(plot, data = data,
                    box_pairs=[box_pairs],
                    test='Mann-Whitney', text_format='star', loc='outside', 
                    verbose=1, comparisons_correction=None, linewidth=2)

    sns.swarmplot(data = data,  
              size = 9, palette = "Set2", linewidth=3 ,dodge = True)

    plot.set_xlabel(' ', fontsize=20)
    plt.xticks(np.arange(2), ('Контроль','ЭС'))
    plot.set_ylabel(ylabel, fontsize=20)
    plt.ylim(ylim)
    sns.despine()

    plot.spines['left'].set_linewidth(2)
    plot.spines['bottom'].set_linewidth(2)
    plot.yaxis.set_major_locator(ticker.MultipleLocator(ytick))


    plt.tight_layout()
    plt.savefig(f'figs/{figname}.tiff', dpi=600, transparent= True, bbox_inches='tight')


def adjust_box_widths(g, fac): # from https://stackoverflow.com/questions/56838187/how-to-create-spacing-between-same-subgroup-in-seaborn-boxplot
    """
    Adjust the withs of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


def boxplot_hue(data, x, y, hue, box_pairs, figname, big_tick, ylim, figsize, loc): 
    figname = f'figs/cell_count/{figname}.tif'
    fig = plt.figure(figsize=figsize) # меняет размер графика
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(font_scale=1.5, style="ticks",context="notebook", rc= custom_params)
    plot = sns.boxplot(data = data, x = x, 
                                    y = y, 
                                    hue = hue, 
                                    palette="vlag", 
                                    linewidth = 3, 
                                    dodge=True)
    
    plot.spines['left'].set_linewidth(2)
    plot.spines['bottom'].set_linewidth(2)
    plot.yaxis.set_major_locator(ticker.MultipleLocator(big_tick))
    plt.ylim(ylim)
    box_pairs = box_pairs
    ax, stat = add_stat_annotation(plot, data=data,x = x, 
                    y = y, hue = hue,
                    box_pairs=box_pairs,
                    test='Mann-Whitney', text_format='star',  
                    verbose=1, comparisons_correction=None, linewidth=2, loc = loc) #loc = 'outside'
    sns.swarmplot(x = x, y = y, 
                        hue = hue, data=data, 
                        size = 9, palette = "Set2", linewidth=3 ,dodge = True)
    handles, labels = plot.get_legend_handles_labels()
    plt.legend(handles[0:2], labels[0:2], loc=2, bbox_to_anchor=(1.03, 1), borderaxespad=0)
    adjust_box_widths(fig, 0.9)
    plot = plot.get_figure()
    plt.tight_layout()                                         
    plot.savefig(figname, dpi=800)
    return stat