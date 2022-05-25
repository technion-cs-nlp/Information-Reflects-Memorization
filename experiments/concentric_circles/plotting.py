import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center", fontsize=7) 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

def plot_circles(data, plt):
    circles0 = data.loc[data.label==0]
    plt.scatter(circles0['x'], circles0['y'])
    circles1 = data.loc[data.label==1]
    plt.scatter(circles1['x'], circles1['y'])

def plot_circles_3d(data):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d')
    circles0 = data.loc[data.label==0]
    circles1 = data.loc[data.label==1]
    ax.scatter(circles0['x'], circles0['y'], circles0['c1'])
    ax.scatter(circles1['x'], circles1['y'], circles1['c1'])

def plot_artifacts(data, plt):
    circles0 = data.loc[data.label==0]
    plt.scatter(circles0['c1'], circles0['c2'])
    plt.show()
    circles1 = data.loc[data.label==1]
    plt.scatter(circles1['c1'], circles1['c2'])
    plt.show()

def plot_activations(factors, division, activations, diff_type, to_save=False, save_suff="", is_bit=False):
    fig, axs = plt.subplots(nrows=len(factors), ncols=len(division),
                            gridspec_kw={'width_ratios':[1]*len(division)})
    fig.set_figheight(2*len(factors))
    fig.set_figwidth(4*len(division))

    for i, div in enumerate(division):
        for j, factor in enumerate(factors):
            acts = activations[factor][div][diff_type]
            if i==len(division)-1 and j==0:
                [[x00, y00], [x01, y01]] = axs[0, -1].get_position().get_points()
                [[x10, y10], [x11, y11]] = axs[-1, -1].get_position().get_points()
                pad = 0.1; width = 0.02
                cbar_ax = fig.add_axes([x11+pad, y10, width, y01-y10])
                g1 = sns.heatmap(acts, ax=axs[j, i], 
                                 cbar_ax=cbar_ax, cbar_kws={'ticks': [-1.0, 0.0, 1.0]},
                                 cmap="coolwarm",
                                 vmin=-1, vmax=1
                                 )
            else:
                g1 = sns.heatmap(acts, ax=axs[j, i], 
                                 cbar=False,
                                 cmap="coolwarm",
                                 vmin=-1, vmax=1
                                 )
            if j == 0:
                g1.set_title(f'Division: {div:.2f}', size=10)
            if i == 0:
                g1.set_ylabel(f'Factor:\n{factor:.2f}', size=10)
            if j != len(factors)-1:
                g1.set_xticks([])
            g1.set_yticks([])
    
    bit_name = "bit-" if is_bit else ""
    if diff_type == 'bias-no':
        plt.suptitle(f"Difference between {bit_name}activations from Bias and No-Bias sets", size=15)
    elif diff_type == 'bias-anti':
        plt.suptitle(f"Difference between {bit_name}activations from Bias and Anti-Bias sets", size=15)
    elif diff_type == 'anti-no':
        plt.suptitle(f"Difference between {bit_name}activations from Anti-Bias and No-Bias sets", size=15)
    
    plt.tight_layout(h_pad=1.08, w_pad=2.5)
    fig.subplots_adjust(top=0.92)
    if to_save:
        if not is_bit:
            plt.savefig(f'results/circles_{diff_type}_aligned_{save_suff}non-bit_activations_all-fact-divs_heatmaps.pdf')
        else:
            plt.savefig(f'results/circles_{diff_type}_aligned_{save_suff}activations_all-fact-divs_heatmaps.pdf')


def plot_interventions(factors, division, model_perf, to_save=False, save_suff="", is_bit=False):
    fig, axs = plt.subplots(nrows=len(factors), ncols=len(division),
                            gridspec_kw={'width_ratios':[1]*len(division)})
    fig.set_figheight(2*len(factors))
    fig.set_figwidth(4*len(division))

    for i, div in enumerate(division):
        for j, factor in enumerate(factors):
            perf = model_perf[factor][div]
            g1 = sns.barplot(data=pd.DataFrame(perf), x="Model", y="Performance", hue="Metric", 
                            ax=axs[j, i], palette="Set2")
            if j == 0:
                g1.set_title(f'Division: {div:.2f}', size=10)
            if i == 0:
                g1.set_ylabel(f'Factor: {factor:.2f}\nPerformance', size=10)
            else:
                g1.set_ylabel(None)
            if j != len(factors)-1:
                g1.set_xlabel(None)
            if j != 0 or i != len(division)-1:
                g1.get_legend().remove()
            if j != len(factors)-1:
                g1.set_xticks([])
    
    plt.suptitle("Controlling bias-encoding neurons", size=15)
    plt.tight_layout(h_pad=1.08, w_pad=2.5)
    fig.subplots_adjust(top=0.92)
    # show_values_on_bars(axs)
    if to_save:
        if not is_bit:
            plt.savefig(f'results/circles_control_bias_{save_suff}non-bit_activations_all-fact-divs_barplots.pdf')
        else:
            plt.savefig(f'results/circles_control_bias_{save_suff}activations_all-fact-divs_barplots.pdf')


def plot_intervention_variation(factors, division, means, nums_to_switch, activations, random=False, 
                                to_save=False, save_suff="", is_bit=False):
    fig, axs = plt.subplots(nrows=len(factors), ncols=len(division),
                            gridspec_kw={'width_ratios':[1]*len(division)})
    fig.set_figheight(2*len(factors))
    fig.set_figwidth(4*len(division))
    
    model_perf = {}
    for i, div in enumerate(division):
        for j, factor in enumerate(factors):
            model_perf[factor] = {}
            model_perf[factor][div] = {"Performance":[], "Metric":[], "Num Switched":[]}
            no_bias_data = pd.read_csv(f'data/circles/aligned_no_circles_factor_{factor}_means_{means[0]}_{means[1]}_{data_size}.csv')
            bias_data = pd.read_csv(f'data/circles/aligned_bias_circles_factor_{factor}_means_{means[0]}_{means[1]}_{data_size}.csv')
            anti_bias_data = pd.read_csv(f'data/circles/aligned_anti_bias_circles_factor_{factor}_means_{means[0]}_{means[1]}_{data_size}.csv')
            
            perf = model_perf[factor][div]
            for num_to_switch in nums_to_switch:
                model = pickle.load(
                    open(f'models/artifact_2/factor_{factor}_means_{means[0]}_{means[1]}_division_{div}_rat_{rat}_{model_data_size}/{model_name}/{seed}.pkl', "rb"))
                if num_to_switch!=0:
                    to_switch_off = activations[factor][div]['bias-no'].mean(0).argsort()[-num_to_switch:]
                    if random:
                        to_switch_off = [i for i in range(16) if i not in to_switch_off]
                    model = switch_off_neurons(model, to_switch_off)
                m_no_bias, m_bias, m_anti_bias \
                = model.score(no_bias_data[['x', 'y', 'c1']], no_bias_data[['label']]), \
                model.score(bias_data[['x', 'y', 'c1']], bias_data[['label']]), \
                model.score(anti_bias_data[['x', 'y', 'c1']], anti_bias_data[['label']])

                model_perf[factor][div]["Performance"].extend([m_no_bias, m_bias, m_anti_bias])
                model_perf[factor][div]["Metric"].extend(["No-Bias", "Bias", "Anti-Bias"])
                model_perf[factor][div]["Num Switched"].extend([num_to_switch]*3)
            g1 = sns.lineplot(data=pd.DataFrame(model_perf[factor][div]), 
                              x="Num Switched", y="Performance", hue="Metric", 
                              ax=axs[j, i], palette="Set2")
            if j == 0:
                g1.set_title(f'Division: {div:.2f}', size=10)
            if i == 0:
                g1.set_ylabel(f'Factor: {factor:.2f}\nPerformance', size=10)
            else:
                g1.set_ylabel(None)
            if j != len(factors)-1:
                g1.set_xlabel(None)
            if j != 0 or i != len(division)-1:
                g1.get_legend().remove()
            if j != len(factors)-1:
                g1.set_xticks([])
    
    plt.suptitle("Influence of Neuron Interventions", size=15)
    plt.tight_layout(h_pad=1.08, w_pad=2.5)
    fig.subplots_adjust(top=0.92)

    if to_save:
        if not is_bit:
            plt.savefig(f'results/circles_control_bias_{save_suff}non-bit_activations_all-fact-divs_barplots.pdf')
        else:
            plt.savefig(f'results/circles_control_bias_{save_suff}activations_all-fact-divs_barplots.pdf')


def plot_decision_boundary(data_file, model, ax):
    data = pd.read_csv(data_file)
    x, y, z = list(data['x']), list(data['y']), list(data['c1'])

    grid = np.meshgrid(np.arange(min(x), max(x), 0.1),
                       np.arange(min(y), max(y), 0.1),
                       np.arange(min(z), max(z), 0.1), sparse=False)
    xx, yy, zz = grid
    datapoints = {'x':[], 'y':[], 'c1':[]}
    xs, ys, zs = xx.shape[0], yy.shape[1], zz.shape[2]
    for i in range(xs):
        for j in range(ys):
            for k in range(zs):
                datapoints['x'].append(xx[i,j,k])
                datapoints['y'].append(yy[i,j,k])
                datapoints['c1'].append(zz[i,j,k])
    data_ = pd.DataFrame(datapoints)
    model_predicted = model.predict(data_)
    
    preds = model_predicted.reshape(xs, ys, zs)
    data_ = data_.values.reshape(xs, ys, zs, -1)
    coords = [[], [], []]
    for i in range(xs-1):
        for j in range(ys-1):
            for k in range(zs-1):
                if abs(preds[i,j,k+1]-preds[i,j,k]) == 1 \
                or abs(preds[i,j+1,k]-preds[i,j,k]) == 1 \
                or abs(preds[i+1,j,k]-preds[i,j,k]) == 1:
                    c = data_[i,j,k]
                    coords[0].append(c[0])
                    coords[1].append(c[1])
                    coords[2].append(c[2])
    
    circles0 = data.loc[data.label==0]
    circles1 = data.loc[data.label==1]

    ax.plot_trisurf(coords[0], coords[1], coords[2], color='red')

    ax.scatter(circles0['x'], circles0['y'], circles0['c1'], 
               alpha=0.03,
               color='tab:orange')
    ax.scatter(circles1['x'], circles1['y'], circles1['c1'], 
               alpha=0.03, 
               color='tab:blue')
    
    ax.title.set_text(model.score(data[['x', 'y', 'c1']], data[['label']]))
