import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_utils import get_binomial_data, get_bernoulli_data, get_tensor_data


def plot_posterior_ax(ax, ax_idxs, params_list, math_params_list, train_data, test_data, post_mean, q1, q2, 
    title, legend, palette, poisson=False, plot_training_points=False):

    x_train, y_train, n_samples, n_trials = get_binomial_data(train_data)
    x_test, y_test, n_samples, n_trials = get_binomial_data(test_data)

    n_params = len(params_list)
    axis = ax[ax_idxs[0]] if n_params==1 else ax[ax_idxs[1]]

    if n_params==1:

        if poisson:

            raise NotImplementedError
            # if plot_training_points:
            #     sns.scatterplot(x=x_train, y=y_train.flatten(), ax=axis, 
            #         label='Training', marker='.', color='black', alpha=0.8, legend=legend, palette=palette, linewidth=0)

            # sns.lineplot(x=x_test.flatten(), y=post_mean, ax=axis, label='Posterior', legend=legend, palette=palette)
            # axis.fill_between(x_test.flatten(), q1, q2, alpha=0.5)

            # axis.set_xlabel(math_params_list[0])
            # axis.set_ylabel('Satisfaction probability')
            # axis.set_title(title)

        else:
            if plot_training_points:
                sns.scatterplot(x=x_train_binomial.flatten(), y=y_train.flatten()/n_trials, ax=axis, 
                    label='Training', marker='.', color='black', alpha=alpha, legend=legend, palette=palette, linewidth=0)

            sns.lineplot(x=x_test.flatten(), y=post_mean, ax=axis, label='Posterior',  legend=legend, palette=palette)
            axis.fill_between(x_test.flatten(), q1, q2, alpha=0.5)

            axis.set_xlabel(math_params_list[0])
            axis.set_ylabel('Satisfaction probability')
            axis.set_title(title)

    elif n_params==2:

        p1, p2 = params_list[0], params_list[1]

        data = pd.DataFrame({p1:x_test[:,0],p2:x_test[:,1],'posterior_preds':post_mean})
        data[p1] = data[p1].apply(lambda x: format(float(x),".2f"))
        data[p2] = data[p2].apply(lambda x: format(float(x),".2f"))
        pivot_data = data.pivot(p1, p2, "posterior_preds")
        pivot_data = pivot_data.reindex(index=data[p1].drop_duplicates(), columns=data[p2].drop_duplicates())
        sns.heatmap(pivot_data, ax=axis, label=f'{title} posterior preds')
        axis.set_title(title)
        axis.set_xlabel(math_params_list[0])
        axis.set_ylabel(math_params_list[1])

    return ax

def plot_validation_ax(ax, params_list, math_params_list, test_data, palette, val_data=None, val_points=20, z=1.96):

    x_val, y_val, n_samples, n_trials = get_binomial_data(val_data)

    n_params = len(params_list)

    if n_params==1:

        if val_data is None:
            for idx in range(len(ax)):
                legend = 'auto' if idx==len(ax)-1 else None
                sns.lineplot(x=x_test.flatten(), y=Poisson_satisfaction_function(x_test).flatten(), ax=ax[idx], 
                    label='true satisfaction',  legend=legend, palette=palette)

        else:
            x_val, y_val_bernoulli = val_data['params'], val_data['labels']
            p = y_val_bernoulli.mean(1).flatten()

            sample_variance = [((param_y-param_y.mean())**2).mean() for param_y in y_val_bernoulli]
            std = np.sqrt(sample_variance).flatten()

            n = n_trials
            errors = (z*std)/np.sqrt(n)

            for idx in range(len(ax)):
                legend = 'auto' if idx==len(ax)-1 else None
                sns.scatterplot(x=x_val.flatten(), y=p.flatten(), ax=ax[idx], label='Validation', 
                    legend=legend, palette=palette,  s=15)
                # ax[idx].errorbar(x=x_val.flatten(), y=p.flatten(), yerr=errors, ls='None', label='Validation')

    elif n_params==2:

        axis = ax[0]
        p1, p2 = params_list[0], params_list[1]

        data = pd.DataFrame({p1:x_val[:,0],p2:x_val[:,1],'val_counts':y_val.flatten()/n_trials})
        data[p1] = data[p1].apply(lambda x: format(float(x),".2f"))
        data[p2] = data[p2].apply(lambda x: format(float(x),".2f"))
        pivot_data = data.pivot(p1, p2, "val_counts")
        pivot_data = pivot_data.reindex(index=data[p1].drop_duplicates(), columns=data[p2].drop_duplicates())
        sns.heatmap(pivot_data, ax=axis, label='Validation')
        axis.set_title("Validation set")
        axis.set_xlabel(math_params_list[0])
        axis.set_ylabel(math_params_list[1])

    return ax


def plot_posterior(params_list, math_params_list, train_data, test_data, post_mean, q1, q2, val_data=None, z=1.96,
    plot_training_points=False):

    palette = sns.color_palette("magma_r", 3)
    sns.set_style("darkgrid")
    sns.set_palette(palette)
    matplotlib.rc('font', **{'size':9, 'weight' : 'bold'})

    x_train, y_train, n_samples, n_trials = get_binomial_data(train_data)
    x_test, y_test, n_samples, n_trials = get_binomial_data(test_data)

    n_params = len(params_list)

    if n_params==1:

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        
        if plot_training_points:
            sns.scatterplot(x=x_train, y=y_train.flatten()/n_trials, ax=axis, 
                label='Training', marker='.', color='black', alpha=0.8, palette=palette, linewidth=0)

        if val_data is None:

            raise NotImplementedError

            # sns.lineplot(x=x_test.flatten(), y=post_mean, ax=ax, label='posterior')
            # ax.fill_between(x_test.flatten(), post_mean-z*post_std, post_mean+z*post_std, alpha=0.5)

            # sns.lineplot(x=x_test.flatten(), y=Poisson_satisfaction_function(x_test).flatten(), ax=ax, 
            #     label='true satisfaction')
            # sns.scatterplot(x=x_train_binomial.flatten(), y=y_train_binomial.flatten()/n_trials_train, ax=ax, 
            #     label='training points', marker='.', color='black')
            
        else:
            sns.lineplot(x=x_test.flatten(), y=post_mean, ax=ax, label='Posterior', palette=palette)
            ax.fill_between(x_test.flatten(), q1, q2, alpha=0.5)

            x_val, y_val, n_params, n_trials = get_tensor_data(train_data)
            p = y_val.mean(1).flatten()

            sample_variance = [((param_y-param_y.mean())**2).mean() for param_y in y_val]
            std = np.sqrt(sample_variance).flatten()
            errors = (z*std)/np.sqrt(n_trials)

            sns.scatterplot(x=x_val.flatten(), y=p.flatten(), ax=ax, label='Validation', palette=palette, s=15)
            # ax.errorbar(x=x_val.flatten(), y=p.flatten(), yerr=errors, ls='None', label='Validation')

    elif n_params==2:
        
        x_val, y_val, n_params, n_trials = get_binomial_data(train_data)

        params_couples_idxs = list(itertools.combinations(range(len(params_list)), 2))

        fig, ax = plt.subplots(len(params_couples_idxs), 2, figsize=(9, 4*len(params_couples_idxs)))

        for row_idx, (i, j) in enumerate(params_couples_idxs):

            p1, p2 = params_list[i], params_list[j]

            axis = ax[row_idx,0] if len(params_couples_idxs)>1 else ax[0]
            data = pd.DataFrame({p1:x_val[:,i],p2:x_val[:,j],'val_counts':y_val.flatten()/n_trials_val})
            data[p1] = data[p1].apply(lambda x: format(float(x),".4f"))
            data[p2] = data[p2].apply(lambda x: format(float(x),".4f"))
            data.sort_index(level=0, ascending=True, inplace=True)

            data = data.pivot(p1, p2, "val_counts")
            sns.heatmap(data, ax=axis, label='validation pts')
            axis.set_title("validation set")

            axis = ax[row_idx,1] if len(params_couples_idxs)>1 else ax[1]

            data = pd.DataFrame({p1:x_test[:,i],p2:x_test[:,j],'posterior_preds':post_mean})
            data[p1] = data[p1].apply(lambda x: format(float(x),".4f"))
            data[p2] = data[p2].apply(lambda x: format(float(x),".4f"))
            data.sort_index(level=0, ascending=True, inplace=True)
            data = data.pivot(p1, p2, "posterior_preds")
            sns.heatmap(data, ax=axis, label='posterior preds')
            axis.set_title("posterior preds")

    plt.tight_layout()
    plt.close()
    return fig
