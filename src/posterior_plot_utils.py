import itertools
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_utils import get_binomial_data, get_bernoulli_data, get_tensor_data


def plot_posterior_ax(ax, ax_idxs, params_list, math_params_list, train_data, test_data, post_mean, q1, q2, 
    title, legend, palette, plot_training_points=False):

    x_train, y_train, n_samples, n_trials = get_binomial_data(train_data)
    x_test, y_test, n_samples, n_trials = get_binomial_data(test_data)

    n_params = len(params_list)
    axis = ax[ax_idxs[0]] if n_params==1 else ax[ax_idxs[1]]

    if n_params==1:

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
        data[p1] = data[p1].apply(lambda x: format(float(x),".3f"))
        data[p2] = data[p2].apply(lambda x: format(float(x),".3f"))
        pivot_data = data.pivot(p1, p2, "posterior_preds")
        pivot_data = pivot_data.reindex(index=data[p1].drop_duplicates(), columns=data[p2].drop_duplicates())
        sns.heatmap(pivot_data, ax=axis, label=f'{title} posterior preds')
        axis.set_title(title)
        axis.set_xlabel(math_params_list[0])
        axis.set_ylabel(math_params_list[1])

    return ax

def plot_validation_ax(ax, params_list, math_params_list, test_data, palette, val_data=None, z=1.96,
    plot_validation_ci=True, val_points_ci=30):

    x_val, y_val, n_samples, n_trials_val = get_binomial_data(val_data)
    n_params = len(params_list)

    if n_params==1:

        if val_data is None:
            for idx in range(len(ax)):
                legend = 'auto' if idx==len(ax)-1 else None
                sns.lineplot(x=x_test.flatten(), y=Poisson_satisfaction_function(x_test).flatten(), ax=ax[idx], 
                    label='true satisfaction',  legend=legend, palette=palette)

        else:

            x_val, y_val_bernoulli = val_data['params'], val_data['labels']

            if plot_validation_ci:
                idxs = np.linspace(0,n_samples-1,val_points_ci).astype(int)
                x_val = x_val[idxs]
                y_val_bernoulli = y_val_bernoulli[idxs]

                p = y_val_bernoulli.mean(1).flatten()
                sample_variance = [((param_y-param_y.mean())**2).mean() for param_y in y_val_bernoulli]
                std = np.sqrt(sample_variance).flatten()
                errors = (z*std)/np.sqrt(n_trials_val)

                for idx in range(len(ax)):
                    legend = 'auto' if idx==len(ax)-1 else None
                    sns.scatterplot(x=x_val.flatten(), y=p.flatten(), ax=ax[idx], label='Test', 
                        legend=legend, palette=palette,  s=15)
                    ax[idx].errorbar(x=x_val.flatten(), y=p.flatten(), yerr=errors, ls='None', elinewidth=1,
                        label='Test')

            else:
                p = y_val_bernoulli.mean(1).flatten()

                for idx in range(len(ax)):
                    sns.scatterplot(x=x_val.flatten(), y=p.flatten(), ax=ax[idx], label='Test', palette=palette, s=15)

    elif n_params==2:

        axis = ax[0]
        p1, p2 = params_list[0], params_list[1]

        data = pd.DataFrame({p1:x_val[:,0],p2:x_val[:,1],'val_counts':y_val.flatten()/n_trials_val})
        data[p1] = data[p1].apply(lambda x: format(float(x),".3f"))
        data[p2] = data[p2].apply(lambda x: format(float(x),".3f"))
        pivot_data = data.pivot(p1, p2, "val_counts")
        pivot_data = pivot_data.reindex(index=data[p1].drop_duplicates(), columns=data[p2].drop_duplicates())
        sns.heatmap(pivot_data, ax=axis, label='Test')
        axis.set_title("Test set")
        axis.set_xlabel(math_params_list[0])
        axis.set_ylabel(math_params_list[1])

    return ax


def plot_posterior(params_list, math_params_list, train_data, test_data, post_mean, q1, q2, val_data=None, z=1.96,
    plot_training_points=False, plot_validation_ci=False, val_points_ci=30):

    palette = sns.color_palette("magma_r", 3)
    sns.set_style("darkgrid")
    sns.set_palette(palette)
    matplotlib.rc('font', **{'size':9, 'weight' : 'bold'})

    x_train, y_train, n_samples, n_trials_train = get_binomial_data(train_data)
    x_test, y_test, n_samples, n_trials_test = get_binomial_data(test_data)
    x_val, y_val, n_samples, n_trials_val = get_binomial_data(val_data)

    n_params = len(params_list)

    if n_params==1:

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        
        if plot_training_points:
            sns.scatterplot(x=x_train, y=y_train.flatten()/n_trials_train, ax=axis, 
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
            
            x_val, y_val_bernoulli = val_data['params'], val_data['labels']

            if plot_validation_ci:
                idxs = np.linspace(0,n_samples-1,val_points_ci).astype(int)
                x_val = x_val[idxs]
                y_val_bernoulli = y_val_bernoulli[idxs]

                p = y_val_bernoulli.mean(1).flatten()
                sample_variance = [((param_y-param_y.mean())**2).mean() for param_y in y_val_bernoulli]
                std = np.sqrt(sample_variance).flatten()
                errors = (z*std)/np.sqrt(n_trials_val)

                sns.scatterplot(x=x_val.flatten(), y=p.flatten(), ax=ax, label='Test', palette=palette, s=15)
                ax.errorbar(x=x_val.flatten(), y=p.flatten(), yerr=errors, ls='None', elinewidth=1, label='Test')

            else:
                p = y_val_bernoulli.mean(1).flatten()
                
                sns.scatterplot(x=x_val.flatten(), y=p.flatten(), ax=ax, label='Test', palette=palette, s=15)

    elif n_params==2:
        
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        p1, p2 = params_list[0], params_list[1]

        data = pd.DataFrame({p1:x_val[:,0],p2:x_val[:,1],'val_counts':y_val.flatten()/n_trials_val})
        data[p1] = data[p1].apply(lambda x: format(float(x),".3f"))
        data[p2] = data[p2].apply(lambda x: format(float(x),".3f"))
        pivot_data = data.pivot(p1, p2, "val_counts")
        pivot_data = pivot_data.reindex(index=data[p1].drop_duplicates(), columns=data[p2].drop_duplicates())
        sns.heatmap(pivot_data, ax=ax[0], label='test pts')
        ax[0].set_title("Test set")
        ax[0].set_xlabel(math_params_list[0])
        ax[0].set_ylabel(math_params_list[1])

        data = pd.DataFrame({p1:x_test[:,0],p2:x_test[:,1],'posterior_preds':post_mean})
        data[p1] = data[p1].apply(lambda x: format(float(x),".3f"))
        data[p2] = data[p2].apply(lambda x: format(float(x),".3f"))
        pivot_data = data.pivot(p1, p2, "posterior_preds")
        pivot_data = pivot_data.reindex(index=data[p1].drop_duplicates(), columns=data[p2].drop_duplicates())
        sns.heatmap(pivot_data, ax=ax[1], label='posterior preds')
        ax[1].set_title("Posterior preds")
        ax[1].set_xlabel(math_params_list[0])
        ax[1].set_ylabel(math_params_list[1])

    plt.tight_layout()
    plt.close()
    return fig
