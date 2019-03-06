"""This module contains useful plotting functionality.
"""
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from predict import predict

def plot_grid(n_rows, n_cols):
    fig_size = 8
    return plt.subplots(
        nrows=n_rows, 
        ncols=n_cols, 
        figsize=(
            fig_size*n_cols, 
            fig_size*n_rows
        )
    )

def plot_mu_t(
        t_grid,
        prior_components,
        posterior_components,
        arrival_time,
        linestyle):

    _, axs = plot_grid(2, 2)
    prior_ax = axs[0][0]
    prior_component_ax = axs[1][0]
    posterior_ax = axs[0][1]
    posterior_component_ax = axs[1][1]

    prior = reduce(np.add, prior_components).reshape(t_grid.shape[0])
    prior_ax.plot(t_grid, prior)
    prior_ax.set_title(r'$\mu_t$ prior')
    prior_ax.set_xlabel('Seconds')
    prior_ax.set_ylabel('Density')
    prior_ax.axvline(x=arrival_time, color='red', label='True arrival time')
    for i, component in enumerate(prior_components):
        prior_component_ax.plot(
            t_grid, component.reshape(len(t_grid), -1),
            label='Trajectory ' + str(i),
            linestyle=linestyle(i)
        )
    prior_component_ax.set_title(r'$\mu_t$ Prior Components')
    prior_component_ax.set_xlabel('Seconds')
    prior_component_ax.set_ylabel('Density')
    prior_component_ax.axvline(x=arrival_time, color='red', label='True arrival time')

    posterior = reduce(np.add, posterior_components).reshape(t_grid.shape[0])
    posterior_ax.plot(t_grid, posterior)
    posterior_ax.set_title(r'$\mu_t$ posterior')
    posterior_ax.set_xlabel('Seconds')
    posterior_ax.set_ylabel('Density')
    posterior_ax.axvline(x=arrival_time, color='red', label='True arrival time')
    for i, component in enumerate(posterior_components):
        posterior_component_ax.plot(
            t_grid, component.reshape(len(t_grid), -1),
            label='Trajectory ' + str(i),
            linestyle=linestyle(i)
        )
    posterior_component_ax.set_title(r'$\mu_t$ Posterior Components')
    posterior_component_ax.set_xlabel('Seconds')
    posterior_component_ax.set_ylabel('Density')
    posterior_component_ax.axvline(x=arrival_time, color='red', label='True arrival time')


def plot_model_probabilities(
        tau_grid,
        model_probs,
        cum_model_probs,
        linestyle):

    _, axs = plot_grid(1, 2)
    model_prob_ax = axs[0]
    cum_model_prob_ax = axs[1]

    seq = enumerate(zip(model_probs, cum_model_probs))
    for i, (model_prob, cum_model_prob) in seq:
        model_prob_ax.set_title('Model probabilities')
        model_prob_ax.set_xlabel(r'$\tau$')
        model_prob_ax.set_ylabel(r'$p(\mathcal{m}_k | X_{obs}')
        model_prob_ax.plot(
            tau_grid, model_prob,
            label='Trajectory' + str(i),
            linestyle=linestyle(i)
        )

        cum_model_prob_ax.set_title(r'Cumulative model probabilities')
        cum_model_prob_ax.set_xlabel(r'$\tau$')
        cum_model_prob_ax.set_ylabel(r'$p(\mathcal{m}_k | X_{obs})')
        cum_model_prob_ax.plot(
            tau_grid, cum_model_prob,
            label='Trajectory' + str(i),
            linestyle=linestyle(i)
        )


def plot_state_model_means(
        models, seg,
        progression,
        linestyle):

    _, axs = plot_grid(3, 2)
    tau_pred_ax = axs[0][0]
    time_left_pred_ax = axs[0][1]
    pos_x_pred_ax = axs[1][0]
    pos_y_pred_ax = axs[1][1]
    vel_x_pred_ax = axs[2][0]
    vel_y_pred_ax = axs[2][1]

    pos = seg[['x', 'y']].values.reshape(-1, 2)
    for i, model in enumerate(models):
        tau, _ = predict(model.g, pos)
        tau_pred_ax.plot(
            progression, tau.T[0],
            label='Trajectory ' + str(i),
            linestyle=linestyle(i)
        )

        time_left, _ = predict(model.h, tau)
        time_left_pred_ax.plot(
            tau, time_left,
            label='Trajectory ' + str(i),
            linestyle=linestyle(i)
        )

        p_x, _ = predict(model.f_p_x, tau)
        pos_x_pred_ax.plot(
            tau, p_x,
            label='Trajectory ' + str(i),
            linestyle=linestyle(i)
        )

        p_y, _ = predict(model.f_p_y, tau)
        pos_y_pred_ax.plot(
            tau, p_y,
            label='Trajectory ' + str(i),
            linestyle=linestyle(i)
        )

        v_x, _ = predict(model.f_v_x, tau)
        vel_x_pred_ax.plot(
            tau, v_x,
            label='Trajectory ' + str(i),
            linestyle=linestyle(i)
        )

        v_y, _ = predict(model.f_v_y, tau)
        vel_y_pred_ax.plot(
            tau, v_y,
            label='Trajectory ' + str(i),
            linestyle=linestyle(i)
        )

    tau_pred_ax.set_title(r'Predicted $\tau$ of input data')
    tau_pred_ax.set_xlabel('Normalised progression')
    tau_pred_ax.set_ylabel(r'$\tau$')
    tau_pred_ax.plot(
        progression, seg.tau,
        label='True trajectory',
        color='red'
    )

    time_left_pred_ax.set_title(r'Predicted $t$ of input data')
    time_left_pred_ax.set_xlabel(r'$\tau$')
    time_left_pred_ax.set_ylabel(r'$t$')
    time_left_pred_ax.plot(
        seg.tau, seg.time_left,
        label='True trajectory',
        color='red'
    )

    pos_x_pred_ax.set_title(r'Predicted $x$-position of input data')
    pos_x_pred_ax.set_xlabel(r'$\tau$')
    pos_x_pred_ax.set_ylabel(r'$p_x$')
    pos_x_pred_ax.plot(
        seg.tau, seg.x,
        label='True trajectory',
        color='red'
    )

    pos_y_pred_ax.set_title(r'Predicted $y$-position of input data')
    pos_y_pred_ax.set_xlabel(r'$\tau$')
    pos_y_pred_ax.set_ylabel(r'$p_y$')
    pos_y_pred_ax.plot(
        seg.tau, seg.y,
        label='True trajectory',
        color='red'
    )

    vel_x_pred_ax.set_title(r'Predicted $x$-velocity of input data')
    vel_x_pred_ax.set_xlabel(r'$\tau$')
    vel_x_pred_ax.set_ylabel(r'$v_x$')
    vel_x_pred_ax.plot(
        seg.tau, seg.dx,
        label='True trajectory',
        color='red'
    )

    vel_y_pred_ax.set_title(r'Predicted $y$-velocity of input data')
    vel_y_pred_ax.set_xlabel(r'$\tau$')
    vel_y_pred_ax.set_ylabel(r'$v_y$')
    vel_y_pred_ax.plot(
        seg.tau, seg.dy,
        label='True trajectory',
        color='red'
    )
