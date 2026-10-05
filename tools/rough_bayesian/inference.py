"""Joint noncentered MCMC: full latent innovations and model parameters.

All expensive recurrences and return likelihoods run as native Numba code.
ESS retains the Gaussian innovation prior exactly. Parameter MH includes the
parameter prior, uses symmetric proposals, and stops adapting after warmup.
The PGAS arm adds an exact terminal-innovation-block conditional update to ESS.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from kernels import coefficients, grid, quadrature


@njit(cache=True, nogil=True)
def log_likelihood(y, logv, noise, rho):
    total = 0.
    conditional = 1 - rho * rho
    for t in range(y.size):
        if not -60 < logv[t] < 60:
            return -np.inf
        residual = y[t] * math.exp(-logv[t] / 2) - rho * noise[t]
        total -= .5 * (logv[t] + math.log(conditional) + residual * residual / conditional)
    return total


@njit(cache=True, nogil=True)
def student_likelihood(y, logv, nu):
    constant = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - .5 * math.log((nu - 2) * math.pi)
    total = 0.
    for t in range(y.size):
        if not -60 < logv[t] < 60:
            return -np.inf
        total += constant - .5 * logv[t] - .5 * (nu + 1) * math.log1p(
            y[t] * y[t] * math.exp(-logv[t]) / (nu - 2))
    return total


@njit(cache=True, nogil=True)
def gaussian_path(phi, innovation, weights, initial, noise, level, scale):
    state = initial.copy()
    h = np.empty(noise.size)
    for t in range(noise.size):
        value = 0.
        for j in range(state.size):
            value += weights[j] * state[j]
        h[t] = level + scale * value
        for j in range(state.size):
            state[j] = phi[j] * state[j] + innovation[j] * noise[t]
    return h, state


@njit(cache=True, nogil=True)
def heston_path(phi, weights, step, initial, noise, level, kappa, eta):
    state = initial.copy()
    logv = np.empty(noise.size)
    theta = math.exp(level)
    step_sum = np.dot(weights, step)
    for t in range(noise.size):
        variance = theta + np.dot(weights, state)
        # Published drift-implicit full-truncation daily discretization.
        positive = max(variance, 1e-10)
        logv[t] = math.log(positive)
        common = eta * math.sqrt(positive) * noise[t]
        predicted = theta
        for j in range(state.size):
            predicted += weights[j] * phi[j] * state[j]
        next_variance = (predicted + step_sum * (kappa * theta + common)) / (1 + kappa * step_sum)
        for j in range(state.size):
            state[j] = phi[j] * state[j] + step[j] * (common - kappa * (next_variance - theta))
    return logv, state


@njit(cache=True, nogil=True)
def heston_gradient(y, phi, weights, step, noise, level, kappa, eta, rho):
    """Exact adjoint of the implemented daily scheme, not an SDE gradient."""
    h, terminal = heston_path(phi, weights, step, np.zeros(phi.size), noise, level, kappa, eta)
    ll = log_likelihood(y, h, noise, rho)
    gradient = np.empty(noise.size)
    adjoint = np.zeros(phi.size)
    step_sum = np.dot(weights, step)
    denominator = 1 + kappa * step_sum
    conditional = 1 - rho * rho
    for t in range(noise.size - 1, -1, -1):
        positive = math.exp(h[t])
        root = math.sqrt(positive)
        residual = y[t] / root - rho * noise[t]
        active = positive > 1.00001e-10
        likelihood_v = (-.5 / positive + .5 * residual * y[t] / (positive * root * conditional)) if active else 0.
        likelihood_w = rho * residual / conditional
        common_v = eta * noise[t] / (2 * root) if active else 0.
        projected = np.dot(step, adjoint)
        gradient[t] = likelihood_w + projected * eta * root / denominator - noise[t]
        for j in range(phi.size):
            adjoint[j] = (phi[j] * adjoint[j]
                          + projected * (common_v * weights[j] - kappa * weights[j] * phi[j]) / denominator
                          + likelihood_v * weights[j])
    return ll - .5 * np.dot(noise, noise), gradient, h, terminal


@njit(cache=True, nogil=True)
def heston_hmc(y, phi, weights, step, noise, level, kappa, eta, rho, epsilon, momentum, uniform):
    current, gradient, _, _ = heston_gradient(y, phi, weights, step, noise, level, kappa, eta, rho)
    proposed = noise.copy()
    velocity = momentum + .5 * epsilon * gradient
    target = -np.inf
    for leapfrog in range(8):
        proposed += epsilon * velocity
        target, gradient, h, terminal = heston_gradient(y, phi, weights, step, proposed, level, kappa, eta, rho)
        if not np.isfinite(target) or not np.all(np.isfinite(gradient)):
            return noise, False
        velocity += epsilon * (.5 if leapfrog == 7 else 1.) * gradient
    difference = target - current + .5 * (np.dot(momentum, momentum) - np.dot(velocity, velocity))
    if math.log(uniform) < difference:
        return proposed, True
    return noise, False


@njit(cache=True, nogil=True)
def gaussian_gradient(y, phi, innovation, weights, root, white, level, scale, rho, nu):
    dimension = phi.size
    noise = white[dimension:]
    h, terminal = gaussian_path(phi, innovation, weights, root @ white[:dimension], noise, level, scale)
    ll = student_likelihood(y, h, nu) if nu > 0 else log_likelihood(y, h, noise, rho)
    gradient = np.empty(white.size)
    adjoint = np.zeros(dimension)
    conditional = 1 - rho * rho
    for t in range(noise.size - 1, -1, -1):
        standardized = y[t] * math.exp(-h[t] / 2)
        if nu > 0:
            square = standardized * standardized
            likelihood_h = -.5 + .5 * (nu + 1) * square / (nu - 2 + square)
            likelihood_w = 0.
        else:
            residual = standardized - rho * noise[t]
            likelihood_h = -.5 + .5 * residual * standardized / conditional
            likelihood_w = rho * residual / conditional
        gradient[dimension + t] = np.dot(innovation, adjoint) + likelihood_w - noise[t]
        for j in range(dimension):
            adjoint[j] = phi[j] * adjoint[j] + scale * weights[j] * likelihood_h
    gradient[:dimension] = root.T @ adjoint - white[:dimension]
    return ll - .5 * np.dot(white, white), gradient


@njit(cache=True, nogil=True)
def gaussian_hmc(y, phi, innovation, weights, root, white, level, scale, rho, nu,
                 epsilon, momentum, uniform):
    current, gradient = gaussian_gradient(y, phi, innovation, weights, root, white, level, scale, rho, nu)
    proposed = white.copy()
    velocity = momentum + .5 * epsilon * gradient
    target = -np.inf
    for leapfrog in range(4):
        proposed += epsilon * velocity
        target, gradient = gaussian_gradient(y, phi, innovation, weights, root, proposed, level, scale, rho, nu)
        if not np.isfinite(target) or not np.all(np.isfinite(gradient)):
            return white, False
        velocity += epsilon * (.5 if leapfrog == 3 else 1.) * gradient
    difference = target - current + .5 * (np.dot(momentum, momentum) - np.dot(velocity, velocity))
    if math.log(uniform) < difference:
        return proposed, True
    return white, False


def configuration(theta, tolerance, kind, with_root=True):
    hurst, kappa, scale, level = theta[0], math.exp(theta[1]), math.exp(theta[2]), theta[3]
    if kind.startswith('fou'):
        phi, innovation, weights, root, evidence = coefficients(hurst, kappa, tolerance, with_root)
        return (phi, innovation, weights, root, level, scale), evidence
    edges, evidence = grid(tolerance)
    rates, weights = quadrature(hurst, edges)
    phi = np.exp(-rates)
    # State is each factor's CONTRIBUTION to variance. Quadrature weights
    # already contain the daily cell integral, so apply that integral once.
    # This scaling also avoids ill-conditioned tiny OU states at fast rates.
    return (phi, np.ones(phi.size), weights, np.zeros(phi.size), level, kappa,
            scale * math.exp(level / 2)), evidence


def trajectory(config, initial_white, noise, kind):
    if kind.startswith('fou'):
        phi, innovation, weights, root, level, scale = config
        return gaussian_path(phi, innovation, weights, root @ initial_white, noise, level, scale)
    return heston_path(*config[:3], config[3], noise, *config[4:])


def prior(theta, level_anchor, leverage):
    if not (0.03 < theta[0] < 0.49 and math.log(1 / 2520) < theta[1] < math.log(.5)
            and math.log(.02) < theta[2] < math.log(3) and abs(theta[3] - level_anchor) < 5):
        return -np.inf
    if leverage and abs(theta[4]) > 2.65:
        return -np.inf
    result = -.5 * ((theta[0] - .15) / .15) ** 2
    result -= .5 * ((theta[1] - math.log(1 / 63)) / 2) ** 2
    result -= .5 * ((theta[2] - math.log(.7)) / 1.5) ** 2
    result -= .5 * ((theta[3] - level_anchor) / 2) ** 2
    if leverage:
        # Gaussian prior on unconstrained atanh(rho), rather than omitting its
        # Jacobian while claiming a uniform rho prior.
        result -= .5 * (theta[4] / 1.2) ** 2
    if theta.size == 6:
        if not math.log(.05) < theta[5] < math.log(98):
            return -np.inf
        result -= .5 * ((theta[5] - math.log(8)) / 1.2) ** 2
    return result


@njit(cache=True, nogil=True)
def _normalize_pick(logw, uniform):
    weights = np.exp(logw - np.max(logw))
    weights /= np.sum(weights)
    total = 0.
    for j in range(weights.size):
        total += weights[j]
        if uniform < total:
            return j
    return weights.size - 1


@njit(cache=True, nogil=True)
def terminal_pgas(y, config_phi, innovation, weights, start_state, reference,
                  level, scale, particles, normals, uniforms):
    """Exact non-Markovian ancestor weights for the full remaining block.

    The reference object is the white driving innovation sequence. Candidate
    ancestors are scored against every future return in the retained block;
    no finite-memory truncation or singular transition density is used.
    """
    length = reference.size
    states = np.empty((particles, start_state.size))
    for p in range(particles):
        states[p] = start_state
    history = np.empty((length, particles))
    ancestry = np.empty((length, particles), np.int64)
    logw = np.zeros(particles)
    ancestor_logw = np.empty(particles)
    next_states = states.copy()
    for t in range(length):
        previous_logw = logw.copy()
        for p in range(particles):
            candidate = states[p].copy()
            suffix = 0.
            for k in range(t, length):
                h = level + scale * np.dot(weights, candidate)
                suffix -= .5 * (h + y[k] * y[k] * math.exp(-h))
                for j in range(candidate.size):
                    candidate[j] = config_phi[j] * candidate[j] + innovation[j] * reference[k]
            ancestor_logw[p] = previous_logw[p] + suffix
        for p in range(particles):
            if p == particles - 1:
                a = _normalize_pick(ancestor_logw, uniforms[t, p])
                z = reference[t]
            else:
                a = _normalize_pick(previous_logw, uniforms[t, p])
                z = normals[t, p]
            ancestry[t, p] = a
            history[t, p] = z
            h = level + scale * np.dot(weights, states[a])
            logw[p] = -.5 * (h + y[t] * y[t] * math.exp(-h))
            for j in range(start_state.size):
                next_states[p, j] = config_phi[j] * states[a, j] + innovation[j] * z
        states, next_states = next_states, states
    index = _normalize_pick(logw, uniforms[-1, -1])
    result = np.empty(length)
    for t in range(length - 1, -1, -1):
        result[t] = history[t, index]
        index = ancestry[t, index]
    return result


def chain(y, seed, tolerance, kind, leverage=False, pgas=False, burn=512, kept=1024, resume=None):
    rng = np.random.default_rng(seed)
    level_anchor = float(np.log(np.mean(y * y)))
    theta = np.array([.12, math.log(1 / 63), math.log(.5), level_anchor, -.25])
    if kind == 'heston':
        theta[2] = math.log(.15)
    student = kind == 'fou_t'
    if student:
        theta = np.r_[theta, math.log(8)]
    theta[0] += rng.uniform(-.03, .03)
    config, evidence = configuration(theta, tolerance, kind)
    initial = np.zeros(config[0].size)
    noise = np.zeros(y.size)
    if resume is not None:
        if burn:
            raise ValueError('resumed sampling cannot readapt warmup')
        rng.bit_generator.state = resume['rng_state']
        theta, initial, noise = resume['theta'].copy(), resume['initial'].copy(), resume['noise'].copy()
        config, evidence = configuration(theta, tolerance, kind)
    h, terminal = trajectory(config, initial, noise, kind)
    rho = math.tanh(theta[4]) if leverage else 0.
    def likelihood(path, driving, parameter, correlation):
        return (student_likelihood(y, path, 2 + math.exp(parameter[5])) if student
                else log_likelihood(y, path, driving, correlation))
    ll = likelihood(h, noise, theta, rho)
    steps = np.array([.006, .055, .035, .035, .035] + ([.06] if student else []))
    accepted = np.zeros(theta.size)
    total_accepted = np.zeros(theta.size)
    draws = np.empty((kept, 8 if student else 7))
    terminal_draws = np.empty((kept, config[0].size))
    parameters = np.empty((kept, theta.size))
    evaluations = 0
    hmc_step = .005 if kind == 'heston' else .03
    hmc_accepted = 0
    hmc_total = 0
    parameter_columns = list(range(5 if leverage else 4)) + ([5] if student else [])
    joint_dimension = len(parameter_columns)
    joint_mean = np.zeros(joint_dimension)
    joint_m2 = np.zeros((joint_dimension, joint_dimension))
    joint_root = np.diag(steps[parameter_columns])
    if resume is not None:
        steps = resume['steps'].copy()
        joint_root = resume['joint_root'].copy()
        hmc_step = resume['hmc_step']
    for iteration in range(burn + kept):
        # Noncentered elliptical slice over both stationary initial Gaussian
        # states and all daily Brownian innovations; no latent path plug-in.
        if kind == 'heston':
            noise, hmc_accept = heston_hmc(y, *config[:3], noise, *config[4:], rho,
                hmc_step, rng.normal(size=noise.size), rng.random())
            hmc_accepted += int(hmc_accept)
            hmc_total += int(hmc_accept)
            if iteration < burn and (iteration + 1) % 32 == 0:
                hmc_step *= math.exp(np.clip(hmc_accepted / 32 - .65, -.15, .15))
                hmc_step = min(.05, max(1e-6, hmc_step))
                hmc_accepted = 0
            h, terminal = trajectory(config, initial, noise, kind)
            ll = likelihood(h, noise, theta, rho)
        else:
            white = rng.normal(size=initial.size)
            driving = rng.normal(size=noise.size)
            prior_h, prior_terminal = trajectory(config, white, driving, kind)
            base = config[4]
            threshold = ll + math.log(rng.random())
            angle = rng.uniform(0, 2 * math.pi)
            lower, upper = angle - 2 * math.pi, angle
            for bracket in range(1000):
                c, s = math.cos(angle), math.sin(angle)
                proposed_initial = initial * c + white * s
                proposed_noise = noise * c + driving * s
                proposed_h = base + (h - base) * c + (prior_h - base) * s
                proposed_terminal = terminal * c + prior_terminal * s
                proposed_ll = likelihood(proposed_h, proposed_noise, theta, rho)
                evaluations += 1
                if proposed_ll > threshold:
                    initial, noise, h, terminal, ll = (proposed_initial, proposed_noise, proposed_h,
                                                     proposed_terminal, proposed_ll)
                    break
                if angle < 0:
                    lower = angle
                else:
                    upper = angle
                angle = rng.uniform(lower, upper)
            else:
                raise ValueError('ESS bracket failed')
            # Native-gradient rejuvenation prevents the global ellipse angle
            # from becoming the sole high-dimensional latent-path bottleneck.
            # Both invariant updates target the same joint Gaussian prior.
            white_vector = np.r_[initial, noise]
            nu = 2 + math.exp(theta[5]) if student else 0.
            white_vector, hmc_accept = gaussian_hmc(y, *config[:4], white_vector, *config[4:], rho, nu,
                hmc_step, rng.normal(size=white_vector.size), rng.random())
            hmc_accepted += int(hmc_accept)
            hmc_total += int(hmc_accept)
            if iteration < burn and (iteration + 1) % 32 == 0:
                hmc_step *= math.exp(np.clip(hmc_accepted / 32 - .65, -.15, .15))
                hmc_step = min(.2, max(1e-6, hmc_step))
                hmc_accepted = 0
            initial, noise = white_vector[:initial.size], white_vector[initial.size:]
            h, terminal = trajectory(config, initial, noise, kind)
            ll = likelihood(h, noise, theta, rho)
        # Componentwise parameter proposals: expensive kernel rebuilding only
        # when H/kappa change. Level/scale/rho re-use the current latent path.
        for j in parameter_columns:
            proposed = theta.copy()
            proposed[j] += steps[j] * rng.normal()
            prior_difference = prior(proposed, level_anchor, leverage) - prior(theta, level_anchor, leverage)
            if not np.isfinite(prior_difference):
                continue
            try:
                if kind == 'heston':
                    proposed_config = list(config)
                    if j == 0:
                        proposed_config, _ = configuration(proposed, tolerance, kind)
                    else:
                        proposed_config[4:] = [proposed[3], math.exp(proposed[1]),
                                              math.exp(proposed[2] + proposed[3] / 2)]
                    if j == 4:
                        proposed_h, proposed_terminal = h, terminal
                    else:
                        proposed_h, proposed_terminal = trajectory(proposed_config, initial, noise, kind)
                elif j < 2:
                    proposed_config, _ = configuration(proposed, tolerance, kind)
                    proposed_h, proposed_terminal = trajectory(proposed_config, initial, noise, kind)
                else:
                    proposed_config = list(config)
                    proposed_config[4] = proposed[3]
                    proposed_config[5] = math.exp(proposed[2])
                    proposed_h = proposed[3] + (h - theta[3]) * math.exp(proposed[2] - theta[2])
                    proposed_terminal = terminal
                proposed_rho = math.tanh(proposed[4]) if leverage else 0.
                proposed_ll = likelihood(proposed_h, noise, proposed, proposed_rho)
            except ValueError:
                continue
            if math.log(rng.random()) < proposed_ll - ll + prior_difference:
                theta, config, h, terminal, ll, rho = (proposed, proposed_config, proposed_h,
                                                     proposed_terminal, proposed_ll, proposed_rho)
                accepted[j] += 1
                total_accepted[j] += 1
        proposed = theta.copy()
        proposed[parameter_columns] += joint_root @ rng.normal(size=joint_dimension)
        prior_difference = prior(proposed, level_anchor, leverage) - prior(theta, level_anchor, leverage)
        if np.isfinite(prior_difference):
            try:
                proposed_config, _ = configuration(proposed, tolerance, kind)
                proposed_h, proposed_terminal = trajectory(proposed_config, initial, noise, kind)
                proposed_rho = math.tanh(proposed[4]) if leverage else 0.
                proposed_ll = likelihood(proposed_h, noise, proposed, proposed_rho)
                if math.log(rng.random()) < proposed_ll - ll + prior_difference:
                    theta, config, h, terminal, ll, rho = (proposed, proposed_config, proposed_h,
                                                         proposed_terminal, proposed_ll, proposed_rho)
            except ValueError:
                pass
        if iteration < burn:
            current_parameters = theta[parameter_columns]
            delta = current_parameters - joint_mean
            joint_mean += delta / (iteration + 1)
            joint_m2 += np.outer(delta, current_parameters - joint_mean)
            if iteration >= 63 and (iteration + 1) % 32 == 0:
                covariance = (2.38 ** 2 / joint_dimension) * (joint_m2 / iteration + np.eye(joint_dimension) * 1e-6)
                joint_root = np.linalg.cholesky(covariance)
        if iteration < burn and (iteration + 1) % 32 == 0:
            steps *= np.exp(np.clip(accepted / 32 - .3, -.2, .2))
            accepted[:] = 0
        if pgas:
            block = min(32, y.size)
            prefix = noise[:-block]
            _, start_state = trajectory(config, initial, prefix, kind)
            noise[-block:] = terminal_pgas(y[-block:], *config[:3], start_state, noise[-block:],
                *config[4:], 8, rng.normal(size=(block, 8)), rng.random(size=(block + 1, 8)))
            h, terminal = trajectory(config, initial, noise, kind)
            ll = log_likelihood(y, h, noise, rho)
        if iteration >= burn:
            index = iteration - burn
            parameters[index] = theta
            terminal_draws[index] = terminal
            draws[index] = np.r_[theta[:4], rho, h[-1], ll,
                                [2 + math.exp(theta[5])] if student else []]
    return {'parameters': parameters, 'terminal': terminal_draws, 'diagnostic_trace': draws,
            'parameter_acceptance': total_accepted / (burn + kept),
            'likelihood_evaluations': evaluations, 'kernel': evidence,
            'hmc_step_size': hmc_step, 'hmc_acceptance': hmc_total / (burn + kept),
            'state': {'theta': theta.copy(), 'initial': initial.copy(), 'noise': noise.copy(),
                      'rng_state': rng.bit_generator.state, 'steps': steps.copy(),
                      'joint_root': joint_root.copy(), 'hmc_step': hmc_step},
            'burn': burn, 'kept': kept}
