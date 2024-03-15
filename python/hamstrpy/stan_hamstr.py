import numpy as np
import os
import sys
import arviz as az
from cmdstanpy import CmdStanModel

import datetime

from .make_stan_dat_hamstr import make_stan_dat_hamstr, get_inits_hamstr

here = os.path.abspath(os.path.dirname(__file__))


def hamstr(
    depth,
    obs_age,
    obs_err,
    min_age=None,
    K_fine=None,
    K_factor=None,
    top_depth=None,
    bottom_depth=None,
    acc_mean_prior=None,
    acc_shape=1.5,
    mem_mean=0.5,
    mem_strength=10,
    model_bioturbation=False,
    n_ind=None,
    L_prior_mean=10,
    L_prior_shape=2,
    L_prior_sigma=None,
    model_displacement=False,
    D_prior_scale=10,
    model_hiatus=False,
    H_top=None,
    H_bottom=None,
    sample_posterior=True,
    hamstr_control={},
    stan_sampler_args={}
):
    if min_age is None:
        min_age = 1950 - datetime.datetime.now().year

    if K_fine is not None:
        if K_fine < 2:
            raise ValueError("A minimum of 2 sections are required")

    if K_factor is not None:
        if K_factor < 2:
            raise ValueError("K_factor must be 2 or greater")
        if abs(K_factor % 1) > 1e-04:
            raise ValueError("K_factor must be an integer")

    stan_dat = make_stan_dat_hamstr(
        depth=depth,
        obs_age=obs_age,
        obs_err=obs_err,
        min_age=min_age,
        K_fine=K_fine,
        K_factor=K_factor,
        top_depth=top_depth,
        bottom_depth=bottom_depth,
        acc_mean_prior=acc_mean_prior,
        acc_shape=acc_shape,
        mem_mean=mem_mean,
        mem_strength=mem_strength,
        model_bioturbation=model_bioturbation,
        n_ind=n_ind,
        L_prior_mean=L_prior_mean,
        L_prior_shape=L_prior_shape,
        L_prior_sigma=L_prior_sigma,
        model_displacement=model_displacement,
        D_prior_scale=D_prior_scale,
        model_hiatus=model_hiatus,
        H_top=H_top,
        H_bottom=H_bottom,
        sample_posterior=sample_posterior,
        hamstr_control=hamstr_control,
        stan_sampler_args=stan_sampler_args
    )
    used_sampler_args = get_stan_sampler_args(**stan_sampler_args)
    np.random.seed(used_sampler_args['seed'])

    inits = [
        get_inits_hamstr(stan_dat) for _ in range(used_sampler_args['chains'])
    ]

    model = CmdStanModel(stan_file=os.path.join(here, "hamstr.stan"))
    fit = None
    if sample_posterior:
        fit = model.sample(
            data=stan_dat,
            inits=inits,
            show_console=True,
            **used_sampler_args,
        )
    else:
        print(
            "Skipping posterior sampling as per the 'sample_posterior' flag."
        )
    stan_dat.update(used_sampler_args)
    stan_dat.pop("hamstr_control")
    stan_dat.pop("stan_sampler_args")
    inference_data = az.from_cmdstanpy(posterior=fit, observed_data=stan_dat)
    # az.to_netcdf(inference_data, "model_results.nc")
    return inference_data


def get_stan_sampler_args(
    chains=4,
    iter=2000,
    warmup=None,
    thin=1,
    seed=None,
    check_data=True,
    sample_file=None,
    diagnostic_file=None,
    verbose=False,
    algorithm=("NUTS", "HMC", "Fixed_param"),
    control=None,
    include=True,
    open_progress=None,
    show_messages=True,
    **kwargs,
):
    if warmup is None:
        warmup = iter // 2
    if seed is None:
        seed = np.random.randint(1, np.iinfo(np.int32).max)
    if open_progress is None:
        open_progress = (
            sys.stdout.isatty() and os.environ.get("RSTUDIO") != "1"
        )

    # Additional arguments passed via kwargs are added to the dictionary
    sampler_args = {
        "chains": chains,
        # "cores": chains, Not available
        # "iter": iter, Does not exists its split into
        # iter_warmup and iter_sampling
        "iter_warmup": warmup,
        "iter_sampling": iter - warmup,
        "thin": thin,
        "seed": seed,
        # "check_data": check_data, not needed since CmdStanPy performs
        # necessary data checks implicitly
        # "sample_file": sample_file,
        "output_dir": sample_file,
        # CmdStanPy provides diagnostic information through its interfaces and
        # the resulting CmdStanMCMC object from the sample() method.
        # While it doesn't directly accept a diagnostic_file argument, you can
        # access diagnostic statistics and perform checks programmatically.
        # Additionally, CmdStanPy saves all sampling output, including
        # diagnostics, to CSV files by default.
        # You can specify the directory where these files are saved using the
        # output_dir argument, and then process these files as needed for
        # diagnostics.
        # For example, to access summary diagnostics from the fit object:
        # print(fit.summary())
        # "diagnostic_file": diagnostic_file,
        # CmdStanPy does not accept verbose as a direct argument, which
        # contrasts with some other interfaces for Stan where verbose might
        # control the amount of output printed to the console during sampling.
        # In CmdStanPy, verbosity can be controlled through the Python logging
        # module
        # import logging
        # logging.basicConfig(level=logging.DEBUG)
        # "verbose": verbose,
        # CmdStanPy does not provide a direct Python argument to switch between
        # algorithms like "NUTS", "HMC", or "Fixed_param".
        # The default and generally recommended algorithm for continuous
        # parameters models is NUTS, which is automatically used by CmdStanPy.
        # "algorithm": algorithm,
        # "control": control,
        # "include": include, Not available
        # "open_progress": open_progress, Not available
        # "show_messages": show_messages, Not available
    }
    sampler_args.update(kwargs)

    return sampler_args
