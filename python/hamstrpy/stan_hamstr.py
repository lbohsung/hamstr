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
    stan_sampler_args={},
    **hamstr_control_kwargs,
):
    """ Fit a hamstr age-depth model

    hamstr is used to fit an age-depth model to a set of age-control points.
    Ages should already be on the desired scale, e.g. calendar ages, and will
    not be calibrated. The function calibrate_14C_age from the calibrate_14C
    module can be used to calibrate radiocarbon dates prior to fitting a hamstr
    model.

    Parameters
    ----------
    depth : array-like
        The depths of observed ages (age control points)
    obs_age : array-like
        Observed age at each depth (age control points)
    obs_err : array-like
        Error associated with each observed age (1 standard error)
    min_age : float, optional
        The minimum age that the first modelled depth can be. Useful if
        extrapolating above the shallowest age control point to e.g. the
        surface. So set min_age to the year the core was collected. E.g. for a
        core collected in 1990, with ages in years BP this would be -40
        (present = 1950 by convention). The default value is the current year
        in calendar age BP, calculated using the datetime.now() return value.
    K_fine : int, optional
        The number of sections at the highest resolution of the model.
    K_factor : float, optional
        The rate at which the thickness of the sections grows between
        subsequent levels.
    top_depth : float, optional
        The top depth of the desired age-depth model. Must encompass the range
        of the data. Defaults to the shallowest and deepest data points.
    bottom_depth : float, optional
        The bottom depths of the desired age-depth model. Must encompass the
        range of the data. Defaults to the shallowest and deepest data points.
    acc_mean_prior : float, optional
        Hyperparameter for the prior on the overall mean accumulation rate for
        the record. Units are obs_age / depth. E.g. if depth is in cm and age
        in years then the accumulation rate is in years/cm. The overall mean
        accumulation rate is given a weak half-normal prior with mean= 0,
        SD = 10 * acc_mean_prior. If left blank, acc_mean_prior is set to the
        mean accumulation rate estimated by fitting a robust linear model using
        rlm.
    acc_shape : float, optional
        Hyperparameter for the shape of the priors on accumulatio rates.
        Defaults to 1.5 - as for Bacon 2.2.
    mem_mean : float, optional
        Hyperparameter; a parameter of the Beta prior distribution on "memory",
        i.e. the autocorrelation parameter in the underlying AR1 model. The
        prior on the correlation between layers is scaled according to the
        thickness of the sediment sections in the highest resolution
        hierarchical layer, *delta_c*, which is determined by the total length
        age-models and the parameter vector *K*. mem_mean sets the mean value
        for *R* (defaults to 0.5), while *w* = R^(delta_c)
    mem_strength : float, optional
        Hyperparameter: sets the strength of the memory prior, defaults to 10
        as in Bacon >= 2.5.1
    model_bioturbation : bool, optional
        If True, n_ind is a required argument and additional uncertainty in
        the observed ages due to sediment mixing (bioturbation) is modelled
        via a latent variable process. The amount of additional uncertainty is
        a function of the mixing depth L, the sedimentation rate, and the
        number of particles (e.g. individual foraminifera) per measured date.
        See description for details. Default is False.
    n_ind : int, optional
        The number of individual particles (e.g. foraminifera) in each sample
        that was dated by e.g. radiocarbon dating. This can be a single value
        or a vector the same length as obs_age. Only used if model_bioturbation
        is True.
    L_prior_mean : float, optional
        Mean of the gamma prior on mixing depth, defaults to 10.
    L_prior_shape : float, optional
        Shape of the gamma prior on the mixing depth. Set only one of
        L_prior_shape and L_prior_sigma, the other will be calculated. If
        either the shape or sigma parameter is set to zero, the mixing depth is
        fixed at the value of L_prior_mean, rather than being sampled.
    L_prior_sigma : float, optional
        Standard deviation of the gamma prior on the mixing depth. Set only one
        of L_prior_shape and L_prior_sigma, the other will be calculated. If
        either the shape or sigma parameter is set to zero, the mixing depth is
        fixed at the value of L_prior_mean, rather than being sampled.
    model_displacement : bool, optional
        Model additional error on observed ages that does not scale with the
        number of individual particles in a sample, for example due to
        incomplete mixing.
    D_prior_scale : float, optional
        Scale of the half-normal prior on additional error on observed ages.
        The mean and standard deviation of a half-normal are equal to the
        scale. Units are those of the depth variable, e.g. cm.
    model_hiatus : bool, optional
        Optionally model a hiatus.
    H_top : float, optional
        Upper limit to the location of a hiatus. By default these are set to
        the top and bottom data points but can be set by the user
    H_bottom : float, optional
        Lower limit to the location of a hiatus. By default these are set to
        the top and bottom data points but can be set by the user
    sample_posterior : bool, optional
        If set to False, hamstr skips sampling the model and returns only the
        data, model structure and prior parameters so that data and prior
        distributions can be plotted and checked without running a model.
    stan_sampler_args : dict, optional
        Additional arguments to the stan sampler passed as dict. E.g.
            {'chains': 8, 'iter': 4000}
        to run 8 MCMC chains of 4000
        iterations instead of the default 4 chains of 2000 iterations.
        Consult the  get_stan_sampler_args function for details.
    hamstr_control_kwargs
        Additional keywords passed to hamstr. May include
            scale_R : bool
                Scale AR1 coefficient by delta_c (as in Bacon) or not.
            nu : int
                Degrees of freedom for the Student-t distributed error model.
                Defaults to 6, which is equivalent to the default
                parametrisation of t.a=3, t.b=4 in Bacon 2.2. Set to a high
                number to approximate a Gaussian error model, (nu = 100 should
                do it).
            scale_shape : bool
                scale the shape parameter according to the number of
                hierarchical levels, to control the total variance of the alpha
                innovations. This defaults to True as of hamstr verion 0.5.
            smooth_s : bool
                Smooth the sedimentation rate used to calculate additional
                error from bioturbation by taking a mean across nearby
                sections.
            inflate_errors : bool
                If set to TRUE, observation errors are inflated so that data
                are consistent with a gamma AR1 age-depth model. This is an
                experimental feature under active development.
            infl_sigma_sd : float or None
                Hyperparameter: sets the standard deviation of the half-normal
                prior on the mean of the additional error terms. Defaults to 10
                times the mean observation error in obs_err.
            infl_shape_shape, infl_shape_mean : float
                Hyperparameter: parametrises the gamma prior on the shape of
                the distribution of the additional error terms.

    Returns
    -------
    arviz.InferenceData
        InferenceData container with output from Stan, data and input
        parameters.
    """
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
        stan_sampler_args=stan_sampler_args,
        **hamstr_control_kwargs,
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
    """ Default Parameters for Sampling Hamstr Models with Stan.
    Returns a dict of parameters for the Stan sampler.
    """
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
