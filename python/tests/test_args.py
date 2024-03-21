import unittest
import numpy as np

from rpy2.robjects import (
    r,
    globalenv,
    conversion,
    default_converter,
    pandas2ri,
    BoolVector,
)

from hamstrpy import make_stan_dat_hamstr
from r_import import run_R_import


class TestHamstrRun(unittest.TestCase):
    run_R_import()

    r(
        '''
        load("../../data/MSB2K.rda")

        MSB2K_cal <- calibrate_14C_age(
            MSB2K,
            age.14C = "age",
            age.14C.se = "error",
        )
        '''
    )

    with (default_converter + pandas2ri.converter).context():
        ref_df = conversion.get_conversion().rpy2py(globalenv['MSB2K_cal'])

    def test_keys(
        self,
        min_age=2024,
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
    ):
        r(
            f'''
            depth <- MSB2K_cal$depth
            obs_age <- MSB2K_cal$age.14C.cal
            obs_err <- MSB2K_cal$age.14C.cal.se
            min_age <- {min_age:d}
            K_fine <- {K_fine if K_fine is not None else "NULL"}
            K_factor <- {K_factor if K_factor is not None else "NULL"}
            top_depth <- {top_depth if top_depth is not None else "NULL"}
            bottom_depth <- {
                bottom_depth if bottom_depth is not None else "NULL"
            }
            acc_mean_prior <- {
                acc_mean_prior if acc_mean_prior is not None else "NULL"
            }
            acc_shape <- {acc_shape}
            mem_mean <- {mem_mean}
            mem_strength <- {mem_strength}
            model_bioturbation <- {str(model_bioturbation).upper()}
            n_ind <- {n_ind if n_ind is not None else "NULL"}
            L_prior_mean <- {L_prior_mean}
            L_prior_shape <- {L_prior_shape}
            bottom_depth <- {
                bottom_depth if bottom_depth is not None else "NULL"
            }
            model_displacement <- {str(model_displacement).upper()}
            D_prior_scale <- {D_prior_scale}
            model_hiatus <- {str(model_hiatus).upper()}
            H_top <- {H_top if H_top is not None else "NULL"}
            H_bottom <- {H_bottom if H_bottom is not None else "NULL"}
            sample_posterior <- {str(sample_posterior).upper()}
            hamstr_control <- list()
            stan_sampler_args <- list()

            stan_dat <- hamstr:::make_stan_dat_hamstr()
            '''
        )
        with (default_converter + pandas2ri.converter).context():
            ref_stan_dat = conversion.get_conversion().rpy2py(
                globalenv['stan_dat']
            )

        stan_dat = make_stan_dat_hamstr(
            depth=self.ref_df['depth'].values,
            obs_age=self.ref_df['age.14C.cal'].values,
            obs_err=self.ref_df['age.14C.cal.se'].values,
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
        )

        for key in stan_dat.keys():
            if isinstance(ref_stan_dat[key], BoolVector):
                ref = np.array(ref_stan_dat[key]).astype(bool)
            else:
                ref = ref_stan_dat[key]

            comp = np.asarray(stan_dat[key])
            if comp.dtype == bool or comp.dtype == str:
                self.assertTrue(
                    np.all(
                        comp == ref
                    ),
                    msg=f'{key} not equal.'
                )
            else:
                self.assertTrue(
                    np.allclose(
                        comp,
                        ref,
                    ),
                    msg=f'{key} not equal.'
                )

    def test_bioturbation_args(self):
        with self.subTest():
            self.test_keys(
                model_bioturbation=True,
                n_ind=5.0,
            )

    def test_L_prior_sigma(self):
        with self.subTest():
            self.test_keys(
                L_prior_sigma=2,
            )


if __name__ == '__main__':
    unittest.main()
