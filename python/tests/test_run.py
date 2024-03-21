import unittest
import numpy as np

from rpy2.robjects import (
    r,
    globalenv,
    conversion,
    default_converter,
    pandas2ri,
)

from hamstrpy import stan_hamstr
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

        hamstr_fit_1 <- hamstr(
            depth = MSB2K_cal$depth,
            obs_age = MSB2K_cal$age.14C.cal,
            obs_err = MSB2K_cal$age.14C.cal.se,
            # the seed argument for the sampler is set here so that
            # this example always returns the same numerical result
            stan_sampler_args = list(seed = 1)
        )

        age_samps <- extract(hamstr_fit_1$fit, 'c_ages')
        '''
    )
    with (default_converter + pandas2ri.converter).context():
        _conv = conversion.get_conversion().rpy2py(globalenv['age_samps'])
        ref_samples = _conv['c_ages']
        ref_df = conversion.get_conversion().rpy2py(globalenv['MSB2K_cal'])

    idata = stan_hamstr.hamstr(
        ref_df['depth'].values,
        ref_df['age.14C.cal'].values,
        ref_df['age.14C.cal.se'].values,
        stan_sampler_args={"seed": 1},
    )

    age_samples = idata.posterior['c_ages'].values
    age_samples = age_samples.reshape(-1, ref_samples.shape[1])

    def test_mean(self):
        ref_mean = self.ref_samples.mean(axis=0)
        age_mean = self.age_samples.mean(axis=0)

        self.assertTrue(
            np.allclose(
                ref_mean,
                age_mean,
                rtol=5e-3,
            )
        )

    def test_std(self):
        ref_std = self.ref_samples.std(axis=0)
        age_std = self.age_samples.std(axis=0)

        self.assertTrue(
            np.allclose(
                ref_std,
                age_std,
                rtol=5e-2,
            )
        )


if __name__ == '__main__':
    unittest.main()
