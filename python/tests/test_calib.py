import unittest
import numpy as np

from rpy2.robjects import (
    r,
    globalenv,
    conversion,
    default_converter,
    pandas2ri,
)

from hamstrpy.calibrate_14C import calibrate_14C_age
from r_import import run_R_import


class TestCalibration(unittest.TestCase):
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
        rc_data = conversion.get_conversion().rpy2py(globalenv['MSB2K'])
        ref_df = conversion.get_conversion().rpy2py(globalenv['MSB2K_cal'])

    calibrate_14C_age(
        rc_data,
    )

    def test_age_calibration(self):
        self.assertTrue(
            np.allclose(
                self.ref_df['age.14C.cal'],
                self.rc_data['t'],
                rtol=5e-2,
            )
        )

    def test_sigma_calibration(self):
        self.assertTrue(
            np.allclose(
                self.ref_df['age.14C.cal.se'],
                self.rc_data['dt'],
                rtol=5e-2,
            )
        )


if __name__ == '__main__':
    unittest.main()
