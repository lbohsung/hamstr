import sys
from pathlib import PurePath

import unittest

# Fetch relative path
tests_path = PurePath(__file__).parent
# By convention the TestLoader discovers TestCases in test_*.py
suite = unittest.TestLoader().discover(tests_path)

if __name__ == '__main__':
    # Set up a test-runner
    runner = unittest.TextTestRunner(verbosity=2)
    # Collect test results
    result = runner.run(suite)
    # If not successful return 1
    sys.exit(not result.wasSuccessful())
