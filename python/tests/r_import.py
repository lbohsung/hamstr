import rpy2.robjects.packages as rpackages


def run_R_import():
    utils = rpackages.importr('utils')
    try:
        rpackages.importr("hamstr")
        rpackages.importr("rstan")

    except rpackages.PackageNotInstalledError:
        from rpy2.rinterface_lib.embedded import RRuntimeError

        try:
            remotes = rpackages.importr("remotes")
        except rpackages.PackageNotInstalledError:
            utils.install_packages("remotes")
            remotes = rpackages.importr("remotes")
        try:
            remotes.install_local('../../')
        except RRuntimeError:
            remotes.install_github(
                "earthsystemdiagnostics/hamstr",
                args="--preclean",
            )

        rpackages.importr("hamstr")
        rpackages.importr("rstan")
