import rpy2.robjects.packages as rpackages


def run_R_import():
    utils = rpackages.importr('utils')
    try:
        rpackages.importr("hamstr")
        rpackages.importr("rstan")

    except rpackages.PackageNotInstalledError:
        try:
            remotes = rpackages.importr("remotes")
        except rpackages.PackageNotInstalledError:
            utils.install_packages("remotes")
            remotes = rpackages.importr("remotes")

        remotes.install_local('../../')

        rpackages.importr("hamstr")
        rpackages.importr("rstan")
