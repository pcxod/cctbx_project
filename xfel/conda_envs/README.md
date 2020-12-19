# Psana Build Instructions

There are two environments in this directory. Psana_environment.yml is suitable
for general use and contains the usual CCTBX dependencies plus psana and its
dependencies. Psana_lcls_environment.yml should only be used at LCLS; it adds
a local build of mpi4py that is required for running jobs on multiple psana
nodes.

The build steps below were tested on Oct 28, 2020. They should be done in a
clean environment (no cctbx installation has been activated, etc.)

Creating the conda environment (step `base`) may take up to ~20 min with no
obvious progress occurring.

## General build

These steps were tested on a CentOS 7.4.1708 machine with 64 cores. In the
bootstrap.py step you should adjust nproc to suit your environment.

```
$ mkdir cctbx; cd cctbx
$ wget https://raw.githubusercontent.com/cctbx/cctbx_project/master/libtbx/auto_build/bootstrap.py
$ wget https://raw.githubusercontent.com/cctbx/cctbx_project/master/xfel/conda_envs/psana_environment.yml
$ python bootstrap.py --builder=dials --use-conda=psana_environment.yml --nproc=64 --python=36
$ source build/conda_setpaths.sh
$ libtbx.python -c "import psana" # Should exit with no output
```

## LCLS build

These steps were tested on an ssh connection to pslogin.slac.stanford.edu. You
will have to remember the location of $INSTALL because a new ssh connection is
opened before the build step.


```
$ ssh psexport
$ cd $INSTALL; mkdir cctbx; cd cctbx
$ wget https://raw.githubusercontent.com/cctbx/cctbx_project/master/libtbx/auto_build/bootstrap.py
$ wget https://raw.githubusercontent.com/cctbx/cctbx_project/master/xfel/conda_envs/psana_lcls_environment.yml
$ python bootstrap.py --builder=dials --use-conda=psana_lcls_environment.yml \
  --no-boost-src --python=37 hot update base
$ exit  # logout of psexport
$ ssh psana
$ cd $INSTALL/cctbx
$ conda  # If you get a usage message, skip the next step.
$ source mc3/etc/profile.d/conda.sh # Activate conda. It could also be found at
                                    # ~/miniconda3/etc/profile.d/conda.sh
$ conda activate `pwd`/base
$ python bootstrap.py --builder=dials --use-conda=psana_lcls_environment.yml \
  --config-flags="--compiler=conda" --config-flags="--use_environment_flags" \
  --config-flags="enable_cxx11" --config-flags="--no_bin_python"             \
  --no-boost-src --python=37 --nproc=10 build
$ source build/conda_setpaths.sh
$ source modules/cctbx_project/xfel/conda_envs/test_psana_lcls.sh
```







