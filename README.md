# scaled_variance_cons_laws

This is a code to compute a scaled variance of number of particles in an
ensemble with conservation laws. A particular application is meant for
heavy ion collisions. The setting is the following:

- In a box of volume V there is a classical Boltzmann gas at temperature T,
  baryonchemical potential muB and strangeness chemical potential muS.
- The gas consists of many particle species
- Custom conservation laws can be imposed on the ensemble:
  energy / baryon number / strangeness / electric charge conservation
- The task is to compute fluctuations of a specific subset of particle species
  in this ensemble.

The problem is very hard in general, but solved in the limit of large volume
in arxiv.org/pdf/0706.3290.pdf by Hauer, Gorenstein and Moroz. This code
implements the formulas from the paper in case of Maxwell-Boltzmann gas.

## Prequisites

1. Eigen3 library (eigen.tuxfamily.org)
2. GNU Scientific Library (GSL) >= 1.15
3. SMASH library (smash-transport.github.io), which requires first two anyway.

## Compiling

1. Set the SMASH_DIR environment variable by executing

      export SMASH_DIR=[...]/smash

- Copy the cmake files to find the libraries for the project

      cp -r $SMASH_DIR/cmake ..

- Run cmake and make

      mkdir build && cd build
      cmake ..
      make

  If Pythia is not found try

     cmake .. -DPythia_CONFIG_EXECUTABLE=[...]/pythia8235/bin/pythia8-config