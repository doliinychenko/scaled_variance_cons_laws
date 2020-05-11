#ifndef SCALED_VARIANCE_H
#define SCALED_VARIANCE_H

#include "smash/decaymodes.h"
#include "smash/particletype.h"

#include <Eigen/Dense>

#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_vector.h>

bool load_particle_types();

class ScaledVarianceCalculator {
 public:
  ScaledVarianceCalculator(const smash::ParticleTypePtrList& types_in_the_box,
                           double T, double mub, double mus, double muq,
                           bool energy_conservation,
                           bool B_conservation,
                           bool S_conservation,
                           bool Q_conservation,
                           bool quantum_statistics) :
    T_ (T),
    mub_ (mub),
    mus_ (mus),
    muq_ (muq),
    energy_conservation_ (energy_conservation),
    B_conservation_ (B_conservation),
    S_conservation_ (S_conservation),
    Q_conservation_ (Q_conservation),
    quantum_statistics_ (quantum_statistics),
    quantum_series_max_terms_(100),
    quantum_series_rel_precision_(1e-12) {
    all_types_in_the_box_.clear();
    for (const smash::ParticleTypePtr t : types_in_the_box) {
      all_types_in_the_box_.push_back(t);
      if (t->is_stable()) {
        stable_particles_.push_back(t);
      }
    }
  }
  void set_T (double T) { T_ = T; }
  void set_mub (double mub) { mub_ = mub; }
  void set_mus (double mus) { mus_ = mus; }
  void set_muq (double muq) { muq_ = muq; }
  void set_B_conservation (bool B_cons) { B_conservation_ = B_cons; }
  void set_S_conservation (bool S_cons) { S_conservation_ = S_cons; }
  void set_Q_conservation (bool Q_cons) { Q_conservation_ = Q_cons; }
  void set_quantum_statistics (bool qs) { quantum_statistics_ = qs; }

  struct solver_params {
    smash::ParticleTypePtrList* types;
    double e;   // energy density
    double nb;  // baryon number density
    double ns;  // strangeness density
    double nq;  // charge density
  };
  static int set_eos_solver_equations(const gsl_vector* x, void* params,
                                      gsl_vector* f);
  std::string print_solver_state(size_t iter,
                                 gsl_multiroot_fsolver* solver) const;
  void setTmu_from_conserved(double Etot, double V,
                             double B, double S, double Q);
  static double symmetric_matrix_determinant_excluding_zero_columns_and_rows(const Eigen::MatrixXd &A);
  void prepare_full_correlation_table_no_resonance_decays();
  void prepare_correlations_after_decays();
  /**
   * Assumes that there is a gas of particles species defined
   * by all_types_in_the_box_. This function computes fluctuations
   * of the total number of particles belonging to a subset of species
   * defined by type_of_interest function. Fluctuations depend on
   * the ensemble and which conservation laws it respects. This is
   * determined by variables energy_conservation_, B_conservation_,
   * S_conservation_, and Q_conservation_.
   *
   * For reference see arxiv.org/pdf/0706.3290.pdf, Eq. (29-40).
   * The approximations of this paper are only valid for large
   * enough volume.
   *
   * \param[in] type_of_interest Function true for particle species, for
   *            which the mean and the scaled variance are computed
   * \param[out] mean density and scaled variance of type_of_interest species
   */

  std::pair<double, double> scaled_variance(
    const smash::ParticleTypePtr type_of_interest);

  double custom_correlation(std::function<int(const smash::ParticleTypePtr)> w1,
    std::function<int(const smash::ParticleTypePtr)> w2,
    bool thermal_correlation);

  void prepare_decays();

  friend std::ostream& operator<< (std::ostream&,
                                   const ScaledVarianceCalculator&);

 private:
  /// Precomputes kappa_NN, kappa_EN, kappa_EE for each species for given T, mu
  void prepare_thermal_arrays();

  /// List of species included in the gas
  smash::ParticleTypePtrList all_types_in_the_box_;
  /**
   *  Structure for holding all decay final state in a form convenient
   *  for statistical caulculations:
   *  resonance ->
   *       vector of decays
   *       each decay = (branching ratio and vector (product -> count))
   *  Sum of branching ratios should be 1 for all particles.
   *  By convention stable particles decay into themselves with brancing ratio 1.
   */
  std::map<smash::ParticleTypePtr,
           std::vector<std::pair<double,
                               std::map<smash::ParticleTypePtr, int>>>>
       all_decay_final_states_;
  /// Auxiliary precalculated expressions for every species in the box
  std::map<smash::ParticleTypePtr, double> kappa_NN_, kappa_EN_, kappa_EE_, thermal_density_;
  /// Full correlation matrix between the species
  Eigen::MatrixXd corr_;
  /// List of stable species, for which correlations after decays are computed
  smash::ParticleTypePtrList stable_particles_;
  /// Correlation matrix after decays
  Eigen::MatrixXd corr_after_decays_;
  /// Temperature of the gas [GeV]
  double T_;
  /// Baryo-chemical potential of the gas [GeV]
  double mub_;
  /// Strangeness chemical potential of the gas [GeV]
  double mus_;
  /// Charge chemical potential of the gas [GeV]
  double muq_;
  /// If ensemble includes energy conservation
  bool energy_conservation_;
  /// If ensemble includes baryon number conservation
  bool B_conservation_;
  /// If ensemble includes strangeness conservation
  bool S_conservation_;
  /// If ensemble includes electric charge conservation
  bool Q_conservation_;
  /// If quantum statistics is included
  bool quantum_statistics_;
  /// Maximal number of terms in the series for quantum formulas
  const unsigned int quantum_series_max_terms_;
  /// Relative precision, at which quantum series summation stops
  const double quantum_series_rel_precision_;
  /// Are thermal arrays kappa_NN_, kappa_EN, etc ready?
  bool thermal_arrays_prepared_ = false;
  /// Is the decays data structure ready?
  bool decays_prepared_ = false;
};

#endif // SCALED_VARIANCE_H
