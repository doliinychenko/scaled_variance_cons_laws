#ifndef SCALED_VARIANCE_H
#define SCALED_VARIANCE_H

#include "smash/decaymodes.h"
#include "smash/particletype.h"

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
    std::function<bool(const smash::ParticleTypePtr)> type_of_interest);

  friend std::ostream& operator<< (std::ostream&,
                                   const ScaledVarianceCalculator&);

 private:
  /// List of species included in the gas
  smash::ParticleTypePtrList all_types_in_the_box_;
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
};

#endif // SCALED_VARIANCE_H
