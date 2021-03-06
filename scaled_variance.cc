#include "scaled_variance.h"

#include "smash/../../scatteractionsfinder.cc"

#include <Eigen/Dense>
#include <fstream>
#include <gsl/gsl_sf_bessel.h>

/// Loads particle table from SMASH
bool load_particle_types() {
  std::cout << "Loading SMASH particle types and decay modes" << std::endl;
  std::string smash_dir(std::getenv("SMASH_DIR"));
  if (smash_dir == "") {
    std::cerr << "Failed to load SMASH particle types. SMASH_DIR is not set."
              << std::endl;
    return true;
  }
  std::ifstream particles_input_file(smash_dir + "/input/particles.txt");
  std::stringstream buffer;
  if (particles_input_file) {
    buffer << particles_input_file.rdbuf();
    smash::ParticleType::create_type_list(buffer.str());
  } else {
    std::cerr << "File with SMASH particle list not found." << std::endl;
    return true;
  }
  std::ifstream decaymodes_input_file(smash_dir + "/input/decaymodes.txt");
  if (decaymodes_input_file) {
    buffer.clear();
    buffer.str(std::string());
    buffer << decaymodes_input_file.rdbuf();
    smash::DecayModes::load_decaymodes(buffer.str());
    smash::ParticleType::check_consistency();
  } else {
    std::cerr << "File with SMASH decaymodes not found." << std::endl;
    return true;
  }
  return false;
}



int ScaledVarianceCalculator::set_eos_solver_equations(
                             const gsl_vector* x, void* params,
                             gsl_vector* f) {
  smash::ParticleTypePtrList* types =
      reinterpret_cast<struct solver_params *>(params)->types;
  double e = reinterpret_cast<struct solver_params *>(params)->e;
  double nb = reinterpret_cast<struct solver_params *>(params)->nb;
  double ns = reinterpret_cast<struct solver_params *>(params)->ns;
  double nq = reinterpret_cast<struct solver_params *>(params)->nq;

  const double T = gsl_vector_get(x, 0);
  const double mub = gsl_vector_get(x, 1);
  const double mus = gsl_vector_get(x, 2);
  const double muq = gsl_vector_get(x, 3);

  double e_tot = 0.0, nb_tot = 0.0, ns_tot = 0.0, nq_tot = 0.0;
  for (const smash::ParticleTypePtr t : (*types)) {
    const double b = t->baryon_number();
    const double s = t->strangeness();
    const double q = t->charge();
    const double z = t->mass() / T;
    const double mu = mub * b + mus * s + muq * q;
    const double mu_m_over_T = (mu - t->mass()) / T;
    const double factor = t->pdgcode().spin_degeneracy() * 4.0 * M_PI *
                          std::pow(T / (2.0 * M_PI * smash::hbarc), 3);
    const double k1 = gsl_sf_bessel_Kn_scaled(1, z);
    const double k2 = gsl_sf_bessel_Kn_scaled(2, z);
    const double ex = std::exp(mu_m_over_T);
    double rho = z * z *  k2 * ex * factor;
    double edens = z * z * (3 * k2 + z * k1) * ex * T * factor;
    e_tot += edens;
    nb_tot += rho * b;
    ns_tot += rho * s;
    nq_tot += rho * q;
  }
  gsl_vector_set(f, 0, e_tot - e);
  gsl_vector_set(f, 1, nb_tot - nb);
  gsl_vector_set(f, 2, ns_tot - ns);
  gsl_vector_set(f, 3, nq_tot - nq);
  return GSL_SUCCESS;
}

std::string ScaledVarianceCalculator::print_solver_state(size_t iter,
     gsl_multiroot_fsolver* solver) const {
  std::stringstream s;
  // clang-format off
  s << "iter = " << iter << ","
    << " x = " << gsl_vector_get(solver->x, 0) << " "
               << gsl_vector_get(solver->x, 1) << " "
               << gsl_vector_get(solver->x, 2) << " "
               << gsl_vector_get(solver->x, 3) << ", "
    << "f(x) = " << gsl_vector_get(solver->f, 0) << " "
                 << gsl_vector_get(solver->f, 1) << " "
                 << gsl_vector_get(solver->f, 2) << " "
                 << gsl_vector_get(solver->f, 3) << " "
                 << std::endl;
  // clang-format on
  return s.str();

}

void ScaledVarianceCalculator::setTmu_from_conserved(double Etot, double V,
                                                double B, double S, double Q) {
  if (quantum_statistics_) {
    std::cout << "WARNING: quantum statistics requested, but not implemented "
              << "for (E, B, S, Q) -> (T, muB, muS, muQ) solver." << std::endl;
    throw std::runtime_error("");
  }
  const gsl_multiroot_fsolver_type *solver_type = gsl_multiroot_fsolver_hybrid;
  gsl_multiroot_fsolver* solver = gsl_multiroot_fsolver_alloc(solver_type, 4);
  gsl_vector* x = gsl_vector_alloc(4);
  int residual_status = GSL_SUCCESS;
  size_t iter = 0;
  std::cout << print_solver_state(iter, solver);

  struct solver_params p = {&all_types_in_the_box_, Etot/V, B/V, S/V, Q/V};
  gsl_multiroot_function f = {&set_eos_solver_equations, 4, &p};

  gsl_vector_set(x, 0, 0.15);
  gsl_vector_set(x, 1, 0.01);
  gsl_vector_set(x, 2, 0.01);
  gsl_vector_set(x, 3, 0.01);

  gsl_multiroot_fsolver_set(solver, &f, x);
  do {
    iter++;
    const auto iterate_status = gsl_multiroot_fsolver_iterate(solver);
    // std::cout << print_solver_state(iter, solver);

    // Avoiding too low temperature
    if (gsl_vector_get(solver->x, 0) < 0.015) {
      T_ = 0.0;
      mub_ = 0.0;
      mus_ = 0.0;
      muq_ = 0.0;
      return;
    }

    // check if solver is stuck
    if (iterate_status) {
      break;
    }
    residual_status = gsl_multiroot_test_residual(solver->f, 1.e-9);
  } while (residual_status == GSL_CONTINUE && iter < 1000);

  if (residual_status != GSL_SUCCESS) {
    std::stringstream solver_parameters;
    solver_parameters << "Solver run with "
                      << "e = " << Etot/V << ", nb = " << B/V
                      << ", ns = " << S/V << ", nq = " << Q/V
                      << std::endl;
    throw std::runtime_error(gsl_strerror(residual_status) +
                             solver_parameters.str() +
                             print_solver_state(iter, solver));
  }
  T_ = gsl_vector_get(solver->x, 0);
  mub_ = gsl_vector_get(solver->x, 1);
  mus_ = gsl_vector_get(solver->x, 2);
  muq_ = gsl_vector_get(solver->x, 3);
  gsl_multiroot_fsolver_free(solver);
  gsl_vector_free(x);
}

std::pair<double, double> ScaledVarianceCalculator::scaled_variance(
    std::function<bool(const smash::ParticleTypePtr)> type_of_interest) {

  constexpr unsigned int m = 5;
  Eigen::MatrixXd k2_tilde = Eigen::MatrixXd::Zero(m, m);
  double kappa1 = 0.0;

  for (const smash::ParticleTypePtr ptype : all_types_in_the_box_) {
    const double b = B_conservation_ ? ptype->baryon_number() : 0;
    const double s = S_conservation_ ? ptype->strangeness() : 0;
    const double q = Q_conservation_ ? ptype->charge() : 0;
    const double z = ptype->mass() / T_;
    const double mu = mub_ * b + mus_ * s + muq_ * q;
    const double mu_m_over_T = (mu - ptype->mass()) / T_;
    if (mu_m_over_T > 0 and quantum_statistics_) {
      std::cout << "Warning: quantum expressions for " << ptype->name() <<
                   " do not converge, m < chemical potential." << std::endl;
    }
    const double factor = ptype->pdgcode().spin_degeneracy() * 4.0 * M_PI *
                          std::pow(T_ / (2.0 * M_PI * smash::hbarc), 3);
    double EE = 0.0, EN = 0.0, NN = 0.0, N1 = 0.0;
    // std::cout << "Computing matrix for " << ptype.name() << std::endl;
    for (unsigned int k = 1; k < quantum_series_max_terms_; k++) {
      if (k > 1 and !quantum_statistics_) {
        break;
      }
      const double k1 = gsl_sf_bessel_Kn_scaled(1, z * k);
      const double k2 = gsl_sf_bessel_Kn_scaled(2, z * k);
      const double x = std::exp(mu_m_over_T * k);
      double N1_summand = z * z / k * k2 * x;
      double NN_summand = z * z * k2 * x;
      double EN_summand = z * z / k * (3 * k2 + k * z * k1) * x;
      double EE_summand = z * z / (k * k) *
                          ((z * z * k * k + 12.0) * k2 + 3.0 * z * k * k1) * x;
      // std::cout << "k = " << k
      //          << ", N1_summand*factor*1000 = " << N1_summand*factor*1000
      //          << ", NN_summand*factor*1000 = " << NN_summand*factor*1000
      //          << ", EN_summand = " << EN_summand
      //          << ", EE_summand = " << EE_summand << std::endl;
      if (k > 1 and
          EE_summand < EE * quantum_series_rel_precision_ and
          EN_summand < EN * quantum_series_rel_precision_ and
          NN_summand < NN * quantum_series_rel_precision_ and
          N1_summand < N1 * quantum_series_rel_precision_) {
        break;
      }
      if (k % 2 == 0 and ptype->pdgcode().is_baryon()) {
        NN_summand = -NN_summand;
        EN_summand = -EN_summand;
        EE_summand = -EE_summand;
        N1_summand = -N1_summand;
      }
      NN += NN_summand;
      EN += EN_summand;
      EE += EE_summand;
      N1 += N1_summand;
    }
    EE *= factor;
    EN *= factor;
    NN *= factor;
    N1 *= factor;

    Eigen::MatrixXd k2_tilde_ptype(m, m);
    // clang-format off
    k2_tilde_ptype << NN,     EN,    q*NN,    b*NN,    s*NN,
                      EN,     EE,    q*EN,    b*EN,    s*EN,
                      q*NN, q*EN,  q*q*NN,  q*b*NN,  q*s*NN,
                      b*NN, b*EN,  b*q*NN,  b*b*NN,  b*s*NN,
                      s*NN, s*EN,  s*q*NN,  s*b*NN,  s*s*NN;
    // clang-format on
    if (!type_of_interest(ptype)) {
      k2_tilde_ptype.row(0).setZero();
      k2_tilde_ptype.col(0).setZero();
      N1 = 0.0;
    }
    kappa1 += N1;
    k2_tilde += k2_tilde_ptype;
  }

  if (!energy_conservation_) {
    k2_tilde.row(1).setZero();
    k2_tilde.col(1).setZero();
  }
  // Remove zero rows and columns before asking for determinant
  // This uses the property of k2_tilde matrix being symmetric
  Eigen::Matrix<bool, 1, Eigen::Dynamic> non_zeros =
      k2_tilde.cast<bool>().colwise().any();
  const unsigned int n = non_zeros.count();
  Eigen::MatrixXd k2_tilde_nz(n, n);
  int i = 0, j;
  for (unsigned int i0 = 0; i0 < m; ++i0) {
    if (!non_zeros(i0)) {
      continue;
    }
    j = 0;
    for (unsigned int j0 = 0; j0 < m; ++j0) {
      if (non_zeros(j0)) {
        k2_tilde_nz(i, j) = k2_tilde(i0, j0);
        j++;
      }
    }
    i++;
  }

  const double rho_type_interest = kappa1;
  // No conservation laws: grand-canonical case
  if (n < 2) {
    return std::make_pair(rho_type_interest,
                          k2_tilde_nz(0,0) / kappa1);
  }

  const double det_k2_tilde = k2_tilde_nz.determinant();
  const double det_k2 = k2_tilde_nz.block(1, 1, n - 1, n - 1).determinant();
  const double expected_scaled_variance = det_k2_tilde / det_k2 /
                                          rho_type_interest;
  return std::make_pair(rho_type_interest, expected_scaled_variance);
}

void ScaledVarianceCalculator::prepare_decays() {
  std::vector<smash::ParticleTypePtr> all_stable_hadrons;
  for (const smash::ParticleType &ptype : smash::ParticleType::list_all()) {
    if (ptype.is_stable()) {
      all_stable_hadrons.push_back(&ptype);
    }
  }

  for (const smash::ParticleTypePtr res : all_types_in_the_box_) {
    smash::decaytree::Node tree(res->name(), 1.0, {res}, {res}, {res}, {});
    constexpr double sqrts = 5.0;  // This is sufficient to add all possible decays
    smash::decaytree::add_decays(tree, sqrts);
    std::vector<smash::FinalStateCrossSection> fs = tree.final_state_cross_sections();
    smash::deduplicate(fs);
    double wsum = 0.0;
    for (const smash::FinalStateCrossSection &xs : fs) {
      wsum += xs.cross_section_;
    }
    assert(std::abs(wsum - 1.0) < 1.e-9);
    std::cout << res->name() << std::endl;
    for (const smash::FinalStateCrossSection &xs : fs) {
      std::map<smash::ParticleTypePtr, int> decay_final_states;
      // For each stable particle count how many of it one finds in the final state
      for (const smash::ParticleTypePtr ptype : all_stable_hadrons) {
        std::string little_str = ptype->name(),
                    big_str = xs.name_;
        // How many times little_str occurs in big_str?
        int nPos = big_str.find(little_str, 0);
        int count = 0;
        while (nPos != std::string::npos) {
          count++;
          nPos = big_str.find(little_str, nPos + little_str.size());
        }
        if (count > 0) {
          decay_final_states[ptype] = count;
        }
      }
      all_decay_final_states_[res].emplace_back(
          std::make_pair(xs.cross_section_, decay_final_states));
      std::cout << "    " << xs.name_ << " " << xs.cross_section_ << ";   ";
      for (const auto &hadron_count : decay_final_states) {
        std::cout << hadron_count.second << hadron_count.first->name() << " ";
      }
      std::cout << std::endl;
    }
  }
}

std::ostream& operator<< (std::ostream& out,
                          const ScaledVarianceCalculator &svc) {
  out << "Scaled variance calculator" << std::endl;
  out << "T [GeV] = " << svc.T_
      << ", muB [GeV] = " << svc.mub_
      << ", muS [GeV] = " << svc.mus_
      << ", muQ [GeV] = " << svc.muq_ << std::endl;
  out << "Quantum statistics = " << svc.quantum_statistics_ << std::endl;
  out << "Included conservation laws: ";
  if (svc.energy_conservation_) {
    out << " energy;";
  }
  if (svc.B_conservation_) {
    out << "  baryon number;";
  }
  if (svc.S_conservation_) {
    out << " strangeness;";
  }
  if (svc.Q_conservation_) {
    out << " electric charge;";
  }
  out << std::endl;
  out << "Species in the box:" << std::endl;
  for (const smash::ParticleTypePtr t : svc.all_types_in_the_box_) {
    out << t->name() << " ";
  }
  out << std::endl;
  return out;
}


int main() {
  load_particle_types();

  // Prepare the set of particle species in the box
  std::vector<smash::ParticleTypePtr> hadrons_in_the_box;
  for (const smash::ParticleType &t : smash::ParticleType::list_all()) {
    if (t.is_hadron() && t.mass() < 2.0) {
      hadrons_in_the_box.push_back(&t);
    }
  }
  std::sort(hadrons_in_the_box.begin(), hadrons_in_the_box.end(),
            [&](smash::ParticleTypePtr ta, smash::ParticleTypePtr tb) {
              return ta->mass() < tb->mass();
            });

  const double Temperature = 0.2;  // [GeV]
  const double muB = 0.0;  // [GeV]
  const double muS = 0.0;  // [GeV]
  const double muQ = 0.0;  // [GeV]
  const bool E_conservation = true;
  const bool B_conservation = true;
  const bool S_conservation = true;
  const bool Q_conservation = true;
  const bool quantum_statistics = false;
  ScaledVarianceCalculator svc(hadrons_in_the_box,
                               Temperature, muB, muS, muQ,
                               E_conservation, B_conservation,
                               S_conservation, Q_conservation,
                               quantum_statistics);
  svc.prepare_decays();
/*
  const double V = 1762.1897;  // [fm^3]
  const double E_tot = 972.4227;  // [GeV]
  const double B_tot = 0.0;
  const double S_tot = 0.0;
  const double Q_tot = 0.0;
  svc.setTmu_from_conserved(E_tot, V, B_tot, S_tot, Q_tot);
  std::cout << svc;

  // Variance of each specie
  for (const smash::ParticleTypePtr t : hadrons_in_the_box) {
    const auto density_and_variance = svc.scaled_variance(
          [&](const smash::ParticleTypePtr t0) { return t0 == t; });
    std::cout << t->name() << " " << density_and_variance.first * V << " "
              << density_and_variance.second << std::endl;
  }

  // Variance of total number
  const auto density_and_variance = svc.scaled_variance(
      [&](const smash::ParticleTypePtr) { return true; });
  std::cout << "Ntot " << density_and_variance.first * V << " "
            << density_and_variance.second << std::endl;
*/
}
