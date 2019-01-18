#include "scaled_variance.h"

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
    if (t.is_hadron() && t.mass() < 1.0) {
      hadrons_in_the_box.push_back(&t);
    }
  }
  std::sort(hadrons_in_the_box.begin(), hadrons_in_the_box.end(),
            [&](smash::ParticleTypePtr ta, smash::ParticleTypePtr tb) {
              return ta->mass() < tb->mass();
            });

  const double V = 1000.0;  // [fm^3]
  const double Temperature = 0.2;  // [GeV]
  const double muB = 0.0;  // [GeV]
  const double muS = 0.0;  // [GeV]
  const double muQ = 0.0;  // [GeV]
  const bool E_conservation = true;
  const bool B_conservation = true;
  const bool S_conservation = true;
  const bool Q_conservation = true;
  const bool quantum_statistics = true;
  ScaledVarianceCalculator svc(hadrons_in_the_box,
                               Temperature, muB, muS, muQ,
                               E_conservation, B_conservation,
                               S_conservation, Q_conservation,
                               quantum_statistics);
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
}
