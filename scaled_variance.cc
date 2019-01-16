#include "scaled_variance.h"

#include <Eigen/Dense>
#include <fstream>
#include <gsl/gsl_sf_bessel.h>

#include "smash/hadgas_eos.h"

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

/**
 * Returns expected scaled variance for a set of species defined by
 * type_of_interest function, in a Boltzmann gas of particles, where species
 * are set by type_in_the_box. For reference see arxiv.org/pdf/0706.3290.pdf,
 * Eq. (29-40). The approximations of this paper are only valid for large
 * enough volume.
 *
 * \param[in] T temperature of the gas
 * \param[in] mub baryo-chemical potential of the gas
 * \param[in] mus strangeness chemical potential of the gas
 * \param[in] type_in_the_box a function, that returns true for species, which
 *            are included in the gas
 * \param[in] type_of_interest a function that returns true for species, for
 *            which the scaled variance is computed
 * \param[in] energy_conservation if ensemble includes energy conservation
 * \param[in] energy_conservation if ensemble includes
 *            baryon number conservation
 * \param[in] energy_conservation if ensemble includes strangeness conservation
 * \param[in] energy_conservation if ensemble includes charge conservation
 * \param[out] mean density and scaled variance of type_of_interest species
 */
std::pair<double, double> scaled_variance(
    double T, double mub, double mus, double muq, bool quantum_statistics,
    std::function<bool(const smash::ParticleType &)> type_in_the_box,
    std::function<bool(const smash::ParticleType &)> type_of_interest,
    bool energy_conservation, bool B_conservation, bool S_conservation,
    bool Q_conservation) {

  constexpr unsigned int m = 5;
  Eigen::MatrixXd k2_tilde = Eigen::MatrixXd::Zero(m, m);

  for (const smash::ParticleType &ptype : smash::ParticleType::list_all()) {
    if (!type_in_the_box(ptype)) {
      continue;
    }
    const double b = B_conservation ? ptype.baryon_number() : 0;
    const double s = S_conservation ? ptype.strangeness() : 0;
    const double q = Q_conservation ? ptype.charge() : 0;
    const double z = ptype.mass() / T;
    const double mu = mub * b + mus * s + muq *q;
    const double mu_m_over_T = (mu - ptype.mass()) / T;
    if (mu_m_over_T > 0 and quantum_statistics) {
      std::cout << "Warning: quantum expressions for " << ptype.name() <<
                   " do not converge, m < chemical potential." << std::endl;
    }
    const double factor = ptype.pdgcode().spin_degeneracy() * 4.0 * M_PI *
                          std::pow(T / (2.0 * M_PI * smash::hbarc), 3);
    double EE = 0.0, EN = 0.0, NN = 0.0;
    constexpr unsigned int maximum_terms = 50;
    // std::cout << "Computing matrix for " << ptype.name() << std::endl;
    for (unsigned int k = 1; k < maximum_terms; k++) {
      if (k > 1 and !quantum_statistics) {
        break;
      }
      const double k1 = gsl_sf_bessel_Kn_scaled(1, z * k);
      const double k2 = gsl_sf_bessel_Kn_scaled(2, z * k);
      const double x = std::exp(mu_m_over_T * k);
      double NN_summand = z * z / k * k2 * x;
      double EN_summand = z * z / (k * k) * (3 * k2 + k * z * k1) * x;
      double EE_summand = z * z * z / (k * k) *
                          ((z * z * k * k + 12.0) * k2 + 3.0 * k1) * x;
      // std::cout << "k = " << k
      //           << ", NN_summand*factor*1000 = " << NN_summand*factor*1000
      //           << ", EN_summand = " << EN_summand
      //           << ", EE_summand = " << EE_summand << std::endl;
      if (k > 1 and
          EE_summand < EE * 1e-12 and
          EN_summand < EN * 1e-12 and
          NN_summand < NN * 1e-12) {
        break;
      }
      if (k % 2 == 0 and ptype.pdgcode().is_baryon()) {
        NN_summand = -NN_summand;
        EN_summand = -EN_summand;
        EE_summand = -EE_summand;
      }
      NN += NN_summand;
      EN += EN_summand;
      EE += EE_summand;
    }
    EE *= factor;
    EN *= factor;
    NN *= factor;

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
    }
    k2_tilde += k2_tilde_ptype;
  }

  if (!energy_conservation) {
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

  double rho_type_interest = k2_tilde_nz(0, 0);

  // No conservation laws: grand-canonical case, therefore
  // Poisson distribution and scaled variance = 1.
  if (n < 2) {
    return std::make_pair(rho_type_interest, 1.0);
  }

  const double det_k2_tilde = k2_tilde_nz.determinant();
  const double det_k2 = k2_tilde_nz.block(1, 1, n - 1, n - 1).determinant();
  const double expected_scaled_variance = det_k2_tilde / det_k2 /
                                          rho_type_interest;
  return std::make_pair(rho_type_interest, expected_scaled_variance);
}

int main() {
  load_particle_types();
  smash::HadronGasEos eos(false);
  const double V = 1000.0;
  const double Energy = 500.0;
  const double B_tot = 0.0, S_tot = 0.0;
  constexpr bool quantum_statistics = false;

  // The included hadron sorts are defined by smash::HadronGasEos::is_eos_particle
  // std::array<double, 3> T_muB_muS = eos.solve_eos(Energy/V, B_tot/V, S_tot/V);
  const double T = 0.2; // T_muB_muS[0];
  const double mub = 0.0; // T_muB_muS[1];
  const double mus = 0.0; // T_muB_muS[2];
  const double muq = 0.0;
  std::cout << "Energy [GeV] = " << Energy
            << ", Volume [fm^3] = " << V
            << ", B_tot = " << B_tot
            << ", S_tot = " << S_tot << std::endl;
  std::cout << "T [GeV] = " << T
            << ", muB [GeV] = " << mub
            << ", muS [GeV] = " << mus
            << ", muQ = " << muq << std::endl;
  std::cout << "Quantum statistics = " << quantum_statistics << std::endl;

  std::vector<smash::PdgCode> pdgs_of_interest;
  for (const smash::ParticleType &t : smash::ParticleType::list_all()) {
    if (smash::HadronGasEos::is_eos_particle(t)) {
      pdgs_of_interest.push_back(t.pdgcode());
    }
  }
  std::sort(pdgs_of_interest.begin(), pdgs_of_interest.end(),
            [&](smash::PdgCode a, smash::PdgCode b) {
              smash::ParticleTypePtr ta = &smash::ParticleType::find(a);
              smash::ParticleTypePtr tb = &smash::ParticleType::find(b);
              return ta->mass() < tb->mass();
            });
  for (const smash::PdgCode &pdg : pdgs_of_interest) {
    const smash::ParticleType &ptype = smash::ParticleType::find(pdg);
    const auto density_and_variance =
        scaled_variance(T, mub, mus, muq, quantum_statistics,
          [&](const smash::ParticleType &type_is_in_the_box) {
            return smash::HadronGasEos::is_eos_particle(type_is_in_the_box);
          },
          [&](const smash::ParticleType &type_condition) {
            return type_condition == ptype;
          },
          true, true, true, true);
    std::cout << ptype.name() << " " << density_and_variance.first * V << " "
              << density_and_variance.second << std::endl;
  }

  // Variance of total number
  const auto density_and_variance = scaled_variance(
      T, mub, mus, muq, quantum_statistics,
      [&](const smash::ParticleType &t) {
        return smash::HadronGasEos::is_eos_particle(t);
      },
      [&](const smash::ParticleType &) { return true; }, true, true, true,
      true);
  std::cout << "Ntot"
            << " " << density_and_variance.first * V << " "
            << density_and_variance.second << std::endl;
}
