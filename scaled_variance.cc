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
    double T, double mub, double mus,
    std::function<bool(const smash::ParticleType &)> type_in_the_box,
    std::function<bool(const smash::ParticleType &)> type_of_interest,
    bool energy_conservation, bool B_conservation, bool S_conservation,
    bool Q_conservation) {

  constexpr unsigned int m = 5;
  Eigen::MatrixXd k2_tilde = Eigen::MatrixXd::Zero(m, m);

  double rho_type_interest = 0.0;
  for (const smash::ParticleType &ptype : smash::ParticleType::list_all()) {
    if (!type_in_the_box(ptype)) {
      continue;
    }
    const double rho_type =
        smash::HadronGasEos::partial_density(ptype, T, mub, mus);
    const double b = B_conservation ? ptype.baryon_number() : 0;
    const double s = S_conservation ? ptype.strangeness() : 0;
    const double q = Q_conservation ? ptype.charge() : 0;
    const double z = ptype.mass() / T;
    const double h =
        z * gsl_sf_bessel_Kn_scaled(3, z) / gsl_sf_bessel_Kn_scaled(2, z);
    const double a2 = energy_conservation ? T * (h - 1.0) : 0.0;
    const double a3 = energy_conservation ? T * T * (3.0 * h + z * z) : 0.0;

    Eigen::MatrixXd k2_tilde_ptype(m, m);
    // clang-format off
    k2_tilde_ptype << 1,   a2,    q,    b,    s,
                     a2,   a3, a2*q, a2*b, a2*s,
                      q, a2*q,  q*q,  q*b,  q*s,
                      b, a2*b,  b*q,  b*b,  b*s,
                      s, a2*s,  s*q,  s*b,  s*s;
    // clang-format on
    if (type_of_interest(ptype)) {
      rho_type_interest += rho_type;
    } else {
      k2_tilde_ptype.row(0).setZero();
      k2_tilde_ptype.col(0).setZero();
    }

    k2_tilde += (k2_tilde_ptype * rho_type);
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

  // No conservation laws: grand-canonical case, therefore
  // Poisson distribution and scaled variance = 1.
  if (n < 2) {
    return std::make_pair(rho_type_interest, 1.0);
  }

  const double det_k2_tilde = k2_tilde_nz.determinant();
  const double det_k2 = k2_tilde_nz.block(1, 1, n - 1, n - 1).determinant();
  const double expected_scaled_variance =
      det_k2_tilde / det_k2 / rho_type_interest;
  return std::make_pair(rho_type_interest, expected_scaled_variance);
}

int main() {
  load_particle_types();
  const double T = 0.129413;
  const double mub = 0.0;
  const double mus = 0.0;
  const double V = 1000.0;
  // Gas of only pions
  const std::vector<smash::PdgCode> pdgs_of_interest = {
      smash::pdg::pi_p, smash::pdg::pi_m, smash::pdg::pi_z};
  for (const smash::PdgCode &pdg : pdgs_of_interest) {
    const smash::ParticleType &ptype = smash::ParticleType::find(pdg);
    const auto density_and_variance =
        scaled_variance(T, mub, mus,
                        [&](const smash::ParticleType &type_is_in_the_box) {
                          return type_is_in_the_box.pdgcode().is_pion();
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
      T, mub, mus,
      [&](const smash::ParticleType &t) { return t.pdgcode().is_pion(); },
      [&](const smash::ParticleType &) { return true; }, true, true, true,
      true);
  std::cout << "Ntot"
            << " " << density_and_variance.first * V << " "
            << density_and_variance.second << std::endl;
}
