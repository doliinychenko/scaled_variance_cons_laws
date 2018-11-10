#ifndef SCALED_VARIANCE_H
#define SCALED_VARIANCE_H

#include "smash/decaymodes.h"
#include "smash/particletype.h"

bool load_particle_types();

std::pair<double, double> scaled_variance(
    double T, double mub, double mus,
    std::function<bool(const smash::ParticleType &)> type_in_the_box,
    std::function<bool(const smash::ParticleType &)> type_of_interest,
    bool energy_conservation, bool B_conservation, bool S_conservation,
    bool Q_conservation);

void initialize_random_number_generator();

#endif // SCALED_VARIANCE_H
