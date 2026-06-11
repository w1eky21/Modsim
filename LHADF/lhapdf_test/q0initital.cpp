#include <iostream>
#include <random>
#include <cmath>
#include <ctime>
// ─────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────
constexpr double PT_MIN  =   50.0;   // GeV - minimum transverse momentum
constexpr double PT_MAX  = 7000.0;   // GeV - maximum transverse momentum (half LHC beam energy)
constexpr double Y_MAX   =    5.0;   // rapidity range [-5, 5]
constexpr double PI      = M_PI;     // pi from cmath

// ─────────────────────────────────────────────
//  Phase space point
// ─────────────────────────────────────────────
struct PhaseSpacePoint {
    double pT1, pT2, pT3;   // transverse momenta of 3 outgoing partons
    double y1,  y2,  y3;    // rapidities of 3 outgoing partons
    double phi1, phi2;      // azimuthal angles of 2 outgoing partons
    double q0;              // proposal density at this point
};

// ─────────────────────────────────────────────
//  Sampler
// ─────────────────────────────────────────────
class Q0Sampler {
public:
    Q0Sampler(unsigned int seed =  30){
        rng_     = std::mt19937(seed);
        uni01_   = std::uniform_real_distribution<double>(0.0, 1.0);
        uni_y_   = std::uniform_real_distribution<double>(-Y_MAX, Y_MAX);
        uni_phi_ = std::uniform_real_distribution<double>(0.0, 2.0 * PI);

        norm_pT_ = 1.0 / PT_MIN - 1.0 / PT_MAX;  // normalisation for 1/pT^2
        den_y_   = 1.0 / (2.0 * Y_MAX);           // uniform rapidity density
        den_phi_ = 1.0 / (2.0 * PI);              // uniform phi density
    }

    PhaseSpacePoint sample() {
        PhaseSpacePoint p;

        p.pT1  = sample_pT();    // draw pT1 from 1/pT^2
        p.pT2  = sample_pT();    // draw pT2 from 1/pT^2
        p.pT3  = sample_pT();    // draw pT3 from 1/pT^2
        p.y1   = uni_y_(rng_);   // draw y1 uniformly on [-5, 5]
        p.y2   = uni_y_(rng_);   // draw y2 uniformly on [-5, 5]
        p.y3   = uni_y_(rng_);   // draw y3 uniformly on [-5, 5]
        p.phi1 = uni_phi_(rng_); // draw phi1 uniformly on [0, 2pi]
        p.phi2 = uni_phi_(rng_); // draw phi2 uniformly on [0, 2pi]
        p.q0   = density(p);     // evaluate proposal density at this point

        return p;
    }

    double density(const PhaseSpacePoint& p) {
        return  pT_den(p.pT1) * pT_den(p.pT2) * pT_den(p.pT3)  // 1/pT^2 densities
              * den_y_   * den_y_   * den_y_                     // uniform rapidity densities
              * den_phi_ * den_phi_;                             // uniform phi densities
    }

private:
    // ── sample pT from 1/pT^2 via exact inverse CDF ──
    double sample_pT() {
        double u = uni01_(rng_);                       // uniform random number in [0,1]
        return 1.0 / (1.0/PT_MIN - u * norm_pT_);     // inverse CDF maps u to pT in [PT_MIN, PT_MAX]
    }

    // ── normalised 1/pT^2 density ──
    double pT_den(double pT) {
        return (1.0 / (pT * pT)) / norm_pT_;          // (1/pT^2) divided by normalisation constant
    }

    std::mt19937                           rng_;
    std::uniform_real_distribution<double> uni01_;
    std::uniform_real_distribution<double> uni_y_;
    std::uniform_real_distribution<double> uni_phi_;
    double norm_pT_;
    double den_y_;
    double den_phi_;
};

// ─────────────────────────────────────────────
//  Main - sample and print one point
// ─────────────────────────────────────────────

int main() {
    // Seed with current time for different results each run
    unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
    Q0Sampler sampler(seed);
    PhaseSpacePoint p = sampler.sample();

    printf("pT1  = %.2f GeV\n", p.pT1);
    printf("pT2  = %.2f GeV\n", p.pT2);
    printf("pT3  = %.2f GeV\n", p.pT3);
    printf("y1   = %.4f\n",     p.y1);
    printf("y2   = %.4f\n",     p.y2);
    printf("y3   = %.4f\n",     p.y3);
    printf("phi1 = %.4f rad\n", p.phi1);
    printf("phi2 = %.4f rad\n", p.phi2);
    printf("q0   = %.4e\n",     p.q0);

    return 0;
}
