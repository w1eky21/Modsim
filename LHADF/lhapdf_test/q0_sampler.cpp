
#include <iostream>
#include <random>
#include <cmath>
#include <ctime>
#include "LHAPDF/LHAPDF.h"

// ─────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────
constexpr double PT_MIN  =   50.0;   // GeV - minimum transverse momentum
constexpr double PT_MAX  = 7000.0;   // GeV - maximum transverse momentum
constexpr double Y_MAX   =    5.0;   // rapidity range [-5, 5]
constexpr double PI      = M_PI;     // pi from cmath
constexpr double X_MIN   =  1e-5;    // minimum parton momentum fraction
constexpr double X_MAX   =    1.0;   // maximum parton momentum fraction
constexpr double MU_F    = 91.2;     // factorisation scale in GeV (Z mass, standard choice)
constexpr double C_ENV   =   12.0;   // envelope bound: xf(x, mu_F) never exceeds this
constexpr int    GLUON   =   21;     // PDG gluon ID

// ─────────────────────────────────────────────
//  Phase space point
// ─────────────────────────────────────────────
struct PhaseSpacePoint {
    double x1,  x2;          // incoming parton momentum fractions
    double pT1, pT2, pT3;    // transverse momenta of 3 outgoing partons
    double y1,  y2,  y3;     // rapidities of 3 outgoing partons
    double phi1, phi2;       // azimuthal angles of 2 outgoing partons
    double q0;               // proposal density at this point
};

// ─────────────────────────────────────────────
//  Sampler
// ─────────────────────────────────────────────
class Q0Sampler {
public:
    Q0Sampler(unsigned int seed = 30) {
        rng_     = std::mt19937(seed);
        uni01_   = std::uniform_real_distribution<double>(0.0, 1.0);
        uni_y_   = std::uniform_real_distribution<double>(-Y_MAX, Y_MAX);
        uni_phi_ = std::uniform_real_distribution<double>(0.0, 2.0 * PI);

        pdf_     = LHAPDF::mkPDF("NNPDF31_lo_as_0118", 0);  // load PDF set

        norm_pT_ = 1.0 / PT_MIN - 1.0 / PT_MAX;  // normalisation for 1/pT^2
        norm_x_  = std::log(X_MAX / X_MIN);       // normalisation for 1/x proposal
        den_y_   = 1.0 / (2.0 * Y_MAX);           // uniform rapidity density
        den_phi_ = 1.0 / (2.0 * PI);              // uniform phi density
    }

    ~Q0Sampler() { delete pdf_; }  // clean up PDF object

    PhaseSpacePoint sample() {
        PhaseSpacePoint p;

        p.x1   = sample_x();     // draw x1 from gluon PDF via rejection sampling
        p.x2   = sample_x();     // draw x2 independently from same PDF
        p.pT1  = sample_pT();    // draw pT1 from 1/pT^2 via inverse CDF
        p.pT2  = sample_pT();    // draw pT2 from 1/pT^2 via inverse CDF
        p.pT3  = sample_pT();    // draw pT3 from 1/pT^2 via inverse CDF
        p.y1   = uni_y_(rng_);   // draw y1 uniformly on [-5, 5]
        p.y2   = uni_y_(rng_);   // draw y2 uniformly on [-5, 5]
        p.y3   = uni_y_(rng_);   // draw y3 uniformly on [-5, 5]
        p.phi1 = uni_phi_(rng_); // draw phi1 uniformly on [0, 2pi]
        p.phi2 = uni_phi_(rng_); // draw phi2 uniformly on [0, 2pi]
        p.q0   = density(p);     // evaluate proposal density at this point

        return p;
    }

    double density(const PhaseSpacePoint& p) {
        return  pdf_den(p.x1) * pdf_den(p.x2)           // f(x1) * f(x2)
              * pT_den(p.pT1) * pT_den(p.pT2) * pT_den(p.pT3)  // 1/pT^2 densities
              * den_y_   * den_y_   * den_y_             // uniform rapidity densities
              * den_phi_ * den_phi_;                     // uniform phi densities
    }

private:
    // ── sample x from gluon PDF via rejection sampling ──
    double sample_x() {
        while (true) {
            double u = uni01_(rng_);
            double x = X_MIN * std::pow(X_MAX / X_MIN, u);  // draw from 1/x proposal
            double accept = pdf_->xfxQ(GLUON, x, MU_F) / C_ENV;  // acceptance probability
            if (uni01_(rng_) < accept) return x;             // accept or retry
        }
    }

    // ── sample pT from 1/pT^2 via exact inverse CDF ──
    double sample_pT() {
        double u = uni01_(rng_);                        // uniform random number in [0,1]
        return 1.0 / (1.0/PT_MIN - u * norm_pT_);      // inverse CDF maps u to pT
    }

    // ── PDF density f(x) = xf(x,mu_F) / x ──
    double pdf_den(double x) {
        return pdf_->xfxQ(GLUON, x, MU_F) / (x * norm_x_);  // divided by 1/x normalisation
    }

    // ── normalised 1/pT^2 density ──
    double pT_den(double pT) {
        return (1.0 / (pT * pT)) / norm_pT_;
    }

    LHAPDF::PDF*                           pdf_;
    std::mt19937                           rng_;
    std::uniform_real_distribution<double> uni01_;
    std::uniform_real_distribution<double> uni_y_;
    std::uniform_real_distribution<double> uni_phi_;
    double norm_pT_;
    double norm_x_;
    double den_y_;
    double den_phi_;
};

// ─────────────────────────────────────────────
//  Main - sample and print one point
// ─────────────────────────────────────────────
int main() {
    unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
    Q0Sampler sampler(seed);
    PhaseSpacePoint p = sampler.sample();

    printf("x1   = %.4e\n",     p.x1);
    printf("x2   = %.4e\n",     p.x2);
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