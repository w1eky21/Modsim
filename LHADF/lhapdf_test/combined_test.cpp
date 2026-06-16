#include <iostream>
#include <random>
#include <cmath>
#include <ctime>
#include <vector>
#include "LHAPDF/LHAPDF.h"
#include "Pythia8/Pythia.h"

// ─────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────
constexpr double PT_MIN  =   50.0;   // GeV
constexpr double PT_MAX  = 7000.0;   // GeV
constexpr double X_MIN   =  1e-5;
constexpr double X_MAX   =    1.0;
constexpr double MU_F    =   91.2;   // factorisation scale GeV
constexpr double C_ENV   =   12.0;   // PDF envelope bound
constexpr int    GLUON   =     21;
constexpr int    N_EVENTS = 10000;
constexpr int    N_ITER   =     8;

// ─────────────────────────────────────────────
//  Physics-based bias function
//  Encodes exactly what q0 knows:
//    - 1/pT^2 from QCD soft singularity
//    - f(x1)*f(x2) from PDFs
//  This is evaluated by Pythia at every phase
//  space point it tries internally
// ─────────────────────────────────────────────
class PhysicsBiasHook : public Pythia8::UserHooks {
public:
    // constructor loads the PDF set once
    PhysicsBiasHook() {
        pdf_     = LHAPDF::mkPDF("NNPDF31_lo_as_0118", 0);
        norm_pT_ = 1.0 / PT_MIN - 1.0 / PT_MAX;
        norm_x_  = std::log(X_MAX / X_MIN);
    }

    ~PhysicsBiasHook() { delete pdf_; }

    // tell Pythia we want to control the bias
    bool canBiasSelection() override { return true; }

    // called by Pythia at every phase space point
    // return q0(pThat, x1, x2) as the bias value
    // Pythia divides this out automatically — weights stay correct
    double biasSelectionBy(const Pythia8::SigmaProcess* sigmaProcessPtr,
                           const Pythia8::PhaseSpace*   phaseSpacePtr,
                           bool                         inEvent) override {

        // get the kinematics Pythia is currently trying
        double pThat = phaseSpacePtr->pTHat();
        double x1    = phaseSpacePtr->x1();
        double x2    = phaseSpacePtr->x2();

        // safety checks — avoid unphysical points
        if (pThat <= PT_MIN) return 1.0;
        if (x1 <= 0.0 || x1 >= 1.0) return 1.0;
        if (x2 <= 0.0 || x2 >= 1.0) return 1.0;

        // ── q0 evaluated at this point ──

        // 1/pT^2 term — from QCD matrix element structure
        double bias_pT = (1.0 / (pThat * pThat)) / norm_pT_;

        // PDF terms f(x1) * f(x2)
        double fx1 = pdf_->xfxQ(GLUON, x1, MU_F) / (x1 * norm_x_);
        double fx2 = pdf_->xfxQ(GLUON, x2, MU_F) / (x2 * norm_x_);

        // full q0 bias — product of all physics-motivated terms
        return bias_pT * fx1 * fx2;
    }

private:
    LHAPDF::PDF* pdf_;
    double norm_pT_;
    double norm_x_;
};

// ─────────────────────────────────────────────
//  Run one batch — returns (pThat, weight) pairs
// ─────────────────────────────────────────────
std::vector<std::pair<double,double>> run_batch(int N) {

    auto hook = std::make_shared<PhysicsBiasHook>();

    Pythia8::Pythia pythia;
    pythia.readString("HardQCD:all = on");
    pythia.readString("SoftQCD:all = off");
    pythia.readString(std::string("PhaseSpace:pThatMin = ")
                      + std::to_string(PT_MIN));
    pythia.readString("Beams:eCM = 13000.");
    pythia.readString("PartonLevel:all = off");
    pythia.readString("Print:quiet = on");
    pythia.setUserHooksPtr(hook);
    pythia.init();

    std::vector<std::pair<double,double>> results;
    results.reserve(N);

    for (int i = 0; i < N; ++i) {
        if (!pythia.next()) continue;
        double pThat = pythia.info.pTHat();
        double w     = pythia.info.weight();
        results.push_back({pThat, w});
    }

    return results;
}

// ─────────────────────────────────────────────
//  Diagnostics
// ─────────────────────────────────────────────
void print_diagnostics(int iter,
                       const std::vector<std::pair<double,double>>& events) {
    double sum_w  = 0.0;
    double sum_w2 = 0.0;
    int    above  = 0;

    for (const auto& e : events) {
        sum_w  += e.second;
        sum_w2 += e.second * e.second;
        if (e.first > 2000.0) ++above;
    }

    double Neff      = (sum_w * sum_w) / (sum_w2 * events.size());
    double frac_2TeV = static_cast<double>(above) / events.size();
    double mean_w    = sum_w / events.size();

    printf("Iter %-4d | Neff/N = %-8.4f | frac>2TeV = %-10.5f | mean_w = %.3e\n",
           iter, Neff, frac_2TeV, mean_w);
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────
int main() {

    printf("\nUsing physics-based q0 bias: f(x1)*f(x2)/pT^2\n");
    printf("%s\n\n", std::string(60, '-').c_str());

    // run with the physics bias — no iteration needed yet
    // q0 is fixed, not adaptive
    // this gives you the baseline performance of the physics bias
    for (int iter = 0; iter < N_ITER; ++iter) {
        auto events = run_batch(N_EVENTS);
        print_diagnostics(iter, events);
    }

    return 0;
}