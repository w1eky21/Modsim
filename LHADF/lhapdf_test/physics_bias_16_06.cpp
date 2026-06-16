#include <iostream>
#include <cmath>
#include <vector>
#include "LHAPDF/LHAPDF.h"
#include "Pythia8/Pythia.h"


/*

g++ -O2 -std=c++17 main.cpp \
    -I/home/stijn/pythia8317/include \
    -L/home/stijn/pythia8317/lib \
    -Wl,-rpath,/home/stijn/pythia8317/lib \
    -lpythia8 \
    $(lhapdf-config --cflags --libs) \
    -o bias_search
*/


// ─────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────
constexpr double PT_MIN   =  100.0;   // GeV - hard floor
constexpr double PT_TARGET = 2000.0;  // GeV - what we want to reach
constexpr double MU_F     =   91.2;   // factorisation scale
constexpr int    GLUON    =    21;
constexpr int    N_EVENTS =  10000;

// ─────────────────────────────────────────────
//  Evaluate q0 bias at a given pThat, x1, x2
//  This is our physics-based proposal:
//    q0 = f(x1) * f(x2) * (1/pT^2)
// ─────────────────────────────────────────────
double q0_bias(LHAPDF::PDF* pdf, double pThat, double x1, double x2) {
    if (pThat <= 0 || x1 <= 0 || x2 <= 0) return 1.0;
    if (x1 >= 1.0 || x2 >= 1.0)           return 1.0;

    double fx1    = pdf->xfxQ(GLUON, x1, MU_F) / x1;  // f(x1)
    double fx2    = pdf->xfxQ(GLUON, x2, MU_F) / x2;  // f(x2)
    double bias   = fx1 * fx2 / (pThat * pThat);       // q0

    // protect against zero or negative PDF values
    if (bias <= 0.0) return 1.0;
    return bias;
}

// ─────────────────────────────────────────────
//  UserHooks — correct Pythia 8.3 interface
// ─────────────────────────────────────────────
class PhysicsBiasHook : public Pythia8::UserHooks {
public:
    PhysicsBiasHook() {
        pdf_ = LHAPDF::mkPDF("NNPDF31_lo_as_0118", 0);
    }
    ~PhysicsBiasHook() { delete pdf_; }

   bool canBiasSelection() override { return true; }

double biasSelectionBy(const Pythia8::SigmaProcess* sigmaProcessPtr,
                       const Pythia8::PhaseSpace*   phaseSpacePtr,
                       bool                         inEvent) override {
    double pThat = phaseSpacePtr->pTHat();
    double x1    = phaseSpacePtr->x1();
    double x2    = phaseSpacePtr->x2();
    return q0_bias(pdf_, pThat, x1, x2);
}
private:
    LHAPDF::PDF* pdf_;
};

// ─────────────────────────────────────────────
//  Diagnostics
// ─────────────────────────────────────────────
void print_diagnostics(const std::vector<std::pair<double,double>>& events) {
    double sum_w  = 0.0;
    double sum_w2 = 0.0;
    double w_max  = 0.0;
    int    above  = 0;

    for (const auto& e : events) {
        sum_w  += e.second;
        sum_w2 += e.second * e.second;
        if (e.second > w_max) w_max = e.second;
        if (e.first > PT_TARGET) ++above;
    }

    int    N        = events.size();
    double Neff     = (sum_w * sum_w) / (sum_w2 * N);
    double frac_2T  = static_cast<double>(above) / N;
    double mean_w   = sum_w / N;

    printf("N = %-6d | Neff/N = %-8.4f | frac>2TeV = %-10.5f | "
           "mean_w = %-10.3e | max_w = %.3e\n",
           N, Neff, frac_2T, mean_w, w_max);
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────
int main() {

    auto hook = std::make_shared<PhysicsBiasHook>();

    Pythia8::Pythia pythia;

    // ── process selection: hard QCD jets ──
    pythia.readString("HardQCD:all = on");
    pythia.readString("SoftQCD:all = off");

    // ── beam setup ──
    pythia.readString("Beams:eCM = 13000.");

    // ── phase space cuts ──
    pythia.readString("PhaseSpace:pThatMin = 100.");   // hard floor
    pythia.readString("PhaseSpace:pThatMax = 7000.");

    // ── no shower for now — matrix element only ──
    pythia.readString("PartonLevel:MPI  = off");
    pythia.readString("PartonLevel:ISR  = off");
    pythia.readString("PartonLevel:FSR  = off");
    pythia.readString("HadronLevel:all  = off");

    // ── suppress banner spam after first iteration ──
    pythia.readString("Print:quiet = off");  // keep first banner
    pythia.readString("Next:numberShowInfo = 0");
    pythia.readString("Next:numberShowProcess = 0");
    pythia.readString("Next:numberShowEvent = 0");

    pythia.setUserHooksPtr(hook);
    pythia.init();

    // ── event loop ──
    std::vector<std::pair<double,double>> events;
    events.reserve(N_EVENTS);

    for (int i = 0; i < N_EVENTS; ++i) {
        if (!pythia.next()) continue;

        double pThat = pythia.info.pTHat();
        double w     = pythia.info.weightSum();  // weight with bias divided out

        events.push_back({pThat, w});
    }

    // ── print pThat distribution summary ──
    printf("\nPhysics-based q0 bias results:\n");
    printf("%s\n", std::string(80, '-').c_str());
    print_diagnostics(events);

    // ── simple pThat histogram ──
    printf("\npThat distribution:\n");
    printf("%-20s %-10s %-10s\n", "pThat range (GeV)", "count", "sum_w");
    double edges[] = {100, 200, 500, 1000, 2000, 3500, 7000};
    int nbins = 6;
    for (int b = 0; b < nbins; ++b) {
        int    count = 0;
        double sumw  = 0.0;
        for (const auto& e : events) {
            if (e.first >= edges[b] && e.first < edges[b+1]) {
                ++count;
                sumw += e.second;
            }
        }
        printf("%-6.0f - %-12.0f %-10d %-10.3e\n",
               edges[b], edges[b+1], count, sumw);
    }

    return 0;
}