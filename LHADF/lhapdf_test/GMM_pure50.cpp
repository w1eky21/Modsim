#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include "LHAPDF/LHAPDF.h"
#include "Pythia8/Pythia.h"

// ─────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────
constexpr double PT_MIN    =  10.0;   // GeV - floor well below target
constexpr double PT_MAX    = 7000.0;  // GeV
constexpr double PT_TARGET =  50.0;   // GeV - what we want events above
constexpr double MU_F      =  91.2;   // factorisation scale
constexpr int    GLUON     =   21;
constexpr int    N_EVENTS  = 50000;
constexpr int    N_ITER    =   100;

// 3D grid dimensions
constexpr int N_PT = 15;
constexpr int N_X  =  8;

// ─────────────────────────────────────────────
//  3D correction grid over (pT, x1, x2)
// ─────────────────────────────────────────────
struct Grid3D {
    const double log_pT_min = std::log(PT_MIN);
    const double log_pT_max = std::log(PT_MAX);
    const double log_x_min  = std::log(1e-5);
    const double log_x_max  = std::log(1.0);

    std::vector<double> corr  = std::vector<double>(N_PT*N_X*N_X, 1.0);
    std::vector<double> sum_w = std::vector<double>(N_PT*N_X*N_X, 0.0);
    std::vector<double> count = std::vector<double>(N_PT*N_X*N_X, 0.0);

    int idx(int bpT, int bx1, int bx2) const {
        return bpT * N_X * N_X + bx1 * N_X + bx2;
    }

    int bin_pT(double pT) const {
        double t = (std::log(pT) - log_pT_min)
                 / (log_pT_max  - log_pT_min);
        return std::max(0, std::min(N_PT-1, (int)(t * N_PT)));
    }

    int bin_x(double x) const {
        double t = (std::log(x) - log_x_min)
                 / (log_x_max  - log_x_min);
        return std::max(0, std::min(N_X-1, (int)(t * N_X)));
    }

    double eval(double pT, double x1, double x2) const {
        return corr[idx(bin_pT(pT), bin_x(x1), bin_x(x2))];
    }

    void accumulate(double pT, double x1, double x2, double w) {
        int i = idx(bin_pT(pT), bin_x(x1), bin_x(x2));
        sum_w[i] += w;
        count[i] += 1.0;
    }

    void reset() {
        std::fill(sum_w.begin(), sum_w.end(), 0.0);
        std::fill(count.begin(), count.end(), 0.0);
    }

    double coverage() const {
        int filled = 0;
        for (int i = 0; i < N_PT*N_X*N_X; ++i)
            if (count[i] > 2) ++filled;
        return (double)filled / (N_PT*N_X*N_X);
    }

    void update() {
        double global_sum = 0.0;
        int    filled     = 0;
        for (int i = 0; i < N_PT*N_X*N_X; ++i) {
            if (count[i] > 2) {
                global_sum += sum_w[i] / count[i];
                ++filled;
            }
        }
        if (filled == 0) { reset(); return; }
        double global_avg = global_sum / filled;

        for (int i = 0; i < N_PT*N_X*N_X; ++i) {
            if (count[i] < 3) continue;
            double avg_w       = sum_w[i] / count[i];
            double ratio       = avg_w / global_avg;
            double reliability = std::sqrt(count[i])
                               / (std::sqrt(count[i]) + 5.0);
            double new_corr    = corr[i] * ratio;
            corr[i] = (1.0 - reliability) * corr[i]
                    +        reliability  * new_corr;
            corr[i] = std::max(1e-6, std::min(corr[i], 1e10));
        }
        reset();
    }
};

// ─────────────────────────────────────────────
//  Bias hook — pure q0 physics only
//  b = f(x1) * f(x2) / pT^2 * correction(pT,x1,x2)
//  No power law boost, no free parameters
// ─────────────────────────────────────────────
class PhysicsBiasHook : public Pythia8::UserHooks {
public:
    Grid3D* grid;

    PhysicsBiasHook(Grid3D* g) : grid(g) {
        pdf_ = LHAPDF::mkPDF("NNPDF31_lo_as_0118", 0);
    }
    ~PhysicsBiasHook() { delete pdf_; }

    bool canBiasSelection() override { return true; }

    double biasSelectionBy(const Pythia8::SigmaProcess*,
                           const Pythia8::PhaseSpace* ps,
                           bool) override {
        double pT = ps->pTHat();
        double x1 = ps->x1();
        double x2 = ps->x2();

        if (pT <= 0 || x1 <= 0 || x2 <= 0) return 1.0;
        if (x1 >= 1.0 || x2 >= 1.0)         return 1.0;

        // pure q0: PDF shapes times QCD soft singularity
        double fx1 = pdf_->xfxQ(GLUON, x1, MU_F) / x1;
        double fx2 = pdf_->xfxQ(GLUON, x2, MU_F) / x2;
        double q0  = fx1 * fx2 / (pT * pT);
        if (q0 <= 0.0) return 1.0;

        // learned 3D correction — starts at 1 everywhere
        double corr = grid->eval(pT, x1, x2);

        return q0 * corr;
    }

private:
    LHAPDF::PDF* pdf_;
};

// ─────────────────────────────────────────────
//  Event struct and batch runner
// ─────────────────────────────────────────────
struct Event { double pT, x1, x2, w; };

std::vector<Event> run_batch(Grid3D& grid, int N, bool quiet) {
    auto hook = std::make_shared<PhysicsBiasHook>(&grid);

    Pythia8::Pythia pythia;
    pythia.readString("HardQCD:all = on");
    pythia.readString("SoftQCD:all = off");
    pythia.readString("Beams:eCM = 13000.");
    pythia.readString("PhaseSpace:pThatMin = 10.");
    pythia.readString("PhaseSpace:pThatMax = 7000.");
    pythia.readString("PartonLevel:MPI = off");
    pythia.readString("PartonLevel:ISR = off");
    pythia.readString("PartonLevel:FSR = off");
    pythia.readString("HadronLevel:all = off");
    pythia.readString("Next:numberShowInfo = 0");
    pythia.readString("Next:numberShowProcess = 0");
    pythia.readString("Next:numberShowEvent = 0");
    if (quiet) pythia.readString("Print:quiet = on");
    pythia.setUserHooksPtr(hook);
    pythia.init();

    std::vector<Event> results;
    results.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (!pythia.next()) continue;
        Event e;
        e.pT = pythia.info.pTHat();
        e.x1 = pythia.info.x1();
        e.x2 = pythia.info.x2();
        e.w  = pythia.info.weightSum();
        results.push_back(e);
        grid.accumulate(e.pT, e.x1, e.x2, e.w);
    }
    return results;
}

// ─────────────────────────────────────────────
//  Print iteration summary
// ─────────────────────────────────────────────
void print_iter(int iter,
                const std::vector<Event>& events,
                const Grid3D& grid) {

    double sum_w = 0.0, sum_w2 = 0.0;
    int    above = 0;
    for (const auto& e : events) {
        sum_w  += e.w;
        sum_w2 += e.w * e.w;
        if (e.pT > PT_TARGET) ++above;
    }
    int    N    = events.size();
    double Neff = (sum_w2 > 0) ? (sum_w*sum_w)/(sum_w2*N) : 0.0;
    double frac = (double)above / N;

    printf("\n── Iter %-3d | Neff/N=%-7.4f | frac>50GeV=%-9.5f | "
           "mean_w=%.3e | coverage=%.2f\n",
           iter, Neff, frac, sum_w/N, grid.coverage());

    // histogram centred around 50 GeV target
    double edges[] = {10, 25, 50, 100, 200, 500, 1000, 7000};
    int    nbins   = 7;
    printf("  %-22s %-8s %-14s\n", "pThat (GeV)", "count", "sum_w");
    for (int b = 0; b < nbins; ++b) {
        int cnt = 0; double sw = 0.0;
        for (const auto& e : events)
            if (e.pT >= edges[b] && e.pT < edges[b+1])
                { ++cnt; sw += e.w; }
        const char* tag = (edges[b] < PT_TARGET
                           && edges[b+1] > PT_TARGET)
                        ? "  <-- target" : "";
        printf("  %-6.0f - %-12.0f %-8d %-14.3e%s\n",
               edges[b], edges[b+1], cnt, sw, tag);
    }
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────
int main() {
    Grid3D grid;

    printf("Pure physics bias: b = f(x1)*f(x2)/pT^2 * corr3D(pT,x1,x2)\n");
    printf("No power law — correction grid learns from data only\n");
    printf("Target: pThat > %.0f GeV\n", PT_TARGET);
    printf("Phase space floor: %.0f GeV\n", PT_MIN);
    printf("%s\n", std::string(60,'-').c_str());

    for (int iter = 0; iter < N_ITER; ++iter) {
        auto events = run_batch(grid, N_EVENTS, iter > 0);
        print_iter(iter, events, grid);
        grid.update();
    }

    return 0;
}
