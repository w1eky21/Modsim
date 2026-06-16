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
constexpr double PT_MIN    =  100.0;
constexpr double PT_MAX    = 7000.0;
constexpr double PT_TARGET = 2000.0;
constexpr double PT_REF    = 1000.0;
constexpr double MU_F      =   91.2;
constexpr int    GLUON     =    21;
constexpr int    N_EVENTS  =  5000;
constexpr int    N_ITER    =   20;

// target Neff/N range — n increases if below low, decreases if above high
constexpr double NEFF_LOW  = 0.10;  // too few effective samples → n too high
constexpr double NEFF_HIGH = 0.50;  // very uniform weights → n could be higher
constexpr double FRAC_TARGET = 0.05; // want at least 5% of events above 2 TeV

// 3D grid
constexpr int N_PT = 15;
constexpr int N_X  =  8;

// ─────────────────────────────────────────────
//  3D correction grid
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

    // returns fraction of cells that were visited this iteration
    // low coverage means n is pushing too hard and missing regions
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
//  Bias hook
//  b = q0(pT,x1,x2) * (pT/PT_REF)^n * correction(pT,x1,x2)
// ─────────────────────────────────────────────
class PhysicsBiasHook : public Pythia8::UserHooks {
public:
    Grid3D* grid;
    double  boost;   // current n — updated between iterations

    PhysicsBiasHook(Grid3D* g, double n) : grid(g), boost(n) {
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

        double fx1 = pdf_->xfxQ(GLUON, x1, MU_F) / x1;
        double fx2 = pdf_->xfxQ(GLUON, x2, MU_F) / x2;
        double q0  = fx1 * fx2 / (pT * pT);
        if (q0 <= 0.0) return 1.0;

        double pT_boost = std::pow(pT / PT_REF, boost);
        double corr     = grid->eval(pT, x1, x2);

        return q0 * pT_boost * corr;
    }

private:
    LHAPDF::PDF* pdf_;
};

// ─────────────────────────────────────────────
//  Diagnostics struct
// ─────────────────────────────────────────────
struct Diagnostics {
    double Neff, frac_2TeV, mean_w, coverage;
    int    n_above;
};

// ─────────────────────────────────────────────
//  Run one batch
// ─────────────────────────────────────────────
struct Event { double pT, x1, x2, w; };

std::vector<Event> run_batch(Grid3D& grid, double boost,
                             int N, bool quiet) {
    auto hook = std::make_shared<PhysicsBiasHook>(&grid, boost);

    Pythia8::Pythia pythia;
    pythia.readString("HardQCD:all = on");
    pythia.readString("SoftQCD:all = off");
    pythia.readString("Beams:eCM = 13000.");
    pythia.readString("PhaseSpace:pThatMin = 100.");
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
//  Compute diagnostics from batch
// ─────────────────────────────────────────────
Diagnostics compute_diag(const std::vector<Event>& events,
                         const Grid3D& grid) {
    double sum_w = 0.0, sum_w2 = 0.0;
    int    above = 0;
    for (const auto& e : events) {
        sum_w  += e.w;
        sum_w2 += e.w * e.w;
        if (e.pT > PT_TARGET) ++above;
    }
    int N = events.size();
    Diagnostics d;
    d.Neff     = (sum_w2 > 0) ? (sum_w*sum_w)/(sum_w2*N) : 0.0;
    d.frac_2TeV = (double)above / N;
    d.mean_w   = sum_w / N;
    d.n_above  = above;
    d.coverage = grid.coverage();
    return d;
}

// ─────────────────────────────────────────────
//  Adapt n based on diagnostics
//
//  Three regimes:
//  1. frac>2TeV == 0 and Neff ok → n too small, increase aggressively
//  2. Neff drops below NEFF_LOW  → n too large, weights uneven, decrease
//  3. frac>2TeV > 0 and Neff ok  → fine-tune n toward target frac
// ─────────────────────────────────────────────
double adapt_boost(double current_boost,
                   const Diagnostics& d) {
    double new_boost = current_boost;

    if (d.frac_2TeV == 0.0 && d.Neff > NEFF_LOW) {
        // no events in tail yet — push harder
        // step size decreases as boost grows to avoid overshoot
        double step = std::max(0.5, 2.0 / std::sqrt(current_boost));
        new_boost = current_boost + step;
        printf("  [n-adapt] No tail events, increasing n: %.2f → %.2f\n",
               current_boost, new_boost);

    } else if (d.Neff < NEFF_LOW) {
        // weights too uneven — bias is too aggressive, pull back
        double step = std::max(0.3, current_boost * 0.15);
        new_boost = current_boost - step;
        printf("  [n-adapt] Neff/N=%.3f too low, decreasing n: %.2f → %.2f\n",
               d.Neff, current_boost, new_boost);

    } else if (d.frac_2TeV > 0.0 && d.frac_2TeV < FRAC_TARGET
               && d.Neff > NEFF_LOW) {
        // tail visible but not enough events yet — gentle increase
        double step = 0.25;
        new_boost = current_boost + step;
        printf("  [n-adapt] frac=%.4f below target, nudging n: %.2f → %.2f\n",
               d.frac_2TeV, current_boost, new_boost);

    } else if (d.frac_2TeV >= FRAC_TARGET && d.Neff > NEFF_HIGH) {
        // doing well — slight decrease to recover more uniform weights
        double step = 0.1;
        new_boost = current_boost - step;
        printf("  [n-adapt] Good frac & Neff, refining n: %.2f → %.2f\n",
               current_boost, new_boost);

    } else {
        printf("  [n-adapt] n=%.2f stable\n", current_boost);
    }

    // hard bounds
    return std::max(0.5, std::min(new_boost, 10.0));
}

// ─────────────────────────────────────────────
//  Print iteration summary
// ─────────────────────────────────────────────
void print_iter(int iter, double boost,
                const Diagnostics& d,
                const std::vector<Event>& events) {
    printf("\n── Iter %-3d | n=%-6.2f | Neff/N=%-7.4f | "
           "frac>2TeV=%-9.5f | mean_w=%.3e | coverage=%.2f\n",
           iter, boost, d.Neff, d.frac_2TeV, d.mean_w, d.coverage);

    double edges[] = {100, 200, 500, 1000, 2000, 3500, 7000};
    printf("  %-22s %-8s %-12s\n", "pThat (GeV)", "count", "sum_w");
    for (int b = 0; b < 6; ++b) {
        int cnt = 0; double sw = 0.0;
        for (const auto& e : events)
            if (e.pT >= edges[b] && e.pT < edges[b+1])
                { ++cnt; sw += e.w; }
        printf("  %-6.0f - %-12.0f %-8d %.3e\n",
               edges[b], edges[b+1], cnt, sw);
    }
}

// ─────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────
int main() {
    Grid3D grid;
    double boost = 2.0;   // start gentle — algorithm finds the right n

    printf("Adaptive bias: b = q0(pT,x1,x2) * (pT/%.0f)^n * corr3D(pT,x1,x2)\n",
           PT_REF);
    printf("n starts at %.1f and is tuned automatically each iteration\n",
           boost);
    printf("Target: Neff/N > %.2f  and  frac>2TeV > %.2f\n",
           NEFF_LOW, FRAC_TARGET);
    printf("%s\n", std::string(65,'-').c_str());

    double best_boost = boost;
    double best_Neff  = 0.0;
    double best_frac  = 0.0;

    for (int iter = 0; iter < N_ITER; ++iter) {

        // run batch with current boost and grid
        auto events = run_batch(grid, boost, N_EVENTS, iter > 0);

        // compute diagnostics
        Diagnostics d = compute_diag(events, grid);

        // print summary
        print_iter(iter, boost, d, events);

        // track best configuration seen so far
        // best = highest frac>2TeV among runs with acceptable Neff
        if (d.Neff > NEFF_LOW && d.frac_2TeV > best_frac) {
            best_frac  = d.frac_2TeV;
            best_boost = boost;
            best_Neff  = d.Neff;
        }

        // update 3D correction grid from accumulated weights
        grid.update();

        // adapt n based on what we observed
        boost = adapt_boost(boost, d);
    }

    printf("\n%s\n", std::string(65,'=').c_str());
    printf("Best configuration found:\n");
    printf("  n          = %.2f\n", best_boost);
    printf("  Neff/N     = %.4f\n", best_Neff);
    printf("  frac>2TeV  = %.5f\n", best_frac);

    return 0;
}