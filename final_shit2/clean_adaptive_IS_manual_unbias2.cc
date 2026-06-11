// Fine-tuned simple adaptive multivariable phase-space bias search for rare high-pT QCD events in Pythia8
//
// What this code does:
//   1. Uses a custom UserHooks bias factor b(Phi) depending on phase-space variables:
//        pTHat, x1*x2, Q2Fac, log(pTHat), a saturating pT term, and a two-region high-pT term.
//   2. Generates a batch of events for each candidate set of bias parameters.
//   3. Manually unbiases every accepted event using
//        manualWeight = pythia.info.weight() / biasFactor.
//   4. Computes both generated fractions and weighted/unbiased fractions.
//   5. Selects the best candidate using a bounded score in [0,1]:
//        score = harmonic_mean(fracPT50, Neff/N)
//              = 2 * fracPT50 * (Neff/N) / (fracPT50 + Neff/N).
//      This rewards both many generated high-pT events and stable weights.
//
// Compile:
//   g++ clean_adaptive_IS_manual_unbias2.cc -o clean2 \
     -I../pythia8317/include -L../pythia8317/lib -lpythia8 \
     -Wl,-rpath,../pythia8317/lib -std=c++17
//
// Run, same arguments as before:
//   ./clean2 10 100000 clean2.csv adaptive_events 0 0.05
//
// Arguments:
//   argv[1] = number of adaptive iterations       default 10
//   argv[2] = generated events per iteration      default 100000
//   argv[3] = summary CSV filename                default adaptive_pt50_simple_summary.csv
//   argv[4] = event output prefix                 default adaptive_events
//   argv[5] = save per-event CSVs? 0/1            default 0
//   argv[6] = minimum acceptable Neff/N           default 0.05

#include "Pythia8/Pythia.h"
#include "Pythia8/UserHooks.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>

#include <fcntl.h>
#include <unistd.h>

using namespace Pythia8;
using namespace std;

// ------------------------------------------------------------
// Explanation:
// Temporarily redirects stdout and stderr to /dev/null. This is used only
// around Pythia initialization/event generation so the Pythia welcome banner
// and event listings do not flood the terminal. Our own iteration summaries
// are printed outside this silencer.
// ------------------------------------------------------------
class TerminalSilencer {
public:
    explicit TerminalSilencer(bool activeIn) : active(activeIn), oldOut(-1), oldErr(-1), nullFd(-1) {
        if (!active) return;
        fflush(stdout);
        fflush(stderr);
        oldOut = dup(STDOUT_FILENO);
        oldErr = dup(STDERR_FILENO);
        nullFd = open("/dev/null", O_WRONLY);
        if (nullFd >= 0) {
            dup2(nullFd, STDOUT_FILENO);
            dup2(nullFd, STDERR_FILENO);
        }
    }

    ~TerminalSilencer() {
        if (!active) return;
        fflush(stdout);
        fflush(stderr);
        if (oldOut >= 0) { dup2(oldOut, STDOUT_FILENO); close(oldOut); }
        if (oldErr >= 0) { dup2(oldErr, STDERR_FILENO); close(oldErr); }
        if (nullFd >= 0) close(nullFd);
    }

private:
    bool active;
    int oldOut;
    int oldErr;
    int nullFd;
};

struct BiasParams {
    double a_pT        = 3.8;
    double b_xprod     = 0.28;
    double c_Q2        = 0.28;
    double d_alphaS    = 0.0;  // kept in output, not used in the hook
    double e_logpT     = 0.0;
    double f_satpT     = 0.0;
    double g_tworegion = 0.0;
};

struct BatchResult {
    int accepted = 0;

    double sumW = 0.0;
    double sumW2 = 0.0;
    double meanW = 0.0;
    double stdW = 0.0;
    double neff = 0.0;
    double neffRatio = 0.0;

    double meanPTHat = 0.0;
    double maxPTHat = 0.0;

    double fracPT10 = 0.0;
    double fracPT20 = 0.0;
    double fracPT50 = 0.0;
    double fracPT100 = 0.0;

    double wFracPT10 = 0.0;
    double wFracPT20 = 0.0;
    double wFracPT50 = 0.0;
    double wFracPT100 = 0.0;

    double score = 0.0;
};

// ------------------------------------------------------------
// Explanation:
// Computes the same bias factor that is returned to Pythia in the UserHook.
// This factor makes selected phase-space points more likely. Since this custom
// hook is corrected manually, the physical event weight is divided by this
// bias factor after the event is accepted.
// ------------------------------------------------------------
static double computeBiasFactor(const BiasParams& p,
                                double pTHat,
                                double xprod,
                                double Q2Fac) {
    const double pT0 = 10.0;
    const double xprod0 = 1.0e-4;
    const double Q20 = 100.0;
    const double pTswitch = 50.0;
    const double maxBias = 1.0e6;

    double bias = 1.0;

    if (p.a_pT != 0.0) {
        bias *= pow(max(pTHat / pT0, 1.0e-12), p.a_pT);
    }
    if (p.b_xprod != 0.0) {
        bias *= pow(max(xprod / xprod0, 1.0e-12), p.b_xprod);
    }
    if (p.c_Q2 != 0.0) {
        bias *= pow(max(Q2Fac / Q20, 1.0e-12), p.c_Q2);
    }
    if (p.e_logpT != 0.0) {
        bias *= pow(max(log(1.0 + pTHat / pT0), 1.0e-12), p.e_logpT);
    }
    if (p.f_satpT != 0.0) {
        const double sat = 1.0 + pTHat / (pTHat + pT0); // between 1 and 2
        bias *= pow(sat, p.f_satpT);
    }
    if (p.g_tworegion != 0.0 && pTHat > pTswitch) {
        bias *= pow(max(pTHat / pTswitch, 1.0e-12), p.g_tworegion);
    }

    if (!isfinite(bias) || bias <= 0.0) bias = maxBias;
    return min(max(1.0, bias), maxBias);
}

// ------------------------------------------------------------
// Explanation:
// Pythia calls this hook during phase-space selection. Returning a number
// larger than 1 makes that point more likely to be selected. The same value is
// later recomputed and used for manual unbiasing.
// ------------------------------------------------------------
class MultiVariableBias : public UserHooks {
public:
    explicit MultiVariableBias(const BiasParams& parsIn) : pars(parsIn) {}

    bool canBiasSelection() override { return true; }

    double biasSelectionBy(const SigmaProcess* sigmaProcessPtr,
                           const PhaseSpace* phaseSpacePtr,
                           bool /*inEvent*/) override {
        if (sigmaProcessPtr->nFinal() != 2) return 1.0;

        const double pTHat = phaseSpacePtr->pTHat();
        const double xprod = phaseSpacePtr->x1() * phaseSpacePtr->x2();
        const double Q2Fac = sigmaProcessPtr->Q2Fac();
        return computeBiasFactor(pars, pTHat, xprod, Q2Fac);
    }

private:
    BiasParams pars;
};

// ------------------------------------------------------------
// Explanation:
// Turns off normal Pythia print settings. The TerminalSilencer is still used
// around init/next because some Pythia output may bypass these settings.
// ------------------------------------------------------------
static void configurePythia(Pythia& pythia) {
    pythia.readString("Print:quiet = on");
    pythia.readString("Init:showChangedSettings = off");
    pythia.readString("Init:showAllSettings = off");
    pythia.readString("Init:showChangedParticleData = off");
    pythia.readString("Init:showAllParticleData = off");
    pythia.readString("Next:numberShowInfo = 0");
    pythia.readString("Next:numberShowProcess = 0");
    pythia.readString("Next:numberShowEvent = 0");

    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13000.");
    pythia.readString("HardQCD:all = on");
    pythia.readString("PhaseSpace:pTHatMin = 0.");
}

// ------------------------------------------------------------
// Explanation:
// Bounded score in [0,1]. It is the harmonic mean of generated high-pT fraction
// and Neff/N. A candidate only scores well if both terms are reasonably large.
// ------------------------------------------------------------
static double balancedScore(double fracPT50, double neffRatio) {
    return fracPT50 * neffRatio;
}

// ------------------------------------------------------------
// Explanation:
// Writes one optional event-level CSV. This is only needed for debugging or for
// making detailed histograms. The summary CSV is always written separately.
// ------------------------------------------------------------
static void writeEventHeader(ofstream& out) {
    out << "event,x1,x2,xprod,sHat,mHat,pTHat,pT2Hat,tHat,uHat,"
        << "Q2Fac,Q2Ren,alphaS,alphaEM,id1,id2,code,"
        << "pythiaWeight,biasFactor,manualWeight,"
        << "maxFinalParticlePT,nFinalParticles,nChargedFinal\n";
}

// ------------------------------------------------------------
// Explanation:
// Runs one batch for one parameter candidate. It computes generated fractions,
// manually weighted fractions, Neff, and the bounded balanced score.
// ------------------------------------------------------------
static BatchResult runBatch(const BiasParams& pars,
                            int nEvents,
                            int iter,
                            const string& eventPrefix,
                            bool saveEvents) {
    // Pythia prints its welcome banner when the object is constructed in some
    // builds, so create/configure/init it while stdout/stderr are redirected.
    unique_ptr<Pythia> pythiaPtr;
    {
        TerminalSilencer silence(true);
        pythiaPtr = make_unique<Pythia>();
        configurePythia(*pythiaPtr);
        pythiaPtr->setUserHooksPtr(make_shared<MultiVariableBias>(pars));
        if (!pythiaPtr->init()) {
            cerr << "Pythia initialization failed in iteration " << iter << "\n";
            return {};
        }
    }
    Pythia& pythia = *pythiaPtr;

    ofstream eventOut;
    if (saveEvents) {
        ostringstream name;
        name << eventPrefix << "_iter_" << setw(3) << setfill('0') << iter << ".csv";
        eventOut.open(name.str());
        writeEventHeader(eventOut);
    }

    BatchResult r;
    double sumPTHat = 0.0;
    int n10 = 0, n20 = 0, n50 = 0, n100 = 0;
    double w10 = 0.0, w20 = 0.0, w50 = 0.0, w100 = 0.0;

    TerminalSilencer silence(true);
    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
        if (!pythia.next()) continue;

        const double x1 = pythia.info.x1();
        const double x2 = pythia.info.x2();
        const double xprod = x1 * x2;
        const double sHat = pythia.info.sHat();
        const double pTHat = pythia.info.pTHat();
        const double Q2Fac = pythia.info.Q2Fac();
        const double biasFactor = computeBiasFactor(pars, pTHat, xprod, Q2Fac);
        const double pythiaWeight = pythia.info.weight();
        const double manualWeight = pythiaWeight / biasFactor;

        double maxFinalPT = 0.0;
        int nFinal = 0;
        int nCharged = 0;
        for (int i = 0; i < pythia.event.size(); ++i) {
            if (!pythia.event[i].isFinal()) continue;
            ++nFinal;
            maxFinalPT = max(maxFinalPT, pythia.event[i].pT());
            if (pythia.event[i].isCharged()) ++nCharged;
        }

        ++r.accepted;
        r.sumW += manualWeight;
        r.sumW2 += manualWeight * manualWeight;
        sumPTHat += pTHat;
        r.maxPTHat = max(r.maxPTHat, pTHat);

        if (pTHat > 10.0)  { ++n10;  w10  += manualWeight; }
        if (pTHat > 20.0)  { ++n20;  w20  += manualWeight; }
        if (pTHat > 50.0)  { ++n50;  w50  += manualWeight; }
        if (pTHat > 100.0) { ++n100; w100 += manualWeight; }

        if (saveEvents) {
            eventOut << iEvent << "," << x1 << "," << x2 << "," << xprod << ","
                     << sHat << "," << sqrt(max(sHat, 0.0)) << "," << pTHat << ","
                     << pTHat * pTHat << "," << pythia.info.tHat() << "," << pythia.info.uHat() << ","
                     << Q2Fac << "," << pythia.info.Q2Ren() << ","
                     << pythia.info.alphaS() << "," << pythia.info.alphaEM() << ","
                     << pythia.info.id1() << "," << pythia.info.id2() << "," << pythia.info.code() << ","
                     << pythiaWeight << "," << biasFactor << "," << manualWeight << ","
                     << maxFinalPT << "," << nFinal << "," << nCharged << "\n";
        }
    }

    if (r.accepted == 0) return r;

    r.meanW = r.sumW / r.accepted;
    const double meanW2 = r.sumW2 / r.accepted;
    r.stdW = sqrt(max(0.0, meanW2 - r.meanW * r.meanW));
    r.neff = (r.sumW2 > 0.0) ? (r.sumW * r.sumW / r.sumW2) : 0.0;
    r.neffRatio = r.neff / r.accepted;

    r.meanPTHat = sumPTHat / r.accepted;
    r.fracPT10 = double(n10) / r.accepted;
    r.fracPT20 = double(n20) / r.accepted;
    r.fracPT50 = double(n50) / r.accepted;
    r.fracPT100 = double(n100) / r.accepted;

    if (r.sumW > 0.0) {
        r.wFracPT10 = w10 / r.sumW;
        r.wFracPT20 = w20 / r.sumW;
        r.wFracPT50 = w50 / r.sumW;
        r.wFracPT100 = w100 / r.sumW;
    }

    r.score = balancedScore(r.fracPT50, r.neffRatio);
    return r;
}

// ------------------------------------------------------------
// Explanation:
// Keeps parameters inside broad but finite ranges. These ranges still search
// all variables used in the bias function, while avoiding absurd values.
// ------------------------------------------------------------
static BiasParams clampParams(BiasParams p) {
    auto clamp = [](double x, double lo, double hi) { return max(lo, min(hi, x)); };
    p.a_pT        = clamp(p.a_pT,        0.0, 8.0);
    p.b_xprod     = clamp(p.b_xprod,     0.0, 2.0);
    p.c_Q2        = clamp(p.c_Q2,        0.0, 2.0);
    p.d_alphaS    = 0.0;
    p.e_logpT     = clamp(p.e_logpT,     0.0, 5.0);
    p.f_satpT     = clamp(p.f_satpT,     0.0, 8.0);
    p.g_tworegion = clamp(p.g_tworegion, 0.0, 8.0);
    return p;
}

// ------------------------------------------------------------
// Explanation:
// Creates the next candidate by making a small local change around the best
// candidate found so far. This is intentionally fine-tuned: it avoids the old
// broad random jump every eighth iteration, because that made parameters such
// as e, f and g suddenly become very large. All phase-space variables are still
// searched, but only one or two are nudged per iteration.
// ------------------------------------------------------------
static BiasParams proposeCandidate(const BiasParams& best, int iter, mt19937& rng) {
    normal_distribution<double> N(0.0, 1.0);
    uniform_real_distribution<double> U(0.0, 1.0);
    BiasParams p = best;

    // Slowly shrink the step size: first explore locally, then refine.
    const double scale = max(0.25, 1.0 - 0.04 * iter);

    // Cycle through parameters so every variable is searched, but smoothly.
    // The extra small random nudge lets combinations still change together.
    const int which = (iter - 1) % 6;
    if (which == 0) p.a_pT        += 0.20 * scale * N(rng);
    if (which == 1) p.b_xprod     += 0.05 * scale * N(rng);
    if (which == 2) p.c_Q2        += 0.05 * scale * N(rng);
    if (which == 3) p.e_logpT     += 0.08 * scale * N(rng);
    if (which == 4) p.f_satpT     += 0.12 * scale * N(rng);
    if (which == 5) p.g_tworegion += 0.12 * scale * N(rng);

    // Small probability of a second tiny nudge, still local.
    if (U(rng) < 0.30) p.a_pT        += 0.06 * scale * N(rng);
    if (U(rng) < 0.20) p.b_xprod     += 0.02 * scale * N(rng);
    if (U(rng) < 0.20) p.c_Q2        += 0.02 * scale * N(rng);
    if (U(rng) < 0.15) p.e_logpT     += 0.03 * scale * N(rng);
    if (U(rng) < 0.15) p.f_satpT     += 0.04 * scale * N(rng);
    if (U(rng) < 0.15) p.g_tworegion += 0.04 * scale * N(rng);

    return clampParams(p);
}

// ------------------------------------------------------------
// Explanation:
// Writes one row per iteration. The weighted fractions are the physical
// unbiasing checks; the generated fractions show sampling efficiency.
// ------------------------------------------------------------
static void writeSummaryHeader(ofstream& out) {
    out << "iteration,accepted,acceptable,bestSoFar,"
        << "a_pT,b_xprod,c_Q2,d_alphaS,e_logpT,f_satpT,g_tworegion,"
        << "sumManualW,sumManualW2,meanManualW,stdManualW,NeffManual,NeffManualOverN,"
        << "meanPTHat,maxPTHat,"
        << "fracPT10,fracPT20,fracPT50,fracPT100,"
        << "wFracPT10,wFracPT20,wFracPT50,wFracPT100,score\n";
}

static void writeSummaryRow(ofstream& out, int iter, const BiasParams& p,
                            const BatchResult& r, bool acceptable, bool bestSoFar) {
    out << iter << "," << r.accepted << "," << int(acceptable) << "," << int(bestSoFar) << ","
        << p.a_pT << "," << p.b_xprod << "," << p.c_Q2 << "," << p.d_alphaS << ","
        << p.e_logpT << "," << p.f_satpT << "," << p.g_tworegion << ","
        << r.sumW << "," << r.sumW2 << "," << r.meanW << "," << r.stdW << ","
        << r.neff << "," << r.neffRatio << ","
        << r.meanPTHat << "," << r.maxPTHat << ","
        << r.fracPT10 << "," << r.fracPT20 << "," << r.fracPT50 << "," << r.fracPT100 << ","
        << r.wFracPT10 << "," << r.wFracPT20 << "," << r.wFracPT50 << "," << r.wFracPT100 << ","
        << r.score << "\n";
}

// ------------------------------------------------------------
// Explanation:
// Main adaptive loop. It keeps the same command-line arguments as before,
// runs one candidate per iteration, writes the summary CSV, and keeps the best
// candidate according to the bounded balanced score.
// ------------------------------------------------------------
int main(int argc, char* argv[]) {
    int nIterations = 10;
    int nEventsPerBatch = 100000;
    string summaryFile = "adaptive_pt50_simple_summary.csv";
    string eventPrefix = "adaptive_events";
    bool saveEvents = false;
    double minNeffRatio = 0.05;

    if (argc > 1) nIterations = atoi(argv[1]);
    if (argc > 2) nEventsPerBatch = atoi(argv[2]);
    if (argc > 3) summaryFile = argv[3];
    if (argc > 4) eventPrefix = argv[4];
    if (argc > 5) saveEvents = (atoi(argv[5]) != 0);
    if (argc > 6) minNeffRatio = atof(argv[6]);

    mt19937 rng(12345);
    BiasParams bestPars = clampParams(BiasParams{});
    BatchResult bestRes;
    double bestScore = -1.0;

    ofstream summary(summaryFile);
    writeSummaryHeader(summary);

    cout << fixed << setprecision(6);
    cout << "Starting simple adaptive manual-unbias search\n";
    cout << "Iterations: " << nIterations << "\n";
    cout << "Events per iteration: " << nEventsPerBatch << "\n";
    cout << "Minimum acceptable Neff/N: " << minNeffRatio << "\n";
    cout << "Summary file: " << summaryFile << "\n";
    cout << "Saving event CSVs: " << (saveEvents ? "yes" : "no") << "\n\n";

    for (int iter = 0; iter < nIterations; ++iter) {
        BiasParams candidate = (iter == 0) ? bestPars : proposeCandidate(bestPars, iter, rng);

        cout << "Iteration " << iter << " candidate: "
             << "a=" << candidate.a_pT
             << ", b=" << candidate.b_xprod
             << ", c=" << candidate.c_Q2
             << ", e=" << candidate.e_logpT
             << ", f=" << candidate.f_satpT
             << ", g=" << candidate.g_tworegion << "\n";

        BatchResult r = runBatch(candidate, nEventsPerBatch, iter, eventPrefix, saveEvents);
        const bool acceptable = (r.accepted > 0 && r.neffRatio >= minNeffRatio);
        const bool isBest = acceptable && (r.score > bestScore);

        if (isBest) {
            bestScore = r.score;
            bestPars = candidate;
            bestRes = r;
        }

        writeSummaryRow(summary, iter, candidate, r, acceptable, isBest);
        summary.flush();

        cout << "  accepted=" << r.accepted
             << ", fracPT50=" << r.fracPT50
             << ", wFracPT50=" << scientific << r.wFracPT50 << fixed
             << ", Neff/N=" << r.neffRatio
             << ", score=" << r.score
             << ", meanW=" << r.meanW
             << ", stdW=" << r.stdW
             << ", meanPTHat=" << r.meanPTHat
             << ", maxPTHat=" << r.maxPTHat;
        if (!acceptable) cout << "  [rejected: low Neff/N]";
        if (isBest) cout << "  [new best]";
        cout << "\n\n";
    }

    cout << "Search finished.\n";
    cout << "Best score                  = " << bestScore << "\n";
    cout << "Best generated fracPT50     = " << bestRes.fracPT50 << "\n";
    cout << "Best weighted wFracPT50     = " << scientific << bestRes.wFracPT50 << fixed << "\n";
    cout << "Best Neff/N                 = " << bestRes.neffRatio << "\n";
    cout << "Best parameters:\n"
         << "  a_pT        = " << bestPars.a_pT << "\n"
         << "  b_xprod     = " << bestPars.b_xprod << "\n"
         << "  c_Q2        = " << bestPars.c_Q2 << "\n"
         << "  d_alphaS    = " << bestPars.d_alphaS << "  (fixed; not used)\n"
         << "  e_logpT     = " << bestPars.e_logpT << "\n"
         << "  f_satpT     = " << bestPars.f_satpT << "\n"
         << "  g_tworegion = " << bestPars.g_tworegion << "\n";

    return 0;
}
