// Fine-tuned simple adaptive multivariable phase-space bias search for rare high-pT QCD events in Pythia8
//
// What this code does:
//   1. Uses a custom UserHooks bias factor b(Phi) depending on phase-space variables:
//        pTHat, x1*x2, Q2Fac, log(pTHat), a saturating pT term, and a two-region high-pT term.
//   2. Generates a batch of events for each candidate set of bias parameters.
//   3. Manually unbiases every accepted event using
//        manualWeight = pythia.info.weight() / biasFactor.
//   4. Computes both generated fractions and weighted/unbiased fractions.
//   5. Selects the best candidate using a simple product score in [0,1]:
//        score = fracPT50 * (Neff/N).
//      This rewards candidates that generate many high-pT events while still
//      keeping the manual-unbiasing weights statistically useful.
//
// Compile:
//   g++ clean_adaptive_IS_lhapdf_sampler.cc -o lhapdf_sampler \
     -I../pythia8317/include -I/path/to/lhapdf/include \
     -L../pythia8317/lib -L/path/to/lhapdf/lib \
     -lpythia8 -lLHAPDF \
     -Wl,-rpath,../pythia8317/lib -Wl,-rpath,/path/to/lhapdf/lib -std=c++17
//
// Run, same arguments as before:
//   ./simple 10 100000 simple.csv adaptive_events 0 0.05
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
#include "LHAPDF/LHAPDF.h"

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

// Temporarily redirects stdout and stderr to /dev/null. This is used only
// around Pythia initialization/event generation so the Pythia welcome banner
// and event listings do not flood the terminal. Our own iteration summaries
// are printed outside this silencer.
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
    // Keep the structure similar to the final_shit2 code, but use the
    // physics-motivated variables from the LHAPDF diagnostics:
    //   a_pT        : strong bias toward hard scattering scale pTHat
    //   b_xprod     : mild bias toward larger tau = x1*x2
    //   c_lumi      : very mild inverse gluon-luminosity bias from LHAPDF
    // The old Q2/log/saturating/two-region terms are removed because they
    // mostly duplicate pTHat or were less useful in the diagnostics.
    double a_pT   = 3.0;
    double b_xprod = 0.2;  // this is tau exponent
    double c_lumi = 0.05;  // inverse gluon luminosity exponent, keep small
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

// Computes the same bias factor that is returned to Pythia in the UserHook.
//
// New physics bias:
//
//   b(Phi) = (pTHat/pT0)^a * (tau/tau0)^b * (gRef/gLumi)^gamma
//
// where tau = x1*x2 and gLumi = g(x1,Q) g(x2,Q) from LHAPDF.
// We use gluon luminosity rather than id-specific luminosity because this is
// safe inside the selection hook and HardQCD at these scales is strongly gluon
// influenced. gamma must stay small because inverse-PDF biases can create
// huge weights if pushed too hard.
static double computeBiasFactor(const BiasParams& p,
                                double pTHat,
                                double xprod,
                                double QFac,
                                const LHAPDF::PDF* pdf) {
    const double pT0 = 10.0;
    const double xprod0 = 1.0e-4;
    const double maxBias = 1.0e5;

    double bias = 1.0;

    if (p.a_pT != 0.0) {
        bias *= pow(max(pTHat / pT0, 1.0e-12), p.a_pT);
    }

    // xprod is tau = x1*x2. This is only a mild correction: high pTHat
    // requires larger partonic energy, but tau alone is not the main driver.
    if (p.b_xprod != 0.0) {
        bias *= pow(max(xprod / xprod0, 1.0e-12), p.b_xprod);
    }

    // Mild inverse gluon-luminosity bias. This uses LHAPDF only if a PDF set
    // was successfully constructed. xfxQ returns x*f(x,Q), so divide by x to
    // get f(x,Q). Constants are protected to avoid log/zero problems.
    if (p.c_lumi != 0.0 && pdf != nullptr) {
        const double x1 = 0.5 * (sqrt(max(xprod, 0.0))); // placeholder not used
        (void)x1;
        // We need the actual x1 and x2 for this term, so this branch is filled
        // by the overload below. Kept only to avoid accidental use.
    }

    if (!isfinite(bias) || bias <= 0.0) bias = maxBias;
    return min(max(1.0, bias), maxBias);
}

static double computeBiasFactor(const BiasParams& p,
                                double pTHat,
                                double x1,
                                double x2,
                                double QFac,
                                const LHAPDF::PDF* pdf) {
    const double pT0 = 10.0;
    const double xprod0 = 1.0e-4;
    const double maxBias = 1.0e5;

    const double xprod = x1 * x2;
    double bias = 1.0;

    if (p.a_pT != 0.0) {
        bias *= pow(max(pTHat / pT0, 1.0e-12), p.a_pT);
    }

    if (p.b_xprod != 0.0) {
        bias *= pow(max(xprod / xprod0, 1.0e-12), p.b_xprod);
    }

    if (p.c_lumi != 0.0 && pdf != nullptr && x1 > 0.0 && x2 > 0.0) {
        const double Q = max(QFac, 1.0); // LHAPDF wants Q, not Q^2

        const double g1 = max(pdf->xfxQ(21, x1, Q) / x1, 1.0e-300);
        const double g2 = max(pdf->xfxQ(21, x2, Q) / x2, 1.0e-300);
        const double gluonLumi = max(g1 * g2, 1.0e-300);

        // Reference luminosity only fixes the normalization before clipping.
        // It is chosen from a typical low-x QCD region. The exact value is not
        // important because maxBias clipping controls the dangerous extremes.
        const double xRef = 1.0e-3;
        const double QRef = 10.0;
        const double gRef1 = max(pdf->xfxQ(21, xRef, QRef) / xRef, 1.0e-300);
        const double gRef2 = gRef1;
        const double lumiRef = max(gRef1 * gRef2, 1.0e-300);

        bias *= pow(max(lumiRef / gluonLumi, 1.0e-12), p.c_lumi);
    }

    if (!isfinite(bias) || bias <= 0.0) bias = maxBias;
    return min(max(1.0, bias), maxBias);
}

// Pythia calls this hook during phase-space selection. Returning a number
// larger than 1 makes that point more likely to be selected. The same value is
// later recomputed and used for manual unbiasing.
class MultiVariableBias : public UserHooks {
public:
    explicit MultiVariableBias(const BiasParams& parsIn, shared_ptr<LHAPDF::PDF> pdfIn) : pars(parsIn), pdf(pdfIn) {}

    bool canBiasSelection() override { return true; }

    double biasSelectionBy(const SigmaProcess* sigmaProcessPtr,
                           const PhaseSpace* phaseSpacePtr,
                           bool /*inEvent*/) override {
        if (sigmaProcessPtr->nFinal() != 2) return 1.0;

        const double pTHat = phaseSpacePtr->pTHat();
        const double xprod = phaseSpacePtr->x1() * phaseSpacePtr->x2();
        const double QFac = sqrt(max(sigmaProcessPtr->Q2Fac(), 1.0));
        return computeBiasFactor(pars, pTHat, phaseSpacePtr->x1(), phaseSpacePtr->x2(), QFac, pdf.get());
    }

private:
    BiasParams pars;
    shared_ptr<LHAPDF::PDF> pdf;
};

// Turns off normal Pythia print settings. The TerminalSilencer is still used
// around init/next because some Pythia output may bypass these settings.
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

// To give our adaptive model a score to optimize we simply multiply
// the 2 values we want to maximize since both are bounded [0,1].
static double balancedScore(double fracPT50, double neffRatio) {
    return fracPT50 * neffRatio;
}

// Writes one optional event-level CSV. This is only needed for debugging or for
// making detailed histograms. The summary CSV is always written separately.
static void writeEventHeader(ofstream& out) {
    out << "event,x1,x2,xprod,sHat,mHat,pTHat,pT2Hat,tHat,uHat,"
        << "Q2Fac,Q2Ren,alphaS,alphaEM,id1,id2,code,"
        << "pythiaWeight,biasFactor,manualWeight,gluonLumi,logGluonLumi,"
        << "maxFinalParticlePT,nFinalParticles,nChargedFinal\n";
}


// Runs one batch for one parameter candidate. It computes generated fractions,
// manually weighted fractions, Neff, and the bounded balanced score.
static BatchResult runBatch(const BiasParams& pars,
                            int nEvents,
                            int iter,
                            const string& eventPrefix,
                            bool saveEvents) {
    // Pythia prints its welcome banner when the object is constructed in some
    // builds, so create/configure/init it while stdout/stderr are redirected.
    shared_ptr<LHAPDF::PDF> pdf;
    unique_ptr<Pythia> pythiaPtr;
    {
        TerminalSilencer silence(true);
        pythiaPtr = make_unique<Pythia>();
        configurePythia(*pythiaPtr);

        // Use any installed LHAPDF set. You can change this to NNPDF31_lo_as_0130,
        // CT18LO, etc., as long as the set is installed and LHAPDF_DATA_PATH is set.
        pdf = shared_ptr<LHAPDF::PDF>(LHAPDF::mkPDF("NNPDF31_lo_as_0118", 0));
        pythiaPtr->setUserHooksPtr(make_shared<MultiVariableBias>(pars, pdf));
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
        const double QFac = sqrt(max(Q2Fac, 1.0));
        const double biasFactor = computeBiasFactor(pars, pTHat, x1, x2, QFac, pdf.get());
        const double pythiaWeight = pythia.info.weight();
        const double manualWeight = pythiaWeight / biasFactor;

        double gluonLumi = 0.0;
        double logGluonLumi = -999.0;
        if (pdf) {
            const double Q = max(QFac, 1.0);
            const double g1 = max(pdf->xfxQ(21, x1, Q) / max(x1, 1.0e-300), 1.0e-300);
            const double g2 = max(pdf->xfxQ(21, x2, Q) / max(x2, 1.0e-300), 1.0e-300);
            gluonLumi = g1 * g2;
            logGluonLumi = log(max(gluonLumi, 1.0e-300));
        }

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
                     << gluonLumi << "," << logGluonLumi << ","
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

// Keeps parameters inside broad but finite ranges. These ranges still search
// all variables used in the bias function, while avoiding absurd values.
static BiasParams clampParams(BiasParams p) {
    auto clamp = [](double x, double lo, double hi) { return max(lo, min(hi, x)); };
    p.a_pT    = clamp(p.a_pT,    0.0, 8.0);
    p.b_xprod = clamp(p.b_xprod, 0.0, 1.0);
    p.c_lumi  = clamp(p.c_lumi,  0.0, 0.3); // keep small for weight stability
    return p;
}

// Creates the next candidate by changing only ONE parameter at a time.
//
// This is a simple coordinate-search / hill-climbing method.
// Instead of randomly moving all parameters at once, the search cycles through
//
//     a_pT -> b_xprod -> c_Q2 -> e_logpT -> f_satpT -> g_tworegion
//
// and changes only that parameter. After all six parameters have been tested
// in the positive direction, the code tests them in the negative direction.
// The step size becomes smaller after each full positive/negative cycle.
//
// This makes the adaptive search easier to interpret:
// if a candidate improves the score, the improvement can be linked mainly to
// the parameter that was changed in that iteration.
static BiasParams proposeCandidate(const BiasParams& best, int iter, mt19937& rng) {
    (void)rng;

    BiasParams p = best;

    // Cycle through only the three useful variables:
    //   a_pT -> b_tau/xprod -> gamma_lumi
    const int paramIndex = (iter - 1) % 3;
    const int pass = (iter - 1) / 3;
    const double direction = (pass % 2 == 0) ? +1.0 : -1.0;

    const int fullCycle = (iter - 1) / 6;
    const double scale = max(0.20, pow(0.75, fullCycle));

    const double stepA = 0.35 * scale;
    const double stepB = 0.05 * scale;
    const double stepC = 0.02 * scale;

    if (paramIndex == 0) p.a_pT    += direction * stepA;
    if (paramIndex == 1) p.b_xprod += direction * stepB;
    if (paramIndex == 2) p.c_lumi  += direction * stepC;

    return clampParams(p);
}

// Writes one row per iteration. The weighted fractions are the physical
// unbiasing checks; the generated fractions show sampling efficiency.
static void writeSummaryHeader(ofstream& out) {
    out << "iteration,accepted,acceptable,bestSoFar,"
        << "a_pT,b_tau,c_lumi,"
        << "sumManualW,sumManualW2,meanManualW,stdManualW,NeffManual,NeffManualOverN,"
        << "meanPTHat,maxPTHat,"
        << "fracPT10,fracPT20,fracPT50,fracPT100,"
        << "wFracPT10,wFracPT20,wFracPT50,wFracPT100,score\n";
}

static void writeSummaryRow(ofstream& out, int iter, const BiasParams& p,
                            const BatchResult& r, bool acceptable, bool bestSoFar) {
    out << iter << "," << r.accepted << "," << int(acceptable) << "," << int(bestSoFar) << ","
        << p.a_pT << "," << p.b_xprod << "," << p.c_lumi << ","
        << r.sumW << "," << r.sumW2 << "," << r.meanW << "," << r.stdW << ","
        << r.neff << "," << r.neffRatio << ","
        << r.meanPTHat << "," << r.maxPTHat << ","
        << r.fracPT10 << "," << r.fracPT20 << "," << r.fracPT50 << "," << r.fracPT100 << ","
        << r.wFracPT10 << "," << r.wFracPT20 << "," << r.wFracPT50 << "," << r.wFracPT100 << ","
        << r.score << "\n";
}

// Main adaptive loop. It keeps the same command-line arguments as before,
// runs one candidate per iteration, writes the summary CSV, and keeps the best
// candidate according to the bounded balanced score.
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
    cout << "Starting one-parameter adaptive manual-unbias search\n";
    cout << "Iterations: " << nIterations << "\n";
    cout << "Events per iteration: " << nEventsPerBatch << "\n";
    cout << "Minimum acceptable Neff/N: " << minNeffRatio << "\n";
    cout << "Summary file: " << summaryFile << "\n";
    cout << "Saving event CSVs: " << (saveEvents ? "yes" : "no") << "\n\n";

    for (int iter = 0; iter < nIterations; ++iter) {
        BiasParams candidate = (iter == 0) ? bestPars : proposeCandidate(bestPars, iter, rng);

        cout << "Iteration " << iter << " candidate: "
             << "a=" << candidate.a_pT
             << ", b_tau=" << candidate.b_xprod
             << ", c_lumi=" << candidate.c_lumi << "\n";

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
         << "  b_tau       = " << bestPars.b_xprod << "\n"
         << "  c_lumi      = " << bestPars.c_lumi << "\n";

    return 0;
}
