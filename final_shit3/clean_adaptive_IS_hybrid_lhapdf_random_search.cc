// Hybrid adaptive multivariable phase-space bias search for rare high-pT QCD events in Pythia8
//
// This version is based on the old simple sampler, but uses a more physics-motivated
// bias function:
//
//   bias = pTHat^a * tau^b * log(pTHat)^e * two-region(pTHat)^g * gluonLumi^(-c)
//
// where
//   tau       = x1*x2
//   gluonLumi = g(x1,Q) * g(x2,Q) from LHAPDF
//
// Removed from the old code:
//   - c_Q2: mostly duplicates pTHat/Q scale information
//   - d_alphaS: already unused
//   - f_satpT: weak extra pT-shaping term
//
// Kept from the old code:
//   - a_pT: strongest high-pT driver
//   - b_tau: old b_xprod, but renamed because xprod = tau
//   - e_logpT: useful extra pT shaping
//   - g_tworegion: useful for specifically enhancing the high-pT tail
//
// Added:
//   - c_lumi: mild LHAPDF gluon-luminosity correction
//
/* 
g++ /home/wiek/Model_sim/modsim_project/LHADF/lhapdf_test/clean_adaptive_IS_hybrid_lhapdf_random_search.cc \
  -o /home/wiek/Model_sim/modsim_project/LHADF/lhapdf_test/hybrid_lhapdf_random_search \
  -I/home/wiek/Model_sim/modsim_project/pythia8317/include \
  $(lhapdf-config --cxxflags) \
  -L/home/wiek/Model_sim/modsim_project/pythia8317/lib \
  $(lhapdf-config --ldflags) \
  -lpythia8 -lLHAPDF \
  -Wl,-rpath,/home/wiek/Model_sim/modsim_project/pythia8317/lib \
  -std=c++17
  */
//
// If LHAPDF data is installed in a non-standard directory, set e.g.:
//   export LHAPDF_DATA_PATH=/path/to/lhapdf/share/LHAPDF
//
// Run:
//   ./hybrid_lhapdf_random_search 8 30000 hybrid_random_summary.csv adaptive_hybrid_events 0 0.05 NNPDF23_lo_as_0130_qed 8 200000
//
// Arguments:
//   argv[1] = number of search rounds             default 8
//   argv[2] = generated events per search run     default 30000
//   argv[3] = summary CSV filename                default adaptive_pt50_hybrid_lhapdf_random_summary.csv
//   argv[4] = event output prefix                 default adaptive_hybrid_events
//   argv[5] = save per-event CSVs? 0/1            default 0
//   argv[6] = minimum acceptable Neff/N           default 0.05
//   argv[7] = LHAPDF set name                     default NNPDF23_lo_as_0130_qed
//   argv[8] = candidates per round                default 8
//   argv[9] = final validation events             default 200000

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
    double a_pT        = 4.818;
    double b_tau       = 0.152;
    double c_lumi      = 0.082;
    double e_logpT     = 0.654;
    double g_tworegion = 0.102;
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
    double fracPT200 = 0.0;
    double fracPT500 = 0.0;
    double fracPT1000 = 0.0;
    double fracPT1500 = 0.0;
    double fracPT2000 = 0.0;

    double wFracPT10 = 0.0;
    double wFracPT20 = 0.0;
    double wFracPT50 = 0.0;
    double wFracPT100 = 0.0;
    double wFracPT200 = 0.0;
    double wFracPT500 = 0.0;
    double wFracPT1000 = 0.0;
    double wFracPT1500 = 0.0;
    double wFracPT2000 = 0.0;

    double score = 0.0;
};

static double safeGluonLuminosity(const LHAPDF::PDF* pdf,
                                  double x1,
                                  double x2,
                                  double Q) {
    // Defensive version: LHAPDF can throw/abort for values outside a PDF grid.
    // If anything goes wrong, return lumiRef (= no luminosity bias for that event).
    const double lumiRef = 1.0e6;
    if (pdf == nullptr) return lumiRef;

    // Keep away from exact 0 and 1. Very high-x points are also numerically fragile.
    const double xmin = 1.0e-7;
    const double xmax = 0.999999;
    x1 = min(max(x1, xmin), xmax);
    x2 = min(max(x2, xmin), xmax);

    // Keep Q in a conservative range. This avoids PDF-grid edge crashes.
    double Qsafe = Q;
    if (!isfinite(Qsafe) || Qsafe <= 0.0) Qsafe = 10.0;
    Qsafe = min(max(Qsafe, 1.0), 1.0e5);

    try {
        // LHAPDF xfxQ returns x*f(x,Q). Divide by x to get f(x,Q).
        const double xg1 = pdf->xfxQ(21, x1, Qsafe);
        const double xg2 = pdf->xfxQ(21, x2, Qsafe);

        double g1 = xg1 / x1;
        double g2 = xg2 / x2;

        if (!isfinite(g1) || g1 <= 0.0) return lumiRef;
        if (!isfinite(g2) || g2 <= 0.0) return lumiRef;

        const double lumi = g1 * g2;
        if (!isfinite(lumi) || lumi <= 0.0) return lumiRef;
        return lumi;
    } catch (...) {
        return lumiRef;
    }
}

static double computeBiasFactor(const BiasParams& p,
                                double pTHat,
                                double tau,
                                double gluonLumi) {
    const double pT0 = 10.0;
    const double tau0 = 1.0e-4;
    const double pTswitch = 50.0;
    const double lumiRef = 1.0e6;
    const double maxBias = 1.0e6;

    double bias = 1.0;

    if (p.a_pT != 0.0) {
        bias *= pow(max(pTHat / pT0, 1.0e-12), p.a_pT);
    }

    if (p.b_tau != 0.0) {
        bias *= pow(max(tau / tau0, 1.0e-12), p.b_tau);
    }

    if (p.e_logpT != 0.0) {
        bias *= pow(max(log(1.0 + pTHat / pT0), 1.0e-12), p.e_logpT);
    }

    if (p.g_tworegion != 0.0 && pTHat > pTswitch) {
        bias *= pow(max(pTHat / pTswitch, 1.0e-12), p.g_tworegion);
    }

    // Mild luminosity bias: low PDF luminosity usually corresponds to rarer high-x events.
    // Keep c_lumi small. Too large makes extreme high-x events dominate and destroys Neff.
    if (p.c_lumi != 0.0) {
        const double lumiRatio = max(lumiRef / max(gluonLumi, 1.0e-300), 1.0e-12);
        bias *= pow(lumiRatio, p.c_lumi);
    }

    if (!isfinite(bias) || bias <= 0.0) bias = maxBias;
    return min(max(1.0, bias), maxBias);
}

class HybridLHAPDFBias : public UserHooks {
public:
    HybridLHAPDFBias(const BiasParams& parsIn, shared_ptr<LHAPDF::PDF> pdfIn)
        : pars(parsIn), pdf(std::move(pdfIn)) {}

    bool canBiasSelection() override { return true; }

    double biasSelectionBy(const SigmaProcess* sigmaProcessPtr,
                           const PhaseSpace* phaseSpacePtr,
                           bool /*inEvent*/) override {
        if (sigmaProcessPtr->nFinal() != 2) return 1.0;

        const double pTHat = phaseSpacePtr->pTHat();
        const double x1 = phaseSpacePtr->x1();
        const double x2 = phaseSpacePtr->x2();
        const double tau = x1 * x2;
        const double Q = sqrt(max(sigmaProcessPtr->Q2Fac(), 1.0));
        const double gluonLumi = safeGluonLuminosity(pdf.get(), x1, x2, Q);

        return computeBiasFactor(pars, pTHat, tau, gluonLumi);
    }

private:
    BiasParams pars;
    shared_ptr<LHAPDF::PDF> pdf;
};

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

static double balancedScore(double fracPT100, double neffRatio) {
    return fracPT100 * neffRatio;
}

static void writeEventHeader(ofstream& out) {
    out << "event,x1,x2,tau,sHat,mHat,pTHat,pT2Hat,tHat,uHat,"
        << "Q2Fac,Q2Ren,alphaS,alphaEM,id1,id2,code,"
        << "gluonLumi,logGluonLumi,"
        << "pythiaWeight,biasFactor,manualWeight,"
        << "maxFinalParticlePT,nFinalParticles,nChargedFinal\n";
}

static BatchResult runBatch(const BiasParams& pars,
                            int nEvents,
                            int iter,
                            const string& eventPrefix,
                            bool saveEvents,
                            const string& pdfSetName) {
    unique_ptr<Pythia> pythiaPtr;
    shared_ptr<LHAPDF::PDF> pdf;

    {
        TerminalSilencer silence(true);
        pdf.reset(LHAPDF::mkPDF(pdfSetName, 0));
        pythiaPtr = make_unique<Pythia>();
        configurePythia(*pythiaPtr);
        pythiaPtr->setUserHooksPtr(make_shared<HybridLHAPDFBias>(pars, pdf));
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
    int n200 = 0, n500 = 0, n1000 = 0, n1500 = 0, n2000 = 0;
    double w10 = 0.0, w20 = 0.0, w50 = 0.0, w100 = 0.0;
    double w200 = 0.0, w500 = 0.0, w1000 = 0.0, w1500 = 0.0, w2000 = 0.0;

    TerminalSilencer silence(true);
    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
        if (!pythia.next()) continue;

        const double x1 = pythia.info.x1();
        const double x2 = pythia.info.x2();
        const double tau = x1 * x2;
        const double sHat = pythia.info.sHat();
        const double pTHat = pythia.info.pTHat();
        const double Q = sqrt(max(pythia.info.Q2Fac(), 1.0));
        const double gluonLumi = safeGluonLuminosity(pdf.get(), x1, x2, Q);
        const double biasFactor = computeBiasFactor(pars, pTHat, tau, gluonLumi);
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

        if (pTHat > 10.0)   { ++n10;   w10   += manualWeight; }
        if (pTHat > 20.0)   { ++n20;   w20   += manualWeight; }
        if (pTHat > 50.0)   { ++n50;   w50   += manualWeight; }
        if (pTHat > 100.0)  { ++n100;  w100  += manualWeight; }
        if (pTHat > 200.0)  { ++n200;  w200  += manualWeight; }
        if (pTHat > 500.0)  { ++n500;  w500  += manualWeight; }
        if (pTHat > 1000.0) { ++n1000; w1000 += manualWeight; }
        if (pTHat > 1500.0) { ++n1500; w1500 += manualWeight; }
        if (pTHat > 2000.0) { ++n2000; w2000 += manualWeight; }

        if (saveEvents) {
            eventOut << iEvent << "," << x1 << "," << x2 << "," << tau << ","
                     << sHat << "," << sqrt(max(sHat, 0.0)) << "," << pTHat << ","
                     << pTHat * pTHat << "," << pythia.info.tHat() << "," << pythia.info.uHat() << ","
                     << pythia.info.Q2Fac() << "," << pythia.info.Q2Ren() << ","
                     << pythia.info.alphaS() << "," << pythia.info.alphaEM() << ","
                     << pythia.info.id1() << "," << pythia.info.id2() << "," << pythia.info.code() << ","
                     << gluonLumi << "," << log(max(gluonLumi, 1.0e-300)) << ","
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
    r.fracPT200 = double(n200) / r.accepted;
    r.fracPT500 = double(n500) / r.accepted;
    r.fracPT1000 = double(n1000) / r.accepted;
    r.fracPT1500 = double(n1500) / r.accepted;
    r.fracPT2000 = double(n2000) / r.accepted;

    if (r.sumW > 0.0) {
        r.wFracPT10 = w10 / r.sumW;
        r.wFracPT20 = w20 / r.sumW;
        r.wFracPT50 = w50 / r.sumW;
        r.wFracPT100 = w100 / r.sumW;
        r.wFracPT200 = w200 / r.sumW;
        r.wFracPT500 = w500 / r.sumW;
        r.wFracPT1000 = w1000 / r.sumW;
        r.wFracPT1500 = w1500 / r.sumW;
        r.wFracPT2000 = w2000 / r.sumW;
    }

    r.score = balancedScore(r.fracPT50, r.neffRatio);
    return r;
}

static BiasParams clampParams(BiasParams p) {
    auto clamp = [](double x, double lo, double hi) { return max(lo, min(hi, x)); };
    p.a_pT        = clamp(p.a_pT,        0.0, 8.0);
    p.b_tau       = clamp(p.b_tau,       0.0, 1.0);
    p.c_lumi      = clamp(p.c_lumi,      0.0, 0.25);
    p.e_logpT     = clamp(p.e_logpT,     0.0, 5.0);
    p.g_tworegion = clamp(p.g_tworegion, 0.0, 8.0);
    return p;
}


static BiasParams proposeRandomCandidate(const BiasParams& best,
                                         int round,
                                         int candidateIndex,
                                         mt19937& rng) {
    // Adaptive random local search:
    // - candidate 0 is always the current best unchanged, so the round cannot get worse.
    // - the other candidates are random perturbations around the current best.
    // - the step scale shrinks each round, so early rounds explore and later rounds refine.
    if (candidateIndex == 0) return best;

    const double scale = max(0.20, pow(0.72, round));

    normal_distribution<double> n01(0.0, 1.0);

    BiasParams p = best;

    // These step sizes are chosen to match the natural scale of each parameter.
    // c_lumi is kept much smaller because a large PDF-luminosity exponent can
    // easily create very large weight fluctuations.
    p.a_pT        += n01(rng) * 0.45 * scale;
    p.b_tau       += n01(rng) * 0.08 * scale;
    p.c_lumi      += n01(rng) * 0.025 * scale;
    p.e_logpT     += n01(rng) * 0.20 * scale;
    p.g_tworegion += n01(rng) * 0.25 * scale;

    // In the first two rounds, add a small chance of broader jumps. This helps
    // escape the initial point if the old hand-tuned values are not optimal with
    // the new luminosity term.
    if (round < 2 && (candidateIndex % 4 == 0)) {
        p.a_pT        += n01(rng) * 0.60;
        p.b_tau       += n01(rng) * 0.12;
        p.c_lumi      += n01(rng) * 0.035;
        p.e_logpT     += n01(rng) * 0.30;
        p.g_tworegion += n01(rng) * 0.35;
    }

    return clampParams(p);
}

static void writeSummaryHeader(ofstream& out) {
    out << "stage,round,candidate,accepted,acceptable,bestSoFar,"
        << "a_pT,b_tau,c_lumi,e_logpT,g_tworegion,"
        << "sumManualW,sumManualW2,meanManualW,stdManualW,NeffManual,NeffManualOverN,"
        << "meanPTHat,maxPTHat,"
        << "fracPT10,fracPT20,fracPT50,fracPT100,fracPT200,fracPT500,fracPT1000,fracPT1500,fracPT2000,"
        << "wFracPT10,wFracPT20,wFracPT50,wFracPT100,wFracPT200,wFracPT500,wFracPT1000,wFracPT1500,wFracPT2000,score\n";
}

static void writeSummaryRow(ofstream& out, const string& stage, int round, int cand,
                            const BiasParams& p, const BatchResult& r,
                            bool acceptable, bool bestSoFar) {
    out << stage << "," << round << "," << cand << ","
        << r.accepted << "," << int(acceptable) << "," << int(bestSoFar) << ","
        << p.a_pT << "," << p.b_tau << "," << p.c_lumi << ","
        << p.e_logpT << "," << p.g_tworegion << ","
        << r.sumW << "," << r.sumW2 << "," << r.meanW << "," << r.stdW << ","
        << r.neff << "," << r.neffRatio << ","
        << r.meanPTHat << "," << r.maxPTHat << ","
        << r.fracPT10 << "," << r.fracPT20 << "," << r.fracPT50 << "," << r.fracPT100 << ","
        << r.fracPT200 << "," << r.fracPT500 << "," << r.fracPT1000 << "," << r.fracPT1500 << "," << r.fracPT2000 << ","
        << r.wFracPT10 << "," << r.wFracPT20 << "," << r.wFracPT50 << "," << r.wFracPT100 << ","
        << r.wFracPT200 << "," << r.wFracPT500 << "," << r.wFracPT1000 << "," << r.wFracPT1500 << "," << r.wFracPT2000 << ","
        << r.score << "\n";
}

static void printCandidateResult(const BatchResult& r, bool acceptable, bool isBest) {
    cout << "  accepted=" << r.accepted
         << ", fracPT50=" << r.fracPT50
         << ", fracPT100=" << r.fracPT100
         << ", fracPT200=" << r.fracPT200
         << ", fracPT500=" << r.fracPT500
         << ", fracPT1000=" << r.fracPT1000
         << ", fracPT2000=" << r.fracPT2000
         << ", wFracPT50=" << scientific << r.wFracPT50
         << ", wFracPT100=" << r.wFracPT100
         << ", wFracPT200=" << r.wFracPT200
         << ", wFracPT500=" << r.wFracPT500
         << ", wFracPT1000=" << r.wFracPT1000
         << ", wFracPT2000=" << r.wFracPT2000 << fixed
         << ", Neff/N=" << r.neffRatio
         << ", score=" << r.score
         << ", meanW=" << r.meanW
         << ", stdW=" << r.stdW
         << ", meanPTHat=" << r.meanPTHat
         << ", maxPTHat=" << r.maxPTHat;
    if (!acceptable) cout << "  [rejected: low Neff/N]";
    if (isBest) cout << "  [new best]";
    cout << "\n";
}

int main(int argc, char* argv[]) {
    int nRounds = 8;
    int nEventsPerSearch = 30000;
    string summaryFile = "adaptive_pt50_hybrid_lhapdf_random_summary.csv";
    string eventPrefix = "adaptive_hybrid_events";
    bool saveEvents = false;
    double minNeffRatio = 0.05;
    string pdfSetName = "NNPDF31_lo_as_0118";
    int candidatesPerRound = 8;
    int nFinalValidationEvents = 200000;

    if (argc > 1) nRounds = atoi(argv[1]);
    if (argc > 2) nEventsPerSearch = atoi(argv[2]);
    if (argc > 3) summaryFile = argv[3];
    if (argc > 4) eventPrefix = argv[4];
    if (argc > 5) saveEvents = (atoi(argv[5]) != 0);
    if (argc > 6) minNeffRatio = atof(argv[6]);
    if (argc > 7) pdfSetName = argv[7];
    if (argc > 8) candidatesPerRound = max(2, atoi(argv[8]));
    if (argc > 9) nFinalValidationEvents = atoi(argv[9]);

    mt19937 rng(12345);
    BiasParams bestPars = clampParams(BiasParams{});
    BatchResult bestRes;
    double bestScore = -1.0;

    ofstream summary(summaryFile);
    writeSummaryHeader(summary);

    cout << fixed << setprecision(6);
    cout << "Starting hybrid LHAPDF adaptive RANDOM search\n";
    cout << "Rounds: " << nRounds << "\n";
    cout << "Candidates per round: " << candidatesPerRound << "\n";
    cout << "Events per search candidate: " << nEventsPerSearch << "\n";
    cout << "Final validation events: " << nFinalValidationEvents << "\n";
    cout << "Minimum acceptable Neff/N: " << minNeffRatio << "\n";
    cout << "PDF set: " << pdfSetName << "\n";
    cout << "Summary file: " << summaryFile << "\n";
    cout << "Saving event CSVs: " << (saveEvents ? "yes" : "no") << "\n\n";

    // Evaluate the initial hand-tuned point once before random search.
    cout << "Initial candidate: "
         << "a=" << bestPars.a_pT
         << ", b_tau=" << bestPars.b_tau
         << ", c_lumi=" << bestPars.c_lumi
         << ", e=" << bestPars.e_logpT
         << ", g=" << bestPars.g_tworegion << "\n";

    {
        BatchResult r = runBatch(bestPars, nEventsPerSearch, 0, eventPrefix, saveEvents, pdfSetName);
        const bool acceptable = (r.accepted > 0 && r.neffRatio >= minNeffRatio);
        const bool isBest = acceptable && (r.score > bestScore);
        if (isBest) {
            bestScore = r.score;
            bestRes = r;
        }
        writeSummaryRow(summary, "initial", -1, 0, bestPars, r, acceptable, isBest);
        summary.flush();
        printCandidateResult(r, acceptable, isBest);
        cout << "\n";
    }

    for (int round = 0; round < nRounds; ++round) {
        cout << "Round " << round << " around current best: "
             << "a=" << bestPars.a_pT
             << ", b_tau=" << bestPars.b_tau
             << ", c_lumi=" << bestPars.c_lumi
             << ", e=" << bestPars.e_logpT
             << ", g=" << bestPars.g_tworegion
             << ", bestScore=" << bestScore << "\n";

        BiasParams roundBestPars = bestPars;
        BatchResult roundBestRes = bestRes;
        double roundBestScore = bestScore;

        for (int cand = 0; cand < candidatesPerRound; ++cand) {
            BiasParams candidate = proposeRandomCandidate(bestPars, round, cand, rng);

            cout << "  candidate " << cand << ": "
                 << "a=" << candidate.a_pT
                 << ", b_tau=" << candidate.b_tau
                 << ", c_lumi=" << candidate.c_lumi
                 << ", e=" << candidate.e_logpT
                 << ", g=" << candidate.g_tworegion << "\n";

            const int eventIterLabel = 1000 * round + cand;
            BatchResult r = runBatch(candidate, nEventsPerSearch, eventIterLabel,
                                     eventPrefix, saveEvents, pdfSetName);

            const bool acceptable = (r.accepted > 0 && r.neffRatio >= minNeffRatio);
            const bool isBest = acceptable && (r.score > bestScore);

            if (acceptable && r.score > roundBestScore) {
                roundBestScore = r.score;
                roundBestPars = candidate;
                roundBestRes = r;
            }

            if (isBest) {
                bestScore = r.score;
                bestPars = candidate;
                bestRes = r;
            }

            writeSummaryRow(summary, "search", round, cand, candidate, r, acceptable, isBest);
            summary.flush();
            printCandidateResult(r, acceptable, isBest);
        }

        // Move to the best point found in this round. Usually this is the global
        // best too, but keeping this explicit makes the search logic clear.
        bestPars = roundBestPars;
        bestRes = roundBestRes;
        bestScore = roundBestScore;

        cout << "End round " << round << ": best "
             << "score=" << bestScore
             << ", a=" << bestPars.a_pT
             << ", b_tau=" << bestPars.b_tau
             << ", c_lumi=" << bestPars.c_lumi
             << ", e=" << bestPars.e_logpT
             << ", g=" << bestPars.g_tworegion << "\n\n";
    }

    cout << "Running final validation for best parameters with "
         << nFinalValidationEvents << " events...\n";

    BatchResult finalRes = runBatch(bestPars, nFinalValidationEvents, 999999,
                                    eventPrefix + "_final", saveEvents, pdfSetName);
    const bool finalAcceptable = (finalRes.accepted > 0 && finalRes.neffRatio >= minNeffRatio);
    writeSummaryRow(summary, "final", nRounds, -1, bestPars, finalRes, finalAcceptable, true);
    summary.flush();

    cout << "\nSearch finished.\n";
    cout << "Best search score             = " << bestScore << "\n";
    cout << "Final validation score        = " << finalRes.score << "\n";
    cout << "Final generated fracPT50      = " << finalRes.fracPT50 << "\n";
    cout << "Final generated fracPT100     = " << finalRes.fracPT100 << "\n";
    cout << "Final generated fracPT200     = " << finalRes.fracPT200 << "\n";
    cout << "Final generated fracPT500     = " << finalRes.fracPT500 << "\n";
    cout << "Final generated fracPT1000    = " << finalRes.fracPT1000 << "\n";
    cout << "Final generated fracPT1500    = " << finalRes.fracPT1500 << "\n";
    cout << "Final generated fracPT2000    = " << finalRes.fracPT2000 << "\n";
    cout << "Final weighted wFracPT50      = " << scientific << finalRes.wFracPT50 << fixed << "\n";
    cout << "Final weighted wFracPT100     = " << scientific << finalRes.wFracPT100 << fixed << "\n";
    cout << "Final weighted wFracPT200     = " << scientific << finalRes.wFracPT200 << fixed << "\n";
    cout << "Final weighted wFracPT500     = " << scientific << finalRes.wFracPT500 << fixed << "\n";
    cout << "Final weighted wFracPT1000    = " << scientific << finalRes.wFracPT1000 << fixed << "\n";
    cout << "Final weighted wFracPT1500    = " << scientific << finalRes.wFracPT1500 << fixed << "\n";
    cout << "Final weighted wFracPT2000    = " << scientific << finalRes.wFracPT2000 << fixed << "\n";
    cout << "Final Neff/N                  = " << finalRes.neffRatio << "\n";
    cout << "Best parameters:\n"
         << "  a_pT        = " << bestPars.a_pT << "\n"
         << "  b_tau       = " << bestPars.b_tau << "\n"
         << "  c_lumi      = " << bestPars.c_lumi << "\n"
         << "  e_logpT     = " << bestPars.e_logpT << "\n"
         << "  g_tworegion = " << bestPars.g_tworegion << "\n";

    return 0;
}

