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
// Compile example, adjust paths for your machine:
//   g++ clean_adaptive_IS_hybrid_lhapdf_sampler.cc -o hybrid_lhapdf_sampler \
//      -I../pythia8317/include -I/path/to/lhapdf/include \
//      -L../pythia8317/lib -L/path/to/lhapdf/lib \
//      -lpythia8 -lLHAPDF \
//      -Wl,-rpath,../pythia8317/lib -Wl,-rpath,/path/to/lhapdf/lib \
//      -std=c++17
//
// If LHAPDF data is installed in a non-standard directory, set e.g.:
//   export LHAPDF_DATA_PATH=/path/to/lhapdf/share/LHAPDF
//
// Run, same style as before:
//   ./hybrid_lhapdf_sampler 30 100000 hybrid_lhapdf_summary.csv adaptive_hybrid_events 0 0.05
//
// Arguments:
//   argv[1] = number of adaptive iterations       default 30
//   argv[2] = generated events per iteration      default 100000
//   argv[3] = summary CSV filename                default adaptive_pt50_hybrid_lhapdf_summary.csv
//   argv[4] = event output prefix                 default adaptive_hybrid_events
//   argv[5] = save per-event CSVs? 0/1            default 0
//   argv[6] = minimum acceptable Neff/N           default 0.05
//   argv[7] = LHAPDF set name                     default NNPDF23_lo_as_0130_qed

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
    double a_pT        = 3.8;
    double b_tau       = 0.14;
    double c_lumi      = 0.05;
    double e_logpT     = 0.65;
    double g_tworegion = 0.35;
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

static double safeGluonLuminosity(const LHAPDF::PDF* pdf,
                                  double x1,
                                  double x2,
                                  double Q) {
    const double xmin = 1.0e-12;
    const double xmax = 1.0 - 1.0e-12;
    const double Qsafe = max(Q, 1.0);

    x1 = min(max(x1, xmin), xmax);
    x2 = min(max(x2, xmin), xmax);

    // LHAPDF xfxQ returns x*f(x,Q). Divide by x to get f(x,Q).
    const double xg1 = pdf->xfxQ(21, x1, Qsafe);
    const double xg2 = pdf->xfxQ(21, x2, Qsafe);

    double g1 = xg1 / x1;
    double g2 = xg2 / x2;

    if (!isfinite(g1) || g1 <= 0.0) g1 = 1.0e-300;
    if (!isfinite(g2) || g2 <= 0.0) g2 = 1.0e-300;

    const double lumi = g1 * g2;
    if (!isfinite(lumi) || lumi <= 0.0) return 1.0e-300;
    return lumi;
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

static double balancedScore(double fracPT50, double neffRatio) {
    return fracPT50 * neffRatio;
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
    double w10 = 0.0, w20 = 0.0, w50 = 0.0, w100 = 0.0;

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

        if (pTHat > 10.0)  { ++n10;  w10  += manualWeight; }
        if (pTHat > 20.0)  { ++n20;  w20  += manualWeight; }
        if (pTHat > 50.0)  { ++n50;  w50  += manualWeight; }
        if (pTHat > 100.0) { ++n100; w100 += manualWeight; }

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

    if (r.sumW > 0.0) {
        r.wFracPT10 = w10 / r.sumW;
        r.wFracPT20 = w20 / r.sumW;
        r.wFracPT50 = w50 / r.sumW;
        r.wFracPT100 = w100 / r.sumW;
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

static BiasParams proposeCandidate(const BiasParams& best, int iter, mt19937& rng) {
    (void)rng;

    BiasParams p = best;

    // Cycle through five parameters:
    //   a_pT -> b_tau -> c_lumi -> e_logpT -> g_tworegion
    const int paramIndex = (iter - 1) % 5;

    // First pass positive, second pass negative, then repeat.
    const int pass = (iter - 1) / 5;
    const double direction = (pass % 2 == 0) ? +1.0 : -1.0;

    const int fullCycle = (iter - 1) / 10;
    const double scale = max(0.20, pow(0.75, fullCycle));

    const double stepA = 0.35 * scale;
    const double stepB = 0.05 * scale;
    const double stepC = 0.02 * scale;
    const double stepE = 0.15 * scale;
    const double stepG = 0.20 * scale;

    if (paramIndex == 0) p.a_pT        += direction * stepA;
    if (paramIndex == 1) p.b_tau       += direction * stepB;
    if (paramIndex == 2) p.c_lumi      += direction * stepC;
    if (paramIndex == 3) p.e_logpT     += direction * stepE;
    if (paramIndex == 4) p.g_tworegion += direction * stepG;

    return clampParams(p);
}

static void writeSummaryHeader(ofstream& out) {
    out << "iteration,accepted,acceptable,bestSoFar,"
        << "a_pT,b_tau,c_lumi,e_logpT,g_tworegion,"
        << "sumManualW,sumManualW2,meanManualW,stdManualW,NeffManual,NeffManualOverN,"
        << "meanPTHat,maxPTHat,"
        << "fracPT10,fracPT20,fracPT50,fracPT100,"
        << "wFracPT10,wFracPT20,wFracPT50,wFracPT100,score\n";
}

static void writeSummaryRow(ofstream& out, int iter, const BiasParams& p,
                            const BatchResult& r, bool acceptable, bool bestSoFar) {
    out << iter << "," << r.accepted << "," << int(acceptable) << "," << int(bestSoFar) << ","
        << p.a_pT << "," << p.b_tau << "," << p.c_lumi << ","
        << p.e_logpT << "," << p.g_tworegion << ","
        << r.sumW << "," << r.sumW2 << "," << r.meanW << "," << r.stdW << ","
        << r.neff << "," << r.neffRatio << ","
        << r.meanPTHat << "," << r.maxPTHat << ","
        << r.fracPT10 << "," << r.fracPT20 << "," << r.fracPT50 << "," << r.fracPT100 << ","
        << r.wFracPT10 << "," << r.wFracPT20 << "," << r.wFracPT50 << "," << r.wFracPT100 << ","
        << r.score << "\n";
}

int main(int argc, char* argv[]) {
    int nIterations = 30;
    int nEventsPerBatch = 100000;
    string summaryFile = "adaptive_pt50_hybrid_lhapdf_summary.csv";
    string eventPrefix = "adaptive_hybrid_events";
    bool saveEvents = false;
    double minNeffRatio = 0.05;
    string pdfSetName = "NNPDF31_lo_as_0118";

    if (argc > 1) nIterations = atoi(argv[1]);
    if (argc > 2) nEventsPerBatch = atoi(argv[2]);
    if (argc > 3) summaryFile = argv[3];
    if (argc > 4) eventPrefix = argv[4];
    if (argc > 5) saveEvents = (atoi(argv[5]) != 0);
    if (argc > 6) minNeffRatio = atof(argv[6]);
    if (argc > 7) pdfSetName = argv[7];

    mt19937 rng(12345);
    BiasParams bestPars = clampParams(BiasParams{});
    BatchResult bestRes;
    double bestScore = -1.0;

    ofstream summary(summaryFile);
    writeSummaryHeader(summary);

    cout << fixed << setprecision(6);
    cout << "Starting hybrid LHAPDF adaptive manual-unbias search\n";
    cout << "Iterations: " << nIterations << "\n";
    cout << "Events per iteration: " << nEventsPerBatch << "\n";
    cout << "Minimum acceptable Neff/N: " << minNeffRatio << "\n";
    cout << "PDF set: " << pdfSetName << "\n";
    cout << "Summary file: " << summaryFile << "\n";
    cout << "Saving event CSVs: " << (saveEvents ? "yes" : "no") << "\n\n";

    for (int iter = 0; iter < nIterations; ++iter) {
        BiasParams candidate = (iter == 0) ? bestPars : proposeCandidate(bestPars, iter, rng);

        cout << "Iteration " << iter << " candidate: "
             << "a=" << candidate.a_pT
             << ", b_tau=" << candidate.b_tau
             << ", c_lumi=" << candidate.c_lumi
             << ", e=" << candidate.e_logpT
             << ", g=" << candidate.g_tworegion << "\n";

        BatchResult r = runBatch(candidate, nEventsPerBatch, iter, eventPrefix, saveEvents, pdfSetName);
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
         << "  b_tau       = " << bestPars.b_tau << "\n"
         << "  c_lumi      = " << bestPars.c_lumi << "\n"
         << "  e_logpT     = " << bestPars.e_logpT << "\n"
         << "  g_tworegion = " << bestPars.g_tworegion << "\n";

    return 0;
}

