// Adaptive multivariable phase-space bias search for rare high-pT QCD events in Pythia8
//
// Goal:
//   Search for bias parameters that increase the generated fraction of events with pTHat > 50 GeV,
//   while keeping the manual-unbiasing weights statistically usable.
//
// Compile with:
//   g++ adaptive_pt50_balanced_manual_unbias_10iter.cc -o adaptive_pt50_balanced_manual_unbias_10iter \
     -I../pythia8317/include \
     -L../pythia8317/lib \
     -lpythia8 \
     -Wl,-rpath,../pythia8317/lib \
     -std=c++17
//
// Example run:
//   ./adaptive_pt50_balanced_manual_unbias_10iter 10 100000 adaptive_pt50_balanced_manual_10_summary.csv adaptive_manual_events 0 0.05
//
// Arguments:
//   argv[1] = number of adaptive iterations       default 10
//   argv[2] = generated events per iteration      default 100000
//   argv[3] = summary CSV filename                default adaptive_pt50_balanced_manual_10_summary.csv
//   argv[4] = event output prefix                 default adaptive_events
//   argv[5] = save per-event CSVs? 0/1            default 0
//   argv[6] = minimum acceptable Neff/N           default 0.05
//
// Important interpretation:
//   fracPT50  = unweighted generated fraction in the biased sample. This is what the search maximizes.
//   wFracPT50 = manually weighted/unbiased physical fraction using pythia.info.weight()/biasFactor.
//               This should be compared with the naive physical result.
//   Neff/N    = effective sample-size fraction computed from the manual unbiasing weights.
//
//   Balanced objective:
//       objective = fracPT50 * NeffManualOverN

#include "Pythia8/Pythia.h"
#include "Pythia8/UserHooks.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>

using namespace Pythia8;
using namespace std;

// ------------------------------------------------------------
// Optional stdout silencer for Pythia init/event listings.
// We do not use it around our own iteration prints.
// ------------------------------------------------------------
class CoutSilencer {
public:
    explicit CoutSilencer(bool activeIn) : active(activeIn), oldBuf(nullptr) {
        if (active) {
            nullStream.open("/dev/null");
            oldBuf = cout.rdbuf(nullStream.rdbuf());
        }
    }
    ~CoutSilencer() {
        if (active && oldBuf) cout.rdbuf(oldBuf);
    }
private:
    bool active;
    streambuf* oldBuf;
    ofstream nullStream;
};

// ------------------------------------------------------------
// Parameter container
// ------------------------------------------------------------
struct BiasParams {
    double a_pT        = 3.8;  // power-law pTHat bias
    double b_xprod     = 0.28; // x1*x2 bias
    double c_Q2        = 0.28; // Q2Fac bias
    double d_alphaS    = 0.0;  // kept for bookkeeping; not safely available in this hook
    double e_logpT     = 0.0;  // logarithmic pTHat bias
    double f_satpT     = 0.0;  // saturating pTHat bias
    double g_tworegion = 0.0;  // extra high-pT two-region bias
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

    // Unweighted/generated fractions in the biased sample.
    double fracPT10 = 0.0;
    double fracPT20 = 0.0;
    double fracPT50 = 0.0;
    double fracPT100 = 0.0;

    // Weighted/unbiased physical fractions.
    double wFracPT10 = 0.0;
    double wFracPT20 = 0.0;
    double wFracPT50 = 0.0;
    double wFracPT100 = 0.0;

    // Objective used by the search.
    double objective = 0.0;
};

// ------------------------------------------------------------
// Bias-factor calculation.
//
// Important:
// Pythia does not appear to include the inverse of this custom UserHooks
// selection bias in pythia.info.weight() for the way this code uses
// biasSelectionBy(). Therefore we recompute the same bias for each accepted
// event and use
//
//     manualWeight = pythia.info.weight() / biasFactor
//
// for all physical/unbiased quantities.
// ------------------------------------------------------------
static double computeBiasFactor(const BiasParams& pars,
                                double pTHat,
                                double xprod,
                                double Q2Fac,
                                double pT0 = 10.0,
                                double xprod0 = 1.0e-4,
                                double Q20 = 100.0,
                                double pTswitch = 50.0,
                                double maxBias = 1.0e6) {
    double bias = 1.0;

    if (pTHat > 0.0 && pars.a_pT != 0.0) {
        bias *= pow(max(pTHat / pT0, 1.0e-12), pars.a_pT);
    }

    if (xprod > 0.0 && pars.b_xprod != 0.0) {
        bias *= pow(max(xprod / xprod0, 1.0e-12), pars.b_xprod);
    }

    if (Q2Fac > 0.0 && pars.c_Q2 != 0.0) {
        bias *= pow(max(Q2Fac / Q20, 1.0e-12), pars.c_Q2);
    }

    // d_alphaS is intentionally not used: alphaS is not safely available here
    // in the same way as pTHat, x1*x2 and Q2Fac are.

    if (pTHat > 0.0 && pars.e_logpT != 0.0) {
        bias *= pow(max(log(1.0 + pTHat / pT0), 1.0e-12), pars.e_logpT);
    }

    if (pTHat > 0.0 && pars.f_satpT != 0.0) {
        const double sat = 1.0 + pTHat / (pTHat + pT0); // between 1 and 2
        bias *= pow(sat, pars.f_satpT);
    }

    if (pTHat > pTswitch && pars.g_tworegion != 0.0) {
        bias *= pow(max(pTHat / pTswitch, 1.0e-12), pars.g_tworegion);
    }

    if (!isfinite(bias) || bias <= 0.0) bias = maxBias;
    bias = max(1.0, bias);
    bias = min(maxBias, bias);

    return bias;
}

// ------------------------------------------------------------
// Custom multivariable phase-space bias
// ------------------------------------------------------------
class MultiVariableBias : public UserHooks {
public:
    BiasParams pars;

    double pT0;
    double xprod0;
    double Q20;
    double pTswitch;
    double maxBias;

    MultiVariableBias(
        const BiasParams& parsIn,
        double pT0In = 10.0,
        double xprod0In = 1e-4,
        double Q20In = 100.0,
        double pTswitchIn = 50.0,
        double maxBiasIn = 1.0e6
    ) : pars(parsIn),
        pT0(pT0In),
        xprod0(xprod0In),
        Q20(Q20In),
        pTswitch(pTswitchIn),
        maxBias(maxBiasIn) {}

    bool canBiasSelection() override { return true; }

    double biasSelectionBy(const SigmaProcess* sigmaProcessPtr,
                           const PhaseSpace* phaseSpacePtr,
                           bool /*inEvent*/) override {

        if (sigmaProcessPtr->nFinal() != 2) return 1.0;

        const double pTHat = phaseSpacePtr->pTHat();
        const double x1 = phaseSpacePtr->x1();
        const double x2 = phaseSpacePtr->x2();
        const double xprod = x1 * x2;
        const double Q2Fac = sigmaProcessPtr->Q2Fac();

        return computeBiasFactor(pars, pTHat, xprod, Q2Fac,
                                 pT0, xprod0, Q20, pTswitch, maxBias);
    }
};

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------
static string eventFileName(const string& prefix, int iter) {
    ostringstream name;
    name << prefix << "_iter_" << setw(3) << setfill('0') << iter << ".csv";
    return name.str();
}

static void writeEventHeader(ofstream& out) {
    out << "event,x1,x2,xprod,sHat,mHat,pTHat,pT2Hat,tHat,uHat,"
        << "Q2Fac,Q2Ren,alphaS,alphaEM,id1,id2,code,"
        << "pythiaWeight,biasFactor,manualWeight,"
        << "maxFinalParticlePT,nFinalParticles,nChargedFinal\n";
}

static void configureQuietPythia(Pythia& pythia) {
    pythia.readString("Print:quiet = on");
    pythia.readString("Init:showChangedSettings = off");
    pythia.readString("Init:showAllSettings = off");
    pythia.readString("Init:showChangedParticleData = off");
    pythia.readString("Init:showAllParticleData = off");
    pythia.readString("Next:numberShowInfo = 0");
    pythia.readString("Next:numberShowProcess = 0");
    pythia.readString("Next:numberShowEvent = 0");
    pythia.readString("Next:showScaleAndVertex = off");
}

static BatchResult runBatch(const BiasParams& pars,
                            int nEvents,
                            int iter,
                            const string& eventPrefix,
                            bool saveEvents,
                            bool silencePythia = true) {

    Pythia pythia;
    configureQuietPythia(pythia);

    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13000.");
    pythia.readString("HardQCD:all = on");
    pythia.readString("PhaseSpace:pTHatMin = 0.");

    shared_ptr<UserHooks> userHooks = make_shared<MultiVariableBias>(
        pars,
        10.0,    // pT0
        1e-4,    // xprod0
        100.0,   // Q20
        50.0,    // pTswitch
        1.0e6    // maxBias
    );
    pythia.setUserHooksPtr(userHooks);

    {
        CoutSilencer silence(silencePythia);
        if (!pythia.init()) {
            cerr << "Pythia initialization failed in iteration " << iter << endl;
            return {};
        }
    }

    ofstream eventOut;
    if (saveEvents) {
        eventOut.open(eventFileName(eventPrefix, iter));
        writeEventHeader(eventOut);
    }

    BatchResult res;
    double sumPTHat = 0.0;

    int nPT10 = 0;
    int nPT20 = 0;
    int nPT50 = 0;
    int nPT100 = 0;

    double wPT10 = 0.0;
    double wPT20 = 0.0;
    double wPT50 = 0.0;
    double wPT100 = 0.0;

    CoutSilencer silence(silencePythia);

    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
        if (!pythia.next()) continue;

        const double x1 = pythia.info.x1();
        const double x2 = pythia.info.x2();
        const double xprod = x1 * x2;

        const double sHat = pythia.info.sHat();
        const double mHat = sqrt(max(sHat, 0.0));
        const double pTHat = pythia.info.pTHat();
        const double pT2Hat = pTHat * pTHat;
        const double tHat = pythia.info.tHat();
        const double uHat = pythia.info.uHat();

        const double Q2Fac = pythia.info.Q2Fac();
        const double Q2Ren = pythia.info.Q2Ren();
        const double alphaS = pythia.info.alphaS();
        const double alphaEM = pythia.info.alphaEM();

        const int id1 = pythia.info.id1();
        const int id2 = pythia.info.id2();
        const int code = pythia.info.code();

        // Pythia's own event weight. For this custom bias hook it is not enough
        // by itself to undo the selection bias, so we manually divide by the
        // same bias factor that was used in biasSelectionBy().
        const double pythiaWeight = pythia.info.weight();
        const double biasFactor = computeBiasFactor(pars, pTHat, xprod, Q2Fac,
                                                    10.0, 1.0e-4, 100.0, 50.0, 1.0e6);
        const double manualWeight = pythiaWeight / max(biasFactor, 1.0e-300);

        double maxFinalParticlePT = 0.0;
        int nFinalParticles = 0;
        int nChargedFinal = 0;

        for (int i = 0; i < pythia.event.size(); ++i) {
            if (pythia.event[i].isFinal()) {
                nFinalParticles++;
                const double pT = pythia.event[i].pT();
                maxFinalParticlePT = max(maxFinalParticlePT, pT);
                if (pythia.event[i].isCharged()) nChargedFinal++;
            }
        }

        res.accepted++;
        res.sumW += manualWeight;
        res.sumW2 += manualWeight * manualWeight;
        sumPTHat += pTHat;
        res.maxPTHat = max(res.maxPTHat, pTHat);

        if (pTHat > 10.0)  { nPT10++;  wPT10  += manualWeight; }
        if (pTHat > 20.0)  { nPT20++;  wPT20  += manualWeight; }
        if (pTHat > 50.0)  { nPT50++;  wPT50  += manualWeight; }
        if (pTHat > 100.0) { nPT100++; wPT100 += manualWeight; }

        if (saveEvents) {
            eventOut << iEvent << ","
                     << x1 << ","
                     << x2 << ","
                     << xprod << ","
                     << sHat << ","
                     << mHat << ","
                     << pTHat << ","
                     << pT2Hat << ","
                     << tHat << ","
                     << uHat << ","
                     << Q2Fac << ","
                     << Q2Ren << ","
                     << alphaS << ","
                     << alphaEM << ","
                     << id1 << ","
                     << id2 << ","
                     << code << ","
                     << pythiaWeight << ","
                     << biasFactor << ","
                     << manualWeight << ","
                     << maxFinalParticlePT << ","
                     << nFinalParticles << ","
                     << nChargedFinal << "\n";
        }
    }

    if (saveEvents) eventOut.close();

    if (res.accepted > 0) {
        res.meanW = res.sumW / res.accepted;
        const double meanW2 = res.sumW2 / res.accepted;
        const double varW = meanW2 - res.meanW * res.meanW;
        res.stdW = sqrt(max(0.0, varW));

        if (res.sumW2 > 0.0) {
            res.neff = (res.sumW * res.sumW) / res.sumW2;
            res.neffRatio = res.neff / res.accepted;
        }

        res.fracPT10 = double(nPT10) / res.accepted;
        res.fracPT20 = double(nPT20) / res.accepted;
        res.fracPT50 = double(nPT50) / res.accepted;
        res.fracPT100 = double(nPT100) / res.accepted;
        res.meanPTHat = sumPTHat / res.accepted;

        if (res.sumW > 0.0) {
            res.wFracPT10 = wPT10 / res.sumW;
            res.wFracPT20 = wPT20 / res.sumW;
            res.wFracPT50 = wPT50 / res.sumW;
            res.wFracPT100 = wPT100 / res.sumW;
        }

        // Balanced objective: reward high generated pTHat > 50 fraction,
        // but penalize candidates with noisy unbiasing weights.
        res.objective = res.fracPT50 * res.neffRatio;
    }

    return res;
}

static double clampDouble(double x, double lo, double hi) {
    return max(lo, min(hi, x));
}

static BiasParams clampParams(BiasParams p) {
    p.a_pT        = clampDouble(p.a_pT,        0.0, 8.0);
    p.b_xprod     = clampDouble(p.b_xprod,     0.0, 2.0);
    p.c_Q2        = clampDouble(p.c_Q2,        0.0, 2.0);
    p.d_alphaS    = 0.0;
    p.e_logpT     = clampDouble(p.e_logpT,     0.0, 5.0);
    p.f_satpT     = clampDouble(p.f_satpT,     0.0, 8.0);
    p.g_tworegion = clampDouble(p.g_tworegion, 0.0, 8.0);
    return p;
}

static BiasParams proposeCandidate(const BiasParams& best,
                                   int iter,
                                   mt19937& rng) {
    uniform_real_distribution<double> U(0.0, 1.0);
    normal_distribution<double> N(0.0, 1.0);

    BiasParams p = best;

    // Every 8th step: global exploration.
    // Other steps: local perturbation around the best accepted point.
    if (iter > 0 && iter % 8 == 0) {
        p.a_pT        = uniform_real_distribution<double>(0.0, 8.0)(rng);
        p.b_xprod     = uniform_real_distribution<double>(0.0, 2.0)(rng);
        p.c_Q2        = uniform_real_distribution<double>(0.0, 2.0)(rng);
        p.e_logpT     = uniform_real_distribution<double>(0.0, 5.0)(rng);
        p.f_satpT     = uniform_real_distribution<double>(0.0, 8.0)(rng);
        p.g_tworegion = uniform_real_distribution<double>(0.0, 8.0)(rng);
    } else {
        // Step sizes shrink slowly so the search first explores, then refines.
        const double scale = max(0.20, 1.0 - 0.015 * iter);

        p.a_pT        += 0.50 * scale * N(rng);
        p.b_xprod     += 0.15 * scale * N(rng);
        p.c_Q2        += 0.15 * scale * N(rng);
        p.e_logpT     += 0.30 * scale * N(rng);
        p.f_satpT     += 0.50 * scale * N(rng);
        p.g_tworegion += 0.50 * scale * N(rng);

        // Occasional directed push to harder pT if still not finding many high-pT events.
        if (U(rng) < 0.35) p.a_pT += 0.25 * scale;
        if (U(rng) < 0.35) p.g_tworegion += 0.35 * scale;
    }

    return clampParams(p);
}

static void writeSummaryHeader(ofstream& summary) {
    summary << "iteration,accepted,acceptable,bestSoFar,"
            << "a_pT,b_xprod,c_Q2,d_alphaS,e_logpT,f_satpT,g_tworegion,"
            << "sumManualW,sumManualW2,meanManualW,stdManualW,NeffManual,NeffManualOverN,"
            << "meanPTHat,maxPTHat,"
            << "fracPT10,fracPT20,fracPT50,fracPT100,"
            << "wFracPT10,wFracPT20,wFracPT50,wFracPT100,"
            << "objective\n";
}

static void writeSummaryRow(ofstream& summary,
                            int iter,
                            const BiasParams& p,
                            const BatchResult& r,
                            bool acceptable,
                            bool bestSoFar) {
    summary << iter << ","
            << r.accepted << ","
            << int(acceptable) << ","
            << int(bestSoFar) << ","
            << p.a_pT << ","
            << p.b_xprod << ","
            << p.c_Q2 << ","
            << p.d_alphaS << ","
            << p.e_logpT << ","
            << p.f_satpT << ","
            << p.g_tworegion << ","
            << r.sumW << ","
            << r.sumW2 << ","
            << r.meanW << ","
            << r.stdW << ","
            << r.neff << ","
            << r.neffRatio << ","
            << r.meanPTHat << ","
            << r.maxPTHat << ","
            << r.fracPT10 << ","
            << r.fracPT20 << ","
            << r.fracPT50 << ","
            << r.fracPT100 << ","
            << r.wFracPT10 << ","
            << r.wFracPT20 << ","
            << r.wFracPT50 << ","
            << r.wFracPT100 << ","
            << r.objective << "\n";
}

// ------------------------------------------------------------
// Main adaptive program
// ------------------------------------------------------------
int main(int argc, char* argv[]) {
    int nIterations = 10;
    int nEventsPerBatch = 100000;
    string summaryFile = "adaptive_pt50_balanced_manual_10_summary.csv";
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

    BiasParams current = clampParams(BiasParams{});
    BiasParams bestPars = current;
    BatchResult bestRes;
    double bestObjective = -1.0;

    ofstream summary(summaryFile);
    writeSummaryHeader(summary);

    cout << fixed << setprecision(6);
    cout << "Starting adaptive pTHat>50 search\n";
    cout << "Iterations: " << nIterations << "\n";
    cout << "Events per iteration: " << nEventsPerBatch << "\n";
    cout << "Minimum acceptable Neff/N: " << minNeffRatio << "\n";
    cout << "Summary file: " << summaryFile << "\n";
    cout << "Saving event CSVs: " << (saveEvents ? "yes" : "no") << "\n\n";

    for (int iter = 0; iter < nIterations; ++iter) {
        BiasParams candidate;

        if (iter == 0) {
            // Start from the best point found in your previous line scan.
            candidate = current;
        } else {
            candidate = proposeCandidate(bestPars, iter, rng);
        }

        cout << "Iteration " << iter << " candidate: "
             << "a=" << candidate.a_pT
             << ", b=" << candidate.b_xprod
             << ", c=" << candidate.c_Q2
             << ", e=" << candidate.e_logpT
             << ", f=" << candidate.f_satpT
             << ", g=" << candidate.g_tworegion << "\n";

        BatchResult res = runBatch(candidate, nEventsPerBatch, iter, eventPrefix, saveEvents, true);

        const bool acceptable = (res.accepted > 0 && res.neffRatio >= minNeffRatio);
        bool isBest = false;

        if (acceptable && res.objective > bestObjective) {
            bestObjective = res.objective;
            bestPars = candidate;
            bestRes = res;
            isBest = true;
        }

        writeSummaryRow(summary, iter, candidate, res, acceptable, isBest);
        summary.flush();

        cout << "  accepted=" << res.accepted
             << ", fracPT50=" << res.fracPT50
             << ", wFracPT50=" << res.wFracPT50
             << ", NeffManual/N=" << res.neffRatio
             << ", objective=" << res.objective
             << ", meanManualW=" << res.meanW
             << ", stdManualW=" << res.stdW
             << ", meanPTHat=" << res.meanPTHat
             << ", maxPTHat=" << res.maxPTHat;

        if (!acceptable) cout << "  [rejected: low Neff/N]";
        if (isBest) cout << "  [new best]";
        cout << "\n\n";
    }

    summary.close();

    cout << "Search finished.\n";
    cout << "Best balanced objective fracPT50*Neff/N = " << bestRes.objective << "\n";
    cout << "Best acceptable generated fracPT50 = " << bestRes.fracPT50 << "\n";
    cout << "Best manual-weighted/unbiased wFracPT50 = " << bestRes.wFracPT50 << "\n";
    cout << "Best manual-weight Neff/N            = " << bestRes.neffRatio << "\n";
    cout << "Best parameters:\n"
         << "  a_pT        = " << bestPars.a_pT << "\n"
         << "  b_xprod     = " << bestPars.b_xprod << "\n"
         << "  c_Q2        = " << bestPars.c_Q2 << "\n"
         << "  d_alphaS    = " << bestPars.d_alphaS << "  (fixed; not used in hook)\n"
         << "  e_logpT     = " << bestPars.e_logpT << "\n"
         << "  f_satpT     = " << bestPars.f_satpT << "\n"
         << "  g_tworegion = " << bestPars.g_tworegion << "\n";

    return 0;
}



