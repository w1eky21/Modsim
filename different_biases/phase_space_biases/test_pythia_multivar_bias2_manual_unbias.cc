// Compile with:
// g++ -std=c++17 test_pythia_multivar_bias2_manual_unbias.cc -o multivar_bias2 \
//   -I../../pythia8317/include \
//   -L../../pythia8317/lib \
//   -lpythia8 \
//   -Wl,-rpath,../../pythia8317/lib
//
// Example run:
// ./multivar_bias2 4.6 0.36 0.36 0 0 0 100000
//
// Output:
// runs/apT_4.6_bx_0.36_cQ2_0.36_elog_0_fsat_0_gtwo_0/events.csv
// runs/apT_4.6_bx_0.36_cQ2_0.36_elog_0_fsat_0_gtwo_0/weights.txt
// runs/apT_4.6_bx_0.36_cQ2_0.36_elog_0_fsat_0_gtwo_0/summary.csv
// runs/apT_4.6_bx_0.36_cQ2_0.36_elog_0_fsat_0_gtwo_0/run.log

#include "Pythia8/Pythia.h"
#include "Pythia8/UserHooks.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

using namespace Pythia8;
using namespace std;
namespace fs = std::filesystem;

// ------------------------------------------------------------
// Custom multivariable phase-space bias
// ------------------------------------------------------------
class MultiVariableBias : public UserHooks {
public:
    // Exponents / strengths
    double a_pT;        // power-law pTHat bias
    double b_xprod;     // x1*x2 bias
    double c_Q2;        // Q2Fac bias
    double d_alphaS;    // currently unused
    double e_logpT;     // logarithmic pTHat bias
    double f_satpT;     // saturating pTHat bias
    double g_tworegion; // two-region pTHat bias

    // Reference scales
    double pT0;
    double xprod0;
    double Q20;
    double alphaS0;
    double pTswitch;
    double maxBias;

    // Debug counters
    long long nBiasCalls = 0;
    long long nBiasGreaterThanOne = 0;
    double minReturnedBias = 1e300;
    double maxReturnedBias = 0.0;
    double sumReturnedBias = 0.0;

    MultiVariableBias(
        double a_pTIn = 1.5,
        double b_xprodIn = 0.5,
        double c_Q2In = 0.0,
        double d_alphaSIn = 0.0,
        double e_logpTIn = 0.0,
        double f_satpTIn = 0.0,
        double g_tworegionIn = 0.0,
        double pT0In = 10.0,
        double xprod0In = 1e-4,
        double Q20In = 100.0,
        double alphaS0In = 0.12,
        double pTswitchIn = 50.0,
        double maxBiasIn = 1000.0
    ) :
        a_pT(a_pTIn),
        b_xprod(b_xprodIn),
        c_Q2(c_Q2In),
        d_alphaS(d_alphaSIn),
        e_logpT(e_logpTIn),
        f_satpT(f_satpTIn),
        g_tworegion(g_tworegionIn),
        pT0(pT0In),
        xprod0(xprod0In),
        Q20(Q20In),
        alphaS0(alphaS0In),
        pTswitch(pTswitchIn),
        maxBias(maxBiasIn)
    {}

    bool canBiasSelection() override {
        return true;
    }

    // Same bias formula used by Pythia and by the manual event unweighting below.
    double computeBias(double pTHat, double xprod, double Q2Fac) const {
        double bias = 1.0;

        // 1. Power-law pTHat bias: (pTHat / pT0)^a
        if (pTHat > 0.0 && a_pT != 0.0) {
            bias *= pow(pTHat / pT0, a_pT);
        }

        // 2. x1*x2 phase-space bias: (x1*x2 / xprod0)^b
        if (xprod > 0.0 && b_xprod != 0.0) {
            bias *= pow(xprod / xprod0, b_xprod);
        }

        // 3. Q2Fac bias: (Q2Fac / Q20)^c
        if (Q2Fac > 0.0 && c_Q2 != 0.0) {
            bias *= pow(Q2Fac / Q20, c_Q2);
        }

        // 4. Inverse alphaS bias currently not used, because SigmaProcess::alphaS()
        // is not available in your Pythia setup.
        // if (alphaS > 0.0 && d_alphaS != 0.0) {
        //     bias *= pow(alphaS0 / alphaS, d_alphaS);
        // }

        // 5. Logarithmic pTHat bias: [log(1 + pTHat/pT0)]^e
        if (pTHat > 0.0 && e_logpT != 0.0) {
            bias *= pow(log(1.0 + pTHat / pT0), e_logpT);
        }

        // 6. Saturating pTHat bias: [1 + pTHat/(pTHat+pT0)]^f
        if (pTHat > 0.0 && f_satpT != 0.0) {
            double sat = 1.0 + pTHat / (pTHat + pT0);
            bias *= pow(sat, f_satpT);
        }

        // 7. Two-region pTHat bias: no extra bias below pTswitch,
        // power-law above pTswitch.
        if (pTHat > pTswitch && g_tworegion != 0.0) {
            bias *= pow(pTHat / pTswitch, g_tworegion);
        }

        // Pythia's biasSelectionBy expects a bias/enhancement >= 1.
        bias = max(1.0, bias);
        bias = min(maxBias, bias);

        return bias;
    }

    double biasSelectionBy(const SigmaProcess* sigmaProcessPtr,
                           const PhaseSpace* phaseSpacePtr,
                           bool /*inEvent*/) override {
        int nFinal = sigmaProcessPtr->nFinal();
        if (nFinal != 2) return 1.0;

        double pTHat = phaseSpacePtr->pTHat();
        double x1 = phaseSpacePtr->x1();
        double x2 = phaseSpacePtr->x2();
        double xprod = x1 * x2;
        double Q2Fac = sigmaProcessPtr->Q2Fac();

        double bias = computeBias(pTHat, xprod, Q2Fac);

        nBiasCalls++;
        if (bias > 1.0) nBiasGreaterThanOne++;
        minReturnedBias = min(minReturnedBias, bias);
        maxReturnedBias = max(maxReturnedBias, bias);
        sumReturnedBias += bias;

        return bias;
    }
};

static string makeRunDirectory(double a_pT, double b_xprod, double c_Q2,
                               double e_logpT, double f_satpT, double g_tworegion) {
    ostringstream dirname;
    dirname << setprecision(6)
            << "runs/"
            << "apT_" << a_pT
            << "_bx_" << b_xprod
            << "_cQ2_" << c_Q2
            << "_elog_" << e_logpT
            << "_fsat_" << f_satpT
            << "_gtwo_" << g_tworegion;
    return dirname.str();
}

// ------------------------------------------------------------
// Main program
// ------------------------------------------------------------
int main(int argc, char* argv[]) {

    // Usage:
    // ./multivar_bias2 a_pT b_xprod c_Q2 e_logpT f_satpT g_tworegion [nEvents] [outdir]
    //
    // Example:
    // ./multivar_bias2 4.6 0.36 0.36 0 0 0 100000

    double a_pT = 1.5;
    double b_xprod = 0.5;
    double c_Q2 = 0.0;
    double d_alphaS = 0.0; // currently unused
    double e_logpT = 0.0;
    double f_satpT = 0.0;
    double g_tworegion = 0.0;
    int nEvents = 100000;

    if (argc > 1) a_pT = atof(argv[1]);
    if (argc > 2) b_xprod = atof(argv[2]);
    if (argc > 3) c_Q2 = atof(argv[3]);
    if (argc > 4) e_logpT = atof(argv[4]);
    if (argc > 5) f_satpT = atof(argv[5]);
    if (argc > 6) g_tworegion = atof(argv[6]);
    if (argc > 7) nEvents = atoi(argv[7]);

    string outdir = makeRunDirectory(a_pT, b_xprod, c_Q2, e_logpT, f_satpT, g_tworegion);
    if (argc > 8) outdir = argv[8];

    fs::create_directories(outdir);

    string eventsFile = outdir + "/events.csv";
    string weightsFile = outdir + "/weights.txt";
    string summaryFile = outdir + "/summary.csv";
    string logFile = outdir + "/run.log";

    ofstream log(logFile);
    log << fixed << setprecision(8);
    log << "a_pT=" << a_pT << "\n";
    log << "b_xprod=" << b_xprod << "\n";
    log << "c_Q2=" << c_Q2 << "\n";
    log << "d_alphaS=" << d_alphaS << "  # currently unused\n";
    log << "e_logpT=" << e_logpT << "\n";
    log << "f_satpT=" << f_satpT << "\n";
    log << "g_tworegion=" << g_tworegion << "\n";
    log << "nEvents=" << nEvents << "\n";
    log << "manualWeight = pythia.info.weight() / manualBias\n";
    log << "manualBias is recomputed from the accepted event using the same bias formula.\n";

    auto biasHook = make_shared<MultiVariableBias>(
        a_pT,
        b_xprod,
        c_Q2,
        d_alphaS,
        e_logpT,
        f_satpT,
        g_tworegion,
        10.0,    // pT0
        1e-4,    // xprod0
        100.0,   // Q20
        0.12,    // alphaS0, unused
        50.0,    // pTswitch
        1000.0   // maxBias
    );

    Pythia pythia;

    // Beam setup
    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13000.");

    // QCD hard processes
    pythia.readString("HardQCD:all = on");

    // Keep inclusive pTHat range
    pythia.readString("PhaseSpace:pTHatMin = 0.");

    // Reduce event printout noise.
    pythia.readString("Next:numberShowInfo = 0");
    pythia.readString("Next:numberShowProcess = 0");
    pythia.readString("Next:numberShowEvent = 0");

    // Attach custom UserHooks bias
    pythia.setUserHooksPtr(biasHook);

    if (!pythia.init()) {
        cerr << "Pythia initialization failed." << endl;
        return 1;
    }

    ofstream out(eventsFile);
    ofstream weightOut(weightsFile);

    // Fixed header: alphaS was removed, so columns no longer shift.
    // New columns:
    //   pythiaWeight = pythia.info.weight()
    //   manualBias   = bias formula evaluated on the accepted event
    //   manualWeight = pythiaWeight / manualBias
    out << "event,x1,x2,xprod,sHat,mHat,pTHat,pT2Hat,tHat,uHat,"
        << "Q2Fac,Q2Ren,alphaEM,id1,id2,code,"
        << "pythiaWeight,manualBias,manualWeight,"
        << "maxFinalParticlePT,nFinalParticles,nChargedFinal\n";

    weightOut << "event,pTHat,pythiaWeight,manualBias,manualWeight\n";

    int accepted = 0;
    int failed = 0;

    double sumPyW = 0.0, sumPyW2 = 0.0;
    double sumManW = 0.0, sumManW2 = 0.0;

    double sumPTHat = 0.0;
    double maxPTHat = 0.0;

    int nPT10 = 0;
    int nPT20 = 0;
    int nPT50 = 0;
    int nPT100 = 0;

    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {

        if (!pythia.next()) {
            failed++;
            continue;
        }

        double x1 = pythia.info.x1();
        double x2 = pythia.info.x2();
        double xprod = x1 * x2;

        double sHat = pythia.info.sHat();
        double mHat = sqrt(max(sHat, 0.0));
        double pTHat = pythia.info.pTHat();
        double pT2Hat = pTHat * pTHat;
        double tHat = pythia.info.tHat();
        double uHat = pythia.info.uHat();

        double Q2Fac = pythia.info.Q2Fac();
        double Q2Ren = pythia.info.Q2Ren();
        double alphaEM = pythia.info.alphaEM();

        int id1 = pythia.info.id1();
        int id2 = pythia.info.id2();
        int code = pythia.info.code();

        // Pythia's own event weight. Your debug test showed this is usually 1.
        double pythiaWeight = pythia.info.weight();

        // Manual unbiasing:
        // Since this hook biases the hard phase-space selection but pythia.info.weight()
        // does not include the compensating factor, we recompute the same bias for
        // the accepted event and divide by it.
        double manualBias = biasHook->computeBias(pTHat, xprod, Q2Fac);
        double manualWeight = pythiaWeight / manualBias;

        double maxFinalParticlePT = 0.0;
        int nFinalParticles = 0;
        int nChargedFinal = 0;

        for (int i = 0; i < pythia.event.size(); ++i) {
            if (pythia.event[i].isFinal()) {
                nFinalParticles++;

                double pT = pythia.event[i].pT();
                if (pT > maxFinalParticlePT) {
                    maxFinalParticlePT = pT;
                }

                if (pythia.event[i].isCharged()) {
                    nChargedFinal++;
                }
            }
        }

        out << accepted << ","
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
            << alphaEM << ","
            << id1 << ","
            << id2 << ","
            << code << ","
            << pythiaWeight << ","
            << manualBias << ","
            << manualWeight << ","
            << maxFinalParticlePT << ","
            << nFinalParticles << ","
            << nChargedFinal << "\n";

        if (accepted < 200) {
            weightOut << accepted << ","
                      << pTHat << ","
                      << pythiaWeight << ","
                      << manualBias << ","
                      << manualWeight << "\n";
        }

        accepted++;

        sumPyW += pythiaWeight;
        sumPyW2 += pythiaWeight * pythiaWeight;
        sumManW += manualWeight;
        sumManW2 += manualWeight * manualWeight;

        sumPTHat += pTHat;
        maxPTHat = max(maxPTHat, pTHat);

        if (pTHat > 10.0) nPT10++;
        if (pTHat > 20.0) nPT20++;
        if (pTHat > 50.0) nPT50++;
        if (pTHat > 100.0) nPT100++;
    }

    out.close();
    weightOut.close();

    double meanPTHat = (accepted > 0) ? sumPTHat / accepted : 0.0;

    double neffPythia = (sumPyW2 > 0.0) ? (sumPyW * sumPyW) / sumPyW2 : 0.0;
    double neffManual = (sumManW2 > 0.0) ? (sumManW * sumManW) / sumManW2 : 0.0;

    double neffPythiaOverN = (accepted > 0) ? neffPythia / accepted : 0.0;
    double neffManualOverN = (accepted > 0) ? neffManual / accepted : 0.0;

    double fracPT10 = (accepted > 0) ? double(nPT10) / accepted : 0.0;
    double fracPT20 = (accepted > 0) ? double(nPT20) / accepted : 0.0;
    double fracPT50 = (accepted > 0) ? double(nPT50) / accepted : 0.0;
    double fracPT100 = (accepted > 0) ? double(nPT100) / accepted : 0.0;

    ofstream summary(summaryFile);
    summary << "accepted,failed,a_pT,b_xprod,c_Q2,d_alphaS,e_logpT,f_satpT,g_tworegion,"
            << "sumPythiaW,sumPythiaW2,NeffPythia,NeffPythiaOverN,"
            << "sumManualW,sumManualW2,NeffManual,NeffManualOverN,"
            << "meanPTHat,maxPTHat,fracPT10,fracPT20,fracPT50,fracPT100,"
            << "biasCalls,biasGreaterThanOne,minReturnedBias,maxReturnedBias,meanReturnedBias\n";

    double meanReturnedBias = (biasHook->nBiasCalls > 0)
        ? biasHook->sumReturnedBias / biasHook->nBiasCalls
        : 0.0;

    summary << accepted << ","
            << failed << ","
            << a_pT << ","
            << b_xprod << ","
            << c_Q2 << ","
            << d_alphaS << ","
            << e_logpT << ","
            << f_satpT << ","
            << g_tworegion << ","
            << sumPyW << ","
            << sumPyW2 << ","
            << neffPythia << ","
            << neffPythiaOverN << ","
            << sumManW << ","
            << sumManW2 << ","
            << neffManual << ","
            << neffManualOverN << ","
            << meanPTHat << ","
            << maxPTHat << ","
            << fracPT10 << ","
            << fracPT20 << ","
            << fracPT50 << ","
            << fracPT100 << ","
            << biasHook->nBiasCalls << ","
            << biasHook->nBiasGreaterThanOne << ","
            << biasHook->minReturnedBias << ","
            << biasHook->maxReturnedBias << ","
            << meanReturnedBias << "\n";
    summary.close();

    log << "accepted=" << accepted << "\n";
    log << "failed=" << failed << "\n";
    log << "eventsFile=" << eventsFile << "\n";
    log << "weightsFile=" << weightsFile << "\n";
    log << "summaryFile=" << summaryFile << "\n";
    log << "NeffPythiaOverN=" << neffPythiaOverN << "\n";
    log << "NeffManualOverN=" << neffManualOverN << "\n";
    log << "biasSelectionBy calls=" << biasHook->nBiasCalls << "\n";
    log << "returned bias > 1 calls=" << biasHook->nBiasGreaterThanOne << "\n";
    log.close();

    pythia.stat();

    cout << fixed << setprecision(6);
    cout << "\nFinished multivariable bias run with manual unbiasing." << endl;
    cout << "Output directory: " << outdir << endl;
    cout << "Events file:      " << eventsFile << endl;
    cout << "Weights file:     " << weightsFile << endl;
    cout << "Summary file:     " << summaryFile << endl;
    cout << "accepted events   = " << accepted << endl;
    cout << "failed events     = " << failed << endl;
    cout << "a_pT              = " << a_pT << endl;
    cout << "b_xprod           = " << b_xprod << endl;
    cout << "c_Q2              = " << c_Q2 << endl;
    cout << "e_logpT           = " << e_logpT << endl;
    cout << "f_satpT           = " << f_satpT << endl;
    cout << "g_tworegion       = " << g_tworegion << endl;
    cout << "Neff/N pythia     = " << neffPythiaOverN << endl;
    cout << "Neff/N manual     = " << neffManualOverN << endl;
    cout << "frac pTHat > 50   = " << fracPT50 << endl;
    cout << "bias calls        = " << biasHook->nBiasCalls << endl;

    return 0;
}

