// Compile with:
// g++ -std=c++17 test_pythia_multivar_bias3.cc -o multivar_bias3 \
   -I../../pythia8317/include \
   -L../../pythia8317/lib \
   -lpythia8 \
   -Wl,-rpath,../../pythia8317/lib
//
// Example run:
// ./multivar_bias2 4.6 0.36 0.36 0 0 0 0
//
// This creates:
// runs/apT_4.6_bx_0.36_cQ2_0.36_dAS_0_elog_0_fsat_0_gtwo_0/
//   events.csv
//   run.log
//   weights.txt
//   summary.csv

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
// Helpers
// ------------------------------------------------------------
string makeRunDirectoryName(double a_pT,
                            double b_xprod,
                            double c_Q2,
                            double d_alphaS,
                            double e_logpT,
                            double f_satpT,
                            double g_tworegion) {
    ostringstream dirname;
    dirname << "runs/"
            << "apT_" << a_pT
            << "_bx_" << b_xprod
            << "_cQ2_" << c_Q2
            << "_dAS_" << d_alphaS
            << "_elog_" << e_logpT
            << "_fsat_" << f_satpT
            << "_gtwo_" << g_tworegion;
    return dirname.str();
}

// ------------------------------------------------------------
// Custom multivariable phase-space bias
// ------------------------------------------------------------
class MultiVariableBias : public UserHooks {
public:
    // Exponents / strengths
    double a_pT;        // power-law pTHat bias
    double b_xprod;     // x1*x2 bias
    double c_Q2;        // Q2Fac bias
    double d_alphaS;    // currently logged but not used; see note below
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

    MultiVariableBias(double a_pTIn = 1.5,
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
                      double maxBiasIn = 1000.0)
        : a_pT(a_pTIn),
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
          maxBias(maxBiasIn) {}

    bool canBiasSelection() override { return true; }

    double biasSelectionBy(const SigmaProcess* sigmaProcessPtr,
                           const PhaseSpace* phaseSpacePtr,
                           bool inEvent) override {
        (void)inEvent;

        int nFinal = sigmaProcessPtr->nFinal();
        if (nFinal != 2) return 1.0;

        double pTHat = phaseSpacePtr->pTHat();
        double x1 = phaseSpacePtr->x1();
        double x2 = phaseSpacePtr->x2();
        double xprod = x1 * x2;
        double Q2Fac = sigmaProcessPtr->Q2Fac();

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

        // 4. alphaS bias is intentionally disabled here.
        // In this Pythia version, SigmaProcess::alphaS() is not available.
        // d_alphaS is still parsed and logged so your scan format remains compatible.

        // 5. Logarithmic pTHat bias: [log(1 + pTHat/pT0)]^e
        if (pTHat > 0.0 && e_logpT != 0.0) {
            bias *= pow(log(1.0 + pTHat / pT0), e_logpT);
        }

        // 6. Saturating pTHat bias: [1 + pTHat/(pTHat+pT0)]^f
        if (pTHat > 0.0 && f_satpT != 0.0) {
            double sat = 1.0 + pTHat / (pTHat + pT0);
            bias *= pow(sat, f_satpT);
        }

        // 7. Two-region pTHat bias: no bias below pTswitch, power-law above pTswitch
        if (pTHat > pTswitch && g_tworegion != 0.0) {
            bias *= pow(pTHat / pTswitch, g_tworegion);
        }

        bias = max(1.0, bias);
        bias = min(maxBias, bias);
        return bias;
    }
};

// ------------------------------------------------------------
// Main program
// ------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Usage:
    // ./multivar_bias2 a_pT b_xprod c_Q2 d_alphaS e_logpT f_satpT g_tworegion [nEvents]
    //
    // Example:
    // ./multivar_bias2 4.6 0.36 0.36 0 0 0 0

    double a_pT = 1.5;
    double b_xprod = 0.5;
    double c_Q2 = 0.0;
    double d_alphaS = 0.0; // parsed/logged, but not used in the bias
    double e_logpT = 0.0;
    double f_satpT = 0.0;
    double g_tworegion = 0.0;
    int nEvents = 100000;

    if (argc > 1) a_pT = atof(argv[1]);
    if (argc > 2) b_xprod = atof(argv[2]);
    if (argc > 3) c_Q2 = atof(argv[3]);
    if (argc > 4) d_alphaS = atof(argv[4]);
    if (argc > 5) e_logpT = atof(argv[5]);
    if (argc > 6) f_satpT = atof(argv[6]);
    if (argc > 7) g_tworegion = atof(argv[7]);
    if (argc > 8) nEvents = atoi(argv[8]);

    string outdir = makeRunDirectoryName(a_pT, b_xprod, c_Q2, d_alphaS,
                                         e_logpT, f_satpT, g_tworegion);
    fs::create_directories(outdir);

    string eventsFile = outdir + "/events.csv";
    string logFile = outdir + "/run.log";
    string weightsFile = outdir + "/weights.txt";
    string summaryFile = outdir + "/summary.csv";

    ofstream log(logFile);
    log << "a_pT = " << a_pT << "\n";
    log << "b_xprod = " << b_xprod << "\n";
    log << "c_Q2 = " << c_Q2 << "\n";
    log << "d_alphaS = " << d_alphaS << "  # parsed/logged, but not used\n";
    log << "e_logpT = " << e_logpT << "\n";
    log << "f_satpT = " << f_satpT << "\n";
    log << "g_tworegion = " << g_tworegion << "\n";
    log << "nEvents = " << nEvents << "\n";
    log << "eventsFile = " << eventsFile << "\n";
    log << "weightsFile = " << weightsFile << "\n";
    log << "summaryFile = " << summaryFile << "\n";

    Pythia pythia;

    // Beam setup
    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13000.");

    // QCD hard processes
    pythia.readString("HardQCD:all = on");

    // Keep inclusive pTHat range
    pythia.readString("PhaseSpace:pTHatMin = 0.");

    // Attach custom UserHooks bias
    shared_ptr<UserHooks> userHooks = make_shared<MultiVariableBias>(
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
        0.12,    // alphaS0, unused while alphaS bias is disabled
        50.0,    // pTswitch
        1000.0   // maxBias
    );

    pythia.setUserHooksPtr(userHooks);

    if (!pythia.init()) {
        cerr << "Pythia initialization failed." << endl;
        log << "Pythia initialization failed.\n";
        return 1;
    }

    ofstream out(eventsFile);
    ofstream weightlog(weightsFile);

    if (!out) {
        cerr << "Could not open output CSV: " << eventsFile << endl;
        return 1;
    }
    if (!weightlog) {
        cerr << "Could not open weight log: " << weightsFile << endl;
        return 1;
    }

    // Fixed header: alphaS removed because no alphaS value is written.
    // This prevents all columns after Q2Ren from being shifted.
    out << "event,x1,x2,xprod,sHat,mHat,pTHat,pT2Hat,tHat,uHat,"
        << "Q2Fac,Q2Ren,alphaEM,id1,id2,code,weight,"
        << "maxFinalParticlePT,nFinalParticles,nChargedFinal\n";

    weightlog << "event,pTHat,weight\n";

    int accepted = 0;
    int printedWeights = 0;

    double sumW = 0.0;
    double sumW2 = 0.0;
    double sumPTHat = 0.0;
    double maxPTHat = 0.0;
    int countPT10 = 0;
    int countPT20 = 0;
    int countPT50 = 0;
    int countPT100 = 0;

    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
        if (!pythia.next()) continue;

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

        // This should include Pythia's compensating bias weight.
        double weight = pythia.info.weight();

        if (printedWeights < 100) {
            weightlog << accepted << "," << pTHat << "," << weight << "\n";
            cout << "event " << accepted
                 << "  pTHat = " << pTHat
                 << "  weight = " << weight << endl;
            printedWeights++;
        }

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
            << weight << ","
            << maxFinalParticlePT << ","
            << nFinalParticles << ","
            << nChargedFinal << "\n";

        accepted++;
        sumW += weight;
        sumW2 += weight * weight;
        sumPTHat += pTHat;
        maxPTHat = max(maxPTHat, pTHat);
        if (pTHat > 10.0) countPT10++;
        if (pTHat > 20.0) countPT20++;
        if (pTHat > 50.0) countPT50++;
        if (pTHat > 100.0) countPT100++;
    }

    out.close();
    weightlog.close();

    double meanW = accepted > 0 ? sumW / accepted : 0.0;
    double meanW2 = accepted > 0 ? sumW2 / accepted : 0.0;
    double varW = max(0.0, meanW2 - meanW * meanW);
    double stdW = sqrt(varW);
    double neff = sumW2 > 0.0 ? (sumW * sumW) / sumW2 : 0.0;
    double neffOverN = accepted > 0 ? neff / accepted : 0.0;
    double meanPTHat = accepted > 0 ? sumPTHat / accepted : 0.0;

    ofstream summary(summaryFile);
    summary << "accepted,a_pT,b_xprod,c_Q2,d_alphaS,e_logpT,f_satpT,g_tworegion,"
            << "sumW,sumW2,meanW,stdW,Neff,NeffOverN,meanPTHat,maxPTHat,"
            << "fracPT10,fracPT20,fracPT50,fracPT100\n";
    summary << accepted << ","
            << a_pT << ","
            << b_xprod << ","
            << c_Q2 << ","
            << d_alphaS << ","
            << e_logpT << ","
            << f_satpT << ","
            << g_tworegion << ","
            << sumW << ","
            << sumW2 << ","
            << meanW << ","
            << stdW << ","
            << neff << ","
            << neffOverN << ","
            << meanPTHat << ","
            << maxPTHat << ","
            << (accepted > 0 ? double(countPT10) / accepted : 0.0) << ","
            << (accepted > 0 ? double(countPT20) / accepted : 0.0) << ","
            << (accepted > 0 ? double(countPT50) / accepted : 0.0) << ","
            << (accepted > 0 ? double(countPT100) / accepted : 0.0) << "\n";
    summary.close();

    pythia.stat();

    log << "accepted = " << accepted << "\n";
    log << "sumW = " << sumW << "\n";
    log << "sumW2 = " << sumW2 << "\n";
    log << "meanW = " << meanW << "\n";
    log << "stdW = " << stdW << "\n";
    log << "Neff = " << neff << "\n";
    log << "NeffOverN = " << neffOverN << "\n";
    log << "meanPTHat = " << meanPTHat << "\n";
    log << "maxPTHat = " << maxPTHat << "\n";
    log.close();

    cout << "\nFinished multivariable bias run." << endl;
    cout << "Output directory: " << outdir << endl;
    cout << "Events CSV:       " << eventsFile << endl;
    cout << "Run log:          " << logFile << endl;
    cout << "Weight check:     " << weightsFile << endl;
    cout << "Summary CSV:      " << summaryFile << endl;
    cout << "accepted    = " << accepted << endl;
    cout << "Neff/N      = " << neffOverN << endl;
    cout << "frac pT>50  = " << (accepted > 0 ? double(countPT50) / accepted : 0.0) << endl;

    return 0;
}

