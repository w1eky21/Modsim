// Compile with:
// g++ test_pythia_multivar_bias.cc -o multivar_bias \
  -I../../pythia8317/include \
  -L../../pythia8317/lib \
  -lpythia8 \
  -Wl,-rpath,../../pythia8317/lib

#include "Pythia8/Pythia.h"
#include "Pythia8/UserHooks.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <memory>
#include <algorithm>

using namespace Pythia8;
using namespace std;

// ------------------------------------------------------------
// Custom multivariable phase-space bias
// ------------------------------------------------------------
class MultiVariableBias : public UserHooks {
public:

    // Exponents / strengths
    double a_pT;       // power-law pTHat bias
    double b_xprod;    // x1*x2 bias
    double c_Q2;       // Q2Fac bias
    double d_alphaS;   // inverse alphaS bias
    double e_logpT;    // logarithmic pTHat bias
    double f_satpT;    // saturating pTHat bias
    double g_tworegion; // two-region pTHat bias

    // Reference scales
    double pT0;
    double xprod0;
    double Q20;
    double alphaS0;
    double pTswitch;
    double maxBias;

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
    ) {
        a_pT = a_pTIn;
        b_xprod = b_xprodIn;
        c_Q2 = c_Q2In;
        d_alphaS = d_alphaSIn;
        e_logpT = e_logpTIn;
        f_satpT = f_satpTIn;
        g_tworegion = g_tworegionIn;

        pT0 = pT0In;
        xprod0 = xprod0In;
        Q20 = Q20In;
        alphaS0 = alphaS0In;
        pTswitch = pTswitchIn;
        maxBias = maxBiasIn;
    }

    bool canBiasSelection() override {
        return true;
    }

    double biasSelectionBy(const SigmaProcess* sigmaProcessPtr,
                           const PhaseSpace* phaseSpacePtr,
                           bool inEvent) override {

        int nFinal = sigmaProcessPtr->nFinal();
        if (nFinal != 2) return 1.0;

        double pTHat = phaseSpacePtr->pTHat();
        double x1 = phaseSpacePtr->x1();
        double x2 = phaseSpacePtr->x2();
        double xprod = x1 * x2;

        double Q2Fac = sigmaProcessPtr->Q2Fac();
        //double alphaS = sigmaProcessPtr->alphaS();

        double bias = 1.0;

        // 1. Power-law pTHat bias:
        // (pTHat / pT0)^a
        if (pTHat > 0.0 && a_pT != 0.0) {
            bias *= pow(pTHat / pT0, a_pT);
        }

        // 2. x1*x2 phase-space bias:
        // (x1*x2 / xprod0)^b
        if (xprod > 0.0 && b_xprod != 0.0) {
            bias *= pow(xprod / xprod0, b_xprod);
        }

        // 3. Q2Fac bias:
        // (Q2Fac / Q20)^c
        if (Q2Fac > 0.0 && c_Q2 != 0.0) {
            bias *= pow(Q2Fac / Q20, c_Q2);
        }

        // 4. Inverse alphaS bias:
        // (alphaS0 / alphaS)^d
        //if (alphaS > 0.0 && d_alphaS != 0.0) {
        //    bias *= pow(alphaS0 / alphaS, d_alphaS);
        //}

        // 5. Logarithmic pTHat bias:
        // [log(1 + pTHat/pT0)]^e
        if (pTHat > 0.0 && e_logpT != 0.0) {
            bias *= pow(log(1.0 + pTHat / pT0), e_logpT);
        }

        // 6. Saturating pTHat bias:
        // [1 + pTHat/(pTHat+pT0)]^f
        if (pTHat > 0.0 && f_satpT != 0.0) {
            double sat = 1.0 + pTHat / (pTHat + pT0);
            bias *= pow(sat, f_satpT);
        }

        // 7. Two-region pTHat bias:
        // no bias below pTswitch, power-law above pTswitch
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
    // ./multivar_bias a b c output.csv
    //
    // Example:
    // ./multivar_bias 1.5 0.5 0.0 multivar_a15_b05_c00.csv

    double a_pT = 1.5;
double b_xprod = 0.5;
double c_Q2 = 0.0;
//double d_alphaS = 0.0;
double e_logpT = 0.0;
double f_satpT = 0.0;
double g_tworegion = 0.0;

string outFile = "multivar_bias.csv";

if (argc > 1) a_pT = atof(argv[1]);
if (argc > 2) b_xprod = atof(argv[2]);
if (argc > 3) c_Q2 = atof(argv[3]);
//if (argc > 4) d_alphaS = atof(argv[4]); //moet argv nummer aanpassen als je wil gebruiken
if (argc > 4) e_logpT = atof(argv[4]);
if (argc > 5) f_satpT = atof(argv[5]);
if (argc > 6) g_tworegion = atof(argv[6]);
if (argc > 7) outFile = argv[7];
    int nEvents = 100000;

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
    shared_ptr<UserHooks> userHooks =
    make_shared<MultiVariableBias>(
        a_pT,
        b_xprod,
        c_Q2,
       // d_alphaS,
        e_logpT,
        f_satpT,
        g_tworegion,
        10.0,    // pT0
        1e-4,    // xprod0
        100.0,   // Q20
       // 0.12,    // alphaS0
        50.0,    // pTswitch
        1000.0   // maxBias
    );

    pythia.setUserHooksPtr(userHooks);

    if (!pythia.init()) {
        cerr << "Pythia initialization failed." << endl;
        return 1;
    }

    ofstream out(outFile);

    out << "event,x1,x2,xprod,sHat,mHat,pTHat,pT2Hat,tHat,uHat,"
        << "Q2Fac,Q2Ren,alphaS,alphaEM,id1,id2,code,weight,"
        << "maxFinalParticlePT,nFinalParticles,nChargedFinal\n";

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
       // double alphaS = pythia.info.alphaS();
        double alphaEM = pythia.info.alphaEM();

        int id1 = pythia.info.id1();
        int id2 = pythia.info.id2();
        int code = pythia.info.code();

        // This includes the compensating bias weight
        double weight = pythia.info.weight();

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

        out << iEvent << ","
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
           // << alphaS << ","
            << alphaEM << ","
            << id1 << ","
            << id2 << ","
            << code << ","
            << weight << ","
            << maxFinalParticlePT << ","
            << nFinalParticles << ","
            << nChargedFinal << "\n";
    }

    out.close();

    pythia.stat();

    cout << "Finished multivariable bias run." << endl;
    cout << "Output file: " << outFile << endl;
    cout << "a_pT       = " << a_pT << endl;
    cout << "b_xprod    = " << b_xprod << endl;
    cout << "c_Q2       = " << c_Q2 << endl;
    //cout << "d_alphaS   = " << d_alphaS << endl;
    cout << "e_logpT    = " << e_logpT << endl;
    cout << "f_satpT    = " << f_satpT << endl;
    cout << "g_tworegion= " << g_tworegion << endl;

    return 0;
}

