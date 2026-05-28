// Compile with:
// g++ test_pythia_multivar_bias.cc -o multivar_bias \
//   -Ipythia8317/include \
//   -Lpythia8317/lib \
//   -lpythia8 \
//   -Wl,-rpath,pythia8317/lib

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

    // Tunable parameters
    double a;        // pTHat power
    double b;        // x1*x2 power
    double c;        // Q2Fac power
    double pT0;
    double xprod0;
    double Q20;
    double maxBias;

    MultiVariableBias(
        double aIn = 1.5,
        double bIn = 0.5,
        double cIn = 0.0,
        double pT0In = 10.0,
        double xprod0In = 1e-4,
        double Q20In = 100.0,
        double maxBiasIn = 1000.0
    ) {
        a = aIn;
        b = bIn;
        c = cIn;
        pT0 = pT0In;
        xprod0 = xprod0In;
        Q20 = Q20In;
        maxBias = maxBiasIn;
    }

    bool canBiasSelection() override {
        return true;
    }

    double biasSelectionBy(const SigmaProcess* sigmaProcessPtr,
                           const PhaseSpace* phaseSpacePtr,
                           bool inEvent) override {

        int nFinal = sigmaProcessPtr->nFinal();

        // Only use pTHat for 2 -> 2 processes
        if (nFinal != 2) return 1.0;

        double pTHat = phaseSpacePtr->pTHat();
        double x1 = phaseSpacePtr->x1();
        double x2 = phaseSpacePtr->x2();
        double xprod = x1 * x2;

        double Q2Fac = sigmaProcessPtr->Q2Fac();

        double bias = 1.0;

        // pTHat bias
        if (pTHat > 0.0 && a != 0.0) {
            bias *= pow(pTHat / pT0, a);
        }

        // x1*x2 bias
        if (xprod > 0.0 && b != 0.0) {
            bias *= pow(xprod / xprod0, b);
        }

        // Q2Fac bias
        if (Q2Fac > 0.0 && c != 0.0) {
            bias *= pow(Q2Fac / Q20, c);
        }

        // Keep bias within safe range
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

    double a = 1.5;
    double b = 0.5;
    double c = 0.0;
    string outFile = "multivar_bias.csv";

    if (argc > 1) a = atof(argv[1]);
    if (argc > 2) b = atof(argv[2]);
    if (argc > 3) c = atof(argv[3]);
    if (argc > 4) outFile = argv[4];

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
            a,      // pTHat power
            b,      // x1*x2 power
            c,      // Q2Fac power
            10.0,   // pT0
            1e-4,   // xprod0
            100.0,  // Q20
            1000.0  // maxBias
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
        double alphaS = pythia.info.alphaS();
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
            << alphaS << ","
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
    cout << "Parameters: a = " << a
         << ", b = " << b
         << ", c = " << c << endl;

    return 0;
}
