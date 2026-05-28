// g++ test_pythia_naiveMC_all_phase_space.cc \
  -Ipythia8317/include \
  -Lpythia8317/lib \
  -lpythia8 \
  -Wl,-rpath,pythia8317/lib \
  -o phase_space

#include "Pythia8/Pythia.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace Pythia8;

int main(int argc, char* argv[]) {

    int nEvents = 100000;
    if (argc > 1) nEvents = atoi(argv[1]);

    std::string outName = "phase_space_naive.csv";
    if (argc > 2) outName = argv[2];

    Pythia pythia;

    // Naive QCD run: no pTHat filter, no bias.
    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13000.");
    pythia.readString("HardQCD:all = on");
    pythia.readString("PhaseSpace:pTHatMin = 0.");
    pythia.readString("Print:quiet = on");
    pythia.readString("Next:numberShowInfo = 0");
    pythia.readString("Next:numberShowProcess = 0");
    pythia.readString("Next:numberShowEvent = 0");

    pythia.init();

    std::ofstream out(outName);
    out << std::setprecision(12);

    out << "event,"
        << "x1,x2,xprod,"
        << "sHat,mHat,pTHat,pT2Hat,tHat,uHat,"
        << "Q2Fac,Q2Ren,alphaS,alphaEM,"
        << "id1,id2,code,weight,"
        << "maxFinalParticlePT,nFinalParticles,nChargedFinal\n";

    int accepted = 0;

    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {

        if (!pythia.next()) continue;

        double maxPT = 0.0;
        int nFinal = 0;
        int nChargedFinal = 0;

        for (int i = 0; i < pythia.event.size(); ++i) {
            const Particle& p = pythia.event[i];

            if (!p.isFinal()) continue;

            ++nFinal;
            if (p.isCharged()) ++nChargedFinal;

            double pT = p.pT();
            if (pT > maxPT) maxPT = pT;
        }

        double x1 = pythia.info.x1();
        double x2 = pythia.info.x2();

        out << accepted << ","
            << x1 << ","
            << x2 << ","
            << x1 * x2 << ","
            << pythia.info.sHat() << ","
            << pythia.info.mHat() << ","
            << pythia.info.pTHat() << ","
            << pythia.info.pT2Hat() << ","
            << pythia.info.tHat() << ","
            << pythia.info.uHat() << ","
            << pythia.info.Q2Fac() << ","
            << pythia.info.Q2Ren() << ","
            << pythia.info.alphaS() << ","
            << pythia.info.alphaEM() << ","
            << pythia.info.id1() << ","
            << pythia.info.id2() << ","
            << pythia.info.code() << ","
            << pythia.info.weight() << ","
            << maxPT << ","
            << nFinal << ","
            << nChargedFinal << "\n";

        ++accepted;
    }

    out.close();

    pythia.stat();

    std::cout << "Wrote " << accepted << " events to " << outName << "\n";

    return 0;
}
