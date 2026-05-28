//g++ test_pythia_fixed_bias.cc \
  -Ipythia8317/include \
  -Lpythia8317/lib \
  -lpythia8 \
  -Wl,-rpath,pythia8317/lib \
  -o fixed_bias

// run like this: ./fixed_bias 100000 2.0 15.0 0.0 fixed_bias_pow2.csv

#include "Pythia8/Pythia.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <string>

using namespace Pythia8;

int main(int argc, char* argv[]) {

    int nEvents = 100000;
    double biasPow = 3.0;
    double biasRef = 15.0;
    double pTHatMin = 0.0;
    double pTHatMax = 0.0;
    std::string outName = "phase_space_fixed_bias.csv";

    if (argc > 1) nEvents = atoi(argv[1]);
    if (argc > 2) biasPow = atof(argv[2]);
    if (argc > 3) biasRef = atof(argv[3]);
    if (argc > 4) pTHatMin = atof(argv[4]);
    if (argc > 5) outName = argv[5];

    Pythia pythia;

    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13000.");
    pythia.readString("HardQCD:all = on");

    pythia.settings.parm("PhaseSpace:pTHatMin", pTHatMin);
    if (pTHatMax > 0.0) {
        pythia.settings.parm("PhaseSpace:pTHatMax", pTHatMax);
    }

    // Fixed importance-sampling bias in hard-process pT.
    pythia.readString("PhaseSpace:bias2Selection = on");
    pythia.settings.parm("PhaseSpace:bias2SelectionPow", biasPow);
    pythia.settings.parm("PhaseSpace:bias2SelectionRef", biasRef);

    pythia.readString("Print:quiet = on");
    pythia.readString("Next:numberShowInfo = 0");
    pythia.readString("Next:numberShowProcess = 0");
    pythia.readString("Next:numberShowEvent = 0");

    if (!pythia.init()) {
        std::cerr << "Pythia init failed.\n";
        return 1;
    }

    std::ofstream out(outName);
    out << std::setprecision(12);

    out << "event,"
        << "biasPow,biasRef,pTHatMin,"
        << "x1,x2,xprod,"
        << "sHat,mHat,pTHat,pT2Hat,tHat,uHat,"
        << "Q2Fac,Q2Ren,alphaS,alphaEM,"
        << "id1,id2,code,weight,"
        << "maxFinalParticlePT,nFinalParticles,nChargedFinal\n";

    int accepted = 0;

    double sumW = 0.0;
    double sumW2 = 0.0;

    int n_pTHat_gt_10 = 0;
    int n_pTHat_gt_20 = 0;
    int n_pTHat_gt_50 = 0;
    int n_pTHat_gt_100 = 0;

    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {

        if (iEvent % 1000 == 0) {
    std::cout << "Processing event "
              << iEvent
              << " / "
              << nEvents
              << std::endl;
}

        if (!pythia.next()) continue;

        double maxPT = 0.0;
        int nFinal = 0;
        int nChargedFinal = 0;

        for (int i = 0; i < pythia.event.size(); ++i) {
            const Particle& p = pythia.event[i];

            if (!p.isFinal()) continue;

            ++nFinal;
            if (p.isCharged()) ++nChargedFinal;

            if (p.pT() > maxPT) maxPT = p.pT();
        }

        double x1 = pythia.info.x1();
        double x2 = pythia.info.x2();
        double pTHat = pythia.info.pTHat();
        double weight = pythia.info.weight();

        sumW += weight;
        sumW2 += weight * weight;

        if (pTHat > 10.0) ++n_pTHat_gt_10;
        if (pTHat > 20.0) ++n_pTHat_gt_20;
        if (pTHat > 50.0) ++n_pTHat_gt_50;
        if (pTHat > 100.0) ++n_pTHat_gt_100;

        out << accepted << ","
            << biasPow << ","
            << biasRef << ","
            << pTHatMin << ","
            << x1 << ","
            << x2 << ","
            << x1 * x2 << ","
            << pythia.info.sHat() << ","
            << pythia.info.mHat() << ","
            << pTHat << ","
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
            << weight << ","
            << maxPT << ","
            << nFinal << ","
            << nChargedFinal << "\n";

        ++accepted;
    }

    out.close();

    double nEff = 0.0;
    if (sumW2 > 0.0) nEff = (sumW * sumW) / sumW2;

    std::cout << "\nFixed bias run finished\n";
    std::cout << "Output file       = " << outName << "\n";
    std::cout << "Generated tries   = " << nEvents << "\n";
    std::cout << "Accepted events   = " << accepted << "\n";
    std::cout << "biasPow           = " << biasPow << "\n";
    std::cout << "biasRef           = " << biasRef << " GeV\n";
    std::cout << "pTHatMin          = " << pTHatMin << " GeV\n";
    std::cout << "sumW              = " << sumW << "\n";
    std::cout << "sumW2             = " << sumW2 << "\n";
    std::cout << "N_eff             = " << nEff << "\n";
    std::cout << "N_eff / N         = " << nEff / accepted << "\n";
    std::cout << "Frac pTHat > 10   = " << double(n_pTHat_gt_10) / accepted << "\n";
    std::cout << "Frac pTHat > 20   = " << double(n_pTHat_gt_20) / accepted << "\n";
    std::cout << "Frac pTHat > 50   = " << double(n_pTHat_gt_50) / accepted << "\n";
    std::cout << "Frac pTHat > 100  = " << double(n_pTHat_gt_100) / accepted << "\n\n";

    pythia.stat();

    return 0;
}
