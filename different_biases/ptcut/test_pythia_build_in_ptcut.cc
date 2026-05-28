#include "Pythia8/Pythia.h"
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cstdlib>

using namespace Pythia8;
using namespace std;

int main(int argc, char* argv[]) {

    int nEvents = 100000;
    double ptCut = 0.0;
    string outName = "qcd_highpt_events_ptcut_0.csv";

    if (argc > 1) nEvents = atoi(argv[1]);
    if (argc > 2) ptCut = atof(argv[2]);
    if (argc > 3) outName = argv[3];

    Pythia pythia;

    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13000.");
    pythia.readString("HardQCD:all = on");

    pythia.settings.parm("PhaseSpace:pTHatMin", ptCut);

    pythia.readString("Print:quiet = on");
    pythia.readString("Next:numberShowInfo = 0");
    pythia.readString("Next:numberShowProcess = 0");
    pythia.readString("Next:numberShowEvent = 0");

    if (!pythia.init()) {
        cerr << "Pythia init failed.\n";
        return 1;
    }

    ofstream out(outName);
    out << setprecision(12);

    out << "event,biasType,biasPow,biasRef,pTHatMin,"
        << "x1,x2,xprod,"
        << "sHat,mHat,pTHat,pT2Hat,tHat,uHat,"
        << "Q2Fac,Q2Ren,alphaS,alphaEM,"
        << "id1,id2,code,weight,"
        << "maxFinalParticlePT,nFinalParticles,nChargedFinal\n";

    int accepted = 0;

    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {

        if (iEvent % 1000 == 0) {
            cout << "Processing event " << iEvent << " / " << nEvents << endl;
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

        out << accepted << ","
            << "ptcut" << ","
            << 0.0 << ","
            << 0.0 << ","
            << ptCut << ","
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

    pythia.stat();

    cout << "Accepted events = " << accepted << endl;
    cout << "Output written to " << outName << endl;

    return 0;
}
