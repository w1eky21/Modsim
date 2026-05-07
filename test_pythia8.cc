//g++ test_pythia8.cc -o test_pythia8_pt \
-Ipythia8317/include \
-Lpythia8317/lib \
-lpythia8 \
-Wl,-rpath,pythia8317/lib

#include "Pythia8/Pythia.h"
#include <iostream>
#include <fstream>

using namespace Pythia8;
using namespace std;

int main() {

    Pythia pythia;

    // Proton-proton collisions at LHC energy
    pythia.readString("Beams:idA = 2212"); // 2212 is proton
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13000."); // 13 TeV

    // Hard QCD jet production
    pythia.readString("HardQCD:all = on"); // alle qcd shit

    // Rare high-pT region
    pythia.readString("PhaseSpace:pTHatMin = 500."); // Alleen >500 GeV HS events are created

    int nEvents = 1000;

    pythia.init();

    ofstream out("qcd_highpt_events.csv");
    out << "event,particle,id,status,px,py,pz,E,pT,eta,phi\n";

    int accepted = 0;

    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {

        if (!pythia.next()) continue;
        accepted++;

        for (int i = 0; i < pythia.event.size(); ++i) {

            if (!pythia.event[i].isFinal()) continue;

            out << iEvent << ","
                << i << ","
                << pythia.event[i].id() << ","
                << pythia.event[i].status() << ","
                << pythia.event[i].px() << ","
                << pythia.event[i].py() << ","
                << pythia.event[i].pz() << ","
                << pythia.event[i].e() << ","
                << pythia.event[i].pT() << ","
                << pythia.event[i].eta() << ","
                << pythia.event[i].phi()
                << "\n";
        }
    }

    out.close();

    pythia.stat();

    cout << "Generated " << accepted << " accepted events." << endl;
    cout << "Output written to qcd_highpt_events.csv" << endl;

    return 0;
}

