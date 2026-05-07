// Compile with:
//
// g++ print_pt.cc -o print_pt \
// -Ipythia8317/include \
// -Lpythia8317/lib \
// -lpythia8 \
// -Wl,-rpath,pythia8317/lib

#include "Pythia8/Pythia.h"
#include <iostream>
#include <fstream>

using namespace Pythia8;
using namespace std;

int main() {

    Pythia pythia;

    // Proton-proton collisions
    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");

    // LHC energy
    pythia.readString("Beams:eCM = 13000.");

    // Hard QCD processes
    pythia.readString("HardQCD:all = on");

    // Generate rare high-pT events
    pythia.readString("PhaseSpace:pTHatMin = 500.");

    int nEvents = 1000;

    pythia.init();

    ofstream out("pt_output.txt");

    out << "# event particle_id name pT\n";

    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {

        if (!pythia.next()) continue;

        cout << "Event " << iEvent << endl;

        for (int i = 0; i < pythia.event.size(); ++i) {

            if (!pythia.event[i].isFinal()) continue;

            double pt = pythia.event[i].pT();

            // print to terminal
            cout << "Particle "
                 << pythia.event[i].name()
                 << "  pT = "
                 << pt
                 << " GeV" << endl;

            // save to file
            out << iEvent << " "
                << pythia.event[i].id() << " "
                << pythia.event[i].name() << " "
                << pt << "\n";
        }
    }

    pythia.stat();

    out.close();

    cout << "\nDone. pT values saved to pt_output.txt\n";

    return 0;
}
