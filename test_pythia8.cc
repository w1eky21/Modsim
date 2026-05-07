// Compile with:
//
// g++ test_pythia8.cc -o test_pythia8 \
 -Ipythia8317/include \
 -Lpythia8317/lib \
 -lpythia8 \
 -Wl,-rpath,pythia8317/lib

#include "Pythia8/Pythia.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace Pythia8;
using namespace std;

int main() {

    Pythia pythia;

    // Proton-proton collisions at LHC energy
    pythia.readString("Beams:idA = 2212"); // proton
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13000."); // 13 TeV

    // Hard QCD jet production
    pythia.readString("HardQCD:all = on");

    // pT cutoff value
    double ptCut = 0.0;

    // Rare high-pT region
    pythia.readString("PhaseSpace:pTHatMin = " + to_string(ptCut));

    int nEvents = 1000;

    pythia.init();

    // Output filenames include pT cutoff
    string csvFilename = "qcd_highpt_events_ptcut_"
                       + to_string((int)ptCut)
                       + ".csv";

    string txtFilename = "pt_output_ptcut_"
                       + to_string((int)ptCut)
                       + ".txt";

    ofstream csvOut(csvFilename);
    ofstream txtOut(txtFilename);

    csvOut << "event,particle,id,status,px,py,pz,E,pT,eta,phi\n";
    txtOut << "# event particle_id name pT\n";

    int accepted = 0;

    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {

        if (!pythia.next()) continue;
        accepted++;

        cout << "Event " << iEvent << endl;

        for (int i = 0; i < pythia.event.size(); ++i) {

            if (!pythia.event[i].isFinal()) continue;

            double pt = pythia.event[i].pT();

            // Print pT information to terminal
            cout << "Particle "
                 << pythia.event[i].name()
                 << "  pT = "
                 << pt
                 << " GeV" << endl;

            // Save full particle information to CSV
            csvOut << iEvent << ","
                   << i << ","
                   << pythia.event[i].id() << ","
                   << pythia.event[i].status() << ","
                   << pythia.event[i].px() << ","
                   << pythia.event[i].py() << ","
                   << pythia.event[i].pz() << ","
                   << pythia.event[i].e() << ","
                   << pt << ","
                   << pythia.event[i].eta() << ","
                   << pythia.event[i].phi()
                   << "\n";

            // Save simpler pT output to TXT
            txtOut << iEvent << " "
                   << pythia.event[i].id() << " "
                   << pythia.event[i].name() << " "
                   << pt << "\n";
        }
    }

    csvOut.close();
    txtOut.close();

    pythia.stat();

    cout << "Generated " << accepted << " accepted events." << endl;
    cout << "CSV output written to " << csvFilename << endl;
    cout << "pT output written to " << txtFilename << endl;

    return 0;
}