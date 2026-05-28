// venv: source pythia/bin/activate
// Compile with:
//
// g++ -O2 -std=c++11 test_pythia_build_in_ptcut.cc   -Ipythia8317/include   -Lpythia8317/lib   -l
//pythia8   -Wl,-rpath,pythia8317/lib   -o ptcut_eventlevel

#include "Pythia8/Pythia.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace Pythia8;
using namespace std;

int main() {

    Pythia pythia;

    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13000.");

    pythia.readString("HardQCD:all = on");

    double ptCut = 0.0;

    pythia.readString("PhaseSpace:pTHatMin = " + to_string(ptCut));

    int nEvents = 1000;

    pythia.init();

    string csvFilename = "qcd_highpt_events_ptcut_"
                       + to_string((int)ptCut)
                       + ".csv";

    string txtFilename = "pt_output_ptcut_"
                       + to_string((int)ptCut)
                       + ".txt";

    ofstream csvOut(csvFilename);
    ofstream txtOut(txtFilename);

    csvOut << "event,particle,id,status,px,py,pz,E,pT,eta,phi,weight,sigmaGen\n";
    txtOut << "# event particle_id name pT\n";

    int accepted = 0;

    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {

        if (!pythia.next()) continue;
        accepted++;

        double weight = pythia.info.weight();
        double sigmaGen = pythia.info.sigmaGen();

        cout << "Event " << iEvent << endl;

        for (int i = 0; i < pythia.event.size(); ++i) {

            if (!pythia.event[i].isFinal()) continue;

            double pt = pythia.event[i].pT();

            cout << "Particle "
                 << pythia.event[i].name()
                 << "  pT = "
                 << pt
                 << " GeV" << endl;

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
                   << pythia.event[i].phi() << ","
                   << weight << ","
                   << sigmaGen
                   << "\n";

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