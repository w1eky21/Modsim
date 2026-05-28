// Compile with:
//
// g++ test_pythia8_static_is.cc -o test_pythia8_static_is \
//   -Ipythia8317/include \
//   -Lpythia8317/lib \
//   -lpythia8 \
//   -Wl,-rpath,pythia8317/lib

#include "Pythia8/Pythia.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

using namespace Pythia8;
using namespace std;

struct PtHatBin {
    double minPt;
    double maxPt;
    int nEvents;
};

struct EventInfo {
    int binIndex;
    int eventNumber;
    double pTHat;
    double hardestFinalPt;
    double sigmaGen;
    double eventWeight;
};

int findHistBin(double x, const vector<double>& edges) {
    for (int i = 0; i < (int)edges.size() - 1; ++i) {
        if (x >= edges[i] && x < edges[i + 1]) return i;
    }
    return -1;
}

int main() {

    // Static importance-sampling bins in partonic pTHat.
    // Each bin is generated separately and later combined with its own cross-section weight.
    vector<PtHatBin> ptHatBins = {
        {0.0,    100.0,  10000},
        {100.0,  250.0,  10000},
        {250.0,  500.0,  10000},
        {500.0, 1000.0,  10000},
        {1000.0, 2000.0, 10000}
    };

    // Histogram edges for the observable: hardest final-state particle pT.
    vector<double> histEdges = {0, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000};
    vector<double> histSumW(histEdges.size() - 1, 0.0);
    vector<double> histSumW2(histEdges.size() - 1, 0.0);

    ofstream summaryOut("static_importance_summary.csv");
    summaryOut << "binIndex,pTHatMin,pTHatMax,nRequested,nAccepted,sigmaGen,eventWeight\n";

    ofstream eventOut("static_importance_events.csv");
    eventOut << "binIndex,event,pTHat,hardestFinalPt,sigmaGen,eventWeight\n";

    vector<EventInfo> allEvents;

    for (int b = 0; b < (int)ptHatBins.size(); ++b) {

        PtHatBin bin = ptHatBins[b];

        Pythia pythia;

        pythia.readString("Beams:idA = 2212");
        pythia.readString("Beams:idB = 2212");
        pythia.readString("Beams:eCM = 13000.");
        pythia.readString("HardQCD:all = on");

        pythia.readString("PhaseSpace:pTHatMin = " + to_string(bin.minPt));
        pythia.readString("PhaseSpace:pTHatMax = " + to_string(bin.maxPt));

        pythia.init();

        vector<EventInfo> eventsInThisBin;
        int accepted = 0;

        for (int iEvent = 0; iEvent < bin.nEvents; ++iEvent) {

            if (!pythia.next()) continue;
            accepted++;

            double hardestPt = 0.0;

            for (int i = 0; i < pythia.event.size(); ++i) {
                if (!pythia.event[i].isFinal()) continue;

                double pt = pythia.event[i].pT();
                if (pt > hardestPt) hardestPt = pt;
            }

            EventInfo ev;
            ev.binIndex = b;
            ev.eventNumber = iEvent;
            ev.pTHat = pythia.info.pTHat();
            ev.hardestFinalPt = hardestPt;
            ev.sigmaGen = 0.0;      // Filled after the run, when sigmaGen is final.
            ev.eventWeight = 0.0;   // Filled after the run.

            eventsInThisBin.push_back(ev);
        }

        double sigmaGen = pythia.info.sigmaGen();
        double eventWeight = 0.0;
        if (accepted > 0) eventWeight = sigmaGen / accepted;

        summaryOut << b << ","
                   << bin.minPt << ","
                   << bin.maxPt << ","
                   << bin.nEvents << ","
                   << accepted << ","
                   << sigmaGen << ","
                   << eventWeight << "\n";

        for (int k = 0; k < (int)eventsInThisBin.size(); ++k) {
            eventsInThisBin[k].sigmaGen = sigmaGen;
            eventsInThisBin[k].eventWeight = eventWeight;

            int hbin = findHistBin(eventsInThisBin[k].hardestFinalPt, histEdges);
            if (hbin >= 0) {
                histSumW[hbin] += eventWeight;
                histSumW2[hbin] += eventWeight * eventWeight;
            }

            eventOut << eventsInThisBin[k].binIndex << ","
                     << eventsInThisBin[k].eventNumber << ","
                     << eventsInThisBin[k].pTHat << ","
                     << eventsInThisBin[k].hardestFinalPt << ","
                     << eventsInThisBin[k].sigmaGen << ","
                     << eventsInThisBin[k].eventWeight << "\n";

            allEvents.push_back(eventsInThisBin[k]);
        }

        cout << "Finished pTHat bin " << bin.minPt << " - " << bin.maxPt << " GeV" << endl;
        cout << "  accepted events = " << accepted << endl;
        cout << "  sigmaGen        = " << sigmaGen << endl;
        cout << "  event weight    = " << eventWeight << endl;

        pythia.stat();
    }

    ofstream histOut("static_importance_histogram.csv");
    histOut << "ptLow,ptHigh,binWidth,sumW,sumW2,dSigma_dpT,relativeError\n";

    for (int i = 0; i < (int)histSumW.size(); ++i) {
        double ptLow = histEdges[i];
        double ptHigh = histEdges[i + 1];
        double width = ptHigh - ptLow;

        double dSigmaDpT = histSumW[i] / width;
        double relErr = 0.0;
        if (histSumW[i] > 0.0) {
            relErr = sqrt(histSumW2[i]) / histSumW[i];
        }

        histOut << ptLow << ","
                << ptHigh << ","
                << width << ","
                << histSumW[i] << ","
                << histSumW2[i] << ","
                << dSigmaDpT << ","
                << relErr << "\n";
    }

    summaryOut.close();
    eventOut.close();
    histOut.close();

    cout << "\nDone. Wrote files:\n";
    cout << "  static_importance_summary.csv\n";
    cout << "  static_importance_events.csv\n";
    cout << "  static_importance_histogram.csv\n";

    return 0;
}
