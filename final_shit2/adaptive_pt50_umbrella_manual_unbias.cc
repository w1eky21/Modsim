// Simple adaptive umbrella-sampling bias search for rare high-pT QCD events in Pythia8
//
// This version keeps the same command-line arguments and the same summary quantities
// as the previous manual-unbias code, but replaces the analytic multi-parameter bias
// by an adaptive umbrella bias in the reaction coordinate pTHat.
//
// Umbrella idea:
//   - pTHat is divided into bins, uniform in log(1+pTHat).
//   - Each bin has an umbrella bias factor B_i.
//   - Pythia is biased with B_i for the current pTHat bin.
//   - Accepted events are manually unbiased with
//         manualWeight = pythia.info.weight() / B_i.
//   - After each iteration, the observed pTHat histogram is compared to a flat
//     target histogram and the umbrella factors are updated smoothly.
//
// Compile:
//   g++ adaptive_pt50_umbrella_manual_unbias.cc -o umbrella \
//      -I../pythia8317/include -L../pythia8317/lib -lpythia8 \
//      -Wl,-rpath,../pythia8317/lib -std=c++17
//
// Run, same arguments as before:
//   ./umbrella 10 100000 umbrella_summary.csv umbrella_events 0 0.05
//
// Arguments:
//   argv[1] = number of adaptive iterations       default 10
//   argv[2] = generated events per iteration      default 100000
//   argv[3] = summary CSV filename                default umbrella_summary.csv
//   argv[4] = event output prefix                 default umbrella_events
//   argv[5] = save per-event CSVs? 0/1            default 0
//   argv[6] = minimum acceptable Neff/N           default 0.05

#include "Pythia8/Pythia.h"
#include "Pythia8/UserHooks.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

using namespace Pythia8;
using namespace std;

/*------------------------------------------------------------
TerminalSilencer

Pythia sometimes prints the welcome banner when the Pythia object is
constructed. This class temporarily redirects stdout and stderr to /dev/null.
It is only used around Pythia construction, initialization and event generation.
------------------------------------------------------------*/
class TerminalSilencer {
public:
    explicit TerminalSilencer(bool activeIn) : active(activeIn), oldOut(-1), oldErr(-1), nullFd(-1) {
        if (!active) return;
        fflush(stdout);
        fflush(stderr);
        oldOut = dup(STDOUT_FILENO);
        oldErr = dup(STDERR_FILENO);
        nullFd = open("/dev/null", O_WRONLY);
        if (nullFd >= 0) {
            dup2(nullFd, STDOUT_FILENO);
            dup2(nullFd, STDERR_FILENO);
        }
    }

    ~TerminalSilencer() {
        if (!active) return;
        fflush(stdout);
        fflush(stderr);
        if (oldOut >= 0) { dup2(oldOut, STDOUT_FILENO); close(oldOut); }
        if (oldErr >= 0) { dup2(oldErr, STDERR_FILENO); close(oldErr); }
        if (nullFd >= 0) close(nullFd);
    }

private:
    bool active;
    int oldOut;
    int oldErr;
    int nullFd;
};

/*------------------------------------------------------------
UmbrellaBiasTable

Stores the adaptive umbrella bias B(pTHat). The bins are uniform in
log(1+pTHat), which gives more resolution at low pT while still covering the
high-pT tail. The bias factor is always at least 1.
------------------------------------------------------------*/
struct UmbrellaBiasTable {
    int nBins = 40;
    double pTMax = 1000.0;
    double maxBias = 1.0e6;
    vector<double> bias;

    UmbrellaBiasTable() : bias(nBins, 1.0) {}

    int binIndex(double pTHat) const {
        const double pT = max(0.0, min(pTHat, pTMax));
        const double y = log1p(pT) / log1p(pTMax);
        int i = int(y * nBins);
        if (i < 0) i = 0;
        if (i >= nBins) i = nBins - 1;
        return i;
    }

    double factor(double pTHat) const {
        return bias[binIndex(pTHat)];
    }
};

struct BatchResult {
    int accepted = 0;

    double sumW = 0.0;
    double sumW2 = 0.0;
    double meanW = 0.0;
    double stdW = 0.0;
    double neff = 0.0;
    double neffRatio = 0.0;

    double meanPTHat = 0.0;
    double maxPTHat = 0.0;

    double fracPT10 = 0.0;
    double fracPT20 = 0.0;
    double fracPT50 = 0.0;
    double fracPT100 = 0.0;

    double wFracPT10 = 0.0;
    double wFracPT20 = 0.0;
    double wFracPT50 = 0.0;
    double wFracPT100 = 0.0;

    double score = 0.0;
    vector<int> hist;
};

/*------------------------------------------------------------
UmbrellaHook

Pythia calls this hook during phase-space selection. Returning B(pTHat) > 1
makes that pTHat bin more likely to be generated. The same B(pTHat) is later
used to manually unbias the event.
------------------------------------------------------------*/
class UmbrellaHook : public UserHooks {
public:
    explicit UmbrellaHook(shared_ptr<UmbrellaBiasTable> tableIn) : table(tableIn) {}

    bool canBiasSelection() override { return true; }

    double biasSelectionBy(const SigmaProcess* sigmaProcessPtr,
                           const PhaseSpace* phaseSpacePtr,
                           bool /*inEvent*/) override {
        if (sigmaProcessPtr->nFinal() != 2) return 1.0;
        return table->factor(phaseSpacePtr->pTHat());
    }

private:
    shared_ptr<UmbrellaBiasTable> table;
};

/*------------------------------------------------------------
configurePythia

Sets up the same QCD process as before and switches off standard Pythia print
settings. TerminalSilencer is still used because some builds print before these
settings take effect.
------------------------------------------------------------*/
static void configurePythia(Pythia& pythia) {
    pythia.readString("Print:quiet = on");
    pythia.readString("Init:showChangedSettings = off");
    pythia.readString("Init:showAllSettings = off");
    pythia.readString("Init:showChangedParticleData = off");
    pythia.readString("Init:showAllParticleData = off");
    pythia.readString("Next:numberShowInfo = 0");
    pythia.readString("Next:numberShowProcess = 0");
    pythia.readString("Next:numberShowEvent = 0");

    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13000.");
    pythia.readString("HardQCD:all = on");
    pythia.readString("PhaseSpace:pTHatMin = 0.");
}

/*------------------------------------------------------------
balancedScore

The score is the generated high-pT fraction times the effective sample-size
ratio. It is proportional to the effective fraction of useful high-pT events.
------------------------------------------------------------*/
static double balancedScore(double fracPT50, double neffRatio) {
    return fracPT50 * neffRatio;
}

/*------------------------------------------------------------
writeEventHeader

Optional event-level output. Useful for debugging the weights and making
histograms. The summary CSV is always written separately.
------------------------------------------------------------*/
static void writeEventHeader(ofstream& out) {
    out << "event,x1,x2,xprod,sHat,mHat,pTHat,pT2Hat,tHat,uHat,"
        << "Q2Fac,Q2Ren,alphaS,alphaEM,id1,id2,code,"
        << "pythiaWeight,biasFactor,manualWeight,"
        << "maxFinalParticlePT,nFinalParticles,nChargedFinal\n";
}

/*------------------------------------------------------------
runBatch

Runs one Pythia batch for the current umbrella table. It computes generated
fractions, manually weighted/unbiased fractions, Neff/N and the score.
------------------------------------------------------------*/
static BatchResult runBatch(shared_ptr<UmbrellaBiasTable> table,
                            int nEvents,
                            int iter,
                            const string& eventPrefix,
                            bool saveEvents) {
    unique_ptr<Pythia> pythiaPtr;
    {
        TerminalSilencer silence(true);
        pythiaPtr = make_unique<Pythia>();
        configurePythia(*pythiaPtr);
        pythiaPtr->setUserHooksPtr(make_shared<UmbrellaHook>(table));
        if (!pythiaPtr->init()) {
            cerr << "Pythia initialization failed in iteration " << iter << "\n";
            return {};
        }
    }
    Pythia& pythia = *pythiaPtr;

    ofstream eventOut;
    if (saveEvents) {
        ostringstream name;
        name << eventPrefix << "_iter_" << setw(3) << setfill('0') << iter << ".csv";
        eventOut.open(name.str());
        writeEventHeader(eventOut);
    }

    BatchResult r;
    r.hist.assign(table->nBins, 0);

    double sumPTHat = 0.0;
    int n10 = 0, n20 = 0, n50 = 0, n100 = 0;
    double w10 = 0.0, w20 = 0.0, w50 = 0.0, w100 = 0.0;

    TerminalSilencer silence(true);
    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
        if (!pythia.next()) continue;

        const double x1 = pythia.info.x1();
        const double x2 = pythia.info.x2();
        const double xprod = x1 * x2;
        const double sHat = pythia.info.sHat();
        const double pTHat = pythia.info.pTHat();
        const double biasFactor = table->factor(pTHat);
        const double pythiaWeight = pythia.info.weight();
        const double manualWeight = pythiaWeight / biasFactor;

        double maxFinalPT = 0.0;
        int nFinal = 0;
        int nCharged = 0;
        for (int i = 0; i < pythia.event.size(); ++i) {
            if (!pythia.event[i].isFinal()) continue;
            ++nFinal;
            maxFinalPT = max(maxFinalPT, pythia.event[i].pT());
            if (pythia.event[i].isCharged()) ++nCharged;
        }

        ++r.accepted;
        r.sumW += manualWeight;
        r.sumW2 += manualWeight * manualWeight;
        sumPTHat += pTHat;
        r.maxPTHat = max(r.maxPTHat, pTHat);
        ++r.hist[table->binIndex(pTHat)];

        if (pTHat > 10.0)  { ++n10;  w10  += manualWeight; }
        if (pTHat > 20.0)  { ++n20;  w20  += manualWeight; }
        if (pTHat > 50.0)  { ++n50;  w50  += manualWeight; }
        if (pTHat > 100.0) { ++n100; w100 += manualWeight; }

        if (saveEvents) {
            const double Q2Fac = pythia.info.Q2Fac();
            eventOut << iEvent << "," << x1 << "," << x2 << "," << xprod << ","
                     << sHat << "," << sqrt(max(sHat, 0.0)) << "," << pTHat << ","
                     << pTHat * pTHat << "," << pythia.info.tHat() << "," << pythia.info.uHat() << ","
                     << Q2Fac << "," << pythia.info.Q2Ren() << ","
                     << pythia.info.alphaS() << "," << pythia.info.alphaEM() << ","
                     << pythia.info.id1() << "," << pythia.info.id2() << "," << pythia.info.code() << ","
                     << pythiaWeight << "," << biasFactor << "," << manualWeight << ","
                     << maxFinalPT << "," << nFinal << "," << nCharged << "\n";
        }
    }

    if (r.accepted == 0) return r;

    r.meanW = r.sumW / r.accepted;
    const double meanW2 = r.sumW2 / r.accepted;
    r.stdW = sqrt(max(0.0, meanW2 - r.meanW * r.meanW));
    r.neff = (r.sumW2 > 0.0) ? (r.sumW * r.sumW / r.sumW2) : 0.0;
    r.neffRatio = r.neff / r.accepted;

    r.meanPTHat = sumPTHat / r.accepted;
    r.fracPT10 = double(n10) / r.accepted;
    r.fracPT20 = double(n20) / r.accepted;
    r.fracPT50 = double(n50) / r.accepted;
    r.fracPT100 = double(n100) / r.accepted;

    if (r.sumW > 0.0) {
        r.wFracPT10 = w10 / r.sumW;
        r.wFracPT20 = w20 / r.sumW;
        r.wFracPT50 = w50 / r.sumW;
        r.wFracPT100 = w100 / r.sumW;
    }

    r.score = balancedScore(r.fracPT50, r.neffRatio);
    return r;
}

/*------------------------------------------------------------
updateUmbrella

Adaptive umbrella update. The target distribution is flat in log(1+pTHat) bins.
If a bin is undersampled compared to the target, its bias is increased. If it is
oversampled, its relative bias is decreased. A damping factor avoids large jumps.
------------------------------------------------------------*/
static void updateUmbrella(UmbrellaBiasTable& table, const vector<int>& hist, int accepted) {
    if (accepted <= 0) return;

    const double target = 1.0 / table.nBins;
    const double damping = 0.35;
    const double pseudo = 1.0; // prevents empty bins from causing infinite updates

    vector<double> updated(table.nBins, 1.0);
    for (int i = 0; i < table.nBins; ++i) {
        const double observed = (hist[i] + pseudo) / (accepted + pseudo * table.nBins);
        const double correction = pow(target / observed, damping);
        updated[i] = table.bias[i] * correction;
    }

    // Smooth in log space so neighbouring pT bins do not get wildly different biases.
    vector<double> smoothed = updated;
    for (int i = 1; i < table.nBins - 1; ++i) {
        const double l = log(updated[i - 1]);
        const double m = log(updated[i]);
        const double r = log(updated[i + 1]);
        smoothed[i] = exp(0.25 * l + 0.50 * m + 0.25 * r);
    }

    // Normalize so the smallest bias is 1, then cap the largest value.
    const double minBias = *min_element(smoothed.begin(), smoothed.end());
    for (double& b : smoothed) {
        b = min(max(1.0, b / minBias), table.maxBias);
    }
    table.bias = smoothed;
}

/*------------------------------------------------------------
Umbrella diagnostics for the summary table.

The old code wrote six analytic bias parameters. To keep the summary format
similar, this code writes six umbrella diagnostics in those columns instead:
    a_pT        -> minimum umbrella bias
    b_xprod     -> maximum umbrella bias
    c_Q2        -> mean umbrella bias
    e_logpT     -> bias in the pTHat > 50 region
    f_satpT     -> number of umbrella bins
    g_tworegion -> pTMax of the umbrella table
------------------------------------------------------------*/
static double meanBias(const UmbrellaBiasTable& table) {
    return accumulate(table.bias.begin(), table.bias.end(), 0.0) / table.bias.size();
}

static double tailBias(const UmbrellaBiasTable& table) {
    int i50 = table.binIndex(50.0);
    double sum = 0.0;
    int n = 0;
    for (int i = i50; i < table.nBins; ++i) {
        sum += table.bias[i];
        ++n;
    }
    return (n > 0) ? sum / n : 1.0;
}

static void writeSummaryHeader(ofstream& out) {
    out << "iteration,accepted,acceptable,bestSoFar,"
        << "a_pT,b_xprod,c_Q2,d_alphaS,e_logpT,f_satpT,g_tworegion,"
        << "sumManualW,sumManualW2,meanManualW,stdManualW,NeffManual,NeffManualOverN,"
        << "meanPTHat,maxPTHat,"
        << "fracPT10,fracPT20,fracPT50,fracPT100,"
        << "wFracPT10,wFracPT20,wFracPT50,wFracPT100,score\n";
}

static void writeSummaryRow(ofstream& out, int iter, const UmbrellaBiasTable& table,
                            const BatchResult& r, bool acceptable, bool bestSoFar) {
    const double minB = *min_element(table.bias.begin(), table.bias.end());
    const double maxB = *max_element(table.bias.begin(), table.bias.end());
    out << iter << "," << r.accepted << "," << int(acceptable) << "," << int(bestSoFar) << ","
        << minB << "," << maxB << "," << meanBias(table) << ",0,"
        << tailBias(table) << "," << table.nBins << "," << table.pTMax << ","
        << r.sumW << "," << r.sumW2 << "," << r.meanW << "," << r.stdW << ","
        << r.neff << "," << r.neffRatio << ","
        << r.meanPTHat << "," << r.maxPTHat << ","
        << r.fracPT10 << "," << r.fracPT20 << "," << r.fracPT50 << "," << r.fracPT100 << ","
        << r.wFracPT10 << "," << r.wFracPT20 << "," << r.wFracPT50 << "," << r.wFracPT100 << ","
        << r.score << "\n";
}

/*------------------------------------------------------------
main

Same command-line interface as the previous code. Each iteration runs one batch,
writes one summary row, keeps the best score, and then updates the umbrella bias
for the next iteration.
------------------------------------------------------------*/
int main(int argc, char* argv[]) {
    int nIterations = 10;
    int nEventsPerBatch = 100000;
    string summaryFile = "umbrella_summary.csv";
    string eventPrefix = "umbrella_events";
    bool saveEvents = false;
    double minNeffRatio = 0.05;

    if (argc > 1) nIterations = atoi(argv[1]);
    if (argc > 2) nEventsPerBatch = atoi(argv[2]);
    if (argc > 3) summaryFile = argv[3];
    if (argc > 4) eventPrefix = argv[4];
    if (argc > 5) saveEvents = (atoi(argv[5]) != 0);
    if (argc > 6) minNeffRatio = atof(argv[6]);

    auto table = make_shared<UmbrellaBiasTable>();
    BatchResult bestRes;
    UmbrellaBiasTable bestTable = *table;
    double bestScore = -1.0;

    ofstream summary(summaryFile);
    writeSummaryHeader(summary);

    cout << fixed << setprecision(6);
    cout << "Starting adaptive umbrella manual-unbias search\n";
    cout << "Iterations: " << nIterations << "\n";
    cout << "Events per iteration: " << nEventsPerBatch << "\n";
    cout << "Minimum acceptable Neff/N: " << minNeffRatio << "\n";
    cout << "Summary file: " << summaryFile << "\n";
    cout << "Saving event CSVs: " << (saveEvents ? "yes" : "no") << "\n\n";

    for (int iter = 0; iter < nIterations; ++iter) {
        const double minB = *min_element(table->bias.begin(), table->bias.end());
        const double maxB = *max_element(table->bias.begin(), table->bias.end());
        cout << "Iteration " << iter << " umbrella: minB=" << minB
             << ", maxB=" << maxB
             << ", meanB=" << meanBias(*table)
             << ", tailB=" << tailBias(*table) << "\n";

        BatchResult r = runBatch(table, nEventsPerBatch, iter, eventPrefix, saveEvents);
        const bool acceptable = (r.accepted > 0 && r.neffRatio >= minNeffRatio);
        const bool isBest = acceptable && (r.score > bestScore);

        if (isBest) {
            bestScore = r.score;
            bestRes = r;
            bestTable = *table;
        }

        writeSummaryRow(summary, iter, *table, r, acceptable, isBest);
        summary.flush();

        cout << "  accepted=" << r.accepted
             << ", fracPT50=" << r.fracPT50
             << ", wFracPT50=" << scientific << r.wFracPT50 << fixed
             << ", Neff/N=" << r.neffRatio
             << ", score=" << r.score
             << ", meanW=" << r.meanW
             << ", stdW=" << r.stdW
             << ", meanPTHat=" << r.meanPTHat
             << ", maxPTHat=" << r.maxPTHat;
        if (!acceptable) cout << "  [rejected: low Neff/N]";
        if (isBest) cout << "  [new best]";
        cout << "\n\n";

        updateUmbrella(*table, r.hist, r.accepted);
    }

    cout << "Search finished.\n";
    cout << "Best score              = " << bestScore << "\n";
    cout << "Best generated fracPT50 = " << bestRes.fracPT50 << "\n";
    cout << "Best weighted wFracPT50 = " << scientific << bestRes.wFracPT50 << fixed << "\n";
    cout << "Best Neff/N             = " << bestRes.neffRatio << "\n";
    cout << "Best umbrella min/max/mean/tail = "
         << *min_element(bestTable.bias.begin(), bestTable.bias.end()) << " / "
         << *max_element(bestTable.bias.begin(), bestTable.bias.end()) << " / "
         << meanBias(bestTable) << " / " << tailBias(bestTable) << "\n";

    return 0;
}

