// Sweep multivariable Pythia bias parameters with manual unbiasing.
//
// Compile with:
// g++ -std=c++17 test_pythia_multivar_bias2_manual_unbias.cc -o sweep_multivar_bias    -I../pythia8317/include    -L../pythia8317/lib    -lpythia8    -Wl,-rpath,../pythia8317/lib
//
// Run, e.g.:
// ./sweep_multivar_bias 20000 0
//
// Arguments:
//   argv[1] = events per candidate, default 20000
//   argv[2] = save full events for every candidate? 0/1, default 0
//
// Output:
//   sweep_runs_manual/sweep_summary.csv
//   sweep_runs_manual/run_XXXX/weights.txt
//   sweep_runs_manual/run_XXXX/run.log
//   optional: sweep_runs_manual/run_XXXX/events.csv

#include "Pythia8/Pythia.h"
#include "Pythia8/UserHooks.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <memory>
#include <algorithm>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <iomanip>

using namespace Pythia8;
using namespace std;
namespace fs = std::filesystem;

struct Candidate {
    double a_pT = 0.0;
    double b_xprod = 0.0;
    double c_Q2 = 0.0;
    double e_logpT = 0.0;
    double f_satpT = 0.0;
    double g_tworegion = 0.0;
    string label = "";
};

struct RunSummary {
    int runId = 0;
    Candidate cand;
    int accepted = 0;
    int failed = 0;
    long double sumPythiaW = 0.0;
    long double sumPythiaW2 = 0.0;
    long double sumManualW = 0.0;
    long double sumManualW2 = 0.0;
    long double sumPTHat = 0.0;
    double maxPTHat = 0.0;
    long long nPT10 = 0;
    long long nPT20 = 0;
    long long nPT50 = 0;
    long long nPT100 = 0;
    long double sumManualWPT50 = 0.0;
    long double sumManualWPT100 = 0.0;
    long long biasCalls = 0;
    long long biasGreaterThanOne = 0;
    double minReturnedBias = 0.0;
    double maxReturnedBias = 0.0;
    double meanReturnedBias = 0.0;
};

class MultiVariableBias : public UserHooks {
public:
    double a_pT;
    double b_xprod;
    double c_Q2;
    double e_logpT;
    double f_satpT;
    double g_tworegion;

    double pT0 = 10.0;
    double xprod0 = 1e-4;
    double Q20 = 100.0;
    double pTswitch = 50.0;
    double maxBias = 1000.0;

    double lastReturnedBias = 1.0;
    long long biasCalls = 0;
    long long biasGreaterThanOne = 0;
    double minReturnedBias = 1e300;
    double maxReturnedBias = 0.0;
    long double sumReturnedBias = 0.0;

    MultiVariableBias(const Candidate& c) {
        a_pT = c.a_pT;
        b_xprod = c.b_xprod;
        c_Q2 = c.c_Q2;
        e_logpT = c.e_logpT;
        f_satpT = c.f_satpT;
        g_tworegion = c.g_tworegion;
    }

    bool canBiasSelection() override { return true; }

    double biasSelectionBy(const SigmaProcess* sigmaProcessPtr,
                           const PhaseSpace* phaseSpacePtr,
                           bool inEvent) override {
        int nFinal = sigmaProcessPtr->nFinal();
        if (nFinal != 2) {
            lastReturnedBias = 1.0;
            return 1.0;
        }

        double pTHat = phaseSpacePtr->pTHat();
        double xprod = phaseSpacePtr->x1() * phaseSpacePtr->x2();
        double Q2Fac = sigmaProcessPtr->Q2Fac();

        double bias = 1.0;

        if (pTHat > 0.0 && a_pT != 0.0) {
            bias *= pow(pTHat / pT0, a_pT);
        }
        if (xprod > 0.0 && b_xprod != 0.0) {
            bias *= pow(xprod / xprod0, b_xprod);
        }
        if (Q2Fac > 0.0 && c_Q2 != 0.0) {
            bias *= pow(Q2Fac / Q20, c_Q2);
        }
        if (pTHat > 0.0 && e_logpT != 0.0) {
            bias *= pow(log(1.0 + pTHat / pT0), e_logpT);
        }
        if (pTHat > 0.0 && f_satpT != 0.0) {
            double sat = 1.0 + pTHat / (pTHat + pT0);
            bias *= pow(sat, f_satpT);
        }
        if (pTHat > pTswitch && g_tworegion != 0.0) {
            bias *= pow(pTHat / pTswitch, g_tworegion);
        }

        bias = max(1.0, bias);
        bias = min(maxBias, bias);

        lastReturnedBias = bias;
        biasCalls++;
        if (bias > 1.0) biasGreaterThanOne++;
        minReturnedBias = min(minReturnedBias, bias);
        maxReturnedBias = max(maxReturnedBias, bias);
        sumReturnedBias += bias;

        return bias;
    }

    double meanReturnedBias() const {
        if (biasCalls == 0) return 0.0;
        return static_cast<double>(sumReturnedBias / biasCalls);
    }
};

static string keyFor(const Candidate& c) {
    ostringstream ss;
    ss << fixed << setprecision(4)
       << c.a_pT << "_" << c.b_xprod << "_" << c.c_Q2 << "_"
       << c.e_logpT << "_" << c.f_satpT << "_" << c.g_tworegion;
    return ss.str();
}

static void addCandidate(vector<Candidate>& candidates, set<string>& seen,
                         double a, double b, double c, double e, double f, double g,
                         const string& label) {
    Candidate cand;
    cand.a_pT = a;
    cand.b_xprod = b;
    cand.c_Q2 = c;
    cand.e_logpT = e;
    cand.f_satpT = f;
    cand.g_tworegion = g;
    cand.label = label;
    string k = keyFor(cand);
    if (seen.insert(k).second) candidates.push_back(cand);
}

static vector<Candidate> makeCandidates() {
    vector<Candidate> candidates;
    set<string> seen;

    addCandidate(candidates, seen, 0,0,0,0,0,0,"naive");

    // Peak region
    for (double a : {3.6,3.8,4.0,4.2,4.4,4.6}) {
      double b = 0.1 * (a - 1.0);  // reproduces old scan
      double c = b;
      addCandidate(candidates, seen, a,b,c,0,0,0,"old_peak");
    }

    // Local grid around the known useful region.
    for (double a : {4.0, 4.6, 5.0}) {
        for (double b : {0.30, 0.36, 0.50}) {
            for (double c : {0.30, 0.36, 0.50}) {
                addCandidate(candidates, seen, a, b, c, 0, 0, 0, "abc_local_grid");
            }
        }
    }

    // Only a few extra-term tests on top of the known good abc bias.
    addCandidate(candidates, seen, 4.6, 0.36, 0.36, 1.0, 0, 0, "best_plus_logpT");
    addCandidate(candidates, seen, 4.6, 0.36, 0.36, 2.0, 0, 0, "best_plus_logpT");
    addCandidate(candidates, seen, 4.6, 0.36, 0.36, 0, 2.0, 0, "best_plus_satpT");
    addCandidate(candidates, seen, 4.6, 0.36, 0.36, 0, 4.0, 0, "best_plus_satpT");
    addCandidate(candidates, seen, 4.6, 0.36, 0.36, 0, 0, 2.0, "best_plus_two_region");
    addCandidate(candidates, seen, 4.6, 0.36, 0.36, 0, 0, 4.0, "best_plus_two_region");

    return candidates;
}

static RunSummary runOne(const Candidate& cand, int runId, int nEvents,
                         const string& baseDir, bool saveEvents) {
    RunSummary s;
    s.runId = runId;
    s.cand = cand;

    ostringstream runName;
    runName << baseDir << "/run_" << setw(4) << setfill('0') << runId;
    string outDir = runName.str();
    fs::create_directories(outDir);

    string eventsFile = outDir + "/events.csv";
    string weightsFile = outDir + "/weights.txt";
    string logFile = outDir + "/run.log";

    ofstream events;
    if (saveEvents) {
        events.open(eventsFile);
        events << "event,x1,x2,xprod,sHat,mHat,pTHat,pT2Hat,tHat,uHat,"
               << "Q2Fac,Q2Ren,alphaEM,id1,id2,code,"
               << "pythiaWeight,manualBias,manualWeight,"
               << "maxFinalParticlePT,nFinalParticles,nChargedFinal\n";
    }

    ofstream weights(weightsFile);
    ofstream log(logFile);

    Pythia pythia;
    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13000.");
    pythia.readString("HardQCD:all = on");
    pythia.readString("PhaseSpace:pTHatMin = 0.");
    pythia.readString("Next:numberShowInfo = 0");
    pythia.readString("Next:numberShowProcess = 0");
    pythia.readString("Next:numberShowEvent = 0");
    pythia.readString("Next:numberCount = 0");

    // Different seed per candidate, reproducible sweep.
    pythia.readString("Random:setSeed = on");
    pythia.readString("Random:seed = " + to_string(1000 + runId));

    auto biasHook = make_shared<MultiVariableBias>(cand);
    pythia.setUserHooksPtr(biasHook);

    log << "label = " << cand.label << "\n"
        << "a_pT = " << cand.a_pT << "\n"
        << "b_xprod = " << cand.b_xprod << "\n"
        << "c_Q2 = " << cand.c_Q2 << "\n"
        << "e_logpT = " << cand.e_logpT << "\n"
        << "f_satpT = " << cand.f_satpT << "\n"
        << "g_tworegion = " << cand.g_tworegion << "\n"
        << "nEvents = " << nEvents << "\n";

    if (!pythia.init()) {
        cerr << "Pythia initialization failed for run " << runId << endl;
        return s;
    }

    weights << "event,pTHat,pythiaWeight,manualBias,manualWeight\n";

    for (int iEvent = 0; s.accepted < nEvents; ++iEvent) {
        if (!pythia.next()) {
            s.failed++;
            continue;
        }

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
        double alphaEM = pythia.info.alphaEM();
        int id1 = pythia.info.id1();
        int id2 = pythia.info.id2();
        int code = pythia.info.code();

        double pythiaWeight = pythia.info.weight();
        double manualBias = max(1.0, biasHook->lastReturnedBias);
        double manualWeight = pythiaWeight / manualBias;

        double maxFinalParticlePT = 0.0;
        int nFinalParticles = 0;
        int nChargedFinal = 0;

        if (saveEvents) {
            for (int i = 0; i < pythia.event.size(); ++i) {
                if (pythia.event[i].isFinal()) {
                    nFinalParticles++;
                    maxFinalParticlePT = max(maxFinalParticlePT, pythia.event[i].pT());
                    if (pythia.event[i].isCharged()) nChargedFinal++;
                }
            }

            events << s.accepted << "," << x1 << "," << x2 << "," << xprod << ","
                   << sHat << "," << mHat << "," << pTHat << "," << pT2Hat << ","
                   << tHat << "," << uHat << "," << Q2Fac << "," << Q2Ren << ","
                   << alphaEM << "," << id1 << "," << id2 << "," << code << ","
                   << pythiaWeight << "," << manualBias << "," << manualWeight << ","
                   << maxFinalParticlePT << "," << nFinalParticles << "," << nChargedFinal << "\n";
        }

        if (s.accepted < 200) {
            weights << s.accepted << "," << pTHat << "," << pythiaWeight << ","
                    << manualBias << "," << manualWeight << "\n";
        }

        s.sumPythiaW += pythiaWeight;
        s.sumPythiaW2 += pythiaWeight * pythiaWeight;
        s.sumManualW += manualWeight;
        s.sumManualW2 += manualWeight * manualWeight;
        s.sumPTHat += pTHat;
        s.maxPTHat = max(s.maxPTHat, pTHat);
        if (pTHat > 10.0) s.nPT10++;
        if (pTHat > 20.0) s.nPT20++;
        if (pTHat > 50.0) { s.nPT50++; s.sumManualWPT50 += manualWeight; }
        if (pTHat > 100.0) { s.nPT100++; s.sumManualWPT100 += manualWeight; }
        s.accepted++;
    }

    if (saveEvents) events.close();
    weights.close();

    s.biasCalls = biasHook->biasCalls;
    s.biasGreaterThanOne = biasHook->biasGreaterThanOne;
    s.minReturnedBias = (biasHook->biasCalls > 0) ? biasHook->minReturnedBias : 0.0;
    s.maxReturnedBias = biasHook->maxReturnedBias;
    s.meanReturnedBias = biasHook->meanReturnedBias();

    double neffManual = (s.sumManualW2 > 0.0)
        ? static_cast<double>((s.sumManualW * s.sumManualW) / s.sumManualW2)
        : 0.0;
    double neffManualOverN = (s.accepted > 0) ? neffManual / s.accepted : 0.0;
    double fracPT50 = (s.accepted > 0) ? static_cast<double>(s.nPT50) / s.accepted : 0.0;
    double scorePT50 = fracPT50 * neffManualOverN;

    log << "accepted = " << s.accepted << "\n"
        << "failed = " << s.failed << "\n"
        << "NeffManualOverN = " << neffManualOverN << "\n"
        << "fracPT50 = " << fracPT50 << "\n"
        << "scorePT50 = " << scorePT50 << "\n"
        << "biasCalls = " << s.biasCalls << "\n";
    log.close();

    return s;
}

static void writeSummaryHeader(ofstream& out) {
    out << "runId,label,accepted,failed,"
        << "a_pT,b_xprod,c_Q2,e_logpT,f_satpT,g_tworegion,"
        << "sumPythiaW,sumPythiaW2,NeffPythia,NeffPythiaOverN,"
        << "sumManualW,sumManualW2,NeffManual,NeffManualOverN,"
        << "meanPTHat,maxPTHat,fracPT10,fracPT20,fracPT50,fracPT100,"
        << "weightedFracPT50,weightedFracPT100,scorePT50,scorePT100,"
        << "biasCalls,biasGreaterThanOne,minReturnedBias,maxReturnedBias,meanReturnedBias\n";
}

static void writeSummaryRow(ofstream& out, const RunSummary& s) {
    double neffPythia = (s.sumPythiaW2 > 0.0)
        ? static_cast<double>((s.sumPythiaW * s.sumPythiaW) / s.sumPythiaW2)
        : 0.0;
    double neffManual = (s.sumManualW2 > 0.0)
        ? static_cast<double>((s.sumManualW * s.sumManualW) / s.sumManualW2)
        : 0.0;
    double neffPythiaOverN = (s.accepted > 0) ? neffPythia / s.accepted : 0.0;
    double neffManualOverN = (s.accepted > 0) ? neffManual / s.accepted : 0.0;
    double meanPTHat = (s.accepted > 0) ? static_cast<double>(s.sumPTHat / s.accepted) : 0.0;
    double fracPT10 = (s.accepted > 0) ? static_cast<double>(s.nPT10) / s.accepted : 0.0;
    double fracPT20 = (s.accepted > 0) ? static_cast<double>(s.nPT20) / s.accepted : 0.0;
    double fracPT50 = (s.accepted > 0) ? static_cast<double>(s.nPT50) / s.accepted : 0.0;
    double fracPT100 = (s.accepted > 0) ? static_cast<double>(s.nPT100) / s.accepted : 0.0;

    double weightedFracPT50 = (s.sumManualW > 0.0) ? static_cast<double>(s.sumManualWPT50 / s.sumManualW) : 0.0;
    double weightedFracPT100 = (s.sumManualW > 0.0) ? static_cast<double>(s.sumManualWPT100 / s.sumManualW) : 0.0;

    double scorePT50 = fracPT50 * neffManualOverN;
    double scorePT100 = fracPT100 * neffManualOverN;

    out << s.runId << ","
        << s.cand.label << ","
        << s.accepted << ","
        << s.failed << ","
        << s.cand.a_pT << ","
        << s.cand.b_xprod << ","
        << s.cand.c_Q2 << ","
        << s.cand.e_logpT << ","
        << s.cand.f_satpT << ","
        << s.cand.g_tworegion << ","
        << static_cast<double>(s.sumPythiaW) << ","
        << static_cast<double>(s.sumPythiaW2) << ","
        << neffPythia << ","
        << neffPythiaOverN << ","
        << static_cast<double>(s.sumManualW) << ","
        << static_cast<double>(s.sumManualW2) << ","
        << neffManual << ","
        << neffManualOverN << ","
        << meanPTHat << ","
        << s.maxPTHat << ","
        << fracPT10 << ","
        << fracPT20 << ","
        << fracPT50 << ","
        << fracPT100 << ","
        << weightedFracPT50 << ","
        << weightedFracPT100 << ","
        << scorePT50 << ","
        << scorePT100 << ","
        << s.biasCalls << ","
        << s.biasGreaterThanOne << ","
        << s.minReturnedBias << ","
        << s.maxReturnedBias << ","
        << s.meanReturnedBias << "\n";
}

int main(int argc, char* argv[]) {
    int nEvents = 20000;
    bool saveEvents = false;

    if (argc > 1) nEvents = atoi(argv[1]);
    if (argc > 2) saveEvents = (atoi(argv[2]) != 0);

    string baseDir = "sweep_runs_manual";
    fs::create_directories(baseDir);

    vector<Candidate> candidates = makeCandidates();
    string summaryFile = baseDir + "/sweep_summary.csv";
    ofstream summary(summaryFile);
    writeSummaryHeader(summary);

    cout << "Running " << candidates.size() << " candidates with "
         << nEvents << " events each.\n";
    cout << "Manual unbiasing: manualWeight = pythia.info.weight() / returnedBias\n";
    cout << "Output summary: " << summaryFile << "\n";

    double bestScore = -1.0;
    RunSummary best;

    for (size_t i = 0; i < candidates.size(); ++i) {
        cout << "[" << (i + 1) << "/" << candidates.size() << "] "
             << candidates[i].label
             << "  a=" << candidates[i].a_pT
             << " b=" << candidates[i].b_xprod
             << " c=" << candidates[i].c_Q2
             << " e=" << candidates[i].e_logpT
             << " f=" << candidates[i].f_satpT
             << " g=" << candidates[i].g_tworegion
             << endl;

        RunSummary s = runOne(candidates[i], static_cast<int>(i), nEvents, baseDir, saveEvents);
        writeSummaryRow(summary, s);
        summary.flush();

        double neffManual = (s.sumManualW2 > 0.0)
            ? static_cast<double>((s.sumManualW * s.sumManualW) / s.sumManualW2)
            : 0.0;
        double neffManualOverN = (s.accepted > 0) ? neffManual / s.accepted : 0.0;
        double fracPT50 = (s.accepted > 0) ? static_cast<double>(s.nPT50) / s.accepted : 0.0;
        double score = fracPT50 * neffManualOverN;

        cout << "    NeffManual/N=" << neffManualOverN
             << " fracPT50=" << fracPT50
             << " score=" << score << "\n";

        if (score > bestScore) {
            bestScore = score;
            best = s;
        }
    }

    summary.close();

    cout << "\nBest candidate by scorePT50 = fracPT50 * NeffManualOverN:\n"
         << "  runId = " << best.runId << "\n"
         << "  label = " << best.cand.label << "\n"
         << "  a_pT = " << best.cand.a_pT << "\n"
         << "  b_xprod = " << best.cand.b_xprod << "\n"
         << "  c_Q2 = " << best.cand.c_Q2 << "\n"
         << "  e_logpT = " << best.cand.e_logpT << "\n"
         << "  f_satpT = " << best.cand.f_satpT << "\n"
         << "  g_tworegion = " << best.cand.g_tworegion << "\n"
         << "  scorePT50 = " << bestScore << "\n"
         << "\nSee: " << summaryFile << "\n";

    return 0;
}
