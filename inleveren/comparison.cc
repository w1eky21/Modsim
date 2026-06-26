
// Run four comparable Pythia8 samples in one executable:
//   1) naive inclusive HardQCD sample
//   2) Pythia pTHatMin = 50 GeV benchmark sample
//   3) Pythia built-in power-law phase-space bias sample
//   4) Hybrid LHAPDF bias sample using the best parameters found so far

/*
g++ four_run_validation.cc -o four_run_validation \
  -I../pythia8317/include \
  $(lhapdf-config --cxxflags) \
  -L../pythia8317/lib \
  $(lhapdf-config --ldflags) \
  -lpythia8 -lLHAPDF \
  -Wl,-rpath,../pythia8317/lib \
  -std=c++17
  */

//   ./four_run_validation 100000 validation 12345 NNPDF31_lo_as_0118

/* Optional arguments:
   argv[1]  nEvents per run              default 100000
   argv[2]  output prefix                default validation
   argv[3]  random seed                  default 12345
   argv[4]  LHAPDF set name              default NNPDF23_lo_as_0130_qed
   argv[5]  a_pT                         default 4.818
   argv[6]  b_tau                        default 0.152
   argv[7]  c_lumi                       default 0.082
   argv[8]  e_logpT                      default 0.654
   argv[9]  g_tworegion                  default 0.102
   argv[10] powerlaw exponent             default 5.0
*/

#include "Pythia8/Pythia.h"
#include "Pythia8/UserHooks.h"
#include "LHAPDF/LHAPDF.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace Pythia8;
using namespace std;

struct BiasParams {
    double a_pT        = 4.818;
    double b_tau       = 0.152;
    double c_lumi      = 0.082;
    double e_logpT     = 0.654;
    double g_tworegion = 0.102;
};

struct RunConfig {
    string label;
    double pTHatMin = 0.0;
    bool useHybridBias = false;
    bool usePowerLawBias = false;
    double powerLawPow = 5.0;
    double powerLawRef = 15.0;
};

struct Summary {
    string label;
    int accepted = 0;
    double sumW = 0.0;
    double sumW2 = 0.0;
    double meanW = 0.0;
    double stdW = 0.0;
    double Neff = 0.0;
    double NeffOverN = 0.0;
    double meanPTHat = 0.0;
    double maxPTHat = 0.0;
    double fracPT10 = 0.0, fracPT20 = 0.0, fracPT50 = 0.0, fracPT100 = 0.0;
    double fracPT200 = 0.0, fracPT500 = 0.0, fracPT1000 = 0.0;
    double wFracPT10 = 0.0, wFracPT20 = 0.0, wFracPT50 = 0.0, wFracPT100 = 0.0;
    double wFracPT200 = 0.0, wFracPT500 = 0.0, wFracPT1000 = 0.0;
};

static double safeGluonLuminosity(const LHAPDF::PDF* pdf, double x1, double x2, double Q) {
    const double lumiRef = 1.0e6;
    if (pdf == nullptr) return lumiRef;

    x1 = min(max(x1, 1.0e-7), 0.999999);
    x2 = min(max(x2, 1.0e-7), 0.999999);
    if (!isfinite(Q) || Q <= 0.0) Q = 10.0;
    Q = min(max(Q, 1.0), 1.0e5);

    try {
        const double xg1 = pdf->xfxQ(21, x1, Q);
        const double xg2 = pdf->xfxQ(21, x2, Q);
        const double g1 = xg1 / x1;
        const double g2 = xg2 / x2;
        const double lumi = g1 * g2;
        if (!isfinite(lumi) || lumi <= 0.0) return lumiRef;
        return lumi;
    } catch (...) {
        return lumiRef;
    }
}

static double computeBiasFactor(const BiasParams& p, double pTHat, double tau, double gluonLumi) {
    const double pT0 = 10.0;
    const double tau0 = 1.0e-4;
    const double pTswitch = 50.0;
    const double lumiRef = 1.0e6;
    const double maxBias = 1.0e6;

    double bias = 1.0;

    if (p.a_pT != 0.0) {
        bias *= pow(max(pTHat / pT0, 1.0e-12), p.a_pT);
    }
    if (p.b_tau != 0.0) {
        bias *= pow(max(tau / tau0, 1.0e-12), p.b_tau);
    }
    if (p.e_logpT != 0.0) {
        bias *= pow(max(log(1.0 + pTHat / pT0), 1.0e-12), p.e_logpT);
    }
    if (p.g_tworegion != 0.0 && pTHat > pTswitch) {
        bias *= pow(max(pTHat / pTswitch, 1.0e-12), p.g_tworegion);
    }
    if (p.c_lumi != 0.0) {
        const double lumiRatio = max(lumiRef / max(gluonLumi, 1.0e-300), 1.0e-12);
        bias *= pow(lumiRatio, p.c_lumi);
    }

    if (!isfinite(bias) || bias <= 0.0) bias = maxBias;
    return min(max(1.0, bias), maxBias);
}

class HybridLHAPDFBias : public UserHooks {
public:
    HybridLHAPDFBias(BiasParams parsIn, shared_ptr<LHAPDF::PDF> pdfIn)
        : pars(parsIn), pdf(std::move(pdfIn)) {}

    bool canBiasSelection() override { return true; }

    double biasSelectionBy(const SigmaProcess* sigmaProcessPtr,
                           const PhaseSpace* phaseSpacePtr,
                           bool /*inEvent*/) override {
        if (sigmaProcessPtr->nFinal() != 2) return 1.0;
        const double pTHat = phaseSpacePtr->pTHat();
        const double x1 = phaseSpacePtr->x1();
        const double x2 = phaseSpacePtr->x2();
        const double tau = x1 * x2;
        const double Q = sqrt(max(sigmaProcessPtr->Q2Fac(), 1.0));
        const double gluonLumi = safeGluonLuminosity(pdf.get(), x1, x2, Q);
        return computeBiasFactor(pars, pTHat, tau, gluonLumi);
    }

private:
    BiasParams pars;
    shared_ptr<LHAPDF::PDF> pdf;
};

static void configureBase(Pythia& pythia, double pTHatMin, int seed) {
    pythia.readString("Print:quiet = on");
    pythia.readString("Init:showChangedSettings = off");
    pythia.readString("Init:showAllSettings = off");
    pythia.readString("Init:showChangedParticleData = off");
    pythia.readString("Init:showAllParticleData = off");
    pythia.readString("Next:numberShowInfo = 0");
    pythia.readString("Next:numberShowProcess = 0");
    pythia.readString("Next:numberShowEvent = 0");

    pythia.readString("Random:setSeed = on");
    pythia.readString("Random:seed = " + to_string(seed));

    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13000.");
    pythia.readString("HardQCD:all = on");
    pythia.readString("PhaseSpace:pTHatMin = " + to_string(pTHatMin));
}

static void writeEventHeader(ofstream& out) {
    out << "sample,event,x1,x2,tau,sHat,mHat,pTHat,pT2Hat,tHat,uHat,"
        << "Q2Fac,Q2Ren,alphaS,alphaEM,id1,id2,code,"
        << "gluonLumi,logGluonLumi,pythiaWeight,biasFactor,manualWeight,analysisWeight,"
        << "maxFinalParticlePT,nFinalParticles,nChargedFinal\n";
}

static void addThresholds(double pT, double w,
                          int& n10, int& n20, int& n50, int& n100, int& n200, int& n500, int& n1000,
                          double& w10, double& w20, double& w50, double& w100, double& w200, double& w500, double& w1000) {
    if (pT > 10.0)   { ++n10;   w10   += w; }
    if (pT > 20.0)   { ++n20;   w20   += w; }
    if (pT > 50.0)   { ++n50;   w50   += w; }
    if (pT > 100.0)  { ++n100;  w100  += w; }
    if (pT > 200.0)  { ++n200;  w200  += w; }
    if (pT > 500.0)  { ++n500;  w500  += w; }
    if (pT > 1000.0) { ++n1000; w1000 += w; }
}

static Summary runSample(const RunConfig& cfg,
                         const BiasParams& pars,
                         int nEvents,
                         const string& outputCsv,
                         const string& pdfSetName,
                         int seed) {
    Summary s;
    s.label = cfg.label;

    shared_ptr<LHAPDF::PDF> pdf(LHAPDF::mkPDF(pdfSetName, 0));

    Pythia pythia;
    configureBase(pythia, cfg.pTHatMin, seed);

    // Built-in Pythia phase-space power-law bias.
    if (cfg.usePowerLawBias) {
        pythia.readString("PhaseSpace:bias2Selection = on");
        pythia.readString("PhaseSpace:bias2SelectionPow = " + to_string(cfg.powerLawPow));
        pythia.readString("PhaseSpace:bias2SelectionRef = " + to_string(cfg.powerLawRef));
    }

    // Custom hybrid bias.
    if (cfg.useHybridBias) {
        pythia.setUserHooksPtr(make_shared<HybridLHAPDFBias>(pars, pdf));
    }
    if (!pythia.init()) {
        cerr << "Pythia initialization failed for sample " << cfg.label << "\n";
        return s;
    }

    ofstream out(outputCsv);
    writeEventHeader(out);

    double sumPTHat = 0.0;
    int n10 = 0, n20 = 0, n50 = 0, n100 = 0, n200 = 0, n500 = 0, n1000 = 0;
    double w10 = 0.0, w20 = 0.0, w50 = 0.0, w100 = 0.0, w200 = 0.0, w500 = 0.0, w1000 = 0.0;

    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
        if (!pythia.next()) continue;

        const double x1 = pythia.info.x1();
        const double x2 = pythia.info.x2();
        const double tau = x1 * x2;
        const double sHat = pythia.info.sHat();
        const double pTHat = pythia.info.pTHat();
        const double Q = sqrt(max(pythia.info.Q2Fac(), 1.0));
        const double gluonLumi = safeGluonLuminosity(pdf.get(), x1, x2, Q);

        double biasFactor = 1.0;
        if (cfg.useHybridBias) {
            biasFactor = computeBiasFactor(pars, pTHat, tau, gluonLumi);
        } else if (cfg.usePowerLawBias) {
            // Pythia should already accounts for this in pythia.info.weight().
            biasFactor = pow(max(pTHat / cfg.powerLawRef, 1.0e-12), cfg.powerLawPow);
        }

        const double pythiaWeight = pythia.info.weight();

        // For pythia powerlae bias only use pythias weight
        // hybrid bias use manual unbiasing 
        const double manualWeight = cfg.useHybridBias ? pythiaWeight / biasFactor : pythiaWeight;
        const double analysisWeight = cfg.useHybridBias ? manualWeight : pythiaWeight;

        double maxFinalPT = 0.0;
        int nFinal = 0;
        int nCharged = 0;
        for (int i = 0; i < pythia.event.size(); ++i) {
            if (!pythia.event[i].isFinal()) continue;
            ++nFinal;
            maxFinalPT = max(maxFinalPT, pythia.event[i].pT());
            if (pythia.event[i].isCharged()) ++nCharged;
        }

        ++s.accepted;
        s.sumW += analysisWeight;
        s.sumW2 += analysisWeight * analysisWeight;
        sumPTHat += pTHat;
        s.maxPTHat = max(s.maxPTHat, pTHat);
        addThresholds(pTHat, analysisWeight, n10, n20, n50, n100, n200, n500, n1000,
                      w10, w20, w50, w100, w200, w500, w1000);

        out << cfg.label << "," << iEvent << ","
            << x1 << "," << x2 << "," << tau << ","
            << sHat << "," << sqrt(max(sHat, 0.0)) << "," << pTHat << ","
            << pTHat * pTHat << "," << pythia.info.tHat() << "," << pythia.info.uHat() << ","
            << pythia.info.Q2Fac() << "," << pythia.info.Q2Ren() << ","
            << pythia.info.alphaS() << "," << pythia.info.alphaEM() << ","
            << pythia.info.id1() << "," << pythia.info.id2() << "," << pythia.info.code() << ","
            << gluonLumi << "," << log(max(gluonLumi, 1.0e-300)) << ","
            << pythiaWeight << "," << biasFactor << "," << manualWeight << "," << analysisWeight << ","
            << maxFinalPT << "," << nFinal << "," << nCharged << "\n";
    }

    if (s.accepted == 0) return s;

    s.meanW = s.sumW / s.accepted;
    const double meanW2 = s.sumW2 / s.accepted;
    s.stdW = sqrt(max(0.0, meanW2 - s.meanW * s.meanW));
    s.Neff = (s.sumW2 > 0.0) ? (s.sumW * s.sumW / s.sumW2) : 0.0;
    s.NeffOverN = s.Neff / s.accepted;
    s.meanPTHat = sumPTHat / s.accepted;

    s.fracPT10 = double(n10) / s.accepted;
    s.fracPT20 = double(n20) / s.accepted;
    s.fracPT50 = double(n50) / s.accepted;
    s.fracPT100 = double(n100) / s.accepted;
    s.fracPT200 = double(n200) / s.accepted;
    s.fracPT500 = double(n500) / s.accepted;
    s.fracPT1000 = double(n1000) / s.accepted;

    if (s.sumW > 0.0) {
        s.wFracPT10 = w10 / s.sumW;
        s.wFracPT20 = w20 / s.sumW;
        s.wFracPT50 = w50 / s.sumW;
        s.wFracPT100 = w100 / s.sumW;
        s.wFracPT200 = w200 / s.sumW;
        s.wFracPT500 = w500 / s.sumW;
        s.wFracPT1000 = w1000 / s.sumW;
    }

    cout << "Finished " << cfg.label << ": accepted=" << s.accepted
         << ", Neff/N=" << s.NeffOverN
         << ", fracPT50=" << s.fracPT50
         << ", wFracPT50=" << s.wFracPT50 << "\n";

    return s;
}

static void writeSummary(const string& path, const vector<Summary>& rows, const BiasParams& p) {
    ofstream out(path);
    out << "sample,accepted,sumW,sumW2,meanW,stdW,Neff,NeffOverN,"
        << "meanPTHat,maxPTHat,"
        << "fracPT10,fracPT20,fracPT50,fracPT100,fracPT200,fracPT500,fracPT1000,"
        << "wFracPT10,wFracPT20,wFracPT50,wFracPT100,wFracPT200,wFracPT500,wFracPT1000,"
        << "a_pT,b_tau,c_lumi,e_logpT,g_tworegion\n";

    out << setprecision(12);
    for (const Summary& s : rows) {
        out << s.label << "," << s.accepted << ","
            << s.sumW << "," << s.sumW2 << "," << s.meanW << "," << s.stdW << ","
            << s.Neff << "," << s.NeffOverN << ","
            << s.meanPTHat << "," << s.maxPTHat << ","
            << s.fracPT10 << "," << s.fracPT20 << "," << s.fracPT50 << ","
            << s.fracPT100 << "," << s.fracPT200 << "," << s.fracPT500 << "," << s.fracPT1000 << ","
            << s.wFracPT10 << "," << s.wFracPT20 << "," << s.wFracPT50 << ","
            << s.wFracPT100 << "," << s.wFracPT200 << "," << s.wFracPT500 << "," << s.wFracPT1000 << ","
            << p.a_pT << "," << p.b_tau << "," << p.c_lumi << "," << p.e_logpT << "," << p.g_tworegion << "\n";
    }
}

int main(int argc, char* argv[]) {
    int nEvents = (argc > 1) ? atoi(argv[1]) : 100000;
    string prefix = (argc > 2) ? argv[2] : "validation";
    int seed = (argc > 3) ? atoi(argv[3]) : 12345;
    string pdfSetName = (argc > 4) ? argv[4] : "NNPDF23_lo_as_0130_qed";

    BiasParams pars;
    if (argc > 5) pars.a_pT        = atof(argv[5]);
    if (argc > 6) pars.b_tau       = atof(argv[6]);
    if (argc > 7) pars.c_lumi      = atof(argv[7]);
    if (argc > 8) pars.e_logpT     = atof(argv[8]);
    if (argc > 9) pars.g_tworegion = atof(argv[9]);
    double powerLawPow = (argc > 10) ? atof(argv[10]) : 5.0;

    cout << "Four-run validation with " << nEvents << " requested events per sample\n";
    cout << "Output prefix: " << prefix << "\n";
    cout << "PDF set: " << pdfSetName << "\n";
    cout << "Bias parameters: a=" << pars.a_pT
         << ", b_tau=" << pars.b_tau
         << ", c_lumi=" << pars.c_lumi
         << ", e_logpT=" << pars.e_logpT
         << ", g_tworegion=" << pars.g_tworegion << "\n";
    cout << "Power-law bias exponent: " << powerLawPow << "\n";

    vector<RunConfig> configs = {
        // label, pTHatMin, useHybridBias, usePowerLawBias, powerLawPow, powerLawRef
        {"naive", 0.0, false, false, powerLawPow, 15.0},
        {"ptcut50", 50.0, false, false, powerLawPow, 15.0},
        {"powerlaw_bias", 0.0, false, true, powerLawPow, 15.0},
        {"hybrid_bias", 0.0, true, false, powerLawPow, 15.0}
    };

    vector<Summary> summaries;
    for (size_t i = 0; i < configs.size(); ++i) {
        const string csv = prefix + "_" + configs[i].label + ".csv";
        summaries.push_back(runSample(configs[i], pars, nEvents, csv, pdfSetName, seed + int(i) * 1000));
    }

    const string summaryPath = prefix + "_summary.csv";
    writeSummary(summaryPath, summaries, pars);
    cout << "Wrote summary: " << summaryPath << "\n";
    return 0;
}


