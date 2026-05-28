// Compile from your Pythia8 root, for example:
// g++ -O2 -std=c++11 test_pythia_power_exp_bias.cc \
//   -Ipythia8317/include \
//   -Lpythia8317/lib \
//   -lpythia8 \
//   -Wl,-rpath,pythia8317/lib \
//   -o power_exp_bias
//
// Run like this:
// ./power_exp_bias 100000 4.0 15.0 500.0 0.0 powexp_n4_p015_cut500.csv
//
// Arguments:
//   1: nEvents
//   2: biasPow = n
//   3: biasRef = p0 [GeV]
//   4: biasCut = pCut [GeV]
//   5: pTHatMin [GeV]
//   6: output CSV filename
//
// Bias function implemented here:
//   b(pTHat) = max(1, (1 + pTHat / p0)^n * exp(-pTHat / pCut))
//
// The max(1, ...) floor means the bias boosts the useful intermediate/high-pTHat
// region, but never actively suppresses very high-pTHat events below the natural
// Pythia sampling rate.

#include "Pythia8/Pythia.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <string>
#include <memory>
#include <algorithm>

using namespace Pythia8;

// -----------------------------------------------------------------------------
// Custom importance-sampling bias:
//
//     b(pTHat) = max(1, (1 + pTHat / p0)^n * exp(-pTHat / pCut))
//
// Compared with a pure power law, the exponential factor damps the extreme tail.
// This should give more high-pTHat events than the soft saturating power law,
// but avoid the catastrophic weight fluctuations seen for very large pure powers.
// -----------------------------------------------------------------------------
class PowerExpBias : public UserHooks {
public:
    PowerExpBias(double powerIn, double p0In, double pCutIn)
        : power(powerIn), p0(p0In), pCut(pCutIn) {}

    bool canBiasSelection() override { return true; }

    double biasSelectionBy(const SigmaProcess* sigmaProcessPtr,
                           const PhaseSpace* phaseSpacePtr,
                           bool inEvent) override {

        (void)sigmaProcessPtr;
        (void)inEvent;

        double pTHat = std::max(0.0, phaseSpacePtr->pTHat());

        // Safety: invalid settings mean no biasing.
        if (p0 <= 0.0 || pCut <= 0.0) {
            selBias = 1.0;
            return selBias;
        }

        double powerPart = std::pow(1.0 + pTHat / p0, power);
        double expPart   = std::exp(-pTHat / pCut);

        selBias = powerPart * expPart;

        // Do not allow the bias to become an anti-bias. This keeps the function
        // as a boost in the target region, rather than suppressing the far tail.
        if (selBias < 1.0) selBias = 1.0;

        // Safety: never return zero, negative, NaN, or infinity.
        if (!std::isfinite(selBias) || selBias <= 0.0) selBias = 1.0;

        return selBias;
    }

private:
    double power;
    double p0;
    double pCut;
};

int main(int argc, char* argv[]) {

    int nEvents = 100000;
    double biasPow = 4.0;
    double biasRef = 15.0;    // p0 in GeV
    double biasCut = 500.0;   // pCut in GeV
    double pTHatMin = 0.0;
    double pTHatMax = 0.0;
    std::string outName = "phase_space_power_exp_bias.csv";

    if (argc > 1) nEvents  = atoi(argv[1]);
    if (argc > 2) biasPow  = atof(argv[2]);
    if (argc > 3) biasRef  = atof(argv[3]);
    if (argc > 4) biasCut  = atof(argv[4]);
    if (argc > 5) pTHatMin = atof(argv[5]);
    if (argc > 6) outName  = argv[6];

    Pythia pythia;

    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13000.");
    pythia.readString("HardQCD:all = on");

    pythia.settings.parm("PhaseSpace:pTHatMin", pTHatMin);
    if (pTHatMax > 0.0) {
        pythia.settings.parm("PhaseSpace:pTHatMax", pTHatMax);
    }

    // Custom damped power-law bias. Do NOT also switch on
    // PhaseSpace:bias2Selection, because that would apply an extra built-in bias.
    std::shared_ptr<PowerExpBias> biasHook =
        std::make_shared<PowerExpBias>(biasPow, biasRef, biasCut);
    pythia.setUserHooksPtr(biasHook);

    pythia.readString("Print:quiet = on");
    pythia.readString("Next:numberShowInfo = 0");
    pythia.readString("Next:numberShowProcess = 0");
    pythia.readString("Next:numberShowEvent = 0");

    if (!pythia.init()) {
        std::cerr << "Pythia init failed.\n";
        return 1;
    }

    std::ofstream out(outName);
    if (!out) {
        std::cerr << "Could not open output file: " << outName << "\n";
        return 1;
    }

    out << std::setprecision(12);

    out << "event,"
        << "biasType,biasPow,biasRef,biasCut,pTHatMin,"
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
            << "power_exp" << ","
            << biasPow << ","
            << biasRef << ","
            << biasCut << ","
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

    std::cout << "\nPower-law with exponential damping bias run finished\n";
    std::cout << "Output file       = " << outName << "\n";
    std::cout << "Generated tries   = " << nEvents << "\n";
    std::cout << "Accepted events   = " << accepted << "\n";
    std::cout << "biasType          = power_exp\n";
    std::cout << "bias function     = max(1, (1 + pTHat / biasRef)^biasPow * exp(-pTHat / biasCut))\n";
    std::cout << "biasPow           = " << biasPow << "\n";
    std::cout << "biasRef / p0      = " << biasRef << " GeV\n";
    std::cout << "biasCut / pCut    = " << biasCut << " GeV\n";
    std::cout << "pTHatMin          = " << pTHatMin << " GeV\n";
    std::cout << "sumW              = " << sumW << "\n";
    std::cout << "sumW2             = " << sumW2 << "\n";
    std::cout << "N_eff             = " << nEff << "\n";
    if (accepted > 0) {
        std::cout << "N_eff / N         = " << nEff / accepted << "\n";
        std::cout << "Frac pTHat > 10   = " << double(n_pTHat_gt_10) / accepted << "\n";
        std::cout << "Frac pTHat > 20   = " << double(n_pTHat_gt_20) / accepted << "\n";
        std::cout << "Frac pTHat > 50   = " << double(n_pTHat_gt_50) / accepted << "\n";
        std::cout << "Frac pTHat > 100  = " << double(n_pTHat_gt_100) / accepted << "\n\n";
    }

    pythia.stat();

    return 0;
}

