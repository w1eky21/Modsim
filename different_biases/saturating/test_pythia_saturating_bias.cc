// Compile from your Pythia8 root, for example:
// g++ -O2 -std=c++11 test_pythia_saturating_bias_fixed.cc \
//   -Ipythia8317/include \
//   -Lpythia8317/lib \
//   -lpythia8 \
//   -Wl,-rpath,pythia8317/lib \
//   -o saturating_bias
//
// Run examples:
// ./saturating_bias 100000 1.0 20.0 0.0 saturating_n1_p020.csv
// ./saturating_bias 100000 2.0 20.0 0.0 saturating_n2_p020.csv
// ./saturating_bias 100000 3.0 20.0 0.0 saturating_n3_p020.csv
//
// Bias function:
//   b(pTHat) = (1 + pTHat / p0)^n
//
// where:
//   n  = biasPow
//   p0 = biasRef

#include "Pythia8/Pythia.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

using namespace Pythia8;

// -----------------------------------------------------------------------------
// Custom importance-sampling bias.
//
// This replaces Pythia's built-in pure power-law bias:
//
//     b(pTHat) ~ pTHat^n
//
// with the softer/saturating power-law form:
//
//     b(pTHat) = (1 + pTHat / p0)^n
//
// This should still increase the number of high-pTHat events, but should avoid
// the very aggressive weight fluctuations seen for large pure-power biases.
// -----------------------------------------------------------------------------
class SaturatingPowerBias : public UserHooks {
public:
    SaturatingPowerBias(double powerIn, double p0In)
        : power(powerIn), p0(p0In) {}

    bool canBiasSelection() override {
        return true;
    }

    double biasSelectionBy(const SigmaProcess* sigmaProcessPtr,
                           const PhaseSpace* phaseSpacePtr,
                           bool inEvent) override {
        (void)sigmaProcessPtr;
        (void)inEvent;

        if (phaseSpacePtr == nullptr || p0 <= 0.0) {
            return 1.0;
        }

        const double pTHat = std::max(0.0, phaseSpacePtr->pTHat());
        double bias = std::pow(1.0 + pTHat / p0, power);

        // Important safety: Pythia must never receive a non-positive,
        // infinite, or NaN bias factor.
        if (!std::isfinite(bias) || bias <= 0.0) {
            bias = 1.0;
        }

        return bias;
    }

private:
    double power;
    double p0;
};

int main(int argc, char* argv[]) {

    int nEvents = 100000;
    double biasPow = 2.0;
    double biasRef = 20.0;   // p0 in GeV
    double pTHatMin = 0.0;
    double pTHatMax = 0.0;
    std::string outName = "phase_space_saturating_bias.csv";

    if (argc > 1) nEvents = std::atoi(argv[1]);
    if (argc > 2) biasPow = std::atof(argv[2]);
    if (argc > 3) biasRef = std::atof(argv[3]);
    if (argc > 4) pTHatMin = std::atof(argv[4]);
    if (argc > 5) outName = argv[5];
    if (argc > 6) pTHatMax = std::atof(argv[6]);

    if (nEvents <= 0) {
        std::cerr << "Error: nEvents must be positive.\n";
        return 1;
    }

    if (biasRef <= 0.0) {
        std::cerr << "Error: biasRef/p0 must be positive.\n";
        return 1;
    }

    Pythia pythia;

    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13000.");
    pythia.readString("HardQCD:all = on");

    pythia.settings.parm("PhaseSpace:pTHatMin", pTHatMin);
    if (pTHatMax > 0.0) {
        pythia.settings.parm("PhaseSpace:pTHatMax", pTHatMax);
    }

    // Use the custom UserHooks bias.
    //
    // Do NOT also enable:
    //   PhaseSpace:bias2Selection = on
    //
    // because that would apply Pythia's built-in pure power-law bias on top of
    // this custom saturating bias.
    std::shared_ptr<SaturatingPowerBias> biasHook =
        std::make_shared<SaturatingPowerBias>(biasPow, biasRef);
    pythia.setUserHooksPtr(biasHook);

    pythia.readString("Print:quiet = on");
    pythia.readString("Next:numberShowInfo = 0");
    pythia.readString("Next:numberShowProcess = 0");
    pythia.readString("Next:numberShowEvent = 0");

    if (!pythia.init()) {
        std::cerr << "Pythia init failed.\n";
        return 1;
    }

    std::ofstream out(outName.c_str());
    if (!out) {
        std::cerr << "Error: could not open output file " << outName << "\n";
        return 1;
    }

    out << std::setprecision(12);

    out << "event,"
        << "biasType,biasPow,biasRef,pTHatMin,pTHatMax,"
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

        if (!pythia.next()) {
            continue;
        }

        double maxPT = 0.0;
        int nFinal = 0;
        int nChargedFinal = 0;

        for (int i = 0; i < pythia.event.size(); ++i) {
            const Particle& p = pythia.event[i];

            if (!p.isFinal()) {
                continue;
            }

            ++nFinal;
            if (p.isCharged()) {
                ++nChargedFinal;
            }

            if (p.pT() > maxPT) {
                maxPT = p.pT();
            }
        }

        const double x1 = pythia.info.x1();
        const double x2 = pythia.info.x2();
        const double pTHat = pythia.info.pTHat();
        const double weight = pythia.info.weight();

        sumW += weight;
        sumW2 += weight * weight;

        if (pTHat > 10.0) ++n_pTHat_gt_10;
        if (pTHat > 20.0) ++n_pTHat_gt_20;
        if (pTHat > 50.0) ++n_pTHat_gt_50;
        if (pTHat > 100.0) ++n_pTHat_gt_100;

        out << accepted << ","
            << "saturating_power" << ","
            << biasPow << ","
            << biasRef << ","
            << pTHatMin << ","
            << pTHatMax << ","
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
    if (sumW2 > 0.0) {
        nEff = (sumW * sumW) / sumW2;
    }

    std::cout << "\nSaturating bias run finished\n";
    std::cout << "Output file       = " << outName << "\n";
    std::cout << "Generated tries   = " << nEvents << "\n";
    std::cout << "Accepted events   = " << accepted << "\n";
    std::cout << "biasType          = saturating_power\n";
    std::cout << "bias function     = (1 + pTHat / biasRef)^biasPow\n";
    std::cout << "biasPow           = " << biasPow << "\n";
    std::cout << "biasRef / p0      = " << biasRef << " GeV\n";
    std::cout << "pTHatMin          = " << pTHatMin << " GeV\n";
    if (pTHatMax > 0.0) {
        std::cout << "pTHatMax          = " << pTHatMax << " GeV\n";
    }
    std::cout << "sumW              = " << sumW << "\n";
    std::cout << "sumW2             = " << sumW2 << "\n";
    std::cout << "N_eff             = " << nEff << "\n";

    if (accepted > 0) {
        std::cout << "N_eff / N         = " << nEff / accepted << "\n";
        std::cout << "Frac pTHat > 10   = " << double(n_pTHat_gt_10) / accepted << "\n";
        std::cout << "Frac pTHat > 20   = " << double(n_pTHat_gt_20) / accepted << "\n";
        std::cout << "Frac pTHat > 50   = " << double(n_pTHat_gt_50) / accepted << "\n";
        std::cout << "Frac pTHat > 100  = " << double(n_pTHat_gt_100) / accepted << "\n\n";
    } else {
        std::cout << "No accepted events, so fractions are undefined.\n\n";
    }

    pythia.stat();

    return 0;
}
