// Compile from your Pythia8 root, for example:
// g++ -O2 -std=c++11 test_pythia_lognormal_bump_bias.cc \
//   -Ipythia8317/include \
//   -Lpythia8317/lib \
//   -lpythia8 \
//   -Wl,-rpath,pythia8317/lib \
//   -o lognormal_bump_bias
//
// Run like this:
// ./lognormal_bump_bias 100000 100.0 100.0 0.8 0.0 bump_A100_pc100_sig08.csv
//
// Arguments:
//   1: nEvents
//   2: amp = A                 strength of the bump
//   3: pCenter = p_c [GeV]     pTHat value where the bump is centered
//   4: sigma                   width in log(pTHat)
//   5: pTHatMin [GeV]
//   6: output CSV filename
//
// Bias function implemented here:
//   b(pTHat) = 1 + A * exp( - (log((pTHat + eps)/pCenter))^2 / (2 sigma^2) )
//
// This targets a finite high-pTHat window instead of endlessly increasing the
// bias into the extreme tail. That should give more high-pTHat events while
// avoiding the worst weight collapse from very strong pure power laws.

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
// Custom lognormal bump importance-sampling bias:
//
//     b(pTHat) = 1 + A * exp( - (log((pTHat + eps)/pCenter))^2 / (2 sigma^2) )
//
// The bias is largest near pCenter and smoothly returns to 1 at very low and
// very high pTHat. This is a targeted rare-event bias rather than a bias that
// keeps growing forever.
// -----------------------------------------------------------------------------
class LognormalBumpBias : public UserHooks {
public:
    LognormalBumpBias(double ampIn, double pCenterIn, double sigmaIn)
        : amp(ampIn), pCenter(pCenterIn), sigma(sigmaIn) {}

    bool canBiasSelection() override { return true; }

    double biasSelectionBy(const SigmaProcess* sigmaProcessPtr,
                           const PhaseSpace* phaseSpacePtr,
                           bool inEvent) override {

        (void)sigmaProcessPtr;
        (void)inEvent;

        double pTHat = std::max(0.0, phaseSpacePtr->pTHat());

        // Invalid settings mean no biasing.
        if (amp <= 0.0 || pCenter <= 0.0 || sigma <= 0.0) {
            selBias = 1.0;
            return selBias;
        }

        // Small epsilon avoids log(0). It does not affect the high-pTHat region.
        const double eps = 1.0e-12;
        double logRatio = std::log((pTHat + eps) / pCenter);
        double exponent = -0.5 * (logRatio * logRatio) / (sigma * sigma);

        selBias = 1.0 + amp * std::exp(exponent);

        // Safety: never return zero, negative, NaN, or infinity.
        if (!std::isfinite(selBias) || selBias <= 0.0) selBias = 1.0;

        return selBias;
    }

private:
    double amp;
    double pCenter;
    double sigma;
};

int main(int argc, char* argv[]) {

    int nEvents = 100000;
    double amp = 100.0;
    double pCenter = 100.0;  // GeV
    double sigma = 0.8;      // width in log(pTHat)
    double pTHatMin = 0.0;
    double pTHatMax = 0.0;
    std::string outName = "phase_space_lognormal_bump_bias.csv";

    if (argc > 1) nEvents  = atoi(argv[1]);
    if (argc > 2) amp      = atof(argv[2]);
    if (argc > 3) pCenter  = atof(argv[3]);
    if (argc > 4) sigma    = atof(argv[4]);
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

    // Custom lognormal bump bias. Do NOT also switch on PhaseSpace:bias2Selection,
    // because that would apply an extra built-in bias.
    std::shared_ptr<LognormalBumpBias> biasHook =
        std::make_shared<LognormalBumpBias>(amp, pCenter, sigma);
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
        << "biasType,biasAmp,biasCenter,biasSigma,pTHatMin,"
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
            << "lognormal_bump" << ","
            << amp << ","
            << pCenter << ","
            << sigma << ","
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

    std::cout << "\nLognormal bump bias run finished\n";
    std::cout << "Output file       = " << outName << "\n";
    std::cout << "Generated tries   = " << nEvents << "\n";
    std::cout << "Accepted events   = " << accepted << "\n";
    std::cout << "biasType          = lognormal_bump\n";
    std::cout << "bias function     = 1 + A * exp(-0.5 * [log((pTHat+eps)/pCenter)/sigma]^2)\n";
    std::cout << "biasAmp / A       = " << amp << "\n";
    std::cout << "biasCenter / pc   = " << pCenter << " GeV\n";
    std::cout << "biasSigma         = " << sigma << "\n";
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

