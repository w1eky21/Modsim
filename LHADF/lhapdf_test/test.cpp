#include "LHAPDF/LHAPDF.h"
#include <iostream>

int main() {
    // Replace with the exact name of your PDF set
    LHAPDF::PDF* pdf = LHAPDF::mkPDF("NNPDF31_lo_as_0118", 0);

    double x = 0.1;
    double Q = 91.2;

    std::cout << "gluon PDF at x=" << x << " Q=" << Q << " : "
              << pdf->xfxQ(21, x, Q) << std::endl;

    delete pdf;
    return 0;
}
