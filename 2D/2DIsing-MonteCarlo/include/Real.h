#ifndef PSIMAG_Real_H
#define PSIMAG_Real_H

//
//          10        20        30        40        50        60
// 1 234567890123456789012345678901234567890123456789012345678901234
// 3.1415926535897932384626433832795028841971693993751058209749445923

/** \file Real.h
 *  \author Thomas C. Schulthess
 *  \brief typedefs and macros for floating-point numbers
 *
 *  \warning The implementation of psimag supports only double literals.
 *  This means that defining Real to be long double precision may result in 
 *  degrading to double precision. No long double literals is supported at
 *  this point. -- see ISO-C++ standard 14882:1998(E), page 18.
 *
 *  \warning defining Real to be float will not degrade the intented precision.
 */

/** \namespace psimag
 */
namespace  psimag {

/** \def SINGLE_PRECISION
 *  \brief To force single precision, else double precision is used
 */

/** \def PI
 *  \brief A literal expression for the ratio of perimeter to diamter
 */

#ifdef SINGLE_PRECISION

typedef float Real;
#undef PI
#define PI 3.14159265358979323846F

#endif /* ifdef SINGLE_PRECISION */
#ifndef SINGLE_PRECISION

typedef double Real;
#undef PI
#define PI 3.141592653589793238462643383279502884197

#endif /* ifndef SINGLE_PRECISION */

} /* namespace psimage */

#endif /* PSIMAG_Real_H */
