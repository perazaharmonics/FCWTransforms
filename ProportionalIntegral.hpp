/* 
* *
* * Filename: ProportionalIntegral.hpp
* *
* * Description:
* *   Proportional-Integral (PI) Loop Filter for Costas Loop and Gardner Timing Recovery.
* *   - Simple discrete-time PI filter with configurable gains.
* *   - Used in CostasLoop and GardnerTimingRecovery classes.
* *
* * Author:
* *   JEP, J. Enrique Peraza
* * Organization:
* *   Trivium Solutions LLC, 9175 Guilford Rd, Suite 220, Columbia, MD 21046
* *
 */
#pragma once
#include <complex>
#include <cmath>


namespace sdr::mdm
{
    template<typename T=float>
    struct PILoopFilter                 // Proportional-Integral Loop Filter
    {
      T kp{T(0.05)};                    // Proportional gain
      T ki{T(0.001)};                   // Integral gain
      T acc{T(0)};                      // Integrator accumulator
      PILoopFilter (void)=default;
      inline void Assemble (
        T p,                            // Proportional gain
        T i)                            // Integrator gain
      {                                 // ~~~~~~~~~~ Assemble ~~~~~~~~~~ //
        kp=p;                           // Set proportional gain
        ki=i;                           // Set integrator gain
      }                                 // ~~~~~~~~~~ Assemble ~~~~~~~~~~ //
      inline void Reset (void)
      {                                 // ~~~~~~~~~~ Reset ~~~~~~~~~~ //
        acc=T(0);                       // Clear integrator accumulator
      }                                 // ~~~~~~~~~~ Reset ~~~~~~~~~~ //
      inline T Step (
        T err)                          // Time step
      {                                 // ~~~~~~~~~~ Step ~~~~~~~~~~ //
        acc+=ki*err;                    // Integrate error
        return kp*err+acc;              // Return PI output
      }                                 // ~~~~~~~~~~ Step ~~~~~~~~~~ //
    }; // PILoopFilter
}