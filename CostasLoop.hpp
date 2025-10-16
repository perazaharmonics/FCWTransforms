/* 
* *
* * Filename: CostasLoop.hpp
* *
* * Description:
* *    Costas Loop class for carrier phase recovery (BPSK/QPSK/SQPSK) using a Numerically Controlled Oscillator (NCO).
* *    Supports decision-directed phase detection and loop PI loop filtering.
* *    PI loop filter drives NCO frequency via rad/s mapping.
* *
* * Author:
* *   JEP, J. Enrique Peraza
* *
 */
#pragma once
#include <complex>
#include <cmath>
#include "../dsp/NCO.hpp"
#include "ProportionalIntegral.hpp"
namespace sdr
{
  namespace mdm
  {
    template<typename T=float>
    class CostasLoop
    {
      public:
        using cplx=std::complex<T>;     // Complex type alias
        CostasLoop (void)=default;      // Default constructor
        ~CostasLoop (void)=default;     // Default destructor
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Prepare the Costas Loop for operation
        // fs: Sample rate (Hz)
        // o: Modulation order: 1=BPSK, 2=QPSK/SQPSK
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        inline void Assemble (
          T sr,                         // Sample rate (Hz)
          int32_t o)                    // Modulation order: 1=BPSK, 2=QPSK/SQPSK
        {                               // ~~~~~~~~~~ Assemble ~~~~~~~~~~ //
           if (sr<=T(0))                // Bad sample rate?
             sr=T(1);                   // Default to 1 Hz
           fs=sr;                       // Set sample rate
           if (o!=1&&o!=2)              // Bad order?
             o=2;                       // Default to QPSK
           ord=o;                       // Set modulation order
           w0=T(0);                     // Start with zero frequency offset
           nco.SetSampleRate(static_cast<double>(fs));// Set NCO sample rate
           nco.SetFrequency(0.0);       // Start NCO at 0 Hz
           nco.Reset();                 // Reset NCO phase to 0
           plf.Reset();                 // Reset loop filter
        }                               // ~~~~~~~~~~ Assemble ~~~~~~~~~~ //
        inline void SetLoopGains (
          T kp,                         // Proportional gain
          T ki)                         // Integrator gain
        {                               // ~~~~~~~~~~ SetLoopGains ~~~~~~~~~~ //
          plf.Assemble(kp,ki);          // Set loop filter gains
        }                               // ~~~~~~~~~~ SetLoopGains ~~~~~~~~~~ //
        inline cplx DownConvert (
          cplx f)                       // The frequency to downconvert (mix) with
        {                               // ~~~~~~~~~~ DownConvert ~~~~~~~~~~ //
          cplx v=nco.Tick();            // Get current NCO phasor and advance
          return f*v;                   // Downconvert input by NCO phasor
        }                               // ~~~~~~~~~~ DownConvert ~~~~~~~~~~ //
        inline void Update (
          cplx y)                       // Input sample after downconversion
        {                               // ~~~~~~~~~~ Update ~~~~~~~~~~ //
          T err{T(0)};                  // Phase error
          if (ord==1)                   // BPSK Modulation order?
          {                             // Yes
            T i=y.real();               // Get the real (In-Phase) part
            T q=y.imag();               // Get the imaginary (In-Quadrature) part
            T s=(i>=T(0))?T(1):T(-1);   // Decision device
            err=q*s;                    // Phase error detector
          }                             // Done with BPSK
          else                          // Else it is QPSK/SQPSK
          {                             //
            T i=y.real();               // Get the real (In-Phase) part
            T q=y.imag();               // Get the imaginary (In-Quadrature) part
            T si=(i>=T(0))?T(1):T(-1);  // Decision device I
            T sq=(q>=T(0))?T(1):T(-1);  // Decision device Q
            err=i*sq-q*si;              // Phase error detector
          }                             // Done with QPSK/SQPSK
          T dw=plf.Step(err);           // PI loop filter step
          w0+=dw;                       // Update NCO frequency (rad/s)
          double f=static_cast<double>(w0/(2.0*M_PI));// Convert to Hz
          nco.SetFrequency(f);          // Update NCO frequency
        }                               // ~~~~~~~~~~ Update ~~~~~~~~~~ //
      private:
        T fs{T(1)};                     // Sampling frequency
        T w0{T(0)};                     // NCO angular frequency (rad/s)
        PILoopFilter<T> plf;            // PI loop filter
        int32_t ord{2};                 // Mode: 1=BPSK, 2=QPSK/SQPSK
        ::sdr::sig::NCO<T> nco;         // NCO instance
    };
  }
}