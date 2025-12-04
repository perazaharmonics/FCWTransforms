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
* * Organization:
* *   Trivium Solutions LLC, 9175 Guilford Rd, Suite 220, Columbia, MD 21046
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
        inline void SetOQPSKMode(bool en)
        {                               // ~~~~~~~~~~ SetOQPSKMode ~~~~~~~~~~ //
          oqpsk=en;                     // Enable/disable OQPSK/SQPSK axis-aware detector
          ord=2;                        // Force modulation order to QPSK/SQPSK
        }                               // ~~~~~~~~~~ SetOQPSKMode ~~~~~~~~~~ //
        inline cplx DownConvert (
          cplx f)                       // The frequency to downconvert (mix) with
        {                               // ~~~~~~~~~~ DownConvert ~~~~~~~~~~ //
          cplx v=nco.Tick();            // Get current NCO phasor and advance
          return f*std::conj(v);        // Downconvert input sample
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
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Axis-aware detector for OQPSK/SQPSK
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            if (oqpsk)                  // Axis-aware detection enabled?
            {                           // Yes
              if (std::abs(i)>=std::abs(q)) // I-axis dominant?
              {                         // Yes
                // ~~~~~~~~~~~~~~~~~~~~ //
                // I-active (treat like BPSK on I)
                // ~~~~~~~~~~~~~~~~~~~~ //
                T s=(i>=T(0))?T(1):T(-1);// Decision device I
                err=q*s;                // Phase error detector
              }                         // Done with I-active   
              else                      // Q-axis dominant?
              {                         // Yes
                // ~~~~~~~~~~~~~~~~~~~~ //
                // Q-active (treat like BPSK on Q)
                // ~~~~~~~~~~~~~~~~~~~~ //
                T s=(q>=T(0))?T(1):T(-1);// Decision device Q
                err=-i*s;               // Phase error detector
              }                         // Done with Q-active
            }                           // Done with axis-aware detection
            else                        // Standard QPSK/SQPSK detection
            {                           // Yes
              T si=(i>=T(0))?T(1):T(-1);// Decision device I
              T sq=(q>=T(0))?T(1):T(-1);// Decision device Q
              err=i*sq-q*si;            // Standard QPSK Costas phase detector
            }                           // Done with standard detection
          }                             // Done with QPSK/SQPSK
          T dw=plf.Step(err);           // PI loop filter step
          w0+=dw;                       // Update NCO frequency (rad/s)
          double f=static_cast<double>(w0/(2.0*M_PI));// Convert to Hz
          nco.SetFrequency(f);          // Update NCO frequency
        }                               // ~~~~~~~~~~ Update ~~~~~~~~~~ //
        inline T GetNCOFrequency (void) const
        {                               // ~~~~~~~~~~ GetNCOFrequency ~~~~~~~~~~ //
          return static_cast<T>(nco.GetFrequency());// Return NCO frequency in Hz
        }                               // ~~~~~~~~~~ GetNCOFrequency ~~~~~~~~~~ //
        inline T GetNCOPhase (void) const
        {                               // ~~~~~~~~~~ GetNCOPhase ~~~~~~~~~~ //
          return nco.GetPhase();        // Return NCO phase in radians
        }                               // ~~~~~~~~~~ GetNCOPhase ~~~~~~~~~~ //
        inline T GetLoopFrequency (void) const
        {                               // ~~~~~~~~~~ GetLoopFrequency ~~~~~~~~~~ //
          return w0/(2.0*M_PI);        // Return loop frequency in Hz
        }                               // ~~~~~~~~~~ GetLoopFrequency ~~~~~~~~~~ //
        inline size_t GetModulationOrder (void) const
        {                               // ~~~~~~~~~~ GetModulationOrder ~~~~~~~~~~ //
          return static_cast<size_t>(ord);// Return modulation order
        }                               // ~~~~~~~~~~ GetModulationOrder ~~~~~~~~~~ //
        inline void Reset (void)
        {                               // ~~~~~~~~~~ Reset ~~~~~~~~~~ //
          w0=T(0);                      // Reset loop frequency to 0 rad/s
          nco.Reset();                  // Reset NCO phase to 0
          plf.Reset();                  // Reset loop filter
        }                               // ~~~~~~~~~~ Reset ~~~~~~~~~~ //
        inline void SetLoopBandwidth (
          T Bn,                         // Loop noise bandwidth (Hz)
          T zeta=T(0.7071))             // Damping factor (default=0.7071, [0..1])
        {                               // ~~~~~~~~~~ SetLoopBandwidth ~~~~~~~~~~~ //
          if (Bn<=T(0))                 // Bad loop bandwidth?
            Bn=fs*T(1e-3);              // Default to 0.1% of sample rate
          bn=Bn;                        // Set loop bandwidth
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Compute time constant and angular frequency
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          const T ts=T(1)/fs;           // Sample period
          const T wn=(T(4)*zeta*bn)/(zeta+T(1)/(T(4)*zeta));// Natural frequency
          if (zeta<=T(0))               // Bad damping factor?
            zeta=T(0.7071);             // Default to 0.7071
          z=zeta;                       // Set damping factor
          const T wnts=wn*ts;           // Normalized natural frequency
          const T d=T(1)+T(2)*zeta*wnts+wnts*wnts;// Denominator
          const T kp=(T(4)*zeta*wnts)/d;// Proportional gain (Kp)
          const T ki=(T(4)*wnts*wnts)/d;// Integral gain (Ki)
          plf.Assemble(kp,ki);          // Set loop filter gains
        }                               // ~~~~~~~~~~ SetLoopBandwidth ~~~~~~~~~~~ //
        inline void SetDampingFactor (T zeta)
        {                               // ~~~~~~~~~~ SetDampingFactor ~~~~~~~~~~~ //
          if (zeta<=T(0))               // Bad damping factor?
            zeta=z;                     // Prevent bad damping factor
          z=zeta;                       // Set new damping factor
          SetLoopBandwidth(bn,z);       // Recompute loop gains with new damping factor
        }                               // ~~~~~~~~~~ SetDampingFactor ~~~~~~~~~~~ //
        inline T GetProportionalGain (void) const
        {                               // ~~~~~~~~~~ GetProportionalGain ~~~~~~~~~~ //
          return plf.kp;                // Return proportional gain
        }                               // ~~~~~~~~~~ GetProportionalGain ~~~~~~~~~~ //
        inline T GetIntegralGain (void) const
        {                               // ~~~~~~~~~~ GetIntegralGain ~~~~~~~~~~ //
          return plf.ki;                // Return integral gain
        }                               // ~~~~~~~~~~ GetIntegralGain ~~~~~~~~~~ //
      private:
        T fs{T(1)};                     // Sampling frequency
        T w0{T(0)};                     // NCO angular frequency (rad/s)
        T bn{T(0.01)};                  // Loop bandwidth (rad/s)
        T z{T(0.7071)};                 // Damping factor
        PILoopFilter<T> plf;            // PI loop filter
        int32_t ord{1};                 // Mode: 1=BPSK, 2=QPSK/SQPSK
        ::sdr::sig::NCO<T> nco;         // NCO instance
        bool oqpsk{false};              // Use OQPSK/SQPSK axis-aware phase detector
    };
  }
}
