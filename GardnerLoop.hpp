/* 
* *
* * Filename: GardnerLoop.hpp
* *
* * Description:
* *   Gardner Timing Error Detector and Loop Filter for symbol timing recovery.
* *   - Operates on complex baseband samples (IQ).
* *   - Uses Farrow FIR interpolator for fractional delay (FarrowFIR.hpp),with selectable order (up to 5).
* *   - TED: e=Re{x(nT+T/2)-x(nT-T/2)}*conj{x(nT)}
* *   - PI loop adjusts nominal symbol period (omega) and fractional phase (mu).
* *
* *   - Output: y(n)=x(nT + mu) * ej(omega * nT)
* *   - omega: nominal symbol period in samples (e.g.,10.0 for 10 samples/symbol)
* *   - mu: fractional timing offset (0 <= mu < 1)
* *   - Kp: proportional gain of the loop filter
* *   - Ki: integral gain of the loop filter
* *
* *  Author:
* *   JEP,J. Enrique Peraza
* *
 */
#pragma once
#include <complex>
#include <cmath>
#include "ProportionalIntegral.hpp"
// Bring in native DelayLine and FarrowInterpolator definitions
#include "DelayLine.hpp"
#include "FarrowFIR.hpp"
namespace sdr::mdm
{
  template<typename T=float,
    size_t MaxLen=4096,
    size_t Order=3>
    class GardnerLoop
    {
      public:
        using cplx=std::complex<T>;     // Complex type alias
        GardnerLoop (void)
        {
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Create our DSP engine components
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          fir=new FarrowInterpolatorCplx<T,MaxLen,Order>(); 
          dl=new DelayLine<cplx,MaxLen>(); // Complex delay line
          pfl=new PILoopFilter<T>();     // Create PI loop filter
        }
        ~GardnerLoop (void)
        {
          if (fir!=nullptr)
          {
            delete fir;
            fir=nullptr;
          }
          if (dl!=nullptr)
          {
            delete dl;
            dl=nullptr;
          }
          if (pfl!=nullptr)
          {
            delete pfl;
            pfl=nullptr;
          }
        }
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Assemble the Gardner Timing Recovery Loop
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        inline void Assemble (
          T sps)                        // Samples per symbol (nominal) (2 for Gardner TED
        {                               // ~~~~~~~~~~ Assemble ~~~~~~~~~~ //
          if (sps<=T(0))                // Bad input?
            sps=T(2);                   // Default to 2 samples/symbol
          this->sps=sps;                // Remember samples per symbol
          mu=T(0);                      // Default fractional timing offset
          w=this->sps;                  // Initialize running sps estimate
          nsamp=0;                     // Reset written sample count
          pfl->Reset();                 // Reset loop filter
          dl->Clear();                  // Clear complex delay line
          fir->SetOrder(Order);         // Set Farrow FIR order
          fir->SetMu(mu);               // Set initial fractional delay
          idel=Order;                   // Integer base delay equals interpolation order
        }                               // ~~~~~~~~~~ Assemble ~~~~~~~~~~ //
        inline void SetLoopGains (
          T kp,                        // Proportional gain
          T ki)                         // Integral gain
        {                               // ~~~~~~~~~~ SetLoopGains ~~~~~~~~~~ //
          pfl->Assemble(kp,ki);         // Set loop filter gains
        }                               // ~~~~~~~~~~ SetLoopGains ~~~~~~~~~~ //
        inline void SetTEDGain (
          T g)                          // Gain factor for TED
        {                               // ~~~~~~~~~~ SetTEDGain ~~~~~~~~~~ //
          this->g=g;                    // Set TED gain
        }                               // ~~~~~~~~~~ SetTEDGain ~~~~~~~~~~ //
        inline void SetSamplesPerSymbol (
          T sps)                        // Set nominal samples per symbol
        {                               // ~~~~~~~~~~ SetSamplesPerSymbol ~~~~~~~~~~ //
          if (sps<=T(0))                // Bad input?
            sps=T(2);                   // Default to 2 samples/symbol
          this->sps=sps;                // Remember samples per symbol
        }                               // ~~~~~~~~~~ SetSamplesPerSymbol ~~~~~~~~~~ //
        inline T GetSamplesPerSymbol (void) const noexcept
        {                               // ~~~~~~~~~~ GetSamplesPerSymbol ~~~~~~~~~~ //
          return sps;                   // Return current samples per symbol estimate
        }                               // ~~~~~~~~~~ GetSamplesPerSymbol ~~~~~~~~~~ //
        inline T GetSamplesPerSymbolEstimate (void) const noexcept
        {
          return w;                     // Return current samples per symbol estimate
        }
        inline T GetMu (void) const noexcept
        {                               // ~~~~~~~~~~ GetMu ~~~~~~~~~~ //
          return mu;                    // Return current fractional timing offset
        }                               // ~~~~~~~~~~ GetMu ~~~~~~~~~~ //
        inline void Reset (void)
        {                               // ~~~~~~~~~~ Reset ~~~~~~~~~~ //
          mu=T(0);                      // Clear fractional timing offset
          w=sps;                        // Reset samples per symbol estimate
          nsamp=0;                     // Reset sample counter
          pfl->Reset();                 // Reset loop filter
          dl->Clear();                  // Clear complex delay line
        }                               // ~~~~~~~~~~ Reset ~~~~~~~~~~ //

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Tick one input sample,possible emit a symbol at correct time step
        // Return true when a decision sample is produced in y
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        inline bool Tick (
          cplx x,                      // Input sample
          cplx* const y,               // Output sample (if available)
          bool* const strobe)           // Strobe signal (if output is valid)
        {                               // ~~~~~~~~~~~ Tick ~~~~~~~~~~~~~~~ //
           if (y==nullptr||strobe==nullptr)// Bad input?
             return false;              // Yes,return error
            dl->Write(x);               // Write complex sample to delay line
            ++nsamp;                    // Count samples written since reset
            if (nsamp<static_cast<size_t>(idel+2)) // Do we have enough history to Farrow FIR interpolate?
            {                           // Not yet,return false.
              *strobe=false;            // Not enough samples yet
              return *strobe;           // Return false
            }
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Advance fractional timing (symbol phase)
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            if (w <= T(0)) w=sps;     // Safety
            mu += T(1) / w;             // Advance by 1 sample relative to sps
            *strobe=false;              // Default no output sample
            if (mu>=static_cast<T>(1))  // Next symbol time?
            {                           // Yes,interpolate adjacent samples from delay line snapshot
              cplx xm,x0,xp;            // Samples at nT+T/2 (newer),nT (center),nT-T/2 (older)
              size_t D=idel;          // Base integer delay for Farrow FIR
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              // Older half-symbol: xp at (D,mu=+0.5)
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              fir->SetMu(static_cast<T>(0.5)); // Set fractional delay to +0.5
              fir->Process(*dl,D,&xp);   // Interpolate older half-symbol
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              // Center: x0 at (D,mu=0.0)
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              fir->SetMu(static_cast<T>(0.0)); // Set fractional delay to 0.0
              fir->Process(*dl,D,&x0);
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              // Newer half-symbol: xm at (D-1,mu=+0.5)
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              if (D>0)                  // Valid integer delay?
              {                         // Yes
                fir->SetMu(static_cast<T>(0.5)); // Set fractional delay to +0.5
                fir->Process(*dl,D-1,&xm); // Interpolate newer half-symbol
              }                         // Done with newer half-symbol 
              else                      // Invalid integer delay?
              {                         // Yes
                xm=x0;                  // Just duplicate center sample
              }                         //
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              // Timing Error Detector (TED)
              // e=Re{[x(nT+T/2)-x(nT-T/2)]*conj{x(nT)}}
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              cplx err=(xm-xp)*std::conj(x0);// Compute timing error
              T e=err.real();           // TED error signal
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              // Loop filter step
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              T dmu=pfl->Step(g*e);     // Get loop filter output
              w+=dmu;                   // Update samples per symbol estimate
              if (w<static_cast<T>(1.2))// Limit minimum sps
                w=static_cast<T>(1.2);  // Limit minimum sps
              if (w>static_cast<T>(16)) // Limit maximum sps
                w=static_cast<T>(16);   // Limit maximum sps
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              // Emit symbol at mu=0
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              mu-=static_cast<T>(1);    // Wrap around mu
              // Emit a symbol each wrap
              *y=x0;                    // Output the interpolated sample
              *strobe=true;             // Indicate output sample is valid
            }                           // Done with symbol time
            return *strobe;             // Return true if strobe is asserted
        }                               // ~~~~~~~~~~~ Tick ~~~~~~~~~~~~~~~ //
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Set loop gains from a normalized timing-loop bandwidth.
        // BN sym: normalized loop noise bandwidth (fraction of symbol rate).
        //   - e.g., BN=0.01 -> 1% of Rs
        //   - BN=0.00f -> 0.05% of Rs (narrower, slower integration).
        // zeta: damping factor (0.5 to 1.0 typical, 0.707 default).
        //
        // NOTE: 
        //   Loop operates at one update per symbol (when mu wraps).
        //   so we design in "symbols" domain with Ts=1.
        //   Effective loop dynamics also scale with TED gain g,
        //   by cranking g up, we make the loop effectively wider.
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        inline void SetLoopBandwidth (
          T Bn,                         // Normalized loop noise bandwidth (fraction of symbol rate)
          T zeta=T(0.7071))             // Damping factor (default=0.7071, [0..1])
        {                               // ~~~~~~~~~~ SetLoopBandwidth ~~~~~~~~~~~ //
          if (Bn<=T(0))                 // Bad loop bandwidth?
            Bn=T(0.01);                 // Failsafe default to 1% of symbol rate
          bn=Bn;                        // Set loop bandwidth
          if (zeta<=T(0))               // Bad damping factor?
            zeta=T(0.7071);             // Default to 0.7071
          z=zeta;                       // Set damping factor
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Compute time constant and angular frequency
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          const T ts=T(1);              // Sample period in symbols domain
          const T wn=(T(4)*zeta*Bn)/(zeta+T(1)/T(4)*zeta);// Natural frequency
          const T wnts=wn*ts;           // Normalized natural frequency
          const T d=T(1)+T(2)*zeta*wnts+wnts+wnts;// Denominator
          const T kp=(T(4)*zeta*wnts)/d;// Proportional gain (Kp)
          const T ki=(T(4)*wnts*wnts)/d;// Integral gain (Ki)
          pfl->Assemble(kp,ki);         // Set loop filter gains
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
          return pfl->kp;               // Return proportional gain
        }                               // ~~~~~~~~~~ GetProportionalGain ~~~~~~~~~~ //
        inline T GetIntegralGain (void) const
        {                               // ~~~~~~~~~~ GetIntegralGain ~~~~~~~~~~ //
          return pfl->ki;               // Return integral gain
        }                               // ~~~~~~~~~~ GetIntegralGain ~~~~~~~~~~ //
      private:
        T sps{T(2)};                    // Samples per symbol (nominal) (2 for Gardner TED)
        T mu{T(0)};                     // Fractional timing offset (0 <= mu < 1)
        T w{T(2)};                      // Running sample per symbol estimate
        T bn{T(0.075)};                  // Noise bandwidth (fraction of symbol rate)
        T z{T(0.7071)};                 // Damping factor
        T g{T(0.1)};                    // Gain factor for TED
        ::sdr::mdm::PILoopFilter<T>* pfl{};// Proportional-Integral Loop Filter
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Our Farrow FIR interpolator and deinterpolator object
        // for fractional delay processing.
        // Very stable and effictient when operating near mu=(0.35-0.45) for order=3
        // and when operating near mu=(0.28-0.32) for order=5.
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        ::sdr::mdm::FarrowInterpolatorCplx<T,MaxLen,Order>* fir{};  // Complex Farrow FIR interpolator
        ::sdr::mdm::DelayLine<cplx,MaxLen>* dl{}; // Complex delay line for input samples
        size_t idel{Order+1};           // Current delay line index
        size_t nsamp{0};                // Samples written since last reset
        
    };
  
}