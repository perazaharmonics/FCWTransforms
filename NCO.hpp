/*
 * *
 * * Filename: NCO.hpp
 * *
 * * Description:
 * *  Numerically Controlled Oscillator (NCO) for complex (IQ) (phasor and real cos/sin) generation.
 * *
 * * Author:
 * *  JEP, J. Enrique Peraza
 * *
 * *
 */
#pragma once
#include <cstdint>
#include <cstddef>
#include <complex>
#include <vector>
#include "../SDRTypes.hpp"

namespace sdr::sig
{
    template<typename T=float>
    class NCO
    {
      public:
        using cplx=std::complex<T>;     // Complex type
        NCO (void)=default;             // Default constructor
        ~NCO (void)=default;            // Default destructor
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Configuration setters/getters
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        inline void SetSampleRate (double sr)
        {                               // ~~~~~~~~~~ SetSampleRate ~~~~~~~~~~ //
          if (sr<0.0)                   // Invalid?
            sr=1.0;                     // Default to 1 Hz
          fs=sr;                        // Set sample rate
          UpdatePhaseInc();             // Update phase increment
        }                               // ~~~~~~~~~~ SetSampleRate ~~~~~~~~~~ //
        inline double GetSampleRate (void) const {return fs;}// Get sample rate
        inline void SetFrequency (double f)
        {                               // ~~~~~~~~~~ SetFrequency ~~~~~~~~~~ //
           if (f<-fs/2.0)               // Below -fs/2?
             f=-fs/2.0;                 // Clamp to -fs/2
           if (f>fs/2.0)                // Above fs/2?
             f=fs/2.0;                  // Clamp to fs/2
           f0=f;                        // Set frequency
           UpdatePhaseInc();            // Update phase increment
        }                               // ~~~~~~~~~~ SetFrequency ~~~~~~~~~~ //
        inline double GetFrequency (void) const {return f0;}// Get frequency
        inline void SetPhase (T p)      // Set the current phase in radians
        {                               // ~~~~~~~~~~ SetPhase ~~~~~~~~~~ //
          ph=Wrap(p);                   // Wrap phase to [-pi,pi) and set
        }                               // ~~~~~~~~~~ SetPhase ~~~~~~~~~~ //
        inline void Reset (void)        // Reset phase to 0
        {                               // ~~~~~~~~~~ Reset ~~~~~~~~~~ //
          ph=T(0);                      // Zero phase
        }                               // ~~~~~~~~~~ Reset ~~~~~~~~~~ //
        inline void ResetPhasor (void)
        {                               // ~~~~~~~~~~ ResetPhasor ~~~~~~~~~~ //
          vr=T(1);                      // Real part = 1
          vi=T(0);                      // Imaginary part = 0
          nstp=0;                       // Reset step counter
        }                               // ~~~~~~~~~~ ResetPhasor ~~~~~~~~~~ //
        inline T GetPhase (void) const  // Get current phase in radians
        {                               // ~~~~~~~~~~ GetPhase ~~~~~~~~~~ //
          return ph;                    // Return current phase
        }                               // ~~~~~~~~~~ GetPhase ~~~~~~~~~~ //
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Tick: Advance one sample and return complex phasor e^{j*ph}
        // Tickle the system forward for one sample tap.
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        inline cplx Tick (void)
        {                               // ~~~~~~~~~~ Tick ~~~~~~~~~~ //
          cplx v=std::polar(T(1),ph);   // Current phasor
          ph=Wrap(ph+phinc);            // Advance phase and wrap
          return v;                     // Return current phasor
        }                               // ~~~~~~~~~~ Tick ~~~~~~~~~~ //
        inline cplx QuickTick (void)
        {                               // ~~~~~~~~~~ QuickTick ~~~~~~~~~~ //
          cplx v(vr,vi);                // Current phasor from rotator state
          // Advance rotator state by one sample
          const T vrn=vr*rc-vi*rs;      // New real part
          const T vin=vr*rs+vi*rc;      // New imaginary part
          vr=vrn;                       // Update real part
          vi=vin;                       // Update imaginary part
          Renormalize();                // Renormalize to unit circle periodically
          ph=Wrap(ph+phinc);            // Advance phase and wrap
          return v;                     // Return current phasor
        }                               // ~~~~~~~~~~ QuickTick ~~~~~~~~~~ //
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // ProcessBlock: Advance N samples and fill out buffer with phasors
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        inline void ProcessBlock (
          cplx* const out,              // Output buffer for N phasors
          size_t N)                     // Number of phasors to generate
        {                               // ~~~~~~~~~~ ProcessBlock ~~~~~~~~~~ //
          if (out==nullptr||N==0)       // Invalid args?
            return;                     // Nothing to do
          for (size_t i=0;i<N;i++)      // For each phasor
          {                             // Generate and store
            out[i]=std::polar(T(1),ph); // Current phasor
            ph=Wrap(ph+phinc);          // Advance phase and wrap
          }                             // Done for N phasors
        }                               // ~~~~~~~~~~ ProcessBlock ~~~~~~~~~~ //
        inline void ProcessBlockQuick (
          cplx* const out,              // Output buffer for N phasors
          size_t N)                     // Number of phasors to generate
        {                               // ~~~~~~~~~~ ProcessBlockQuick ~~~~~~~~~~ //
          if (out==nullptr||N==0)       // Invalid args?
            return;                     // Nothing to do
          for (size_t i=0;i<N;i++)      // For each phasor
          {                             // Generate and store
            out[i]=cplx(vr,vi);         // Current phasor from rotator state
            // Advance rotator state by one sample
            const T vrn=vr*rc-vi*rs;    // New real part
            const T vin=vr*rs+vi*rc;    // New imaginary part
            vr=vrn;                     // Update real part
            vi=vin;                     // Update imaginary part
            if (((i+1)&(RENORM_PERIOD-1))==0)// Time to renormalize?
              Renormalize();            // Renormalize to unit circle periodically
          }                             // Done for N phasors
          ph=Wrap(ph+static_cast<T>(N)*phinc);// Advance phase by N samples and wrap
        }                               // ~~~~~~~~~~ ProcessBlockQuick ~~~~~~~~~~ //
        // Backward compatibility alias: some callers (e.g., Mixer USE_AF path)
        // expect an API named Process(out, N). Provide a thin inline wrapper.
        inline void Process (
          cplx* const out,
          size_t N)
        {
          ProcessBlockQuick(out,N);
        }
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Mix down input buffer in place takes in a buffer of complex samples
        // and mixes them down in place using the NCO phasors to downconvert to
        // baseband.
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        inline void MixDownInPlace (
         std::vector<cplx>* in)         // Input buffer to downconvert in place
        {                               // ~~~~~~~~~~ MixDownInPlace ~~~~~~~~~~ //
          if (in==nullptr||in->empty()) // Bad args?
            return;                     // Nothing to do
          for (size_t n=0;n<in->size();++n)// For each input sample
            (*in)[n]*=std::conj(this->Tick());// Mix down in place
        }                               // ~~~~~~~~~~ MixDownInPlace ~~~~~~~~~~ //
        inline void MixDownBlock (
          const cplx* __restrict in,    // Input buffer to downconvert
          cplx* const __restrict o,     // Output buffer for downconverted samples
          size_t N)                     // Number of samples to process
        {                               // ~~~~~~~~~~ MixDownBlock ~~~~~~~~~~ //
          if (in==nullptr||o==nullptr||N==0)// Bad args?
            return;                     // Nothing to do
          T lvr=vr;                     // Local copy of rotator real
          T lvi=vi;                     // Local copy of rotator imaginary
          T lrc=rc;                     // Local copy of rotator cosine
          T lrs=rs;                     // Local copy of rotator sine
          int lstp=nstp;                // Local copy of step counter
          for (size_t i=0;i<N;++i)      // For each input sample....
          {                             // Downconvert....
            const T xr=in[i].real();    // Get the real part of input signal
            const T xi=in[i].imag();    // Get the imaginary part of input signal
            const T I=xr*lvr+xi*lvi;    // In-phase component
            const T Q=xi*lvr-xr*lvi;    // Quadrature component
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Mix to baseband: y[n]=x[n]*conj(v) = (xr +jxi)*(lvr - jlvi)
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            o[i]=cplx(I,Q);             // Downconverted output sample
            // ~~~~~~~~~~~~~~~~~~~~~~~ //
            // Rotate phase vector: v*=rot
            // ~~~~~~~~~~~~~~~~~~~~~~~ //
            const T vrn=lvr*lrc-lvi*lrs;// New real part
            const T vin=lvr*lrs+lvi*lrc;// New imaginary part
            lvr=vrn;                    // Update real part
            lvi=vin;                    // Update imaginary part
            if (++lstp>=RENORM_PERIOD)  // Time to renormalize?
            {                           // Yes
              lstp=0;                   // Reset step counter
              const T r2=lvr*lvr+lvi*lvi;// Current radius squared
              const T rinv=static_cast<T>(1.0/std::sqrt(r2));// Inverse radius
              lvr*=rinv;                // Renormalize real
              lvi*=rinv;                // Renormalize imaginary
            }                           // Done renormalization
          }                             // Done traversing signal
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Update internal phase rotator state
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          vr=lvr;                       // Update rotator real
          vi=lvi;                       // Update rotator imaginary
          nstp=lstp;                    // Update step counter
          ph=Wrap(ph+static_cast<T>(N)*phinc);// Advance phase by N samples and wrap
        }
      private:
        double fs{1.0};                 // Sample rate in Hz
        double f0{0.0};                 // Frequency in Hz
        T ph{T(0)};                     // Current phase in radians
        T phinc{T(0)};                  // Phase increment per sample in radians
        T vr{T(1)};                     // Current real e^{j*w} state (cos(w))
        T vi{T(0)};                     // Current imaginary e^{j*w} state (sin(w))
        T rc{T(1)};                     // rotator e^{j*phinc} (cos(phinc)
        T rs{T(0)};                     // rotator e^{j*phinc} (sin(phinc))
        int nstp{0};                    // Step counter for renormalization
        static constexpr int RENORM_PERIOD=4096;// Renormalization period
        inline void RebuildRotator (void)
        {                               // ~~~~~~~~~ RebuildRotator ~~~~~~~~~~ //
          rc=static_cast<T>(std::cos(phinc));// Rebuild rotator cosine
          rs=static_cast<T>(std::sin(phinc));// Rebuild rotator sine
        }                               // ~~~~~~~~~ RebuildRotator ~~~~~~~~~~ //
        inline void Renormalize (void)
        {                               // ~~~~~~~~~~ Renormalize ~~~~~~~~~~ //
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Mantain (vr,vi) on unit circle without per sample sqrt() calls
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          if (++nstp>=RENORM_PERIOD)    // Time to renormalize?
          {                             // Yes
            nstp=0;                     // Reset step counter
            const T r2=vr*vr+vi*vi;     // Current radius squared
            // Newton-Raphson would be more to much here
            const T rinv=static_cast<T>(1.0/std::sqrt(r2));// Inverse radius
            vr*=rinv;                   // Renormalize real
            vi*=rinv;                   // Renormalize imaginary
          }                             // Done renormalization
        }                               // ~~~~~~~~~~ Renormalize ~~~~~~~~~~ //
        inline void UpdatePhaseInc (void)
        {                               // ~~~~~~~~~ UpdatePhaseInc ~~~~~~~~~~ //
          phinc=static_cast<T>(2.0*M_PI*f0/fs);// Phase increment per sample
          RebuildRotator();             // Rebuild rotator values
        }                               // ~~~~~~~~~ UpdatePhaseInc ~~~~~~~~~~ //
        static inline T Wrap (T x)      // The phase to wrap.
        {                               // ~~~~~~~~~ Wrap ~~~~~~~~~~~~~~~~~~~~ //
          while (x>=static_cast<T>(M_PI)) // Above +pi?
            x-=static_cast<T>(2.0*M_PI);// Wrap around
          while (x<static_cast<T>(-M_PI))// Below -pi?
            x+=static_cast<T>(2.0*M_PI);// Wrap around
          return x;                     // Return wrapped phase
        }                               // ~~~~~~~~~ Wrap ~~~~~~~~~~~~~~~~~~~~ //
    };
}