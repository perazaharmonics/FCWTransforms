/* 
* *
* * Filename: AMIRIGB.hpp
* *
* * Description:
* *  IRIGB-122 1 kHz Amplitude Modulated Time Code Decoder. Same as DCLSIRIGB but
* *  with amplitude modulation instead of DC level shift keying. Uses MJB's Timespec
* *  class for time representation, so our timestamps are coherent throughout the CATT.
* *
* *  Based on "IRIG Standard 200-04, Inter-range Instrumentation Group (IRIG)
* *  Standard Time Code Formats", Section 4.2.2 Amplitude Modulated Time Code
* *  (pages 15-18).
* *
* *
* * Author:
* *  JEP, J. Enrique Peraza
* *  MJB, Matthew J. Bienemann
* *
 */
#pragma once
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include "TimeSpec.h"
#include "DCLSIRIGB.hpp"

namespace sdr::time
{
  // More than likely defining it as AMIRIGB "has a" DCLSIRIGB decoder, rather than AMIRIGB "is a" DCLSIRIGB decoder.
  class AMIRIGB
  {
    public:
      struct Parameters
      {
        double sr{48000.0};             // Sample rate in Hz.
        double envlpfc{80.0};           // Envelope LPF cutoff (Hz).
        double agclpfc{5.0};            // AGC LPF cutoff (Hz).
        double schmon{1.6};             // Schmitt trigger ON (env>=on*mean -> HIGH voltage).
        double schmoff{1.2};            // Schmitt trigger OFF (env<=off*mean -> LOW voltage).
        uint32_t wupsamp{2000};         // Warmup samples to skip before decoding.
      };
    public:
      AMIRIGB (void)
      {
        Configure(prm);
      }
      ~AMIRIGB (void)=default;
      void SetParameters (const Parameters& pa)
      {                                 // ~~~~~~~~~~ SetParameters ~~~~~~~~~~~~~~~~~ //
        prm=pa;                         // Store new parameter struct.
        Configure(prm);                 // Configure our IRIGB-122 decoder.
        Reset();                        // Reset internal states.
      }                                 // ~~~~~~~~~~ SetParameters ~~~~~~~~~~~~~~~~~ //
      void Reset (void) 
      {                                 // ~~~~~~~~~~ Reset ~~~~~~~~~~~~~~~~~~~~~~~~~ //
        env=0.0;                        // Zeroize envelope value
        agc=1e-6;                       // Zeroize AGC value.
        hgh=false;                      // Zeroize current Schmitt trigger state.
        seen=0;                         // Zeroize frames/samples seen.
        dec->Reset();                   // Zeroize internal DC Level Shift IRIGB decoder state.
      }                                 // ~~~~~~~~~~ Reset ~~~~~~~~~~~~~~~~~~~~~~~~~ //
      void AttachDecoder (DCLSIRIGB* d)
      {                                 // ~~~~~~~~~~ AttachDecoder ~~~~~~~~~~~~~~~~~~ //
        if (d!=nullptr)                 // Valid decoder?
          dec=d;                        // Yes, set it.
        return;                         // Explicit return.
      }                                 // ~~~~~~~~~~ AttachDecoder ~~~~~~~~~~~~~~~~~~ //
      // Process a block of AM samples in [-1,1]; ts0 is the TimeSpec of sample 0.
      void ProcessBlock (
        const float* x,                 // Input samples buffer
        size_t n,                       // Number of frames in block.
        const ::TimeSpec& t0)           // Timestamp of sample 0.
      {                                 // ~~~~~~~~~ ProcessBlock ~~~~~~~~~~~~~~~~~~~ //
        if (x==nullptr||dec==nullptr)   // No input buffer, and no decoder?
          return;                       // Can't do much so just return.
        const double fs=prm.sr;         // Get the operational sample rate.
        for (size_t i=0;i<n;++i)        // For the number of frames or samples in this block....
        {                               // Process the signal.....
          double v=std::fabs(static_cast<double>(x[i])); // Get the sample in double.
          env+=enva*(v-env);            // Update envelope detector.
          agc+=agca*(env-agc);          // Update AGC level.
          double mean=std::max(agc,1e-6); // Mean level (avoid div by zero).
          double on=prm.schmon*mean;    // Schmitt trigger ON level.
          double off=prm.schmoff*mean;  // Schmitt trigger OFF level.
          bool nh=(hgh);                // New high state.
          if (!hgh&&env>=on)            // Currently LOW, but envelope >= ON level?
            nh=true;                    // Switch to HIGH voltage level.
          if (hgh&&env<=off)            // Currently HIGH, but envelope <= OFF level?
            nh=false;                   // Switch to LOW voltage level.
          if (seen<prm.wupsamp)         // Still in warm up period?
          {                             // Yes
            seen++;                     // Count this sample.
            hgh=nh;                     // Set new HIGH level.
            continue;                   // Keep processing.
          }                             // Done with warmup samples.
          if (nh!=hgh)                  // Was there a bit transition?
          {                             // Yes
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // edge time = ts0 + i/fs in nanoseconds
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            ::TimeSpec ets=AddSamples(t0,i,fs); // Edge time in TimeSpec format.
            dec->OnEdge(nh?1:0,ets);    // Notify the DC Level Shift IRIGB decoder.
            hgh=nh;                     // Set new HIGH level.
          }                             // Done with bit transition.
        }                               // Done with for all frames/samples in this block.
      }                                 // ~~~~~~~~~ ProcessBlock ~~~~~~~~~~~~~~~~~~~ //
    private:
      static double Alpha (
        double fc,                      // The frequency of the carrier.
        double fs)                      // The rate we are sampling at.
      {                                 // ~~~~~~~~~~~~ Alpha ~~~~~~~~~~~~~~~~ //
        if (fc<=0.0)                    // Negative frequencies?
          return 1.0;                   // We haven't normalized bw, so no neg freq. Return 1.
        double alph=std::exp(-2.0*M_PI*fc/fs); // Calculate alpha.
        if (alph<1e-6)                  // Too small?
          alph=1e-6;                    // Clamp it.
        if (alph>1.0)                   // Too big?
          alph=1.0;                     // Clamp it.
        return alph;                    // Return alpha.
      }                                 // ~~~~~~~~~~~~ Alpha ~~~~~~~~~~~~~~~~ //
      void Configure (const Parameters& pa)
      {                                 // ~~~~~~~~~~ Configure ~~~~~~~~~~~~~~ //
        enva=Alpha(pa.envlpfc,pa.sr);   // Envelope alpha.
        agca=Alpha(pa.agclpfc,pa.sr);   // AGC alpha.
      }                                 // ~~~~~~~~~~ Configure ~~~~~~~~~~~~~~ //
      static ::TimeSpec AddSamples (
        const ::TimeSpec& t0,           // Initial reference time.
        uint64_t idx,                   // Sample index to add.
        double sr)                      // Sample rate.
      {                                 // ~~~~~~~~~ AddSamples ~~~~~~~~~~~~~~ //
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Convert idx/fs to ns using integer math
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        long double addns=static_cast<long double>(idx)*(1.0e9L/sr);
        uint64_t add=static_cast<uint64_t>(addns+0.5L);
        uint64_t nstot=static_cast<uint64_t>(t0.tv_nsec)+add;
        ::TimeSpec tout=t0;             // Copy initial time.
        tout.tv_sec+=(nstot/1000000000ULL); // Add seconds.
        tout.tv_nsec=(nstot%1000000000ULL); // Remainder nanoseconds.
        return tout;                    // Return new time.
      }                                 // ~~~~~~~~~ AddSamples ~~~~~~~~~~~~~~ //
    private:
      Parameters prm{};                 // Decoder parameters.
      DCLSIRIGB* dec{nullptr};          // DC Level Shift Keying IRIGB decoder.
      double env{0.0};                  // Current envelope value.
      double agc{1e-6};                 // Current AGC level.
      double enva{0.0};                 // Envelope alpha.
      double agca{0.0};                 // AGC alpha.
      bool hgh{false};                  // Current Schmitt trigger state.
      uint32_t seen{0};                 // Lifetime samples seen.
  };
}
