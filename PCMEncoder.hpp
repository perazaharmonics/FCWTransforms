/* 
* *
* * Filename: PCMEncoder.hpp
* *
* * Description:
* *  Pulse Code Modulation encoder: NRZ-L/M/S and BiPhase-L/M/S. Input bits -> +- 1 symbol(s).
* *
* * Author:
* *  JEP, J. Enrique Peraza
* *
* * Organization:
* *  Trivium Solutions LLC, 9175 Guilford Rd, Suite 220, Columbia, MD 21046
* *
*/
#pragma once
#include <vector>
#include <cstdint>
#include <complex>
#include <limits>
#include <algorithm>
#include "../logger/Logger.h"

namespace sdr::mdm
{
  enum class PCMType
  {
    NRZ_L,                              // Non Return to Zero - Level
    NRZ_M,                              // Non Return to Zero - Mark
    NRZ_S,                              // Non Return to Zero - Space
    BIPHASE_L,                          // BiPhase - Level
    BIPHASE_M,                          // BiPhase - Mark
    BIPHASE_S                           // BiPhase - Space
  }; // PCMTypes enum
  template<typename T=float>
  class PCMEncoder
  {
    public:
      PCMEncoder (PCMType pt)
        : ptyp(pt), llvl(T(1))
      {
        lg=logx::Logger::NewLogger();   // Own a live logger instance
      }
      ~PCMEncoder (void)
      {
        if (lg)
          lg->ExitLog("PCMEncoder destructor called");
      }
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Encode input bits into PCM symbols
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      inline void Encode (
        const std::vector<uint8_t>* in, // Input bits
        std::vector<T>* const eo)       // Output encoded symbols
      {                                 // ~~~~~~~~~~ Encode ~~~~~~~~~~ //
        if (in==nullptr||eo==nullptr)   // Bad args?
          return;                       // Nothing to do so return.
        if (lg)
          lg->Deb("PCM Encode: Type=%d, InputBits=%zu",static_cast<int>(ptyp),in->size());
        eo->clear();                    // Clear output buffer
        eo->reserve(in->size()*2);      // Reserve space (max 2 symbols per bit for BiPhase)
        for (const auto& b:(*in))       // For each input bit in input vector....
        {                               // Pulse Code Modulate into symbols.... well levels.... well PCM symbols........ 
          switch (ptyp)                 // Encode according to PCM type
          {
            case PCMType::NRZ_L:        // Non-Return-to-Zero Level?
            {                           // Yes
              T v=b?static_cast<T>(-1):static_cast<T>(1);// Determine level
              eo->push_back(v);         // Output level
              if (lg)
                lg->Deb(" PCM Encode NRZ-L: InputBit=%d, OutputLevel=%f",b,v);
              break;                    // Done encoding so break.
            }                           // Done NRZ-L
            case PCMType::NRZ_M:        // Non-Return-to-Zero Mark?
            {                           // Yes
              if (b!=0)                 // Is the input bit a 0?
                llvl=-llvl;             // No, toggle last level
              eo->push_back(llvl);      // Output level
              if (lg)
                lg->Deb(" PCM Encode NRZ-M: InputBit=%d, OutputLevel=%f",b,llvl);
              break;                    // Done encoding so break.
            }                           // Done NRZ-M
            case PCMType::NRZ_S:        // Non-Return-to-Zero Space?
            {                           // Yes, inverse of NRZ-M
              if (b==0)                 // Is the input bit a 0?
                llvl=-llvl;             // Yes, toggle last level
              eo->push_back(llvl);      // Output level
              if (lg)
                lg->Deb(" PCM Encode NRZ-S: InputBit=%d, OutputLevel=%f",b,llvl);
              break;                    // Done encoding so break.
            }                           // Done NRZ-S
            case PCMType::BIPHASE_L:    // BiPhase Level?
            {                           // Yes, so determine levels
              T a=b?static_cast<T>(-1):static_cast<T>(1);// First half level
              eo->push_back(a);         // Output first half level
              eo->push_back(-a);        // Output second half level (inverted)
              if (lg)
                lg->Deb(" PCM Encode BiPhase-L: InputBit=%d, OutputLevels=(%f,%f)",b,a,-a);
              break;                    // Done encoding so break.
            }                           // Done BiPhase-L
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Transition at mid-cell always. Additional transition at start if bit=1
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            case PCMType::BIPHASE_M:    // BiPhase Mark?
            {                           // Yes
              T a=llvl;                 // First half level is laft level
              if (b!=0)                 // Is input bit a 0?
                a=-a;                   // No, so toggle first half level
              eo->push_back(a);         // Output first half level
              llvl=-a;                  // Update last level to second half level
              eo->push_back(llvl);      // Output second half level
              if (lg)
                lg->Deb(" PCM Encode BiPhase-M: InputBit=%d, OutputLevels=(%f,%f)",b,a,llvl);
              break;                    // Done encoding so break.
            }                           // Done BiPhase-M
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Transition at mid-cell always. Additional transition at start if bit=0
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            case PCMType::BIPHASE_S:    // BiPhase Space?
            {                           // Yes
              T a=llvl;                 // First half is last level....
              if (b==0)                 // Is input bit a 0?
                a=-a;                   // Yes, so toggle first half level
              eo->push_back(a);         // Output first half level
              llvl=-a;                  // Update last level to second half level
              eo->push_back(llvl);      // Output second half level
              if (lg)
                lg->Deb(" PCM Encode BiPhase-S: InputBit=%d, OutputLevels=(%f,%f)",b,a,llvl);
              break;                    // Done encoding so break.
            }                           // Done BiPhase-S
          }                             // Done encoding this input bit
        }                               // Done for each input bit
      }                                 // ~~~~~~~~~~ Encode ~~~~~~~~~~ //
      // Manchester polarity probe
      inline bool ProbeManchesterPolarity (
        const std::vector<T>& chs,     // Input soft chips
        size_t ppr=64,                 // Probe pairs of chips (max 64)
        T ptol=T(0.0),                 // Probe tolerance
        T mfr=T(0.60))                 // Min fraction of matching pairs to decide polarity
      {                                // ~~~~~~~~~~ ProbeManchesterPolarity ~~~~~~~~~~ //
        size_t N=chs.size();           // Number of input chips
        if (N<2)                       // Not enough chips?
          return false;                // Can't determine polarity, return false.
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Lambda function for mathematical sgn function
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        auto sgn=[](T v)->T             // Return T(1) for v>=0, T(-1) for v<0
        {                               //
          return v>=T(0)?T(1):T(-1);    // Return sign
        };                              // End sgn lambda
        size_t mp=std::min(ppr,N/2);    // Max number of probe pairs
        size_t posneg{0};               // (+,-) matching pairs count
        size_t negpos{0};               // (-,+) matching pairs count
        size_t i=0;                     // Input chip index
        size_t seen{0};                 // Number of valid pairs seen
        while (seen<mp&&i+1<N)          // While not done probing and still chips to process....
        {
          const T a=chs[i+0];           // First chip
          const T b=chs[i+1];           // Second chip
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Check for valid Manchester pair
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          const bool ssgn=(sgn(a)==sgn(b));// Same sign?
          const bool ninv=(std::abs(a+b)<=ptol);// Near inversion?
          if (ssgn&&!ninv)              // Same sign and not near inversion?
          {                             // Yes, invalid Manchester pair
            i+=1;                       // Slip one chip and retry
            continue;                   // Continue to next iteration
          }                             // Done checking for valid Manchester pair
          if (a>T(0)&&b<T(0))           // (+,-) pair?
            ++posneg;                   // Yes, increment (+,-) count
          else if (a<T(0)&&b>T(0))      // (-,+) pair?
            ++negpos;                   // Yes, increment (-,+) count
          ++seen;                       // Increment valid pairs seen
          i+=2;                         // Advance input index by two chips
        }                               // Done probing input chips
        if (seen==0)                    // No valid pairs seen?
          return false;                 // Can't determine polarity, return false.
        const T fracpn=static_cast<T>(posneg)/static_cast<T>(seen);// (+,-) fraction
        const T fracnp=static_cast<T>(negpos)/static_cast<T>(seen);// (-,+) fraction
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // If (-,+) dominates, that suggests inverted Manchester relative to (+,-) as bit-0
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        const bool inv=(fracnp>=mfr)&&(fracnp>fracpn);// Inverted Manchester?
        if (lg)
          lg->Deb(" PCM ProbeManchesterPolarity: ProbedPairs=%zu, (+,-)Fraction=%f, (-,+)Fraction=%f, Inverted=%d",seen,fracpn,fracnp,inv?1:0);
        return inv;                    // Return inversion result
      }                                // ~~~~~~~~~~ ProbeManchesterPolarity ~~~~~~~~~~ //
      // Encode Manchester from bits to chips:
      inline void EncodeManchester (
        const std::vector<uint8_t>* in, // Input bits
        std::vector<uint8_t>* const co)  // Output chips (0/1)
      {                                 // ~~~~~~~~~~ EncodeManchester ~~~~~~~~~~ //
        if (in==nullptr||co==nullptr)   // Bad args?
          return;                       // Nothing to do so return.
        co->clear();                    // Clear output buffer
        co->reserve(in->size()*2);      // Reserve space (2 chips per bit)
        for (const auto& b:(*in))       // For each input bit in input vector....
        {                               // Manchester encode into chips....
          T a=b?T(-1):T(1);             // First chip
          co->push_back(a);             // Output first chip
          co->push_back(-a);            // Output second chip (inverted)
          if (lg!=nullptr)
            lg->Deb(" PCM Encode Manchester: InputBit=%d, OutputChips=(%d,%d)",b,
              (a<0)?1:0,(-a<0)?1:0);
        }                               // Done for each input bit
      }                                 // ~~~~~~~~~~ EncodeManchester ~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Decode Machester from soft chips:
      // Soft input symbols (preferred): 'chs' are the *I* soft symbols at the
      // recovered symbol rate. For a bit every two chips:
      //  metric m=chip0-chip1; bit = (m<0)?1:0; confidence = |m|
      // If chip1 is not the inverse of chip0, we slip by one chip and retry.
      //
      // Hard input symbols (0/1 from slicer): pair chips {c0,c1}. We map 1->(+), 0->(-).
      // bit=(c0?0:1). Same slip logic as soft path.
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      inline void DecodeManchesterSoft (
        const std::vector<T>* chs,      // Soft input symbols (+/- values)
        std::vector<uint8_t>* const bo, // Output bits
        std::vector<T>* const conf=nullptr,// Optional confidence values
        T sstol=T(0.0))                 // Same sign tolerance for slip detection
      {                                 // ~~~~~~~~~ DecodeManchesterSoft ~~~~~~~~~~ //
        if (chs==nullptr||bo==nullptr)  // Bad args?
          return;                       // Nothing to do so return.
        bo->clear();                    // Clear output buffer
        if (conf!=nullptr)              // Confidence levels requested?
          conf->clear();                // Clear confidence buffer
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Lambda function for mathematical sgn function
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        auto sgn=[](T v)->T             // Return T(1) for v>=0, T(-1) for v<0
        {                               //
          return v>=T(0)?T(1):T(-1);    // Return sign
        };                              // End sgn lambda
        size_t i=0;                     // Input symbol index
        size_t N=chs->size();           // Number of input symbols
        while (i+1<N)                   // While at least two symbols to process...
        {
          T a=(*chs)[i+0];              // First chip
          T b=(*chs)[i+1];              // Second chip
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Check for valid Manchester pair
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          if (sgn(b)==sgn(a)&&std::abs(b+a)>sstol)// Same sign and above tolerace?
          {                             // Yes, invalid Manchester pair
            if (lg!=nullptr)
              lg->War(" PCM DecodeManchester: Invalid Manchester pair at input index %zu: (%f,%f), slipping 1 chip",i,a,b);
            i+=1;                       // Slip one chip and retry
            continue;                   // Continue to next iteration
          }                             // Done checking for valid Manchester pair
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Valid Manchester pair; decode bit
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          T m=a-b;                      // Decision metric
          uint8_t bit=(m<T(0))?1u:0u;   // Determine output bit
          bo->push_back(bit);           // Output bit
          if (conf!=nullptr)            // Confidence levels requested?
            conf->push_back(std::abs(m));// Output confidence
          if (lg!=nullptr)
            lg->Deb(" PCM DecodeManchester: InChips=(%f,%f) Metric=%f Bit=%u",a,b,m,bit);
          i+=2;                         // Advance input index by two chips           
        }                               // Done while at least two input symbols
      }                                 // ~~~~~~~~~ DecodeManchesterSoft ~~~~~~~~~~ //
      inline void DecodeManchesterHard (
        const std::vector<uint8_t>* chh,// Hard input symbols (0/1)
        std::vector<uint8_t>* const bo) // Output bits
      {                                 // ~~~~~~~~~ DecodeManchesterHard ~~~~~~~~~~ //
        if (chh==nullptr||bo==nullptr)  // Bad args?
          return;                       // Nothing to do so return.
        bo->clear();                    // Clear output buffer
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Lambda function to map 1 => '+', 0 => '-'
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        auto pos=[](uint8_t h)->bool    // Map hard bit to positive/negative
        {
          return h!=0;                  // 1=> true (+), 0=> false (-)
        };                              // End pos lambda
        size_t i=0;                     // Input symbol index
        size_t N=chh->size();           // Number of input symbols
        while (i+1<N)                   // While at least two symbols to process...
        {
          uint8_t c0=(*chh)[i+0];       // First chip
          uint8_t c1=(*chh)[i+1];       // Second chip
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Manchester-L wants:
          // (+,-) -> bit 0
          // (-,+) -> bit 1
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          if (pos(c0)==pos(c1))         // Same sign?
          {                             // Yes, invalid Manchester pair
            if (lg!=nullptr)
              lg->War(" PCM DecodeManchester: Invalid Manchester pair at input index %zu: (%u,%u), slipping 1 chip",i,c0,c1);
            i+=1;                       // Slip one chip and retry
            continue;                   // Continue to next iteration
          }                             // Done checking for invalid Manchester pair
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Else, valid Manchester pair; decode bit
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          uint8_t bit=pos(c0)?0u:1u;    // Determine output bit.
          bo->push_back(bit);           // Output bit
          if (lg!=nullptr)
            lg->Deb(" PCM DecodeManchester: InChips=(%u,%u) Bit=%u",c0,c1,bit);
          i+=2;                         // Advance input index by two chips
        }                               // Done while at least two input symbols
      }                                 // ~~~~~~~~~ DecodeManchesterHard ~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Decode Manchester Soft Gated with optional auto-polarity detection.
      // Applies pair validity check with tolerance.
      // Computes sof confidence per pair c=|a-b| / (|a|+|b|+eps)
      // If c < cgate, marks an erasure (confidence=0) but still emits a bit.
      // so downstream indexing remains aligned.
      // If auto-polarity is enabled, probes initial segment for polarity.
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      inline void DecodeManchesterSoftGated (
        const std::vector<T>* chs,      // Input soft chips
        std::vector<uint8_t>* const bo, // Out bits
        std::vector<T>* const conf,     // Out confidence values (pass nullptr to skip)
        std::vector<uint8_t>* const eras,// Out erasure flags (1=erased)
        T ptol=T(0.0),                  // Same sign tolerance for slip detection
        T cgate=T(0.25),                // Confidence gate; lower -> stricter gating.
        bool apol=true,                 // Probe and invert mapping if true
        bool* polinv=nullptr,          // Output if we did polarity inversion
        size_t ppr=64)                  // Number of pairs to probe for polarity
      {                                 // ~~~~~~~~~ DecodeManchesterSoftGated ~~~~~~~~~ //
        if (chs==nullptr||bo==nullptr)  // Bad args?
          return;                       // Nothing to do so return.
        bo->clear();                    // Clear output buffer
        if (conf!=nullptr)              // Confidence levels requested?
          conf->clear();                // Clear confidence buffer
        if (eras!=nullptr)              // Erasure flags requested?
          eras->clear();                // Clear erasure buffer
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Decide polarity if requested
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        bool inv=false;                 // No inversion by default
        if (apol)                       // Auto-polarity requested?
          inv=ProbeManchesterPolarity(*chs,ppr,ptol,T(0.60)); // Probe polarity
        if (polinv!=nullptr)            // User wants inversion result?
          *polinv=inv;                  // Return inversion result
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Lambda function for mathematical sgn function
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        auto sgn=[](T v)->T             // Return T(1) for v>=0, T(-1) for v<0
        {                               //
          return v>=T(0)?T(1):T(-1);    // Return sign
        };                              // End sgn lambda
        const T eps=std::numeric_limits<T>::epsilon();// Small value to avoid div by zero
        size_t i=0;                     // Input symbol index
        const size_t N=chs->size();     // Number of input symbols
        while (i+1<N)                   // While at least two symbols to process...
        {
          const T a=(*chs)[i+0];        // First chip
          const T b=(*chs)[i+1];        // Second chip
          const bool ssgn=(sgn(a)==sgn(b));// Same sign?
          const bool ninv=(std::abs(a+b)<=ptol);// Near inversion?
          if (ssgn&&!ninv)              // Same sign and not near inversion?
          {
            if (lg!=nullptr)
              lg->War(" PCM DecodeManchesterGated: Invalid Manchester pair at input index %zu: (%f,%f), slipping 1 chip",i,a,b);
            i+=1;                       // Slip one chip and retry
            continue;                   // Continue to next iteration
          }                             // Done checking for valid Manchester pair
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Soft metric and normalized confidence
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          const T m=a-b;                // Decision metric
          const T den=std::abs(a)+std::abs(b)+eps;// Denominator for confidence
          T c=std::abs(m)/den;          // Normalized confidence
          if (c>T(1))                   // Confidence > 1?
            c=T(1);                     // Clamp to 1
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Map to bit with optional polarity inversion
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          uint8_t bit=(m<T(0))?1u:0u;   // Determine output bit
          if (inv)                      // Inversion requested?
            bit^=1u;                    // Invert bit
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Apply soft erasure gate
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          const bool iseras=(c<cgate);  // Is this an erasure?
          bo->push_back(bit);           // Output bit
          if (conf!=nullptr)            // Confidence levels requested?
            conf->push_back(iseras?T(0):c);// Output confidence or zero if erasure
          if (eras!=nullptr)            // Erasure flags requested?
            eras->push_back(iseras?1u:0u);// Output erasure flag
          if (lg!=nullptr)              // Log decode info
            lg->Deb(" PCM DecodeManchesterGated: InChips=(%f,%f) Metric=%f Confidence=%f%s Bit=%u",a,b,m,c,
              iseras?" ERASURE":"",bit);
          i+=2;                         // Advance input index by two chips
        }                               // Done while at least two input symbols
      }                                 // ~~~~~~~~~ DecodeManchesterSoftGated ~~~~~~~~~ //
      // Decode PCM symbols into bits
      // in:  NRZ-* -> 1 symbol/bit; BIPHASE-* -> 2 symbols/bit
      // bo:  Output bits
      // Caveat: Differential families (NRZ-M/S, BIPHASE-M/S) need the previous
      // end-of-bit level to resolve the first bit. Seed llvl with a known value
      // (typically from a preamble) or accept that the first decoded bit may be ambiguous.
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      inline void Decode(
        const std::vector<T>* in,       // Input symbols (+/-1), 1 per bit (NRZ-*), 2 per bit (BIPHASE-*)
        std::vector<uint8_t>* const bo) // Output bits
      {                                 // ~~~~~~~~~~ Decode ~~~~~~~~~~ //
        if (in==nullptr || bo==nullptr) // Bad args?
          return;                       // Nothing to do so return.
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Lambda function for sgn function
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        auto sgn=[](T v)->T             // Sign function
        {                               // Return +1 for v>=0, -1 for v<0
           return v>=T(0)?T(1):T(-1);   // Return sign
        };                              // End sgn lambda
        bo->clear();                    // Clear output buffer
        if (lg!=nullptr)
          lg->Deb("PCM Decode: Type=%d, InputSymbols=%zu", static_cast<int>(ptyp), in->size());
        switch (ptyp)                   // Decode according to PCM type
        {                               //
          case PCMType::NRZ_L:          // Non-Return-to-Zero Level?
          {                             // Yes
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Encoder: bit 0 -> +1, bit 1 -> -1
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            bo->reserve(in->size());    // Reserve space
            for (const auto& s:*in)     // For each input symbol
            {                           // Decode symbol
              T v=sgn(s);               // Determine sign
              uint8_t b=(v<T(0))?1u:0u; // Determine bit according to level
              bo->push_back(b);         // Output bit
              if (lg!=nullptr)
                lg->Deb(" PCM Decode NRZ-L: In=%f V=%f Bit=%u", s, v, b);
            }                           // Done for each input symbol
            break;                      // Done decoding so break.
          }                             // Done NRZ-L
          case PCMType::NRZ_M:          // Non-Return-to-Zero Mark?
          {                             // Yes
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Encoder toggles level on bit=1
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            bo->reserve(in->size());    // Reserve space
            for (const auto& s:*in)     // For each input symbol
            {                           // Decode 
              T v=sgn(s);               // Determine sign
              uint8_t b=(v!=llvl)?1u:0u;// Determine bit according to level
              bo->push_back(b);         // Output bit
              llvl=v;                   // Current symbol becomes last level
              if (lg!=nullptr)
                lg->Deb(" PCM Decode NRZ-M: In=%f V=%f Bit=%u NewLL=%f",s,v,b,llvl);
            }                           // Done for each input symbol
            break;                      // Done decoding so break.
          }                             // Done NRZ-M
          case PCMType::NRZ_S:          // Non-Return-to-Zero Space?
          {                             // Yes
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Encoder toggles level on bit=0
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            bo->reserve(in->size());    // Reserve space
            for (const auto& s:*in)     // For each input symbol
            {                           // Decode
              T v=sgn(s);               // Determine sign
              uint8_t b=(v==llvl)?1u:0u;// Determine output bit according to level
              bo->push_back(b);         // Output bit
              llvl=v;                   // Current symbol becomes last level
              if (lg!=nullptr)          //
                lg->Deb(" PCM Decode NRZ-S: In=%f V=%f Bit=%u NewLL=%f",s,v,b,llvl);
            }                           // Done for each input symbol
            break;                      // Done decoding so break.
          }                             // Done NRZ-S
          case PCMType::BIPHASE_L:      // BiPhase Level?
          {                             // Yes
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Encoder: (a, -a), with a = (+1 for bit=0, -1 for bit=1)
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            size_t nbits=in->size()/2;  // Number of output bits
            bo->reserve(nbits);         // Reserve space
            for (size_t i=0;i+1<in->size();i+=2)// For each pair of input symbols
            {                           // Decode pair
              T a=sgn((*in)[i]);        // First half-level
              T b=sgn((*in)[i+1]);      // Second half-level
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              // Optional consistency check: b should be -a
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              uint8_t bit=(a<T(0))?1u:0u;// Determine bit according to first half-level
              bo->push_back(bit);       // Output bit
              if (lg) lg->Deb(" PCM Decode BiPhase-L: In=(%f,%f) AB=(%f,%f) Bit=%u",(*in)[i],(*in)[i+1],a,b,bit);
            }                           // Done for each pair of input symbols
            break;                      // Done decoding so break.
          }                             // Done BiPhase-L
          case PCMType::BIPHASE_M:      // BiPhase Mark?
          {                             // Yes
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Encoder: mid-bit transition always; start-of-bit transition if bit=1.
            // a = first half-level; prior end-of-bit level is llvl. bit = (a != llvl) ? 1 : 0
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            size_t nbits=in->size()/2;  // Number of output bits
            bo->reserve(nbits);         // Reserve space
            for (size_t i=0;i+1<in->size();i+=2) // For each pair of input symbols
            {                           // Decode pair
              T a=sgn((*in)[i]);        // first half-level
              T b=sgn((*in)[i+1]);      // end-of-bit level
              uint8_t bit=(a!=llvl)?1u:0u;// Determine bit according to first half-level and last level
              bo->push_back(bit);       // Output bit
              llvl=b;                   // track end-of-bit for next decision
              if (lg!=nullptr)          //
                lg->Deb(" PCM Decode BiPhase-M: In=(%f,%f) AB=(%f,%f) Bit=%u NewLL=%f",(*in)[i],(*in)[i+1],a,b,bit,llvl);
            }                           // Done for each pair of input symbols
            break;                      // Done decoding so break.
          }                             // Done BiPhase-M
          case PCMType::BIPHASE_S:      // BiPhase Space?
          {                             // Yes
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Encoder: mid-bit transition always; start-of-bit transition if bit=0.
            // bit = (a != llvl) ? 0 : 1
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            size_t nbits=in->size()/2;  // Number of output bits
            bo->reserve(nbits);         // Reserve space
            for (size_t i=0;i+1<in->size();i+=2) // For each pair of input symbols
            {                           // Decode pair
              T a=sgn((*in)[i]);        // first half-level
              T b=sgn((*in)[i+1]);      // end-of-bit level
              uint8_t bit=(a!=llvl)?0u:1u;// Determine bit according to first half-level and last level
              bo->push_back(bit);       // Output bit
              llvl=b;                   // track end-of-bit for next decision
              if (lg!=nullptr)
              lg->Deb(" PCM Decode BiPhase-S: In=(%f,%f) AB=(%f,%f) Bit=%u NewLL=%f",(*in)[i],(*in)[i+1],a,b,bit, llvl);
            }                           // Done for each pair of input symbols
            break;                      // Done decoding so break.
          }                             // Done BiPhase-S
        }                               // Done decoding all input symbols
      }                                 // ~~~~~~~~~~ Decode ~~~~~~~~~~ //
      // Getters and Setters
      inline PCMType GetPCMType (void) const
      {
        return ptyp;
      }
      inline void SetPCMType (PCMType pt)
      {
        ptyp=pt;
      }
      inline T GetLastLevel (void) const
      {
        return llvl;
      }
      // PCMTYpe index to string
      static inline std::string PCMTypeToString (PCMType pt)
      {
        switch (pt)
        {
          case PCMType::NRZ_L:    return "NRZ-L";
          case PCMType::NRZ_M:    return "NRZ-M";
          case PCMType::NRZ_S:    return "NRZ-S";
          case PCMType::BIPHASE_L:return "BiPhase-L";
          case PCMType::BIPHASE_M:return "BiPhase-M";
          case PCMType::BIPHASE_S:return "BiPhase-S";
          default:                return "Unknown";
        }
      }
    private:
      PCMType ptyp{PCMType::NRZ_L};     // PCM encoding type
      T llvl{T(1)};                     // Last level for NRZ
      std::unique_ptr<logx::Logger> lg{};      
  };
}
