/* 
* *
* * Filename: DCLSIRGIGB.hpp
* * 
* * Description:
* *  DCLS (DC Level Shift) IRIG-B time code decoder (100 bps, 10 ms per bit)
* *  Feed level change edges with host timestamps. It classifies high-pulse widths, assembles
* *  bits & markers, decodes BCD fields, and exposes a fresh DCLS Irigic Decoded time each frame.
* *  Uses MJB's TimeSpec structure for timestamps and time differences so that timestamps are coherent
* *  throughout the CATT.
* *
* * Clocking model:
* *   - Call OnEdge(level,sec,nsec) for every GPIO transitions of the IRIG TTL input
* *   - The decodes measures HIGH durations to classify 0/1/P and assumes overall bit time = 10 ms
* *     (checks rising edge spacing ~10 ms to retain lock)
* *
* * NOTE: This is the Digital Clock Locking System IRIG-B decoder. 
* *  The 1 kHz AM Tone IRIG-B Decoder will come afterwards, and be the likely used.
* *
* * Author:
* *   JEP, J. Enrique Peraza
* *   MJB, Matthew J. Bienemann
* * 
 */
#pragma once
#include <cstdint>
#include <cstddef>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include "TimeSpec.h"


namespace sdr::time
{
  struct IRIGDecoded
  {
    uint16_t yr{0};                     // eg., ... 2025
    uint16_t doy{0};                    // 1 .. 366
    uint16_t hr{0},mm{0},ss{0};         // 0..23, 0..59, 0..59
    bool valid{false};                  // True if we decoded a valid timestamp.
  };
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // DCLS IRIG-B decoder using TimeSpec edge timestamps.
  // Pulse widths for a HIGH around: 0 -> 2 ms; 1 -> 5 ms; P -> 8 ms
  // Bit period: 10 ms (100 bits per second)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  class DCLSIRIGB
  {
    public:
      struct Tolerances
      {
        int64_t btus{10000};           // Bit time microseconds (10 ms)
        int64_t w0us{2000};            // Width for bit 0 microseconds
        int64_t w1us{5000};            // Width for bit 1 microseconds
        int64_t wpus{8000};            // Width for position marker microseconds
        int64_t tol0us{700};           // Tolerance for still a bit 0 in microseconds.
        int64_t tol1us{900};           // Tolerance for still a bit 1 in microseconds.
        int64_t tolpus{900};           // Tolerance for still a position marker in microseconds.
        int64_t edgetolus{1200};       // Tolerance for edge to still be in bit time in microseconds.
      };
      DCLSIRIGB (void)
      {
        //Reset();
      }
      ~DCLSIRIGB (void) = default;
      void Reset (void)
      {                                 // ~~~~~~~~~~~~~ Reset ~~~~~~~~~~~~~~~ //
        lockd=false;                    // Clear lock
        bidx=0;                         // Clear bit index
        hrise=false;                    // Clear have rising edge flag.
        fbts.fill(0);                   // Clear bits buffer.
        frmm.fill(0);                   // Clear marker buffer.
        ndec.store(false);              // Clear decoded flag
      }                                 // ~~~~~~~~~~~~~ Reset ~~~~~~~~~~~~~~~ //
      void Assemble (const Tolerances& t)
      {                                 // ~~~~~~~~~~~~ Assemble ~~~~~~~~~~~~~~ //
        tol=t;                          // Store tolerances
      }                                 // ~~~~~~~~~~~~ Assemble ~~~~~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // OnEdge: Feed a level change edge with timestamp (1=rising, 0=falling) Timespec ts.
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      void OnEdge (
        int lvl,                        // Level: 1=rising, 0=falling
        const ::TimeSpec& ts)           // Timespec structure timestamp of edge
      {                                 // ~~~~~~~~~~~ OnEdge ~~~~~~~~~~~~~~~~ //
        if (lvl==1)                     // Rising edge?
        {                               // Yes.
          if (hrise)                    // Did we see a prior rising edge?
          {                             // Most likely, it is a digital clock after all.
            auto dus=TimespecDifferenceMicroSeconds(lrise,ts); // Get microseconds since last rising edge
            if (std::llabs(dus-tol.btus)>tol.edgetolus) // Too far off bit time?
            {                           // Yes
              lockd=false;              // No lock
              bidx=0;                   // Clear bit index
            }                           // Done checking bit time
          }                             // Done checking prior rising edge
          lrise=ts;                     // Store last rising edge timestamp
          hrise=true;                   // We have a rising edge now
          llvl=1;                       // Store last level
        }                               // Done for rising edge..
        else                            // Otherwise it is trailing edge trigger
        {                               // So we will process it that way.
          if (!hrise||llvl!=1)          // No rising edge before this trailing edge?
          {                             // An anomaly, cannot do much.
            llvl=0;                     // Zeroize last level.
            return;                     // Ignore spurious falling edge
          }                             // Done checking for no prior rising edge.
          auto wus=TimespecDifferenceMicroSeconds(lrise,ts); // Get pulse width in microseconds
          llvl=0;                       // Store last level.
          int bit=ClassifyWidths(wus);  // Classify width into bit code
          if (bit<0)                    // Invalid bit code?
          {                             // Yes
            lockd=false;                // Lose lock
            bidx=0;                     // Clear bit index
            return;                     // Early return
          }                             // Done checking for invalid bit code
          PushBit(bit);                 // Push the bit code into buffers
          if (bidx>=100)                // Got the whole 100 bits (full frame)?
          {                             // Yes
            TryDecode();                // Try to decode the frame
            bidx=0;                     // Clear bit index for next frame.
          }                             // Done checking for full frame.
        }                               // Done with trailing edge trigger.
      }                                 // ~~~~~~~~~~~ OnEdge ~~~~~~~~~~~~~~~~ //
      bool GetDecodedFrame (
        IRIGDecoded* const of)          // Buffer to place output frame in.
      {                                 // ~~~~~~~~~ GetDecodedFrame ~~~~~~~~~~ //
        if (of==nullptr)                // Do we have a buffer to place results in?
          return false;                 // No, return false. Can't do much.
        if (!ndec.load())               // Do we have a new decoded frame?
          return false;                 // No, return false.
        *of=ldec;                       // Copy decoded frame to output buffer
        ndec.store(false);              // Clear new decoded frame flag
        return of->valid;               // Return validity of decoded frame
      }                                 // ~~~~~~~~~ GetDecodedFrame ~~~~~~~~~~ //
    private:
      static inline int64_t TimespecDifferenceMicroSeconds (
        const ::TimeSpec& a,            // First timestamp we want to compare to
        const ::TimeSpec& b)            // Second timestamp to compare to
    {                                   // ~~ TimespecDifferenceMicroSeconds ~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Both are UTC epochs in seconds + nanoseconds
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      const int64_t dsec=b.tv_sec-a.tv_sec; // Seconds difference from timeval
      const int64_t dnan=b.tv_nsec-a.tv_nsec;// Nanoseconds difference from timeval
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Combine to total nanoseconds
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      const long double nans=static_cast<long double>(dsec)*1.0e9L+static_cast<long double>(dnan);
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Return microseconds
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      return static_cast<int64_t>(std::llround(nans/1000.0L));
    }                                   // ~~ TimespecDifferenceMicroSeconds ~~ //
    int ClassifyWidths (
      int64_t wus)
    {                                   // ~~~~~~~~~~ ClassifyWidths ~~~~~~~~~ //
      if (Near(wus,tol.w0us,tol.tol0us)) return 0; // Bit 0
      if (Near(wus,tol.w1us,tol.tol1us)) return 1; // Bit 1
      if (Near(wus,tol.wpus,tol.tolpus)) return 2; // Position Marker
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Otherwise determine approximate and map it
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      int c=Nearest(wus,tol.w0us,tol.w1us,tol.wpus);
      int64_t dmin=std::llabs(wus-(c==0?tol.w0us:(c==1?tol.w1us:tol.wpus)));
      return (dmin>2000)?-1:c;          // Return closest if too far off
    }                                   // ~~~~~~~~~~ ClassifyWidths ~~~~~~~~~ //
    static inline bool Near (
      int64_t v,                        // The value to compare
      int64_t c,                        // The reference value
      int64_t tol)                      // The tolerance
    {                                   // ~~~~~~~~~~~ Near ~~~~~~~~~~~~~~~~~~ //
      return (v>=c-tol)&&(v<=c+tol);    // Within tolerance? 
    }                                   // ~~~~~~~~~~~ Near ~~~~~~~~~~~~~~~~~~ //
    static inline int Nearest (
      int64_t v,                        // The value to compare
      int64_t a,                        // Possible value a (closest to v - n)
      int64_t b,                        // Possible value b (closest to v)
      int64_t c)                        // Possible value c (closest to v + n)
    {                                   // ~~~~~~~~~~ Nearest ~~~~~~~~~~~~~~~ //
      int64_t da=std::llabs(v-a);       // distance to a
      int64_t db=std::llabs(v-b);       // distance to b
      int64_t dc=std::llabs(v-c);       // distance to c
      if ((da<=db)&&(da<=dc)) return 0; // a is closest
      if ((db<=da)&&(db<=dc)) return 1; // b is closest
      return 2;                         // c is closest
    }                                   // ~~~~~~~~~ Nearest ~~~~~~~~~~~~~~~~ //
    void PushBit (int bc)               // The bit code: 0,1, 2(P)
    {                                   // ~~~~~~~~~ PushBit ~~~~~~~~~~~~~~~~ //
      if (!lockd)                       // Unlocked?
      {                                 // Yes
        if (bc==2)                      // Is this a position marker?
        {                               // Yes
          bidx=0;                       // Clear bit index for this bit code
          frmm[bidx]=1;                 // Store marker bit
          fbts[bidx]=0;                 // Store data bit as 0
          lockd=true;                   // We are now locked
          bidx++;                       // Advance bit index
        }                               // Done setting lock
        return;                         // Early return.
      }
      bool ismarker=(bc==2);            // Is this a position marker?
      frmm[bidx]=ismarker?1:0;          // Store marker bit
      fbts[bidx]=(bc==1)?1:0;           // Store data bit
      if (ismarker)                     // Is this a marker code?
      {                                 // Yes
        if (!(bidx%10==0||bidx==0))     // Valid marker position?
        {                               // No,
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Invalid marker position, lose lock
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          lockd=false;                  // Set no lock
          bidx=0;                       // Clear bit index for this code
          return;                       // Can't do much, so return.
        }                               // Done checking marker position
      }                                 // Done checking if it's a marker.
      bidx++;                           // Increment marker position if we ever get here
    }                                   // ~~~~~~~~~ PushBit ~~~~~~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Index into Marker and Bit arrays helpers
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    inline uint8_t BitAt (size_t i) const     // Index into bit array
    {                                   // ~~~~~~~~~ BitAt ~~~~~~~~~~~~~~~~~ //
      return (i<100)?fbts[i]:0;         // Return the element if idx is valid.
    }                                   // ~~~~~~~~~ BitAt ~~~~~~~~~~~~~~~~~ //
    inline uint8_t MarkerAt (size_t i) const  // Index into marker array
    {                                   // ~~~~~~~~ MarkerAt ~~~~~~~~~~~~~~ //
      return (i<100)?frmm[i]:0;         // Return the element if idx is valid.
    }                                   // ~~~~~~~~ MarkerAt ~~~~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Binary Coded Data Format Helpers
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    static inline uint32_t BCD4 (
      const DCLSIRIGB* src,             // Source decoder
      size_t base)                      // The base to start reading from
    {                                   // ~~~~~~~~~~~~ BCD4 ~~~~~~~~~~~~~~~~ //
      return (src->BitAt(base+0)?1:0)|  // LSB
        (src->BitAt(base+1)?2:0)|       // Next bit
        (src->BitAt(base+2)?4:0)|       // Next bit
        (src->BitAt(base+3)?8:0);       // MSB
    }                                   // ~~~~~~~~~~~~ BCD4 ~~~~~~~~~~~~~~~~ //
    // BDCT3 Helper
    static inline uint32_t BCDT3 (
      const DCLSIRIGB* src,             // Source decoder
      size_t base)                      // The base to start reading from
    {                                   // ~~~~~~~~~~~~ BCDT3 ~~~~~~~~~~~~~~~~ //
      return (src->BitAt(base+0)?10:0)| // LSB
        (src->BitAt(base+1)?20:0)|      // Next bit
        (src->BitAt(base+2)?40:0);      // MSB
    }                                   // ~~~~~~~~~~~~ BCDT3 ~~~~~~~~~~~~~~~~ //
    static uint32_t BCDT2 (
      const DCLSIRIGB* src,             // Source decoder
      size_t base)                      // The base to start reading from
    {                                   // ~~~~~~~~~~~~ BCDT2 ~~~~~~~~~~~~~~~~ //
      return (src->BitAt(base+0)?10:0)| // LSB
        (src->BitAt(base+1)?20:0);      // MSB
    }                                   // ~~~~~~~~~~~~ BCDT2 ~~~~~~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Day of Year H2 Helper
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    static uint32_t DOYH2 (
      const DCLSIRIGB* src,             // Source decoder
      size_t base)                      // The base to start reading from
    {                                   // ~~~~~~~~~~~~ DOYH2 ~~~~~~~~~~~~~~~~ //
      return (src->BitAt(base+0)?200:0)|// 200s
        (src->BitAt(base+1)?100:0);     // 100s
    }                                   // ~~~~~~~~~~~~ DOYH2 ~~~~~~~~~~~~~~~~ //
    static inline bool InRange (
      uint32_t v,                       // Value to check.
      uint32_t lo,                      // Lower boundary
      uint32_t hi)                      // Upper boundary
    {                                   // ~~~~~~~~~~ InRange ~~~~~~~~~~~~~~~~ //
      return (v>=lo&&v<=hi);            // In range?
    }                                   // ~~~~~~~~~~ InRange ~~~~~~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Try to decode current frame bits into ldec
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    void TryDecode (void)
    {                                   // ~~~~~~~~~ TryDecode ~~~~~~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Pivot indices for fields
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      static const int piv[10]={0,9,19,29,39,49,59,69,79,89}; // Position index values
      for (int k=0;k<10;k++)            // For all marker positions
      {                                 // Verify if markers are present at pivot.
        if (!MarkerAt(piv[k]))          // Is the marker present?
        {                               // No, so no lock either.
         lockd=false;                   // Set lock to false
         return;                        // Not much to do, so return.
        }                               // Done checking positional marker values.
      }                                 // Done for all pivots of the matrix.
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Call BCD4 and BCDTn helpers to extract fields located at known positions
      // found by the object in-itself (using BitAt(idx) and MarkerAt(idx) helpers.
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      uint32_t sec=BCD4(this,1)+BCDT3(this,6); // Seconds: 0..59
      uint32_t min=BCD4(this,10)+BCDT3(this,15); // Minutes: 0..59
      uint32_t hr=BCD4(this,20)+BCDT2(this,25); // Hours: 0..23
      uint32_t doy=BCD4(this,30)+BCD4(this,35)+DOYH2(this,40); // Day of Year: 1..366
      uint32_t yr2=BCD4(this,50)+BCD4(this,55); // Year within century: 0..99
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Validate ranges
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      if (!InRange(sec,0,59)||!InRange(min,0,59)||!InRange(hr,0,23)||!InRange(doy,1,366))
        return;                         // Invalid range, return.
      uint16_t yrfull=static_cast<uint16_t>((yr2>=70)?(1900+yr2):(2000+yr2));
      ldec={yrfull,static_cast<uint16_t>(doy),static_cast<uint8_t>(hr),static_cast<uint8_t>(min),
        static_cast<uint8_t>(sec),true};
      ndec.store(true);                 // Decode successful, set new decode available.
    }                                   // ~~~~~~~~~ TryDecode ~~~~~~~~~~~~~~~ // 
    private:
      Tolerances tol{};                 // Our tolerances struct
      bool lockd{false};                // True if we locked to this IRIGB frame.
      bool hrise{false};                // True if we have rising edge.
      int llvl{0};                      // Last level seen.
      ::TimeSpec lrise{};               // Last rising edge timestamp.
      size_t bidx{0};                   // Bit index in current frame (0..99)
      std::array<uint8_t,100> fbts{};   // Frame bits storage (100 bits)
      std::array<uint8_t,100> frmm{};   // Frame Markers storage (100 bits)
      IRIGDecoded ldec{};               // Last decoded time
      std::atomic<bool> ndec{false};    // True if new decoded time available.
  };
}
