/* 
* *
* * Filename: PSKMaps.hpp
* *
* * Description:
* *   BPSK/QPSK/SQPSK bit<-> symbol Gray mapping utilities.
* *   - BPSK: 0 -> +1, 1 -> -1
* *   - QPSK: 00 -> (1,1), 01 -> (-1,1), 11 -> (-1,-1), 10 -> (1,-1)
* *   - SQPSK: 00 -> (1,0), 01 -> (0,1), 11 -> (-1,0), 10 -> (0,-1)
* *
* * Author:
* *   JEP, J. Enrique Peraza
* * Organization:
* *   Trivium Solutions LLC, 9175 Guilford Rd, Suite 220, Columbia, MD 21046
* *
 */
#pragma once
#include <complex>
#include <cstdint>
#include <cmath>

namespace sdr::mdm
{
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // BPSK symbol mapping to Gray code
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  template<typename T=float>
  static inline std::complex<T> BPSKMap ( uint8_t b) // The bit to map (0 or 1)
  {                                     // ~~~~~~~~~~ BPSKMap ~~~~~~~~~~ //
    if (b&0x01)                         // Bit is 1?
      return std::complex<T>(-1.0f,0.0f);// Yes, map to -1
    return std::complex<T>(1.0f,0.0f);  // No, map to +1
  }                                     // ~~~~~~~~~~ BPSKMap ~~~~~~~~~~ //
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // QPSK symbol mapping to Gray code
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  template<typename T=float>
  static inline std::complex<T> QPSKMap (
    uint8_t b0,                         // Bit 0 (LSB)
    uint8_t b1)                         // Bit 1 (MSB)
  {                                     // ~~~~~~~~~~ QPSKMap ~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Gray: 00->(1,1); 01->(-1,1); 11->(-1,-1); 10->(1,-1),
    // then normalize by 1/sqrt(2)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    int idx=(b0<<1)|b1;                 // Form index
    T s=static_cast<T>(1.0/std::sqrt(2.0));// Scale factor
    switch (idx)                        // Map index to symbol
    {                                   // and normalize
      case 0:return std::complex<T>(s,s);// 00 -> (1,1)/sqrt(2)
      case 1:return std::complex<T>(-s,s);// 01 -> (-1,1)/sqrt(2)
      case 3:return std::complex<T>(-s,-s);// 11 -> (-1,-1)/sqrt(2)
      default:return std::complex<T>(s,-s);// 10 -> (1,-1)/sqrt(2)
    }                                   // End switch
  }                                     // ~~~~~~~~~~ QPSKMap ~~~~~~~~~~ //
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // SQPSK symbol mapping to Gray code
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  template<typename T=float>
  static inline std::complex<T> SQPSKMap (
    uint8_t b0,                         // Bit 0 (LSB)
    uint8_t b1)                         // Bit 1 (MSB)
  {                                     // ~~~~~~~~~~ SQPSKMap ~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Gray: 00->(1,0); 01->(0,1); 11->(-1,0); 10->(0,-1)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    int idx=((b1&0x01)<<1)|(b0&0x01);   // MSB:LSB to index
    switch (idx)                        // Map index to symbol
    {                                   // and normalize
      case 0:return std::complex<T>(T(1),T(0));// 00 -> (1,0)
      case 1:return std::complex<T>(T(0),T(1));// 01 -> (0,1)
      case 2:return std::complex<T>(T(0),T(-1));// 10 -> (0,-1)
      default:return std::complex<T>(T(-1),T(0));// 11 -> (-1,0)
    }                                   // End switch
  }                                     // ~~~~~~~~~~ SQPSKMap ~~~~~~~~~~ //
} // namespace sdr::mdm
