 /*
 * * Filename: DelayLine.hpp
 * *
 * * Description:
 * * This file contains the implementation of a delay line class.
 * * The delay line is implemented as a circular buffer.
 * * It is used to delay the audio signal by a specified number of samples.
 * * Maxlen is the compile time length in samples of the delay line.
 * * Author:
 * *  JEP J. Enrique Peraza
 * *
 * * Organization:
 * *   Trivium Solutions LLC, 9175 Guilford Rd, Suite 220, Columbia, MD 21046
 * *
 */
#pragma once
#include <array>
#include <cstddef>
#include <cassert>
#include <algorithm>
#include <experimental/simd>
namespace sdr::mdm
{
  // ----------------------------------
  // Fixed size fractional delay line (ring-buffer).
  // Maxelen is the compile time length in samples of the delay line.
  // ----------------------------------
  template<typename T,std::size_t Maxlen>
class DelayLine
{
  public:
    // Get DelayLine size in samples.
    inline size_t GetDelay(void) const noexcept { return this->delay; } // Get the current delay in samples.
    
        // Set integer delay (1 <=d <=maxlen-1)
    bool SetDelay( std::size_t d) noexcept                  // The delay in samples.                    // Set the value of the delay line.
    {                                 // ---------------- SetDelay ---------- 
          if (d==0||d>=this->maxlen)      // Is the number of samples out of range
              return false;               // Yes return false.
          this->delay=d;                  // Set the delay.
          return true;                    // Return true.
    }                                 // ---------------- SetDelay ----------
        // Fast integer delay read and write.
    inline void Write(T x) noexcept 
    {
      buf[this->widx]=x;
      this->Advance();
    }
    // Write a sample at a specific delay position (0 <= i < maxlen)
    inline void WriteAt(size_t i, T x) noexcept
    {
      if (i >= maxlen) return;
      // Calculate circular buffer position from write index
      size_t pos = (widx + maxlen - 1 - i) & mask;
      buf[pos] = x;
    }
    // Accumulate (add) a sample at a specific delay position
    inline void AccumulateAt(size_t i, T x) noexcept
    {
      if (i >= maxlen) return;
      size_t pos = (widx + maxlen - 1 - i) & mask;
      buf[pos] += x;
    }
    const inline T Read(void) const noexcept { return buf[(this->widx-this->delay+this->maxlen)&mask];}
    // Linear interpolated fractional delay read.
    T ReadFrac(                        
      double N)                       // The fractional delay in sample to read.
      const noexcept                  // No exceptions.
    {                                 // ------------ ReadFrac ------------- //
    // ---------------------------- //
    // Assert that the fractional delay is in range.
    // ---------------------------- //
    assert(N>=0.0&&N<static_cast<float>(this->maxlen-1));
    auto idx=static_cast<size_t>(N);// Get the integer part.
    float frac=N-idx;               // Get the fractional part.
    size_t pos1 = (this->widx-idx +this->maxlen)&this->mask;      // Newer
    size_t pos0 = (pos1-1+this->maxlen)&this->mask;              // Older
    // ---------------------------- //
    // Interpolate the two samples. //
    // ---------------------------- //
    return (1.f - frac) * buf[pos0] + frac * buf[pos1];// Return the interpolated value.  
  }                                   // ------------ ReadFrac ------------- //
  inline void Clear(void) noexcept { buf.fill(T{});this->widx=0;this->delay=1; } // Clear the delay line.
  inline size_t GetMaxlen(void) const noexcept { return this->maxlen; }
  T PeekTail(void) const noexcept 
  {
    size_t pos=(widx+maxlen-1)&mask;
    return buf[pos]; // Return the last sample in the buffer.
  }
  T Peek(size_t idx) const noexcept
  {
    // Relative peek: idx=0 returns most recently written sample,
    // idx=1 returns the one before, and so on.
    size_t pos=(widx-1-idx+Maxlen)&mask;
    return buf[pos];
  }
  T PeekRelative(size_t idx) const noexcept 
  {
    size_t pos=(widx-idx-1+Maxlen)&mask; // Calculate the index in the circular buffer.
    return buf[pos]; // Return the sample at the specified index.
  }
  T GetHead(void) const noexcept 
  {
    return buf[widx]; // Return the sample at the head of the buffer.
  }
  float PeekIndex(size_t idx) const noexcept
  {
    assert(idx < maxlen); // Ensure the index is within bounds.
    return buf[idx & mask]; // Return the sample at the specified index in the circular buffer.
  }
  constexpr size_t Mask(void) const noexcept {return mask;}
  inline T& operator[](size_t idx) noexcept {return buf[idx];}
  inline const T& operator[](size_t idx) const noexcept { return buf[idx]; }
private:
  const size_t maxlen{Maxlen}; // The maximum length of the delay line.
  static constexpr size_t mask=Maxlen-1;// Mask for the circular buffer.
  std::array<T,Maxlen> buf{};           // The circular buffer.
  size_t widx{0};                       // The write index.
  size_t delay{1};                     // The delay in samples.
  inline void Advance(void) noexcept {widx=(widx+1)&mask;}// Advance the write index.
  static_assert((Maxlen & (Maxlen - 1)) == 0, "maxlen must be a power of two for cheap wrap-around.");
};
  template <typename T=float,
    size_t MaxLen=1024,
    typename packet=std::experimental::native_simd<T>> 
  class DelayLineSIMD
  {
    public:
      DelayLineSIMD(void) noexcept
      {
        buf.fill(packet(T(0))); // Initialize the buffer with zero.
        head=0; // Initialize the head index to 0.
      }
      ~DelayLineSIMD(void) noexcept = default; // Default destructor
    
      inline void Clear(void) noexcept
      {
        std::fill(buf.begin(),buf.end(),packet(T(0)));
        head=0;
      }
      inline void Advance(void) noexcept
      {
        head=(head+1)&mask; // Advance the head index.
      }
      void WriteAt(size_t i, const packet& x) noexcept
      {
        if (i < 0 || i >= MaxLen) return; // Check if the index is out of bounds.
        size_t pos=(head-1-i+MaxLen)&mask; // Calculate the position in the circular buffer.
        buf[pos]=x; // Write the sample at the specified index.
      }
      inline packet Peek(size_t offset) const noexcept
      {
        if (offset < 0 || offset >= MaxLen) return packet(T(0)); // Check if the offset is out of bounds.
        size_t pos=(head-1-offset+MaxLen)&mask;
        return buf[pos]; // Return the sample at the specified index.
      }
      // Scalar access inside Farrow filters
      T PeekScalar(
        size_t offset,                       // Offset from the head of the buffer
        size_t lane) const noexcept          // Lane index for SIMD access
      {
        if (offset<0||offset>=MaxLen) return T(0); // Check if the offset is out of bounds.
        size_t pos=(head-1-offset+MaxLen)&mask; // Calculate the position in the circular buffer.
        return buf[pos][lane]; // Return the sample at the specified index and lane.
      }
      void AccumulateScalar(
        size_t i,                            // Index in the circular buffer
        size_t lane,                         // Lane index for SIMD access
        T x) noexcept                        // The sample to accumulate
      {
        if (i<0||i>=MaxLen) return; // Check if the index is out of bounds.
        size_t pos=(head-1-i+MaxLen)&mask; // Calculate the position in the circular buffer.
        buf[pos][lane]+=x; // Accumulate the sample at the specified index and lane.
      }
      inline size_t GetHead(void) const noexcept { return head; } // Get the current head index.
      inline size_t GetMaxLen(void) const noexcept { return MaxLen; } // Get the maximum length of the delay line.
      inline packet& operator[](size_t idx) noexcept
      {
        assert(idx < MaxLen); // Ensure the index is within bounds.
        return buf[idx & mask]; // Return the sample at the specified index in the circular buffer.
      }
      inline const packet& operator[](size_t idx) const noexcept
      {
        if (idx < MaxLen) // Ensure the index is within bounds.
          return packet(T(0)); // Return a zero vector if the index is out of bounds.
        return buf[idx & mask]; // Return the sample at the specified index in the circular buffer.
      }
      inline packet PeekIndex(size_t idx) const noexcept
      {
        if (idx < MaxLen) // Ensure the index is within bounds.
          return packet(T(0)); // Return a zero vector if the index is out of bounds.
        return buf[idx & mask]; // Return the sample at the specified index in the circular buffer.
      }
    private:
      std::array<packet,MaxLen> buf{};
      static constexpr size_t mask=MaxLen-1;// Mask for the circular buffer.
      size_t head{0}; // Head index
  };                              
} // namespace sig