 /* 
* *
* * Filename: FCWSIMD.hpp
* *
* * Description:
* *  Fast C++ SIMD-optimized (AVX2) implementations of:
* *   - MDCT/IMDCT (Modified Discrete Cosine Transform) via DCT-IV
* *   - DCT-I, DCT-II, DCT-III
* *   - FFT/IFFT
* *   - STFT/ISTFT (Short-Time Fourier Transform)
* *   - Wavelet transforms (DWT/IDWT/CWT/ICWT)
* *   - PSD (Periodogram, Welch)
* *
* * Author:
* *  JEP, J. Enrique Peraza
* *
* * Organizations:
* *  Trivium Solutions, LLC, 9175 Guilford Rd, Suite 220, Columbia, MD 21046
*/
#pragma once
#include <vector>
#include <complex>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <functional>
#include <string>
#include <type_traits>
#include <algorithm>
#if defined(__x86_64__)||defined(_M_X64)||defined(__i386__)
  #include <immintrin.h>
#endif
#include "FCWTransforms.h"
#include "DSPWindows.h"

// Minimal status type (no exceptions). If caller defines sdr::Status elsewhere,
// this identical enum should be ABI-compatible; otherwise this provides it.
namespace sdr{enum class Status:int{Ok=0,Err=1,Invalid=2,Unimpl=3};}

namespace sig::spectral::simd{
  using std::vector;using std::complex;using sdr::Status;
  using ::sig::spectral::WaveletOps;using ::sig::spectral::Window;

  // ================================
  // Tiny SIMD helpers (x86 AVX2)
  // ================================
  namespace detail
  {
    inline bool HasAVX2 (void) noexcept
    {
      #if defined(__AVX2__)
        return true;
      #else
        return false;
      #endif
    }
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  inline float Dot8f (
      const float* a,                   // Input signal
      const float* b,                   // Input signal 2
      size_t n) noexcept                // Length of vectors
    {                                   // ~~~~~~~~~~~ Dot8 ~~~~~~~~~~~~~ //
      float acc=0.0f;                   // Accumulator
      size_t i=0;                       // Loop index
      #if defined(__AVX2__)             // SIMD enabled?
        __m256 vacc=_mm256_setzero_ps();// Zero accumulator
        for(;i+8<=n;i+=8)               // Process 8 at a time
        {
            __m256 va=_mm256_loadu_ps(a+i);// Load 8 floats
            __m256 vb=_mm256_loadu_ps(b+i);// Load 8 floats
            vacc=_mm256_fmadd_ps(va,vb,vacc);// Fused multiply-add
        }
        alignas(32)                     // Align for storage in 32-byte boundary
        float tmp[8];                   // Temp storage
          _mm256_storeu_ps(tmp,vacc);     // Store accumulator (unaligned for safety)
        // Horizontal add
        acc=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
      #endif
      for(;i<n;++i)                     // Remainder
        acc+=a[i]*b[i];                 // Scalar dot
      return acc;                       // Return result
    }                                   // ~~~~~~~~~~~ Dot8 ~~~~~~~~~~~~~ // 
  inline double Dot4d (
      const double* a,                  // Input signal
      const double* b,                  // Input signal 2
      size_t n) noexcept                // Length of vectors
    {                                   // ~~~~~~~~~~~ Dot4d ~~~~~~~~~~~~~ //
      double acc=0.0;                   // Accumulator
      size_t i=0;                       // Loop index
      #if defined(__AVX2__)             // SIMD enabled?
        __m256d vacc=_mm256_setzero_pd();// Zero accumulator
        for(;i+4<=n;i+=4)               // Process 4 at a time
        {
            __m256d va=_mm256_loadu_pd(a+i);// Load 4 doubles
            __m256d vb=_mm256_loadu_pd(b+i);// Load 4 doubles
            vacc=_mm256_fmadd_pd(va,vb,vacc);// Fused multiply-add
        }
        alignas(32)                     // Align for storage in 32-byte boundary
        double tmp[4];                  // Temp storage
          _mm256_storeu_pd(tmp,vacc);     // Store accumulator (unaligned for safety)
        acc=tmp[0]+tmp[1]+tmp[2]+tmp[3];// Horizontal add
      #endif
      for(;i<n;++i)                     // Remainder
        acc+=a[i]*b[i];                 // Scalar dot
      return acc;                       // Return result
    }                                   // ~~~~~~~~~~~ Dot4d ~~~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Fused AXPY: y=a*x+b (fused when AVX2 FMA is present)
    // y[i]=a[i]*c+b[i];
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    inline void AxpyMul (
      float* y,                         // Output
      const float* a,                   // Input signal
      const float* b,                   // Input signal 2
      float c,                          // Scalar
      size_t n) noexcept                // Length of vectors
      {                                 // ~~~~~~~~~~~ AxpyMul ~~~~~~~~~~~~~ //
      size_t i=0;                       // Loop index
      #if defined(__AVX2__)             // SIMD enabled?
        __m256 vc=_mm256_set1_ps(c);    // Broadcast scalar
        for(;i+8<=n;i+=8)               // Process 8 at a time
        {                               // Loop body
            __m256 va=_mm256_loadu_ps(a+i);// Load 8 floats
            __m256 vb=_mm256_loadu_ps(b+i);// Load 8 floats
            __m256 vy=_mm256_fmadd_ps(va,vc,vb);// Fused multiply-add
            _mm256_storeu_ps(y+i,vy);   // Store result
        }                               // Loop body
      #endif
      for(;i<n;++i)                     // Remainder
        y[i]=a[i]*c+b[i];               // Scalar
    }                                   // ~~~~~~~~~~~ AxpyMul ~~~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // y += a*x (fused when AVX2 FMA is present)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    inline void Axpy (
      float* y,                         // Output
      const float* x,                   // Input signal
      float a,                          // Scalar
      size_t n) noexcept
    {                                   // ~~~~~~~~~~~ Axpy ~~~~~~~~~~~~~ //
      size_t i=0;                       // Loop index
      #if defined(__AVX2__)             // SIMD enabled?
        __m256 va=_mm256_set1_ps(a);    // Broadcast scalar
        for(;i+8<=n;i+=8)               // Process 8 at a time
        {                               // Loop body
          __m256 vx=_mm256_loadu_ps(x+i);// Load 8 floats
          __m256 vy=_mm256_loadu_ps(y+i);// Load 8 floats
          vy=_mm256_fmadd_ps(vx,va,vy); // Fused multiply-add
          _mm256_storeu_ps(y+i,vy);     // Store result
        }                               // Loop body
      #endif
      for(;i<n;++i)                     // Remainder
        y[i]+=a*x[i];                   // Scalar
    }                                   // ~~~~~~~~~~~ Axpy ~~~~~~~~~~~~~ //
    inline void Mul (
      float* y,                         // Output
      const float* x,                   // Input signal
      const float* w,                   // Weights
      size_t n) noexcept                // Length of vectors
    {                                   // ~~~~~~~~~~~ Mul ~~~~~~~~~~~~~ //
      size_t i=0;                       // Loop index
      #if defined(__AVX2__)
        for(;i+8<=n;i+=8)               // Process 8 at a time
        {                               // Loop body
          __m256 vx=_mm256_loadu_ps(x+i);// Load 8 floats
          __m256 vw=_mm256_loadu_ps(w+i);// Load 8 floats
          __m256 vy=_mm256_mul_ps(vx,vw);// Element-wise multiply
          _mm256_storeu_ps(y+i,vy);     // Store result
        }                               // Loop body
      #endif
      for(;i<n;++i)                     // Remainder
        y[i]=x[i]*w[i];                 // Scalar
    }                                   // ~~~~~~~~~~~ Mul ~~~~~~~~~~~~~ //
    inline void Square (
      const float* x,                   // Input
      float* y,                         // Output
      size_t n) noexcept                // Length of vectors
    {                                   // ~~~~~~~~~~~ Square ~~~~~~~~~~~~~ //
      size_t i=0;                       // Loop index
      #if defined(__AVX2__)
        for(;i+8<=n;i+=8)               // Process 8 at a time
        {                               // Loop body
          __m256 vx=_mm256_loadu_ps(x+i);// Load 8 floats
          __m256 vy=_mm256_mul_ps(vx,vx);// Element-wise square
          _mm256_storeu_ps(y+i,vy);     // Store result
        }                               // Loop body
      #endif
      for(;i<n;++i)                     // Remainder
        y[i]=x[i]*x[i];                 // Scalar
    }                                   // ~~~~~~~~~~~ Square ~~~~~~~~~~~~~ //
    inline void Square (
      const double* x,
      double* y,
      size_t n) noexcept
    {
      for(size_t i=0;i<n;++i)
        y[i]=x[i]*x[i];
    }
    inline size_t NextPow2(size_t n) noexcept 
    {
      if(n==0)
        return 1;
      --n;
      for(size_t s=1;s<sizeof(size_t)*8;s<<=1)
        n|=n>>s;
      return n+1;
    }
  }

  // =====================================================================
  // WaveletOps SIMD (focus on hot inner loops: thresholding & Haar steps)
  // =====================================================================
  template<typename T=double>
  class WaveletOpsSIMD
  {
    public:
      using WT=typename WaveletOps<T>::WaveletType;
      using TT=typename WaveletOps<T>::ThresholdType;
      // Vector hard-threshold detail coefficients (SIMD where possible)
      static Status HardThreshold (
        const vector<T>& det,           // Detail coeffs
        T thr,                          // Threshold
        vector<T>& out) noexcept        // Output coeffs
      {                                 // ~~~~~~~~~~~ HardThreshold ~~~~~~~~~~~~~ //
        size_t n=det.size();            // Length of vector
        out.resize(n);                  // Resize output vector
        if constexpr(std::is_same<T,float>::value)
        {                               // Is it a float?
          const float t=static_cast<float>(thr);// cast threshold to float
          size_t i=0;                   // Loop index
          #if defined(__AVX2__)         // AVX2 path
            __m256 vt=_mm256_set1_ps(t);// Broadcast threshold
            for(;i+8<=n;i+=8)           // Process 8 at a time
            {                           // Loop body
              __m256 vx=_mm256_loadu_ps(det.data()+i);// Load 8 floats
              __m256 av=_mm256_andnot_ps(_mm256_set1_ps(-0.0f),vx);// abs(x)
              __m256 m=_mm256_cmp_ps(av,vt,_CMP_LT_OQ);// mask abs(x)<t
              __m256 zr=_mm256_setzero_ps();// zero
              __m256 vy=_mm256_blendv_ps(vx,zr,m);// blend
              _mm256_storeu_ps(out.data()+i,vy);// store result
            }                           // Loop body
          #endif
          for(;i<n;++i)                 // Remainder
          {                             // Loop body
            float v=static_cast<float>(det[i]);// Load value
            out[i]=(std::fabs(v)<t)?0.0f:v;// Threshold
          }                             // Loop body
          return Status::Ok;            // Success
        }                               // Done with float samples
        else                            // Else it is a double
        {                               // Double path (no SIMD)
          for(size_t i=0;i<n;++i)       // For the # of samples in the signal
          {                             // threshold.....
            T v=det[i];                 // Load value
            out[i]=(std::fabs(v)<thr)?T(0):v;// Threshold
          }                             // Done thresholding
          return Status::Ok;            // Success
        }                               // Done with double samples
      }                                 // ~~~~~~~~~~~ HardThreshold ~~~~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Vector soft-threshold detail coefficients (SIMD where possible)
      // y=sign(x)*max(|x|-thr,0)
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      static Status SoftThreshold (
        const vector<T>& det,           // Detail coeffs
        T thr,                          // Threshold
        vector<T>& out) noexcept        // Output coeffs
      {                                 // ~~~~~~~~~~~ SoftThreshold ~~~~~~~~~~~~~ //
        size_t n=det.size();            // Length of vector
        out.resize(n);                  // Resize output vector
        if constexpr(std::is_same<T,float>::value)// Is it a float?
        {                               // Float path
          const float t=static_cast<float>(thr);// cast threshold to float
          size_t i=0;                   // Loop index
          #if defined(__AVX2__)         // AVX2 path
            __m256 vt=_mm256_set1_ps(t);// Broadcast threshold
            __m256 vz=_mm256_setzero_ps();// Zero
            for(;i+8<=n;i+=8)           // Process 8 at a time
            {                           // Loop body
              __m256 vx=_mm256_loadu_ps(det.data()+i);// Load 8 floats
              __m256 s=_mm256_and_ps(_mm256_set1_ps(-0.0f),vx);// sign(x)
              __m256 av=_mm256_andnot_ps(_mm256_set1_ps(-0.0f),vx);// abs(x)
              __m256 sh=_mm256_sub_ps(av,vt);// |x|-t
              sh=_mm256_max_ps(sh,vz);  // max(|x|-t,0)
              __m256 vy=_mm256_or_ps(sh,s);// sign(x)*max(|x|-t,0)
              _mm256_storeu_ps(out.data()+i,vy);// store result
            }                           // Loop body
          #endif
          for(;i<n;++i)                 // Remainder
          {                             // Loop body
            float v=static_cast<float>(det[i]);// Load value
            float a=std::fabs(v)-t;     // |x|-t
            out[i]=(a>0.0f)?std::copysign(a,v):0.0f;// Threshold
          }                             // Done with remainder thresholding
          return Status::Ok;            // Success
        }                               // Done with float samples
        else                            // Else it is a double
        {                               // Double path (no SIMD)
          for(size_t i=0;i<n;++i)       // For the # of samples in the signal
          {                             // threshold.....
            T v=det[i];                 // Load detail coefficient
            T a=std::fabs(v)-thr;       // |x|-t
            out[i]=(a>0)?std::copysign(a,v):T(0);// Threshold
          }                             // Done with thresholding
          return Status::Ok;            // Success
        }                               // Done with double samples
      }                                 // ~~~~~~~~~~~ SoftThreshold ~~~~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // One-level Haar DWT (approx,detail) using averaging and differencing
      // sig length must be even
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      static Status HaarLevel (
        const vector<T>& sig,           // Input signal
        vector<T>& app,                 // Approximation coefficients
        vector<T>& det) noexcept        // Detail coefficients
      {                                 // ~~~~~~~~~~~ HaarLevel ~~~~~~~~~~~~~ //
        size_t n=sig.size();            // Length of signal
        if(n<2)                         // Must be at least 2 samples
          return Status::Invalid;       // Invalid
        if(n%2!=0)                      // Must be even length
          return Status::Invalid;       // Invalid
        size_t h=n/2;                   // Half-length
        app.resize(h);                  // Resize output
        det.resize(h);                  // Resize output
        for(size_t i=0;i<h;++i)         // For each output coeff
        {                               // Haar step
          T s0=sig[2*i];                // Even index: approx coeffs
          T s1=sig[2*i+1];              // Odd index: detail coeffs
          app[i]=T(1/std::sqrt(2.0))*(s0+s1);// 1/sqrt(2)
          det[i]=T(1/std::sqrt(2.0))*(s0-s1);// 1/sqrt(2)
        }                               // Done Haar steps
        return Status::Ok;              // Success
      }                                 // ~~~~~~~~~~~ HaarLevel ~~~~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // One-level inverse Haar (reconstruct signal from approx,detail)
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      static Status HaarInverse (
        const vector<T>& app,
        const vector<T>& det,
        vector<T>& sig) noexcept 
      {
        size_t h=app.size();
        if(h==0||h!=det.size())
          return Status::Invalid;
        sig.resize(h*2);
        if constexpr(std::is_same<T,float>::value)
        {
          for(size_t i=0;i<h;++i)
          {
            float a=static_cast<float>(app[i]);
            float d=static_cast<float>(det[i]);
            sig[2*i]=static_cast<T>(0.7071067811865475f*(a+d));
            sig[2*i+1]=static_cast<T>(0.7071067811865475f*(a-d));
          }
          return Status::Ok;
        }
        else
        {
          for(size_t i=0;i<h;++i)
          {
            T a=app[i];
            T d=det[i];
            sig[2*i]=T(0.70710678118654752440)*(a+d);
            sig[2*i+1]=T(0.70710678118654752440)*(a-d);
          }
          return Status::Ok;
        }
      }

      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Additional filter banks (wrap)
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Helper: convert vector<T> -> vector<double>
      static inline vector<double> ToD (
        const vector<T>& x) noexcept
      {
        vector<double> y(x.size());
        for(size_t i=0;i<x.size();++i)
          y[i]=static_cast<double>(x[i]);
        return y;
      }
      static inline vector<T> FromD (
        const vector<double>& x) noexcept
      {
        vector<T> y(x.size());
        for(size_t i=0;i<x.size();++i)
          y[i]=static_cast<T>(x[i]);
        return y;
      }
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Db1 (alias Haar)
      static Status Db1Level (
        const vector<T>& sig,
        vector<T>& app,
        vector<T>& det) noexcept
      {
        return HaarLevel(sig, app, det);
      }
      static Status Db1Inverse (
        const vector<T>& app,
        const vector<T>& det,
        vector<T>& sig) noexcept
      {
        return HaarInverse(app, det, sig);
      }

      // Db6 via legacy WaveletOps<double> forward/backward (scalar correctness path)
      static Status Db6Level (
        const vector<T>& sig,                  // Input signal
        vector<T>& app,                        // Approximation coefficients
        vector<T>& det) noexcept               // Detail coefficients
      {
        vector<double> sd=ToD(sig);
        WaveletOps<double> w;
        auto pr=w.db6(sd);
        app=FromD(pr.first);
        det=FromD(pr.second);
        return Status::Ok;
      }
      
      static Status Db6Inverse (
        const vector<T>& app,
        const vector<T>& det,
        vector<T>& sig) noexcept
      {
        vector<double> ad=ToD(app),dd=ToD(det);
        WaveletOps<double> w;
        vector<double> rd=w.inverse_db6(ad,dd);
        sig=FromD(rd);
        return Status::Ok;
      }

      // Sym5
      static Status Sym5Level (
        const vector<T>& sig,
        vector<T>& app,
        vector<T>& det) noexcept
      {
        vector<double> sd=ToD(sig);
        WaveletOps<double> w;
        auto pr=w.sym5(sd);
        app=FromD(pr.first);
        det=FromD(pr.second);
        return Status::Ok;
      }
      static Status Sym5Inverse (
        const vector<T>& app,
        const vector<T>& det,
        vector<T>& sig) noexcept
      {
        vector<double> ad=ToD(app),dd=ToD(det);
        WaveletOps<double> w;
        vector<double> rd=w.inverse_sym5(ad,dd);
        sig=FromD(rd);
        return Status::Ok;
      }

      // Sym8
      static Status Sym8Level (
        const vector<T>& sig,
        vector<T>& app,
        vector<T>& det) noexcept
      {
        vector<double> sd=ToD(sig);
        WaveletOps<double> w;
        auto pr=w.sym8(sd);
        app=FromD(pr.first);
        det=FromD(pr.second);
        return Status::Ok;
      }
      static Status Sym8Inverse (
        const vector<T>& app,
        const vector<T>& det,
        vector<T>& sig) noexcept
      {
        vector<double> ad=ToD(app),dd=ToD(det);
        WaveletOps<double> w;
        vector<double> rd=w.inverse_sym8(ad,dd);
        sig=FromD(rd);
        return Status::Ok;
      }

      // Coif5
      static Status Coif5Level (
        const vector<T>& sig,
        vector<T>& app,
        vector<T>& det) noexcept
      {
        vector<double> sd=ToD(sig);
        WaveletOps<double> w;
        auto pr=w.coif5(sd);
        app=FromD(pr.first);
        det=FromD(pr.second);
        return Status::Ok;
      }
      static Status Coif5Inverse (
        const vector<T>& app,
        const vector<T>& det,
        vector<T>& sig) noexcept
      {
        vector<double> ad=ToD(app),dd=ToD(det);
        WaveletOps<double> w;
        vector<double> rd=w.inverse_coif5(ad,dd);
        sig=FromD(rd);
        return Status::Ok;
      }
    // DWT multilevel
    static Status DWTMultilevel (
        const vector<T>& sig,           // Input signal
        WT wtype,                       // Wavelet type
        size_t levels,                  // Decomposition levels
        TT ttype,                       // Threshold type
        T thr,                          // Threshold
        vector<vector<T>>& app,         // Approx coeffs per level
        vector<vector<T>>& det) noexcept// Detail coeffs per level
    {                                   // ~~~~~~~~~~~ DWTMultilevel ~~~~~~~~~~~~~ //
      vector<T> s=sig;                  // Copy input signal
      size_t n=s.size();                // Length of signal
      size_t np=detail::NextPow2(n);    // Next power of 2
      if(np!=n)                         // Is the signal length a power of 2?
        s.resize(np,T(0));              // No, pad with zeros
      app.clear();                      // Clear output approx coeffs
      det.clear();                      // Clear output detail coeffs
      app.reserve(levels);              // Reserve space for approx coeffs
      det.reserve(levels);              // Reserve space for detail coeffs
      for(size_t l=0;l<levels;++l)      // For each level
      {                                 // Decompose the signal
        vector<T> a,d;                  // Approx and detail coeffs
        Status st;                      // Status
        switch (wtype)                  // Dispatch on wavelet type
        {
          case WT::Haar:st=HaarLevel(s,a,d);break;
          case WT::Db6:st=Db6Level(s,a,d);break;
          case WT::Sym5:st=Sym5Level(s,a,d);break;
          case WT::Sym8:st=Sym8Level(s,a,d);break;
          case WT::Coif5:st=Coif5Level(s,a,d);break;
          default:return Status::Unimpl;
        }
        if (st!=Status::Ok)             // Check status
          return st;                    // Return on error
        if (ttype!=TT::None)            // Thresholding requested?
        {                               // Yes
          vector<T> dt;                 // Temp storage
          switch (ttype)                // Dispatch on threshold type
          {
            case TT::Hard:st=HardThreshold(d,thr,dt);break;
            case TT::Soft:st=SoftThreshold(d,thr,dt);break;
            default:return Status::Unimpl;
          }
          if (st!=Status::Ok)           // Check status
            return st;                  // Return on error
          d=std::move(dt);              // Move thresholded detail coeffs
        }                               // Done thresholding
        app.push_back(std::move(a));    // Store approx coeffs
        det.push_back(std::move(d));    // Next input is current approx coeffs
        s=app.back();                   // for next level
      }                                 // Done levels
      return Status::Ok;                // Success
    }                                   // ~~~~~~~~~~~ DWTMultilevel ~~~~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Inverse DWT multilevel (reconstruct signal from app,det coeffs)
      // Assumes app and det are vectors of vectors, each inner vector is
      // the coeffs for that level, with the last element of app being
      // the coarsest approximation.
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    static Status IDWTMultilevel (
      const vector<vector<T>>& app,     // Approx coeffs per level
      const vector<vector<T>>& det,     // Detail coeffs per level
      WT wtype,                         // Wavelet type
      vector<T>& sig) noexcept          // Output signal
    {                                   // ~~~~~~~~~~~ IDWTMultilevel ~~~~~~~~~~~~~ //
      if(app.empty()||app.size()!=det.size())// Bad args?
        return Status::Invalid;         // Yes, return invalid status
      size_t n=app.back().size();       // Length of coarsest approx coeffs
      sig=app.back();                   // Start with coarsest approx coeffs
      // Iterate from coarsest to finest level
      for(std::ptrdiff_t l=static_cast<std::ptrdiff_t>(app.size())-1;l>=0;--l)
      {
        vector<T> s;                    // Temp storage
        Status st;                      // Status variable
        switch (wtype)                  // Dispatch on wavelet type
        {                               // to reconstruct signal
          case WT::Haar:st=HaarInverse(sig,det[l],s);break;
          case WT::Db6:st=Db6Inverse(sig,det[l],s);break;
          case WT::Sym5:st=Sym5Inverse(sig,det[l],s);break;
          case WT::Sym8:st=Sym8Inverse(sig,det[l],s);break;
          case WT::Coif5:st=Coif5Inverse(sig,det[l],s);break;
          default:return Status::Unimpl;
        }                               // Check status
        if (st!=Status::Ok)             // Error?
          return st;                    // Yes, return error status
        sig=std::move(s);               // Move reconstructed signal
      }                                 // Done levels
      sig.resize(n);                    // Trim to original length
      return Status::Ok;                // Success
    }                                   // ~~~~~~~~~~~ IDWTMultilevel ~~~~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Continuous Wavelet Transforms (CWT)
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    static Status CWTMultilevel (
        const vector<T>& sig,           // Input signal
        const vector<T>& sca,           // Scales
        std::function<vector<T>(const vector<T>&,T)> wfunc,// Wavelet function
        T thr,                          // Threshold
        const std::string& ttype,       // Threshold type: "soft" or "hard"
        vector<vector<T>>& coeffs)noexcept
      {
        size_t n=sig.size();            // Length of signal
        size_t np=detail::NextPow2(n);  // Next power of 2
        vector<T> pad=sig;              // Copy input signal
        if(np!=n)                       // Is the signal length a power of 2?
          pad.resize(np,T(0));          // No, pad with zeros
        coeffs.clear();                 // Clear output coeffs
        coeffs.reserve(sca.size());     // Reserve space for coeffs
        for (const auto& s:sca)         // For each scale
        {                               // Compute CWT at this scale
          vector<T> cw=wfunc(pad,s);    // CWT coeffs at this scale
          if (ttype=="soft")            // Soft thresholding?
          {                             // Yes
            vector<T> out;              // Temp storage
            SoftThreshold(cw,thr,out);  // Soft threshold
            coeffs.push_back(std::move(out));// Move to output
          }                             // Done soft thresholding
          else                          // Else hard thresholding
          {                             // Yes
            vector<T> out;              // Temp storage
            HardThreshold(cw,thr,out);  // Hard threshold
            coeffs.push_back(std::move(out));// Move to output
          }                             // Done hard thresholding
        }                               // Done scales
        return Status::Ok;              // Success
      }                                 // ~~~~~~~~~~~ CWTMultilevel ~~~~~~~~~~~~~ //
      static Status ICWTMultilevel (
        const vector<vector<T>>& coeffs,// CWT coeffs per scale
        std::function<vector<T>(const vector<T>&,T)> wfunc,// Wavelet function
        const vector<T>& sca,vector<T>& sig) noexcept // Inverse CWT
      {                                 // ~~~~~~~~~~~ ICWTMultilevel ~~~~~~~~~~~~~ //
        if(coeffs.empty()||coeffs.size()!=sca.size())// Bad args?
          return Status::Invalid;       // Yes, return invalid status
        size_t n=coeffs[0].size();      // Length of coeffs
        sig.assign(n,T(0));             // Start with zero signal
        for(size_t i=0;i<coeffs.size();++i)// For each scale
        {                               // Reconstruct contribution at this scale
          vector<T> rec=wfunc(coeffs[i],sca[i]);// Reconstructed signal at this scale
          if(rec.size()!=n)             // Bad size?
            return Status::Invalid;     // Yes, return invalid status
          for(size_t k=0;k<n;++k)       // Accumulate
            sig[k]+=rec[k];             // into output signal
        }                               // Done scales
        return Status::Ok;              // Success
      }                                 // ~~~~~~~~~~~ ICWTMultilevel ~~~~~~~~~~~~~ //
      static Status ScalogramReal (
        const vector<vector<T>>& cwt,   // CWT coeffs per scale
        vector<vector<T>>& out) noexcept // Scalogram (squared magnitude)
      {                                 // ~~~~~~~~~~~ ScalogramReal ~~~~~~~~~~~~~ //
        out.clear();                    // Clear output
        out.reserve(cwt.size());        // Reserve space for output
        for(const auto& v:cwt)          // For each scale
        {                               // Square coeffs
          vector<T> m(v.size());        // Magnitude squared
          if constexpr(std::is_same<T,float>::value)// Is it a float?
          {                             // Yes
            // SIMD square
            detail::Square(reinterpret_cast<const float*>(v.data()),reinterpret_cast<float*>(m.data()),v.size());
          }
          else                          // Else double
          {
            // No SIMD square
            detail::Square(reinterpret_cast<const double*>(v.data()),reinterpret_cast<double*>(m.data()),v.size());
          }
          out.push_back(std::move(m));  // Move to output
        }
        return Status::Ok;              // Success
      }                                 // ~~~~~~~~~~~ ScalogramReal ~~~~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // CWT wavelet PSI (Pointwise) functions
      // Pointwise wavelets
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Morlet: psi(t)=(exp(-t^2/2))*cos(w0*t), w0=5 (default)
      // Mexican Hat: psi(t)=(1-(t^2)/a^2)*exp(-t^2/(2*a^2)), a=1 (default)
      // Meyer: see MeyerVx and MeyerPsi below
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      static inline T MorletPsi (
        T x,                            // Input
        T w0=T(5)) noexcept             // Wavelet parameter
      {
        return std::exp(-x*x/T(2))*std::cos(w0*x);
      }
      static inline T MexicanHatPsi (
        T x,
        T a=T(1)) noexcept
      {
        T a2=a*a;
        return (T(1)-(x*x)/a2)*std::exp(-(x*x)/(T(2)*a2));
      }
      static inline T MeyerVx(T x) noexcept
      {
        if(x<0)
          return T(0);
        else if (x<1)
        {
          T c1=T(35)/T(32),c2=T(35)/T(16),c3=T(21)/T(16),c4=T(5)/T(8);
          return x*x*x*(c1-c2*x+c3*x*x-c4*x*x*x);
        }
        else return T(1);
      }
      static inline T MeyerPsi(T x) noexcept
      {
        T ax=std::fabs(x);
        if (ax<T(1)/T(3))
          return T(1);
        if (ax<=T(2)/T(3))
          return std::sin(T(M_PI)/T(2)*MeyerVx(T(3)*ax-T(1)))*std::cos(T(M_PI)*x);
        return T(0);
      }

      template<typename Psi>
      static Status CWTForward (
        const vector<T>& sig,           // Input signal
        T sc,                           // Scale
        Psi psi,                        // Wavelet function
        vector<T>& cof) noexcept        // Output coeffs
      {                                 // ~~~~~~~~~~~ CWTForward ~~~~~~~~~~~~~ //
        size_t n=sig.size();            // Length of signal
        if (n==0||sc<=T(0))             // Bad args?
          return Status::Invalid;       // Yes, return invalid status
        cof.assign(n,T(0));             // Resize output coeffs
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Build reversed psi kernel for 'same' convolution: h_rev[j]=psi(((n-1)-j)/sc)
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        vector<T> href(n);              // Reversed wavelet
        for (size_t j=0;j<n;++j)        // For each tap
        {                               // Compute reversed wavelet
          T x=(T(n-1)-T(j))/sc;         // Scaled position
          href[j]=psi(x);               // Wavelet value
        }                               // Done building reversed wavelet
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Zero-pad signal on both sides (length n-1 on each) to extract 'same' windows
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        vector<T> pad;                  // Padded signal
        pad.resize(2*n-1,T(0));         // Resize and zero
        for (size_t i=0;i<n;++i)        // Copy signal
          pad[n-1+i]=sig[i];            // to center of padded signal
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Convolve signal with wavelet by sliding dot product over windows of length n
        // Slide dot over windows of length n
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        const T inv=T(1)/std::sqrt(sc); // Normalization factor
        if constexpr(std::is_same<T,float>::value)// Is it a float?
        {                               // Yes
          const float* k=reinterpret_cast<const float*>(href.data());// Wavelet kernel
          const float* p=reinterpret_cast<const float*>(pad.data());// Padded signal
          for(size_t i=0;i<n;++i)      // For each output coeff
          {                            // Compute dot product
            float v=detail::Dot8f(p+i,k,n);// Dot product
            cof[i]=static_cast<T>(v)*inv;// Scale
          }                             // Done output coeff
        }                               // Done float path
        else                            // Else double
        {                               // Double path
          const double* k=reinterpret_cast<const double*>(href.data()); // Wavelet kernel
          const double* p=reinterpret_cast<const double*>(pad.data()); // Padded signal
          for(size_t i=0;i<n;++i)       // For each output coeff
          {                             // Compute dot product
            double v=detail::Dot4d(p+i,k,n);// Dot product
            cof[i]=static_cast<T>(v)*inv;// Scale
          }                             // Done output coeff
        }                               // Done double path
        return Status::Ok;              // Success
      }                                 // ~~~~~~~~~~~ CWTForward ~~~~~~~~~~~~~ //
      // Specific wavelet wrappers
      static Status CWTForwardMorlet (
        const vector<T>& sig,
        T sc,vector<T>& cof,
        T w0=T(5)) noexcept
      {
            return CWTForward(sig,sc,[&](T x){return MorletPsi(x,w0);},cof);
      }
      static Status CWTForwardMexicanHat (
        const vector<T>& sig,
        T sc,vector<T>& cof,
        T a=T(1)) noexcept
      {
            return CWTForward(sig,sc,[&](T x){return MexicanHatPsi(x,a);},cof);
      }
      static Status CWTForwardMeyer (
        const vector<T>& sig,
        T sc,vector<T>& cof) noexcept
      {
            return CWTForward(sig,sc,[&](T x){return MeyerPsi(x);},cof);
      }

      // Approx/Detail style decompositions
      static Status MorletDecompose (
        const vector<T>& s,
        T w0,
        vector<T>& app,
        vector<T>& det) noexcept
        {
          size_t n=s.size();
          app.resize(n);
          det.resize(n);
          T A=T(1)/std::sqrt(T(2)*T(M_PI));
          T si=T(1)/w0;
          for(size_t i=0;i<n;++i)
          {
            T t=T(i)-T(n)/T(2);
            T mor=A*std::exp(-(t*t)/(T(2)*si*si))*std::cos(w0*t/si);
            app[i]=s[i]*mor;
            det[i]=s[i]*(T(1)-mor);
          }
            return Status::Ok;
        }
      static Status MexicanHatDecompose (
        const vector<T>& s,
        T a,
        vector<T>& app,
        vector<T>& det) noexcept
      {
        size_t n=s.size();
        app.resize(n);
        det.resize(n);
        T A=T(2)/(std::sqrt(T(3)*a)*std::pow(T(M_PI),T(0.25)));
        T a2=a*a;
        for(size_t i=0;i<n;++i)
        {
          T t=T(i)-T(n)/T(2);
          T t2=t*t;
          T mex=A*(T(1)-t2/a2)*std::exp(-t2/(T(2)*a2));
          app[i]=s[i]*mex;
          det[i]=s[i]*(T(1)-mex);
        }
        return Status::Ok;
      }
      static Status MeyerDecompose (
        const vector<T>& s,
        vector<T>& app,
        vector<T>& det) noexcept
      {
        size_t n=s.size();
        app.resize(n);
        det.resize(n);
        for(size_t i=0;i<n;++i)
        {
          T t=T(i)-T(n)/T(2);
          T mey=T(0);
          T at=std::fabs(t);
          if (at<T(1)/T(3))
            mey=T(1);
          else if (at<=T(2)/T(3))
            mey=std::sin(T(M_PI)/T(2)*MeyerVx(T(3)*at-T(1)))*std::cos(T(M_PI)*t);
          app[i]=s[i]*mey;
          det[i]=s[i]*(T(1)-mey);
        }
        return Status::Ok;
      }

      // Inverse reconstructions
      static Status InverseMorlet (
        const vector<T>& app,
        const vector<T>& det,
        T w0,
        vector<T>& rec) noexcept
      {
        size_t n=app.size();
        if(det.size()!=n)
          return Status::Invalid;
        rec.assign(2*n,T(0));
        T A=T(1)/std::sqrt(T(2)*T(M_PI));
        T si=T(1)/w0;
        for(size_t i=0;i<n;++i)
        {
          T t=T(2)*T(i)-T(n);
          T mor=A*std::exp(-(t*t)/(T(2)*si*si))*std::cos(w0*t/si);
          rec[2*i]+=app[i]*mor;
          rec[2*i+1]+=det[i]*(T(1)-mor);
        }
        return Status::Ok;
      }
      static Status InverseMexicanHat (
        const vector<T>& app,
        const vector<T>& det,
        T a,
        vector<T>& rec) noexcept
      {
        size_t n=app.size();
        if(det.size()!=n)
          return Status::Invalid;
        rec.assign(2*n,T(0));
        T A=T(2)/(std::sqrt(T(3)*a)*std::pow(T(M_PI),T(0.25)));
        T a2=a*a;
        for(size_t i=0;i<n;++i)
        {
          T t=T(2)*T(i)-T(n);
          T t2=t*t;
          T mex=A*(T(1)-t2/a2)*std::exp(-t2/(T(2)*a2));
          rec[2*i]+=app[i]*mex;
          rec[2*i+1]+=det[i]*(T(1)-mex);
        }
        return Status::Ok;
      }
      static Status InverseMeyer (
        const vector<T>& app,
        const vector<T>& det,
        vector<T>& rec) noexcept
      {
        size_t n=app.size();
        if(det.size()!=n)
          return Status::Invalid;
        rec.assign(2*n,T(0));
        for(size_t i=0;i<n;++i)
        {
          T t=T(2)*T(i)-T(n);
          T at=std::fabs(t);
          T mey=T(0);
          if(at<T(1)/T(3))
            mey=T(1);
          else if(at<=T(2)/T(3))
            mey=std::sin(T(M_PI)/T(2)*MeyerVx(T(3)*at-T(1)))*std::cos(T(M_PI)*t);
          rec[2*i]+=app[i]*mey;
          rec[2*i+1]+=det[i]*(T(1)-mey);
        }
        return Status::Ok;
      }
  };

  // ======================================================
  // DCT SIMD (DCT-II/III/IV via cosine-matrix dot-products)
  // ======================================================
  template<typename T=float>
  class DCTSIMD
  {
    public:
      // Compute DCT-II (orthonormal variant). O(N^2) using SIMD dot-products.
      static Status DCTII (
        const vector<T>& x,
        vector<T>& X) noexcept
      {
        size_t n=x.size();
        if(n==0)
          return Status::Invalid;
        X.assign(n,T(0));
        const T s0=std::sqrt(T(1)/T(n));
        const T s=std::sqrt(T(2)/T(n));
        // Precompute cosines row-wise for cache locality
        vector<T> cosv(n*n,T(0));
        for(size_t k=0;k<n;++k)
        {
          for(size_t i=0;i<n;++i)
          {
            cosv[k*n+i]=std::cos((T(M_PI)/T(n))*(T(i)+T(0.5))*T(k));
          }
        }
        if constexpr(std::is_same<T,float>::value)
        {
          for(size_t k=0;k<n;++k)
          {
            float sc=(k==0)?static_cast<float>(s0):static_cast<float>(s);
            X[k]=static_cast<T>(sc*detail::Dot8f(reinterpret_cast<const float*>(x.data()),reinterpret_cast<const float*>(&cosv[k*n]),n));
          }
        }
        else
        {
          for(size_t k=0;k<n;++k)
          {
            double sc=(k==0)?static_cast<double>(s0):static_cast<double>(s);
            X[k]=static_cast<T>(sc*detail::Dot4d(reinterpret_cast<const double*>(x.data()),reinterpret_cast<const double*>(&cosv[k*n]),n));
          }
        }
        return Status::Ok;
      }
      // DCT-I (orthonormal): X[k]=sqrt(2/(N-1))*a(k)*sum_{n}a(n)*x[n]*cos(pi*n*k/(N-1))
      static Status DCTI (
        const vector<T>& x,
        vector<T>& X) noexcept
      {
        size_t n=x.size();
        if(n<2)
          return Status::Invalid;
        X.assign(n,T(0));
        const T fac=std::sqrt(T(2)/T(n-1));
        vector<T> ax(n,T(0));
        for(size_t i=0;i<n;++i)
        {
          T an=(i==0||i==n-1)?T(1/std::sqrt(2.0)):T(1);
          ax[i]=an*x[i];
        }
        vector<T> cosv(n*n,T(0));
        for(size_t k=0;k<n;++k)
        {
          for(size_t i=0;i<n;++i)
          {
            cosv[k*n+i]=std::cos(T(M_PI)*T(i)*T(k)/T(n-1));
          }
        }
        if constexpr(std::is_same<T,float>::value)
        {
          for(size_t k=0;k<n;++k)
          {
            float ak=(k==0||k==n-1)?float(1/std::sqrt(2.0)):1.0f;
            float sc=float(fac)*ak;
            X[k]=static_cast<T>(sc*detail::Dot8f(reinterpret_cast<const float*>(ax.data()),reinterpret_cast<const float*>(&cosv[k*n]),n));
          }
        }
        else
        {
          for(size_t k=0;k<n;++k)
          {
            double ak=(k==0||k==n-1)?(1/std::sqrt(2.0)):1.0;
            double sc=double(fac)*ak;
            X[k]=static_cast<T>(sc*detail::Dot4d(reinterpret_cast<const double*>(ax.data()),reinterpret_cast<const double*>(&cosv[k*n]),n));
          }
        }
        return Status::Ok;
      }
      // DCT-III (inverse of DCT-II under the same orthonormalization)
      static Status DCTIII (
        const vector<T>& X,
        vector<T>& x) noexcept
      {
        size_t n=X.size();
        if(n==0)
          return Status::Invalid;
        x.assign(n,T(0));
        const T s0=std::sqrt(T(1)/T(n));
        const T s=std::sqrt(T(2)/T(n));
        vector<T> cosv(n*n,T(0));
        for(size_t i=0;i<n;++i)
        {
          for(size_t k=0;k<n;++k)
          {
            cosv[i*n+k]=std::cos((T(M_PI)/T(n))*(T(i)+T(0.5))*T(k));
          }
        }
        vector<T> sck(n,T(0));
        for(size_t k=0;k<n;++k)
        {
          sck[k]=((k==0)?s0:s)*X[k];
        }
        if constexpr(std::is_same<T,float>::value)
        {
          for(size_t i=0;i<n;++i)
          {
            x[i]=static_cast<T>(detail::Dot8f(reinterpret_cast<const float*>(sck.data()),reinterpret_cast<const float*>(&cosv[i*n]),n));
          }
        }
        else
        {
          for(size_t i=0;i<n;++i)
          {
            x[i]=static_cast<T>(detail::Dot4d(reinterpret_cast<const double*>(sck.data()),reinterpret_cast<const double*>(&cosv[i*n]),n));
          }
        }
        return Status::Ok;
      }
      // DCT-IV (orthonormal). Also O(N^2) using SIMD where possible.
      static Status DCTIV (
        const vector<T>& x,
        vector<T>& X) noexcept
      {
        size_t n=x.size();
        if(n==0)
          return Status::Invalid;
        X.assign(n,T(0));
        const T s=std::sqrt(T(2)/T(n));
        vector<T> cosv(n*n,T(0));
        for(size_t k=0;k<n;++k)
        {
          for(size_t i=0;i<n;++i)
          {
            cosv[k*n+i]=std::cos((T(M_PI)/T(n))*(T(i)+T(0.5))*(T(k)+T(0.5)));
          }
        }
        if constexpr(std::is_same<T,float>::value)
        {
          for(size_t k=0;k<n;++k)
          {
            X[k]=static_cast<T>(static_cast<float>(s)*detail::Dot8f(reinterpret_cast<const float*>(x.data()),reinterpret_cast<const float*>(&cosv[k*n]),n));
          }
        }
        else
        {
          for(size_t k=0;k<n;++k)
          {
            X[k]=static_cast<T>(static_cast<double>(s)*detail::Dot4d(reinterpret_cast<const double*>(x.data()),reinterpret_cast<const double*>(&cosv[k*n]),n));
          }
        }
        return Status::Ok;
      }

      // Dispatcher similar to legacy Transform(Type)
      enum class Type{I,II,III,IV};
      static Status Transform (
        const vector<T>& in,
        Type t,
        vector<T>& out) noexcept
      {
        switch(t)
        {
          case Type::I: return DCTI(in,out);
          case Type::II: return DCTII(in,out);
          case Type::III: return DCTIII(in,out);
          case Type::IV: return DCTIV(in,out);
        }
        return Status::Invalid;
      }

      // --------------------- MDCT/IMDCT ---------------------
      // MDCT: input timeBlock size 2N, output N; window must be 2N (PB pair like MLTSine)
      static Status MDCT (
        const vector<T>& tb,
        const Window<T>& win,
        vector<T>& X) noexcept
      {
        size_t L=tb.size();
        if(L==0||win.Size()!=L||L%2!=0)
          return Status::Invalid;
        size_t N=L/2;
        X.assign(N,T(0));
        vector<T> xw(L,0);
        for(size_t i=0;i<L;++i)
          xw[i]=tb[i]*win.GetData()[i];
        const T c=T(M_PI)/T(N);
        // Precompute cosine matrix row-major: k in 0..N-1, n in 0..2N-1
        vector<T> C(N*L,0);
        for(size_t k=0;k<N;++k)
        {
          for(size_t n=0;n<L;++n)
          {
            C[k*L+n]=std::cos(c*(T(n)+T(0.5)+T(N/2))* (T(k)+T(0.5)));
          }
        }
        if constexpr(std::is_same<T,float>::value)
        {
          for(size_t k=0;k<N;++k)
          {
            X[k]=static_cast<T>(detail::Dot8f(reinterpret_cast<const float*>(xw.data()),reinterpret_cast<const float*>(&C[k*L]),L));
          }
        }
        else
        {
          for(size_t k=0;k<N;++k)
          {
            X[k]=static_cast<T>(detail::Dot4d(reinterpret_cast<const double*>(xw.data()),reinterpret_cast<const double*>(&C[k*L]),L));
          }
        }
        return Status::Ok;
      }
      // IMDCT: input N coeffs, output 2N samples; caller should OLA with hop N
      static Status IMDCT (
        const vector<T>& X,
        const Window<T>& win,
        vector<T>& tb) noexcept
      {
        size_t N=X.size();
        size_t L=win.Size();
        if(N==0||L!=2*N)
          return Status::Invalid;
        tb.assign(L,T(0));
        const T c=T(M_PI)/T(N);
        // Precompute cosine matrix column-major for dot-usage: n rows, k cols
        vector<T> C(L*N,0);
        for(size_t n=0;n<L;++n)
        {
          for(size_t k=0;k<N;++k)
          {
            C[n*N+k]=std::cos(c*(T(n)+T(0.5)+T(N/2))* (T(k)+T(0.5)));
          }
        }
        if constexpr(std::is_same<T,float>::value)
        {
          for(size_t n=0;n<L;++n)
          {
            tb[n]=static_cast<T>(detail::Dot8f(reinterpret_cast<const float*>(X.data()),reinterpret_cast<const float*>(&C[n*N]),N));
          }
        }
        else
        {
          for(size_t n=0;n<L;++n)
          {
            tb[n]=static_cast<T>(detail::Dot4d(reinterpret_cast<const double*>(X.data()),reinterpret_cast<const double*>(&C[n*N]),N));
          }
        }
        // Apply synthesis window and normalization (2/N ensures PB perfect recon with OLA)
        vector<T> w=win.GetData();
        T s=T(2)/T(N);
        for(size_t n=0;n<L;++n)
        {
          tb[n]=tb[n]*w[n]*s;
        }
        return Status::Ok;
      }
  };

  // ===================================================================
  // SpectralOps SIMD: windowed DFT (naive) + helpers using Window SIMD
  // ===================================================================
  template<typename T=float>
  class SpectralOpsSIMD
  {
    public:
      using WT=typename Window<T>::WindowType;

      // Apply window into out buffer: out[i]=in[i]*w[i] (SIMD via Window helpers)
      static Status ApplyWindow (
        const vector<T>& in,
        const Window<T>& win,
        vector<T>& out) noexcept
      {
        size_t n=in.size();
        if(n!=win.Size())
          return Status::Invalid;
        out.resize(n);
        if constexpr(std::is_same<T,float>::value)
        {
          detail::Mul(out.data(),in.data(),win.GetData().data(),n);
          return Status::Ok;
        }
        else if constexpr(std::is_same<T,double>::value)
        {
          for(size_t i=0;i<n;++i)
            out[i]=in[i]*win.GetData()[i];
          return Status::Ok;
        }
        else
        {
          for(size_t i=0;i<n;++i)
            out[i]=in[i]*win.GetData()[i];
          return Status::Ok;
        }
      }

      // Naive DFT (scalar for stability). For test-sized N this is fine.
      static Status DFT (
        const vector<std::complex<T>>& x,
        vector<std::complex<T>>& X) noexcept
      {
        size_t n=x.size();
        if(n==0)
          return Status::Invalid;
        X.assign(n,{});
        const T tw=-T(2)*T(M_PI)/T(n);
        // Separate real/imag for better vectorization
        vector<T> xr(n,0),xi(n,0);
        for(size_t i=0;i<n;++i)
        {
          xr[i]=x[i].real();
          xi[i]=x[i].imag();
        }
        for(size_t k=0;k<n;++k)
        {
          T rr=T(0),ri=T(0);
          for(size_t i=0;i<n;++i)
          {
            T ang=tw*T(k)*T(i);
            T c=std::cos(ang);
            T s=std::sin(ang);
            rr += xr[i]*c - xi[i]*s;
            ri += xr[i]*s + xi[i]*c;
          }
          X[k]=std::complex<T>(rr,ri);
        }
        return Status::Ok;
      }

      // Element-wise complex multiply: y=x.*h
      static Status CMul (
        const vector<std::complex<T>>& x,
        const vector<std::complex<T>>& h,
        vector<std::complex<T>>& y) noexcept
      {
        if(x.size()!=h.size())
          return Status::Invalid;
        size_t n=x.size();
        y.resize(n);
        for(size_t i=0;i<n;++i)
        {
          T ar=x[i].real(),ai=x[i].imag(),br=h[i].real(),bi=h[i].imag();
          y[i]=std::complex<T>(ar*br-ai*bi,ar*bi+ai*br);
        }
        return Status::Ok;
      }

      // Magnitude squared for complex vector (SIMD friendly path)
      static Status Mag2 (
        const vector<std::complex<T>>& x,
        vector<T>& m) noexcept
      {
        size_t n=x.size();
        m.resize(n);
        if constexpr(std::is_same<T,float>::value)
        {
          const float* p=reinterpret_cast<const float*>(x.data());
          size_t i=0;
          size_t nn=2*n;
          #if defined(__AVX2__)
            for(;i+8<=nn;i+=8)
            {
              __m256 v=_mm256_loadu_ps(p+i);
              __m256 vr=_mm256_permute_ps(v,0b11011000); // shuffle to pair re/im (approx)
              // Fallback to scalar tail for accuracy; AVX2 pairing for AoS is awkward without more shuffles; keep simple
              break;
            }
          #endif
          for(size_t k=0;k<n;++k)
          {
            float a=x[k].real();
            float b=x[k].imag();
            m[k]=a*a+b*b;
          }
          return Status::Ok;
        }
        else
        {
          for(size_t k=0;k<n;++k)
          {
            double a=x[k].real();
            double b=x[k].imag();
            m[k]=a*a+b*b;
          }
          return Status::Ok;
        }
      }

      // Real convolution (valid mode), small N; AVX used in inner multiply-accumulate
      static Status ConvolveValid (
        const vector<T>& a,
        const vector<T>& b,
        vector<T>& y) noexcept
      {
        if(a.size()<b.size()||b.empty())
          return Status::Invalid;
        size_t na=a.size(),nb=b.size(),ny=na-nb+1;
        y.assign(ny,T(0));
        if constexpr(std::is_same<T,float>::value)
        {
          for(size_t i=0;i<ny;++i)
            y[i]=detail::Dot8f(reinterpret_cast<const float*>(a.data()+i),reinterpret_cast<const float*>(b.data()),nb);
          return Status::Ok;
        }
        else
        {
          for(size_t i=0;i<ny;++i)
            y[i]=static_cast<T>(detail::Dot4d(reinterpret_cast<const double*>(a.data()+i),reinterpret_cast<const double*>(b.data()),nb));
          return Status::Ok;
        }
      }

      // ================== FFT / IFFT (radix-2 iterative, SIMD for float) ==================
      static Status FFT (
        const vector<std::complex<T>>& x,    // Input
        vector<std::complex<T>>& X) noexcept // Output
      {                                 // ~~~~~~~~~~~ FFT ~~~~~~~~~~~~~ //
        size_t n=x.size();              // Length of input
        if (n==0)                       // Bad args?
          return Status::Invalid;       // Yes, return invalid status
        if ((n&(n-1))!=0)               // Not power of 2?
          return Status::Invalid;       // Yes, return invalid status
        X=x;                            // copy input
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Bit-reversal permutation
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        size_t l=0;                     // Number of bits
        for (size_t t=n;t>1;t>>=1)      // Count bits
          ++l;                          // Each halving adds a bit
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Bit-reversal lambda
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        auto brev=[&](size_t v)         
        {                               // Bit-reverse v in l bits
          size_t r=0;                   // Result
          for (size_t i=0;i<l;++i)      // For each bit
          {                             // Extract and insert
            r=(r<<1)|((v>>i)&1);        // Extract bit i and insert
          }                             // Done bits
          return r;                     // Return result
        };                              // Done with lambda bit reversal
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Do bit-reversal permutation
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        for(size_t i=0;i<n;++i)         // For the length of the signal...
        {                               // Bit-reverse-permute
          size_t j=brev(i);             // Bit-reverse index i
          if(j>i)                       // Avoid double swap and self-swap
            std::swap(X[i],X[j]);       // Swap
        }                               // Done bit-reversal permutation
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Butterfly stages
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        for(size_t m=2;m<=n;m<<=1)      // For each section of the butterfly
        {                               // m=butterfly size
          size_t h=m>>1;                // Half-size
          T ang=-T(2)*T(M_PI)/T(m);     // twiddle angle
          T cw=std::cos(ang);           // twiddle cosine
          T sw=std::sin(ang);           // twiddle sine
          for(size_t k=0;k<n;k+=m)      // For each butterfly 
          {                             // starting at k
            T wr=T(1);                  // Initial twiddle
            T wi=T(0);                  // Initial twiddle
            size_t j=0;                 // For each butterfly index
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // SIMD path for float using AVX2
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            if constexpr(std::is_same<T,float>::value) // Float path
            {
              #if defined(__AVX2__)
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              // Process j in chunks of 4 using SIMD. Twiddle changes per j; precompute 4 then vectorize butterflies.
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              for(;j+4<=h;j+=4)         // For each j in chunks of 4
              {                         // Process 4 at a time
                // ~~~~~~~~~~~~~~~~~~~~ //
                // Precompute twiddles for j..j+3 via recurrence
                // ~~~~~~~~~~~~~~~~~~~~ //
                float wrv[4];           // Store for SIMD
                float wiv[4];           // Store for SIMD
                wrv[0]=wr;              // j twiddle
                wiv[0]=wi;              // j twiddle
                // ~~~~~~~~~~~~~~~~~~~~ //
                // Advance wr,wi to j+1
                // j+1
                // ~~~~~~~~~~~~~~~~~~~ //
                float nwr=wr*static_cast<float>(cw)-wi*static_cast<float>(sw);
                float nwi=wr*static_cast<float>(sw)+wi*static_cast<float>(cw);
                wr=nwr;                 // Advance wr to j+1
                wi=nwi;                 // Advance wi to j+1
                wrv[1]=wr;              // j+1
                wiv[1]=wi;              // j+1
                // ~~~~~~~~~~~~~~~~~~~~ //
                // Advance wr,wi to j+2
                // j+2
                // ~~~~~~~~~~~~~~~~~~~ //
                nwr=wr*static_cast<float>(cw)-wi*static_cast<float>(sw);
                nwi=wr*static_cast<float>(sw)+wi*static_cast<float>(cw);
                wr=nwr;                 // Advance wr to j+2
                wi=nwi;                 // Advance wi to j+2
                wrv[2]=wr;              // j+2
                wiv[2]=wi;              // j+2
                // ~~~~~~~~~~~~~~~~~~~~ //
                // Advance wr,wi to j+3
                // j+3
                // ~~~~~~~~~~~~~~~~~~~ //
                nwr=wr*static_cast<float>(cw)-wi*static_cast<float>(sw);
                nwi=wr*static_cast<float>(sw)+wi*static_cast<float>(cw);
                wr=nwr;                 // Advance wr to j+3
                wi=nwi;                 // Advance wi to j+3
                wrv[3]=wr;              // j+3
                wiv[3]=wi;              // j+3
                // ~~~~~~~~~~~~~~~~~~~~ //
                // Now process 4 butterflies at j..j+3 using wrv,wiv
                // Gather u and v (SoA to feed SIMD)
                // ~~~~~~~~~~~~~~~~~~~~ //
                float ur[4];            // u real
                float ui[4];            // u imag
                float vr[4];            // v real
                float vi[4];            // v imag
                for (int t4=0;t4<4;++t4)// t4=0..3
                {                       // Gather u and v
                  auto u=X[k+(j+t4)];   // u
                  auto v=X[k+(j+t4)+h]; // v
                  ur[t4]=u.real();      // u real
                  ui[t4]=u.imag();      // u imag
                  vr[t4]=v.real();      // v real
                  vi[t4]=v.imag();      // v imag
                }                       // Done gather
                __m128 UR=_mm_loadu_ps(ur);// load u real
                __m128 UI=_mm_loadu_ps(ui);// load u imag
                __m128 VR=_mm_loadu_ps(vr);// load v real
                __m128 VI=_mm_loadu_ps(vi);// load v imag
                __m128 WR=_mm_loadu_ps(wrv);// load wr
                __m128 WI=_mm_loadu_ps(wiv);// load wi
                // ~~~~~~~~~~~~~~~~~~~~ //
                // Compute t=wr*v - wi*vi; t=wr*vi + wi*vr
                // tr = wr*vr - wi*vi;  ti = wr*vi + wi*vr
                // o1 = u + t; o2 = u - t
                // ~~~~~~~~~~~~~~~~~~~~ //
                __m128 TR=_mm_sub_ps(_mm_mul_ps(WR,VR), _mm_mul_ps(WI,VI));// tr
                __m128 TI=_mm_add_ps(_mm_mul_ps(WR,VI), _mm_mul_ps(WI,VR));// ti
                __m128 O1R=_mm_add_ps(UR,TR); __m128 O1I=_mm_add_ps(UI,TI);// o1=u+t
                __m128 O2R=_mm_sub_ps(UR,TR); __m128 O2I=_mm_sub_ps(UI,TI);// o2=u-t
                // ~~~~~~~~~~~~~~~~~~~~ //
                // Store back
                // ~~~~~~~~~~~~~~~~~~~~ //
                _mm_storeu_ps(ur,O1R);
                _mm_storeu_ps(ui,O1I);
                _mm_storeu_ps(vr,O2R);
                _mm_storeu_ps(vi,O2I);
                // Store results
                for (int t4=0;t4<4;++t4)
                { 
                  X[k+(j+t4)]=std::complex<float>(ur[t4],ui[t4]);
                }
                for (int t4=0;t4<4;++t4)
                {
                  X[k+(j+t4)+h]=std::complex<float>(vr[t4],vi[t4]);
                }
                // ~~~~~~~~~~~~~~~~~~~~ //
                // wr,wi already advanced to j+3; advance once more to point at j+4
                // ~~~~~~~~~~~~~~~~~~~~ //
                nwr=wr*static_cast<float>(cw)-wi*static_cast<float>(sw);
                nwi=wr*static_cast<float>(sw)+wi*static_cast<float>(cw);
                wr=nwr;                 // Advance wr to j+4
                wi=nwi;                 // Advance wi to j+4
              }                         // Done chunks of 4
              #endif
            }
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // End SIMD path for float
            // Scalar tail or full scalar path
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            for(;j<h;++j)               // For each j in the half-butterfly
            {                           // Scalar path
              std::complex<T> u=X[k+j]; // u
              std::complex<T> v=X[k+j+h];// v
              T vr=v.real();            // v real
              T vi=v.imag();            // v imag
              T tr=wr*vr-wi*vi;         // tr
              T ti=wr*vi+wi*vr;         // ti
              X[k+j]=std::complex<T>(u.real()+tr,u.imag()+ti);
              X[k+j+h]=std::complex<T>(u.real()-tr,u.imag()-ti);
              T nwr=wr*cw-wi*sw;        // Advance wr,wi
              T nwi=wr*sw+wi*cw;        // to next j
              wr=nwr;                   // Next wr
              wi=nwi;                   // Next wi
            }                           // Done j in half-butterfly
          }                             // Done butterflies in section
        }                               // Done sections of butterfly
        return Status::Ok;              // All done
      }                                 // ~~~~~~~~~~~ FFT ~~~~~~~~~~~~~ //
      static Status IFFT (
        const vector<std::complex<T>>& X,
        vector<std::complex<T>>& x) noexcept
        {
        size_t n=X.size();
        if (n==0)
          return Status::Invalid;
        vector<std::complex<T>> tmp=X;
        for(size_t i=0;i<n;++i)
          tmp[i]=std::conj(tmp[i]);
        vector<std::complex<T>> out;
        auto s=FFT(tmp,out);
        if(s!=Status::Ok)
          return s;
        x.resize(n);
        for(size_t i=0;i<n;++i)
          x[i]=std::conj(out[i])/T(n);
        return Status::Ok;
      }

      // ================== STFT / ISTFT ==================
      static Status STFT (
        const vector<std::complex<T>>& sig, // Input signal
        const Window<T>& win,               // Analysis window
        int winSize,                        // Window size
        float ov,                           // Overlap (0..1)
        vector<vector<std::complex<T>>>& frames) noexcept // Output frames
        {                                   // ~~~~~~~~~~~ STFT ~~~~~~~~~~~~~ //
        if(winSize<=0||win.GetWindowsize()!=size_t(winSize))// Bad args?
          return Status::Invalid;           // Yes, return invalid status
        size_t N=winSize;                   // Window size
        size_t H=std::max<size_t>(1,size_t(N*(1.0f-ov)));// Hop size
        if(H==0)                            // Ensure hop size is at least 1
          H=1;                              // Set to 1
        size_t L=sig.size();                // Signal length
        if(L<N)                             // Signal too short?
          return Status::Invalid;           // Yes, return invalid status
        size_t nf=1+(L-N)/H;                // Number of frames
        frames.assign(nf,vector<std::complex<T>>(N));// Allocate output
        vector<T> w=win.GetData();          // Get window data
        vector<std::complex<T>> buf(N);     // Temp buffer
        for(size_t f=0;f<nf;++f)            // For each frame
        {                                   // Frame f
          size_t s=f*H;                     // Start sample
          for(size_t i=0;i<N;++i)           // For each window sample
          {                                 // Apply window
            T wr=w[i];                      // Window value
            auto v=(s+i<L)?sig[s+i]:std::complex<T>(0,0);// Signal value or 0
            buf[i]=std::complex<T>(v.real()*wr,v.imag()*wr);// Windowed value
          }                                 // Done windowing
          FFT(buf,frames[f]);               // FFT into frame
        }                                   // Done frames
        return Status::Ok;                  // return OK
      }                                     // ~~~~~~~~~~ STFT ~~~~~~~~~~~~~ //
      static Status ISTFT (
        const vector<vector<std::complex<T>>>& frames, // Input frames
        const Window<T>& win,           // Synthesis window
        int winSize,                    // Window size
        float ov,                       // Overlap (0..1)
        vector<std::complex<T>>& sig) noexcept // Output signal
      {                                 // ~~~~~~~~~~ ISTFT ~~~~~~~~~~~~~ //
        if(frames.empty())              // No frames?
          return Status::Invalid;       // Yes, return invalid
        size_t N=winSize;               // Window size
        size_t nf=frames.size();        // Number of frames
        size_t H=std::max<size_t>(1,size_t(N*(1.0f-ov)));// Hop size
        vector<T> w=win.GetData();      // Get window data
        size_t L=N+(nf-1)*H;            // Output signal length
        sig.assign(L,std::complex<T>(0,0));// Allocate output
        vector<std::complex<T>> buf(N); // Temp buffer
        for(size_t f=0;f<nf;++f)        // For each frame
        {                               // ISTFT
          IFFT(frames[f],buf);          // IFFT into buffer
          size_t s=f*H;                 // Start sample
          for(size_t i=0;i<N;++i)       // For each window sample
          {                             // Apply window and overlap-add
            T wr=w[i];                  // Window value
            std::complex<T> v=buf[i]*wr;// Windowed value
            if(s+i<L)                   // Within output range?
              sig[s+i]+=v;              // Overlap-add
          }                             // Done windowing.
        }                               // Done OLA for all frames.
        return Status::Ok;              // return OK
      }                                 // ~~~~~~~~~~ ISTFT ~~~~~~~~~~~~~ //

      // ================== PSD (periodogram & Welch) ==================
      static Status PSDPeriodogram (
        const vector<T>& x,             // Input signal
        WT wtype,                       // Window type
        int nfft,                       // FFT size
        vector<T>& psd) noexcept        // Output PSD
      {                                 // ~~~~~~~~~~ PSD ~~~~~~~~~~~~~ //
        if (nfft<=0)                    // Bad args?     
          return Status::Invalid;       // Yes, return invalid status
        size_t N=nfft;                  // FFT size
        vector<std::complex<T>> cx(N);  // Complex buffer
        for(size_t i=0;i<N;++i)         // For each sample in the signal.
        {                               // Copy to complex buffer
          T v=(i<x.size())?x[i]:T(0);   // Get value from signal or zero-pad it
          cx[i]=std::complex<T>(v,0);   // Store in complex buffer
        }                               // Done copy
        Window<T> w(wtype,N);           // Create window
        vector<T> wd=w.GetData();       // Get samples of window
        for(size_t i=0;i<N;++i)         // Apply window
          cx[i]=std::complex<T>(cx[i].real()*wd[i],0);// Windowed value
        vector<std::complex<T>> X;      // where to store Spectrum
        FFT(cx,X);                      // Compute FFT
        psd.resize(N/2+1);              // Allocate output
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // PSD normalization: sum(w^2)
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        T s=T(0);                       // Sum of squares of window
        for(size_t i=0;i<N;++i)         // For the length of the signal
          s+=wd[i]*wd[i];               // Accumulate square
        T norm=T(1)/s;                  // Normalization factor
        for(size_t k=0;k<=N/2;++k)      // For each positive freq bin
        {                               // Compute PSD
          T a=X[k].real();              // Real part
          T b=X[k].imag();              // Imag part
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Magnitude squared times normalization
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          psd[k]=(a*a+b*b)*norm;        // PSD value
        }                               // Done freq bins
        return Status::Ok;              // return OK
      }                                 // ~~~~~~~~~~ PSD ~~~~~~~~~~~~~ //
      static Status PSDWelch (
        const vector<T>& x,             // Input signal
        WT wtype,                       // Window type
        int seg,                        // Segment size
        int overlap,                    // Overlap size
        int nfft,                       // FFT size
        vector<T>& psd) noexcept        // Output PSD
      {                                 // ~~~~~~~~~~ PSD Welch ~~~~~~~~~~~~~ //
        if (seg<=0||nfft<=0||overlap<0||overlap>=seg)// Bad args?
          return Status::Invalid;       // Yes, return invalid status
        size_t N=seg;                   // Segment size
        size_t H=seg-overlap;           // Hop size
        if (x.size()<N)                 // Signal too short?
          return Status::Invalid;       // Yes, return invalid status
        Window<T> w(wtype,N);           // Create window
        vector<T> wd=w.GetData();       // Get samples of window
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // PSD normalization: sum(w^2)
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        T ws=T(0);                      // Sum of squares of window
        for(auto v:wd)                  // For the length of the signal
          ws+=v*v;                      // Accumulate square
        if (ws==T(0))                   // All-zero window?
          return Status::Invalid;       // Yes, return invalid
        T wnorm=T(1)/ws;                // Normalization factor
        size_t nf=1+(x.size()-N)/H;     // Number of frames
        vector<T> acc(nfft/2+1,T(0));   // Accumulator
        vector<std::complex<T>> buf(nfft);// FFT buffer
        for (size_t f=0;f<nf;++f)       // For each frame
        {                               // Calculate Welch's PSD
          size_t s=f*H;                 // Start sample
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Prepare buffer: windowed signal + zero-pad
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          for (size_t i=0;i<(size_t)nfft;++i) // For each FFT bin within the frame
          {                             // Prepare buffer
            T v=(i<N)?x[s+i]*wd[i]:T(0);// Windowed value or zero-pad
            buf[i]=std::complex<T>(v,0);// Store in complex buffer
          }                             // Done preparing buffer
          vector<std::complex<T>> X;    // where to store Spectrum
          FFT(buf,X);                   // Compute FFT
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Accumulate magnitude-squared
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          for (size_t k=0;k<=size_t(nfft/2);++k)// For each positive freq bin
          {                             // Accumulate magnitude-squared
            T a=X[k].real();            // Real part
            T b=X[k].imag();            // Imag part
            acc[k]+= (a*a+b*b)*wnorm;   // Accumulate PSD value
          }                             // Done freq bins
        }                               // Done frames
        psd.resize(nfft/2+1);           // Allocate output
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Average and scale by 1/numFrames to get final PSD
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        T sc=T(1)/T(nf);                // Scale factor
        for (size_t k=0;k<psd.size();++k)// For each freq bin
          psd[k]=acc[k]*sc;             // Average PSD value
        return Status::Ok;              // return OK
      }                                 // ~~~~~~~~~~ PSD Welch ~~~~~~~~~~~~~ //
  }; // class SpectralOpsSIMD
} // namespace SDR

