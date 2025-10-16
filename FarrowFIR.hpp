/*
 * *
 * * Filename: FarrowDelayLine.hpp
 * *
 * * Description:
 * * A time-varying fractional delay using Farrow's method for Lagrange interpolation.
 * *  instead of using the Direct Form FIR filter, we use a Farrow structure
 * *  to compute the fractional delay. A Farrow filter rewrites any polynomial
 * *  based fractional delay FIR as a sum of fixed sub-filter whore outputs are
 * *  combined by powers of mu. It's nice and very stable for audio-rate modulation
 * *  of delay lines, like pitch bends. It's less compuationally consumptive too.
 * *     y[n, mu] = sum_{m=0}^M (mu^m * v_m[n]), with v_m[n] = sum_{k=0}^K (c_{m,k} * x[n - D - k])
 *  * Note: SIMD capable.
 * * 
 * * Where:
 * *   M = polynoimial order (for an N-th order Lagrance Fractional Delay M=N).
 * *   c_{m,k} are the fixed FIR taps (constant at run-time).
 * *   All time-varying cost is in the Horner-style evaluation of sum_{m=0}^M (mu^m * v_m[n]).
 * * Author:
 * *  JEP J.Enrique Peraza
 * *
  */
#pragma once
#include <array>
#include <cstddef>
#include <algorithm>
#include <complex>
#include <cmath>
#include <experimental/simd>
#include "DelayLine.hpp"


template<class pack> inline pack ipow(pack base, size_t n) noexcept
{
  pack r=pack(1);
  for(; n; --n) r*=base;
  return r;
}
namespace sdr::mdm
{

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Helper: Compile-time 2-D array for coefficients matrix (max order +1)^2
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  template<typename T, size_t MaxOrder>
  using CoeffMatrix=std::array<std::array<T,MaxOrder+1>,MaxOrder+1>;
  using cplx=std::complex<float>;
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Polynomial helpers - naive because orders are tine (<=5)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  namespace detail
  {
    // ~~~~~~~~~~~~~~~~~~~~~~~- //
    // Expand product (x-m) for all m in the provided list, normalize by
    // the denominator.
    // ~~~~~~~~~~~~~~~~~~~~~~~- //
    template <typename T, size_t Max>
    void MultiplyMonomial(              // Multiply a monomial (x-m) for all m in the provided list
      std::array<T,Max>& p,               // Polynomial coefficients array
      size_t order,                // Order of the polynomial
      T root,                           // The root to multiply by
      T denom) noexcept                 // The denominator to normalize by
    {                                   // ~~~~~~~~ MultiplyMonomial ~~~~~~~~~~~~ //
        // ~~~~~~~~~~~~~~~~~~~~~~ //
        // Poly holds coefficients up to current degree (inclusive)
        // newpoly=poly*(x-root)/denominator
        // ~~~~~~~~~~~~~~~~~~~~~~ //
        for (size_t i=std::min<std::size_t>(order+1,Max-1);i-->0;)  // For each coefficient in the polynomial.
        {                               // Compute new polynomial coefficients.
            p[i+1]+=p[i]/denom;         // Compute the new coefficient.
            p[i]*=(-root)/denom;        // Update the next coefficient.
        }                               // Done with the new polynomial coefficients.
    }                                   // ~~~~~~~~ MultiplyMonomial ~~~~~~~~~~~~ //
  // Note: The generic template above works for both real and complex types,
  // so a separate complex overload is unnecessary and caused a redefinition.

    // ~~~~~~~~~~~~~~~~~~~~~~~- //
    // Build Lagrance basis polynomial L-k(x) of degree N (0<=k<=N)
    // ~~~~~~~~~~~~~~~~~~~~~~~- //
    template <typename T,size_t Max>
    std::array<T,Max> BuildLagrangeCoeffs(
      size_t k,                         // The index of the Lagrange polynomial to build
      size_t N) noexcept                // The order of the Lagrange polynomial
    {                                   // ~~~~~~~~ BuildLagrangeCoeffs ~~~~~~~~-- //
      std::array<T,Max> coefs{}; // Coefficients of the Lagrange polynomial
      coefs[0]=T(1);                    // Initialize the first coefficient to 1
      size_t deg=0;                     // Degree of the polynomial 
      for (size_t m=0;m<=N;++m)         // For the order of the polynomial
      {                                 // Compute the coefficients of the Lagrange polynomial
        if (m==k)                       // Is the current index equal to K?
          continue;                     // Yes not this one.
        MultiplyMonomial(coefs,deg,static_cast<T>(m),static_cast<T>(k)-static_cast<T>(m));// Multiply the monomial (x-m) for all m in the provided list
        ++deg;                          // Increment the degree of the polynomial
      }                                 // Done with the coefficients of the Lagrange polynomial
      return coefs;                     // Return the coefficients of the Lagrange polynomial   
    }                                   // ~~~~~~~~ BuildLagrangeCoeffs ~~~~~~~~-- //

  }                                     // namespace detail
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Farrow-structure Lagrange Fractional-Delay Filter (order <= MaxOrder)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  template<typename T=float,
  size_t MaxLen=1024,
  size_t MaxOrder=5>
  class FarrowInterpolator
  {
    public:
      using cplx=std::complex<T>;
      constexpr static size_t MAXORDER=MaxOrder;
      ~FarrowInterpolator(void) noexcept = default; // Default destructor
      void SetOrder(
        size_t N) noexcept              // Set the order of the Lagraange Interpolator
      {                                 // ~~~~~~~~ SetOrder ~~~~~~~~~~~~ //                 
        order=std::min<size_t>(N,MaxOrder); 
        BuildCoefficients();           // Build the coefficients for the Lagrange filter
      }                                // ~~~~~~~~ SetOrder ~~~~~~~~~~~~ //
      void SetMu(                      // Set the fractional part of the delay line
       T mu ) noexcept                // The fractional part of the delay 
      {                                  // ~~~~~~~~ SetMu ~~~~~~~~~~~~ //
        this->mu=mu-std::floor(mu);      // Set the fractional part of the delay
      }
      template<typename DL>
      bool Process (
        const DL& minidl,             // Delay line like object
        size_t D,                     // The total delay encoded in size_t
      T* const out) noexcept          // The output stream
      {                               // ~~~~~~~- Process ~~~~~~~~~~~ //
        if (D<order || D>MaxLen-1) return false;
        std::array<T,MaxOrder+1> v{}; // Zero the output buffer
        // ~~~~~~~~~~~~~~~~~~~ //
        // Calculate the coefficients using Horner's method fo mu^m
        // ~~~~~~~~~~~~~~~~~~~ //
        for (size_t k=0;k<=order;++k)// Rows
          v[k]=minidl->Peek(D-k);        // Get the sample at index D+k from the delay line
        // ~~~~~~~~~~~~~~~~~~~- //
        // Polynomial accumulation: Horner's evaluation for current mu
        // y=(((v_N)*mu+n_{N-1})*mu+ ...
        // +v_0)*mu+v_0
        // ~~~~~~~~~~~~~~~~~~~- //
        T cum=coeff[order][order]*v[0];     // seed with highest µ-power term
        for (int m=static_cast<int>(order-1);m>=0;--m) // For each coefficient in the Lagrange polynomial
          cum=cum*mu+coeff[m][order]*v[order-m]; // Compute the output sample by summing the products of the coefficients and the corresponding samples from the delay line
        *out=cum;                    // Store the output sample in the output buffer
        return true;                 // Return true to indicate success
      }    
    bool Process(
      const DelayLine<T,MaxLen>& dl,   // The delay line to process
      size_t D,                     // The fractional delay in samples
      T* const y) noexcept          // Output buffer.
    {                               // ~~~~~~~- Process ~~~~~~~~~~~ //
      if (D<order || D > MaxLen - 1) return false;
      size_t maxD=MaxLen-1-order;   // Maximum delay length.
      if (D>maxD) D=maxD;              // Clamp the delay to the
      // ~~~~~~~~~~~~~~~~~~~- //
      // v_m=S_k (C[m][k]*x[n-D-k]))
      // ~~~~~~~~~~~~~~~~~~~- //
      std::array<T,MaxOrder+1> v{}; // Zero the output buffer
      for (size_t k=0;k<=order;++k) // For each coefficient in the Lagrange polynomial
      {
        const T xk=dl.Peek(D-k);    // Get the sample at index D+k from the delay line
        for (size_t m=0;m<=order;++m)
          v[m]+=coeff[m][k]*xk;    // Compute the output sample by summing the products of the coefficients and the corresponding samples from the delay line
      }                            // Done with the output samples.
      // ~~~~~~~~~~~~~~~~~~~ //
      // Perform Horner's evalueation of coeffiecient expansion
      // y=(((v_N)*mu+n_{N-1})*mu+ ... +v_0)*mu+v_0
      // ~~~~~~~~~~~~~~~~~~~ //
      *y=0;                        // Initialize the output sample with the last coefficient
      for (int i=static_cast<int>(order);i>=0;--i)  // For each coefficient in the Lagrange polynomial
        *y=(*y)*mu+v[i];           // Compute the output sample by summing the products of the coefficients and the corresponding samples from the delay line
      return true;                 // Return true to indicate success
    }                              // ~~~~~~~- Process ~~~~~~~~~~~ //
    bool Process (
      const DelayLine<cplx,MaxLen>& dl, // Delay line like object
      size_t D,                     // The total delay encoded in size_t
      cplx* const out) noexcept     // The output stream
    {                               // ~~~~~~~- Process ~~~~~~~~~~~ //
      if (D<order || D>MaxLen-1) return false;
      std::array<cplx,MaxOrder+1> v{}; // Zero the output buffer
      // ~~~~~~~~~~~~~~~~~~~ //
      // Calculate the coefficients using Horner's method fo mu^m
      // ~~~~~~~~~~~~~~~~~~~ //
      for (size_t k=0;k<=order;++k)// Rows
        v[k]=dl.Peek(D-k);            // Get the sample at index D+k from the delay line
      // ~~~~~~~~~~~~~~~~~~~- //
      // Polynomial accumulation: Horner's evaluation for current mu
      // y=(((v_N)*mu+n_{N-1})*mu+ ...
      // +v_0)*mu+v_0
      // ~~~~~~~~~~~~~~~~~~~- //
      cplx cum=coeff[order][order]*v[0];// seed with highest µ-power term
      for (int m=static_cast<int>(order-1);m>=0;--m) // For each coefficient in the Lagrange polynomial
        cum=cum*mu+coeff[m][order]*v[order-m]; // Compute the output sample by summing the products of the coefficients and the corresponding samples from the delay line
      *out=cum;                    // Store the output sample in the output buffer
      return true;                 // Return true to indicate success
    }
    inline const std::array<T, MaxOrder+1>& operator[](size_t i) const noexcept // ~~~~~~~~ GetCoefficients ~~~~~~~~~~~~ //
    {
      assert(i < MaxOrder+1);
      return coeff[i];
    }
    inline std::array<T, MaxOrder+1>& operator[](size_t i) noexcept // ~~~~~~~~ GetCoefficients ~~~~~~~~~~~~ //
    {
      assert(i < MaxOrder+1);
      return coeff[i];
    }
    private:
    size_t order{3};               // Order of the Lagrange filter, default is 3.
    T mu=T(0);                     // Fractional part of the delay, default is 0.
    CoeffMatrix<T,MaxOrder> coeff{}; // Coefficients for the Lagrange filter, size is order+1.
    void BuildCoefficients(void) noexcept // Build the coefficients for the Lagrange filter
    {                                   // ~~~~~~~~ BuildCoefficients ~~~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~-- //
      //  coeff[m][k]=polynomial coefficients of mu^m for tap k
      // ~~~~~~~~~~~~~~~~~~~~~~-- //
      for (size_t k=0;k<=order;++k)     // For each coefficient in the interpolator
      {
        auto c=detail::BuildLagrangeCoeffs<T,MaxOrder+1>(k,order);
        for (size_t m=0;m<=order;++m)   // For the order of the filter.
          coeff[m][k]=c[m];             // Store the coefficients in the matrix
      }                                 // Done with the coefficients.                   
    }                                   // ~~~~~~~~ BuildCoefficients ~~~~~~~~~~~~ //
  };                                    // class FarrowInterpolator
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Complex Farrow Interpolator
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  template<typename T=float,
    size_t MaxLen=1024,
    size_t MaxOrder=5>
  class FarrowInterpolatorCplx
  {
    public:
      using cplx=std::complex<T>;
      constexpr static size_t MAXORDER=MaxOrder;
      ~FarrowInterpolatorCplx(void) noexcept = default; // Default destructor
      void SetOrder(
        size_t N) noexcept              // Set the order of the Lagraange Interpolator
      {                                 // ~~~~~~~~ SetOrder ~~~~~~~~~~~~ //                 
        order=std::min<size_t>(N,MaxOrder); 
        BuildCoefficients();           // Build the coefficients for the Lagrange filter
      }                                // ~~~~~~~~ SetOrder ~~~~~~~~~~~~ //
      void SetMu(                      // Set the fractional part of the delay line
       T mu ) noexcept                // The fractional part of the delay 
      {                                  // ~~~~~~~~ SetMu ~~~~~~~~~~~~ //
        this->mu=mu-std::floor(mu);      // Set the fractional part of the delay
      }
      bool Process (
        const DelayLine<cplx,MaxLen>& dl, // Delay line like object
        size_t D,                     // The total delay encoded in size_t
        cplx* const out) noexcept       // The output stream
      {                                 // ~~~~~~~- Process ~~~~~~~~~~~ //
        if (D<order || D>MaxLen-1) return false;
        std::array<cplx,MaxOrder+1> v{}; // Sub-filter outputs v_m
        for (size_t k=0;k<=order;++k) // Gather P+1 integer taps
        {
          const cplx xk = dl.Peek(D - k);
          for (size_t m=0;m<=order;++m)
            v[m] += coeff[m][k] * xk;  // Accumulate sub-filter outputs
        }
        // Horner evaluation across mu
        cplx y = v[order];
        for (int m=static_cast<int>(order)-1; m>=0; --m)
          y = y * mu + v[m];
        *out = y;
        return true;
      }                              // ~~~~~~~- Process ~~~~~~~~~~~ //
    private:
      size_t order{3};               // Order of the Lagrange filter, default is 3.
      T mu=T(0);                     // Fractional part of the delay, default is 0.
      CoeffMatrix<cplx,MaxOrder> coeff{}; // Coefficients for the Lagrange filter, size is order+1.
      void BuildCoefficients(void) noexcept // Build the coefficients for the Lagrange filter
      {                                   // ~~~~~~~~ BuildCoefficients ~~~~~~~~~~~~ //
        // ~~~~~~~~~~~~~~~~~~~~~~-- //
        //  coeff[m][k]=polynomial coefficients of mu^m for tap k
        // ~~~~~~~~~~~~~~~~~~~~~~-- //
        for (size_t k=0;k<=order;++k)     // For each coefficient in the interpolator
        {
          auto c=detail::BuildLagrangeCoeffs<cplx,MaxOrder+1>(k,order);
          for (size_t m=0;m<=order;++m)   // For the order of the filter.
            coeff[m][k]=c[m];             // Store the coefficients in the matrix
        }                                 // Done with the coefficients.                   
      }                                   // ~~~~~~~~ BuildCoefficients ~~~~~~~~~~~~ //
  };                                    // class FarrowInterpolatorCplx




  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Farrow Deinterpolator to estimate the fractional delay of where to
  // write the sample in the delay line. As opposed to the interpolator, that determins
  // what the output sample iss, the deinterpolator determines where the input sample
  // should be written in the delay line.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  template<typename T=float,
    size_t MaxLen=1024,
    size_t MaxOrder=5>
  class FarrowDeinterpolator
  {
    public:
      FarrowDeinterpolator(void) noexcept
      {
        BuildCoefficients();           // Build the coefficients for the Lagrange filter
      }
      ~FarrowDeinterpolator(void) noexcept = default; // Default destructor
      /// Note that the order MUST be the same as in the interpolator.
      void SetOrder(size_t N) noexcept // Set the deinterpolator order
      {
        order=std::min<size_t>(N,MaxOrder);// Clamp the order to the maximum order
        BuildCoefficients();           // Build the coefficients for the Lagrange filter
      }                                // ~~~~~~~~ SetOrder ~~~~~~~~~~~~ //
      T GetMu(void) const noexcept    // Get the fractional part of the delay
      {
        return mu;                     // Return the fractional part of the delay
      }
      // Apply the inverse Farrow filter one write (distribute x into fractional slots)
      bool Process(
        T x,                          // The sample to write to the delay line
        DelayLine<T,MaxLen>& dl,      // The delay line to write to (mutable)
        size_t D) noexcept            // The total delay in samples
        {
          if (D<order || D>MaxLen-1) return false;
          std::array<T,MaxOrder+1> v{};   // Zero the output buffer
          // ~~~~~~~~~~~~~~~~~~~ //
          // Calculate the coefficients using Horner's method fo mu^m
          // ~~~~~~~~~~~~~~~~~~~ //
          for (size_t m=0;m<=order;++m)// Rows
          {                            
            for (size_t k=0;k<=order;++k)// Cols
            {
              if (k==0)
                v[m]=coeff[m][k]*x; // Initialize the first coefficient
              else
                v[m]+=coeff[m][k]*std::pow(mu,k); // Compute the coefficient for the current order and tap
            }
          }                              // Done with the coefficients.
          // ~~~~~~~~~~~~~~~~~~~ //
          // Perform additive write to the delay line
          // ~~~~~~~~~~~~~~~~~~~ //
          for (size_t k=0;k<=order;++k)
            dl.WriteAt(D - k, dl.Peek(D - k) + v[k]);
          return true;                 // Return true to indicate success
        }   
        // Coefficient access
      const auto& GetCoeff(void) const noexcept { return coeff; }                             // ~~~~~~~- Process ~~~~~~~~~~~ //
      /// Set fractional delay parameter mu
      void SetMu(T mu_val) noexcept { mu = mu_val - std::floor(mu_val); }
    private:
      size_t order{3};                // Order MUST be the same as in the interpolator.
      T mu{0};                        // Fractional part of the delay, default is 0.
      CoeffMatrix<T,MaxOrder>  coeff{}; // Coefficients for the Lagrange filter, size is order+1.
      void BuildCoefficients(void) noexcept
      {
        for (size_t k=0;k<=order;++k)
        {                              // For each coefficient in the filter...
          auto c=detail::BuildLagrangeCoeffs<T,MaxOrder+1>(k,order);
          for (size_t m=0;m<=order;++m)
            coeff[m][k]=T(c[m]);         // Store the coefficients in the matrix
        }
      }
  };                                  // class FarrowDeinterpolator
  
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Farrow Deinterpolator Complex
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  template<typename T=cplx,
    size_t MaxLen=1024,
    size_t MaxOrder=5>
  class FarrowDeinterpolatorCplx
  {
    public:
      FarrowDeinterpolatorCplx(void) noexcept
      {
        BuildCoefficients();           // Build the coefficients for the Lagrange filter
      }
      ~FarrowDeinterpolatorCplx(void) noexcept = default; // Default destructor
      /// Note that the order MUST be the same as in the interpolator.
      void SetOrder(size_t N) noexcept // Set the deinterpolator order
      {
        order=std::min<size_t>(N,MaxOrder);// Clamp the order to the maximum order
        BuildCoefficients();           // Build the coefficients for the Lagrange filter
      }                                // ~~~~~~~~ SetOrder ~~~~~~~~~~~~ //
      T GetMu(void) const noexcept    // Get the fractional part of the delay
      {
        return mu;                     // Return the fractional part of the delay
      }
      // Apply the inverse Farrow filter one write (distribute x into fractional slots)
      bool Process(
        T x,                            // The sample to write to the delay line
        DelayLine<T,MaxLen>& dl,        // The delay line to write to (mutable)
        size_t D) noexcept              // The total delay in samples
      {                                 // ~~~~~~~~~~~~~ Process ~~~~~~~~~~~~~~ //
        if (D<order || D>MaxLen-1) return false;
        std::array<T,MaxOrder+1> v{};   // Zero the output buffer
        // ~~~~~~~~~~~~~~~~~~~ //
        // Calculate the coefficients using Horner's method fo mu^m
        // ~~~~~~~~~~~~~~~~~~~ //
        for (size_t m=0;m<=order;++m)// Rows
        {                            
          for (size_t k=0;k<=order;++k)// Cols
          {
            if (k==0)
              v[m]=coeff[m][k]*x; // Initialize the first coefficient
            else
              v[m]+=coeff[m][k]*std::pow(mu,k); // Compute the coefficient for the current order and tap
          }
        }                               // Done with the coefficients.
        // ~~~~~~~~~~~~~~~~~~~ //
        // Perform additive write to the delay line
        // This equation does all the work.
        // ~~~~~~~~~~~~~~~~~~~ //
        for (size_t k=0;k<=order;++k)
          dl.WriteAt(D-k,dl.Peek(D-k)+v[k]);
        return true;                 // Return true to indicate success
      }                              // ~~~~~~~~ Process ~~~~~~~~~~~ //
    // Coefficient access
      const auto& GetCoeff(void) const noexcept { return coeff; }                             
      /// Set fractional delay parameter mu
      void SetMu(T mu_val) noexcept { mu = mu_val - std::floor(mu_val); }
    private:
      size_t order{3};                // Order MUST be the same as in the interpolator.
      T mu{0};                        // Fractional part of the delay, default is 0.
      CoeffMatrix<T,MaxOrder>  coeff{}; // Coefficients for the Lagrange filter, size is order+1.
      void BuildCoefficients(void) noexcept
      {
        for (size_t k=0;k<=order;++k)
        {                              // For each coefficient in the filter...
          auto c=detail::BuildLagrangeCoeffs<T,MaxOrder+1>(k,order);
          for (size_t m=0;m<=order;++m)
            coeff[m][k]=T(c[m]);         // Store the coefficients in the matrix
        }
      }
  };                                  // class FarrowDeinterpolatorCplx
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Variable fractional delay line using Farrow's approach to Lagrange interpolation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  template<typename T=float,
    size_t MaxLen=1024,
    size_t MaxOrder=5,
    typename vT = std::experimental::native_simd<T>>
  class FarrowInterpolatorSIMD
  {
    static_assert(MaxOrder <= 5, "MaxOrder must be <= 5 for FarrowInterpolatorSIMD");
    static constexpr size_t VL=vT::size();
    using Coeff=std::array<std::array<vT,MaxOrder+1>,MaxOrder+1>;
    public:
      constexpr static size_t MAXORDER=MaxOrder;
      ~FarrowInterpolatorSIMD(void) noexcept = default; // Default destructor
      void SetOrder(
        size_t N) noexcept              // Set the order of the Lagraange Interpolator
      {                                 // ~~~~~~~~ SetOrder ~~~~~~~~~~~~ //                 
        order=std::min<size_t>(N,MaxOrder); 
        BuildCoefficients();           // Build the coefficients for the Lagrange filter
      }                                // ~~~~~~~~ SetOrder ~~~~~~~~~~~~ // 
    bool Prepare(
      size_t order,
      float f) noexcept
    {
      if (order==0||f<0.0||f>=static_cast<float>(order))
        return false;                // Sanitize input: can't have zero order or invalid delay.
      this->order=std::min<size_t>(order,MaxOrder); // Set the order of the Farrow filter
      mu=vT(f-std::floor(f));         // Set the fractional delay.
      BuildCoefficients();            // Calculate the coefficients for the Farrow filter.
      return true;                    // Return true if preparation was successful.
    }
    /**
     * @brief Processes a sample(s) using the Farrow interpolator.
     * @param dl The delay line to process.
     * @param D The fractional delay in samples.
     * @param y Pointer to output buffer
     * @return 0 is good? value.
     */
    inline vT ProcessFrame(
      const DelayLineSIMD<T,MaxLen,vT>& dl, // The delay line to process
      size_t D) noexcept        // The total delay
    {                               // ~~~~~~~- Process ~~~~~~~~~~~ //
      if (D>order && D<MaxLen-1) return vT(0);
      std::array<vT,MaxOrder+1> v{}; // Zero the output buffer
      size_t maxD=MaxLen-1-order;   // Maximum delay length.
      // ~~~~~~~~~~~~~~~~~~~- //
      // v_m=S_k (C[m][k]*x[n-D-k])) <-Gether P+1 integer taps
      // ~~~~~~~~~~~~~~~~~~~- //
      for (size_t k=0;k<=order;++k) // For each coefficient in the Lagrange polynomial
      {
        const vT xk=dl.Peek(D-k);    // Get the sample at index D+k from the delay line
        for (size_t m=0;m<=order;++m)
          v[m]+=coeffvT[m][k]*xk;    // Compute the output sample by summing the products of the coefficients and the corresponding samples from the delay line
      }                            // Done with the output samples.
      // ~~~~~~~~~~~~~~~~~~~ //
      // Perform Horner's evalueation of coeffiecient expansion
      // y=(((v_N)*mu+n_{N-1})*mu+ ... +v_0)*mu+v_0
      // ~~~~~~~~~~~~~~~~~~~ //
      vT y = vT(0);                 // Initialize the output sample to zero
      for (int m=static_cast<int>(order); m>=0; --m)  // For each coefficient in the Lagrange polynomial
        y*=*mu+v[m];           // Accumulate using Horner's method
     return y;                 // Return the processed output value
    }                              // ~~~~~~~- Process ~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~ //
    // Process a block of frames using the Farrow interpolator.
    // ~~~~~~~~~~~~~~~~~~~ //
    inline vT ProcessBlock(
      const DelayLineSIMD<T,MaxLen,vT>& dl,
    size_t D,                     // The total delay in samples
    size_t nFrames,               // The number of frames to process
    vT* const y) noexcept         // The output buffer
    {                             // ~~~~~~~- ProcessBlock ~~~~~~~~~~~ //
      if (D<order || D>MaxLen-1) return vT(0);
      for (size_t i=0;i<nFrames;++i)// For each row (frame) in the block...
        y[i]=ProcessFrame(dl, D); // Process each frame using the Farrow interpolator
      return vT(0);               // Return zero as a placeholder
    }
    inline const std::array<vT, MaxOrder+1>& operator[](size_t i) const noexcept // ~~~~~~~~ GetCoefficients ~~~~~~~~~~~~ //
    {
      assert(i < MaxOrder+1);
      return coeffvT[i];
    }
    inline std::array<vT, MaxOrder+1>& operator[](size_t i) noexcept // ~~~~~~~~ GetCoefficients ~~~~~~~~~~~~ //
    {
      assert(i < MaxOrder+1);
      return coeffvT[i];
    }
    private:
      size_t order{3};               // Order of the Lagrange filter, default is 3.
      vT mu{vT(0)};                  // Fractional part of the delay, default is 0.
      Coeff coeffvT{}; // coeffvTicients for the Lagrange filter, size is order+1.
      void BuildCoefficients(void) noexcept // Build the coefficients for the Lagrange filter
      {                                   // ~~~~~~~~ BuildCoefficients ~~~~~~~~~~~~ //
        // ~~~~~~~~~~~~~~~~~~~~~~-- //
        //  coeff[m][k]=polynomial coefficients of mu^m for tap k
        // ~~~~~~~~~~~~~~~~~~~~~~-- //
        for (size_t m=0;m<=order;++m)   // For each order
        {
         auto c=detail::BuildLagrangeCoeffs<T,MaxOrder+1>(0,order); // Build the coefficients for the Lagrange filter
         for (size_t k=0;k<=order;++k) // For each tap
           coeffvT[m][k]=VT(c[m][k]); // Store the coefficients in the matrix
        }

      }
  };
  template<typename T=float,
    size_t MaxLen=1024,
    size_t MaxOrder=5,
    typename vT=std::experimental::native_simd<T>>
  class FarrowDeinterpolatorSIMD
  {
    static_assert(MaxOrder <= 5, "MaxOrder must be <= 5 for FarrowInterpolatorSIMD");
    static constexpr size_t VL=vT::size();
    using Coeff=std::array<std::array<vT,MaxOrder+1>,MaxOrder+1>;
    public:
      constexpr static size_t MAXORDER=MaxOrder;
      ~FarrowDeinterpolatorSIMD(void) noexcept = default; // Default destructor
      void SetOrder(
        size_t N) noexcept              // Set the order of the Lagraange Interpolator
      {                                 // ~~~~~~~~ SetOrder ~~~~~~~~~~~~ //                 
        order=std::min<size_t>(N,MaxOrder); 
        BuildCoefficients();           // Build the coefficients for the Lagrange filter
      }                                // ~~~~~~~~ SetOrder ~~~~~~~~~~~~ //
      void SetMu(const vT& m) noexcept  // Set the fract part of the dela
      {
        mu=m-vT(std::floor(m));         // Set the fractional part of the delay
      }
      vT GetMu(void) const noexcept    // Get the fractional part of the delay
      {
        return mu;                     // Return the fractional part of the delay
      }
      // Apply the inverse Farrow filter one write (distribute x into fractional slots)
      bool Process(
        const vT& x,                  // The sample to write to the delay line
        DelayLineSIMD<T,MaxLen,vT>& dl, // The delay line to write to (mutable)
        size_t D) noexcept            // The total delay in samples
      {
        if (D<order || D>MaxLen-1) return false;
        std::array<vT,MaxOrder+1> v{};
        // ~~~~~~~~~~~~~~~~~~~ //
        // Calculate the coefficients using Horner's method fo mu^m
        // ~~~~~~~~~~~~~~~~~~~ //
        for (size_t m=0;m<=order;++m)// Rows
        {
          for (size_t k=0;k<=order;++k)// Cols
          {
            if (k==0)
              v[m]=coeffvT[m][k]*x; // Initialize the first coeffvTicient
            else
              v[m]+=coeffvT[m][k]*std::pow(mu,k); // Compute the coefficient
          }
        }                              // Done with the coefficients.
        // ~~~~~~~~~~~~~~~~~~~ //
        // Perform additive write to the delay line
        // ~~~~~~~~~~~~~~~~~~~ //
        for (size_t k=0;k<=order;++k) // For each coefficient in
          dl.WriteAt(D-k,dl.Peek(D-k)+v[k]);// Additive write to the delay line
        return true;                 // Return true to indicate success
      }                                // ~~~~~~~- Process ~~~~~~~~~~~ //
    inline vT ProcessFrame(
      const DelayLineSIMD<T,MaxLen,vT>& dl, // The delay line to process
      size_t D) noexcept         // Output buffer.
    {                               // ~~~~~~~- Process ~~~~~~~~~~~ //
      if (D<order || D > MaxLen - 1) return vT(0);
      size_t maxD=MaxLen-1-order;   // Maximum delay length.
      if (D>maxD) D=maxD;              // Clamp the delay to the
      // ~~~~~~~~~~~~~~~~~~~- //
      // v_m=S_k (C[m][k]*x[n-D-k]))
      // ~~~~~~~~~~~~~~~~~~~- //
      std::array<vT,MaxOrder+1> v{}; // Zero the output buffer
      for (size_t k=0;k<=order;++k) // For each coefficient in the Lagrange polynomial
      {
        const vT xk=dl.Peek(D-k);    // Get the sample at index D+k from the delay line
        for (size_t m=0;m<=order;++m)
          v[m]+=coeffvT[m][k]*xk;    // Compute the output sample
      }                            // Done with the output samples.
      vT y=v[order];
      // ~~~~~~~~~~~~~~~~~~~ //
      // Perform Horner's evalueation of coeffiecient expansion
      // y=(((v_N)*mu+n_{N-1})*mu+ ... +v_0)*mu+v_0
      // ~~~~~~~~~~~~~~~~~~~ //
      for (int m=static_cast<int>(order)-1;m>=0;--m)  // For each coefficient in the Lagrange polynomial
        y=y*mu+v[m];           // Compute the output sample by summing the products of the coefficients and the corresponding samples from the delay line
      return y;                 // Return true to indicate success
    }                              // ~~~~~~~- Process ~~~~~~~~~~~ //
    inline vT ProcessBlock(
      const DelayLineSIMD<T,MaxLen,vT>& dl,    // Our Delay line buffer to deinterpolate to
      size_t D,                      // The total delay in samples
      size_t nFrames,               // The number of frames to process
      vT* const y) noexcept         // The output buffer
    {
      if (D<order||D>MaxLen-1) return false;
      // ~~~~~~~~~~~~~~~~~~~~~~- // 
      // A block is just a slice of arrays bro and they contain signals.
      // ~~~~~~~~~~~~~~~~~~~~~~- //
      for (size_t n=0;n<nFrames;++n)
        y[n]=ProcessFrame(dl,D+n);
      return vT(0);               // Return zero as a placeholder
    }
    inline const std::array<vT, MaxOrder+1>& operator[](size_t i) const noexcept // ~~~~~~~~ GetCoefficients ~~~~~~~~~~~~ //
    {
      assert(i < MaxOrder+1);
      return coeffvT[i];
    }
    inline std::array<vT, MaxOrder+1>& operator[](size_t i) noexcept // ~~~~~~~~ GetCoefficients ~~~~~~~~~~~~ //
    {
      assert(i < MaxOrder+1);
      return coeffvT[i];
    }
    private:
      size_t order{3};                // Order MUST be the same as in the interpolator.
      vT mu{vT(0)};                    // Fractional part of the delay
      Coeff coeffvT{}; // Coefficients for the Lagrange filter, size is order+1.
      void BuildCoefficients(void) noexcept
      {
        for (size_t k=0;k<=order;++k)
        {                              // For each coefficient in the filter...
          auto c=detail::BuildLagrangeCoeffs<vT,MaxOrder+1>(k,order);
          for (size_t m=0;m<=order;++m)
            coeffvT[m][k]=vT(c[m]);         // Store the coeffvTicients in the matrix
        }
      }                                // ~~~~~~~~ BuildCoefficients ~~~~~~~~~~~~ //
  };

} // namespace sig::wg