/* 
 * *
 * * Filename: ReedSolomon.hpp
 * *
 * * Description:
 * *   CCSDS Reed-Solomon error correction encoder/decoder for RS(255,223) with
 * *   support for both encoding and decoding operations. , t=16 over GF(256) with
 * *   generator polynomial: G(x)= x^8 +x^7 +x^2 +1 (0x187).
 * * 
 * * Encoding: 223 data bytes -> 32 parity bytes -> 255 byte codeword.
 * * Decoding: Correct up to 16 random symbol errors; erasureless path
 * * Shortening: k<=223 -> n=k+32 by prefeeding (223-k) zero bytes.
 * *
 * * Author:
 * *  JEP, J. Enrique Peraza
 * *
 * *
 */
#pragma once
#include<cstdint>
#include<vector>
#include<cstring>
#include<cstdarg>
#include<cstdio>
#include<memory>
#include"../logger/Logger.h"

// Mutual-exclusion guard: ensure RS API types are defined only once
#ifndef SDR_MDM_RS_API_GUARD
#define SDR_MDM_RS_API_GUARD

namespace sdr::mdm
{
  struct RSStatus
  {
    int32_t corr{0};                    // Number of corrected symbols
    bool ok{false};                     // True if decode successful
    RSStatus (void)=default;            // Default constructor
  };
  struct GF256
  {
    uint8_t alog[512];                  // Anti-log table
    uint8_t log[256];                   // Log table
    GF256(void)                         // Constructor builds tables
    {
      uint16_t p=0x187;                 // Primitive polynomial
      uint16_t x=1;                     // Start with alpha^0=1
      std::memset(alog,0,sizeof(alog)); // Clear anti-log table
      std::memset(log,0,sizeof(log));   // Clear log table
      for (uint16_t i=0;i<255;++i)       // For each exponent
      {                                 // Build tables
        alog[i]=static_cast<uint8_t>(x);// Store anti-log
        log[x]=i;                       // Store log
        x<<=1;                          // Multiply by alpha (x)
        if (x&0x100)                     // Overflow?
          x^=p;                         // Reduce modulo primitive polynomial
      }                                 // Done building tables
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Perform a symmetric extension of the anti-log table
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      for (uint16_t i=255;i<512;++i)     // Extend anti-log table with period 255
        alog[i]=alog[i-255];            // allow direct indexing without explicit mod
      log[0]=0;                         // Table log[0] is undefined, set to 0
    }                                   // End constructor
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // GF(256) arithmetic operations existing under the contour of the given primitive
    // polynomial G(x)= x^8 +x^7 +x^2 + x + 1 (0x187).
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    inline uint8_t Add (uint8_t a,uint8_t b)const{return a^b;} // GF(256) addition
    inline uint8_t Subtract (uint8_t a,uint8_t b)const{return a^b;} // GF(256) subtraction
    inline uint8_t Multiply (uint8_t a,uint8_t b)const // GF(256) multiplication
    {
      if (a==0||b==0)                    // Zero factor?
        return 0;                       // Yes, return zero
      uint16_t s=static_cast<uint16_t>(log[a])+static_cast<uint16_t>(log[b]);
      return alog[s%255];               // Lookup anti-log
    }                                   // End Multiply
    inline uint8_t Inverse (uint8_t a) const // GF(256) multiplicative inverse
    {
      if (a==0)                          // Zero has no inverse
        return 0;                       // Return zero
      return alog[255-log[a]];          // Inverse is alpha^{255 - log(a)}
    }                                   // ~~~~~~~~~~ Inverse ~~~~~~~~~~ //
    inline uint8_t Power (uint8_t a,
      uint8_t e) const // ~~~~~~~~~~ Power ~~~~~~~~~~ //
    {
      if (e==0)                         // Any element to the power of 0 is 1
        return 1;                       // So just return 1.
      if (a==0)                         // 0 to any power is 0
        return 0;                       // So just return 0.
      int32_t r=(log[a]*e)%255;         // Compute exponent mod 255
      if (r<0)                          // Negative?
        r+=255;                         // Wrap around
      return alog[r];                   // Return result
    }                                   // ~~~~~~~~~~ Power ~~~~~~~~~~ //
  }; // End struct GF256
  // Reed-Solomon RS(255,223) encoder/decoder class
  class RS255223
  {
  public:
    static constexpr uint8_t N=255;    // Code length
    static constexpr uint8_t K=223;    // Data length
    static constexpr uint8_t T=16;     // Error correction capability
    static constexpr uint8_t ROOT_START=112; // CCSDS: roots alpha^112 to alpha^143

    RS255223 (void)                     // Constructor
    {
      // Initialize logger first so it's available for any early logging
      lg=logx::Logger::NewLogger();    // Create new logger instance
      Assemble();                      // Assemble generator polynomial
      sto=new RSStatus();              // Create status object
      // Note: Logging writes to stdout and to /home/ljt/Projects/SDR/src/logs/log.txt
    }                                  // End constructor
    ~RS255223 (void)
    {
      if (sto!=nullptr)                // Did we create a status object?
      {                                // Yes, delete it
        delete sto;                    // Delete status object
        sto=nullptr;                   // Remember we deleted it
      }                                // Done deleting status object
      if (lg)                          // If logger initialized
      {
        lg->Shutdown();                // Gracefully shutdown logger
        lg.reset();                    // Release resource
      }
    }
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Assemble generator polynomial G(x)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    inline void Assemble (void)
    {                                 // ~~~~~~~~~ Assemble ~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // g(x) = Prod_{i=112..143} (x + alpha^i)
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      gen.assign(33,0);                // Generator polynomial coefficients [0..32]
      gen[0]=1;                        // Start with constant 1
      for (int i=0;i<32;++i)            // Current resulting degree == i
      {                                // For each root
        uint8_t a=gf.alog[ROOT_START+i]; // alpha^{112+i}
        for (int j=32;j>=1;--j)         // Update degrees j down to 1
          gen[j]=gf.Add(gen[j-1],gf.Multiply(a,gen[j]));
        gen[0]=gf.Multiply(a,gen[0]);  // constant term
      }                                // Done assembling generator polynomial
      {
        char buf[1024];
        int off=0;
        off+=std::snprintf(buf+off,sizeof(buf)-off,"RS Assemble: gen[0..32]=");
        for (int i=0;i<=32;i++)
          off+=std::snprintf(buf+off,sizeof(buf)-off,"%s%02X",(i==0?"":" "),gen[i]);
        if (lg)
          lg->Inf("%s",buf);
      }
    }                                  // ~~~~~~~~~ Assemble ~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Encode data bytes into RS(255,223) codeword
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    inline void Encode (
      const uint8_t* const dat,        // data to encode
      int32_t k,                       // Data length
      std::vector<uint8_t>* const o)   // Output code word
    {                                  // ~~~~~~~~~~ Encode ~~~~~~~~~~ //
      if (dat==nullptr||o==nullptr)    // Bad args?
        return;                        // Yes, nothing to do. Return.
      if (k<=0||k>K)                   // Invalid k?
        k=K;                           // Clamp to max 223
      uint8_t poly[N]={0};             // Parity bytes buffer
      std::memcpy(poly+(N-k),dat,k);   // Copy data to high-degree coefficients
      for (int i=N-k;i<N;++i)          // Process data bytes
      {                                // Encode
        uint8_t fb=poly[i];            // Compute feedback byte
        if (fb!=0)                     // Non-zero feedback?
        {
          for (int j=1;j<=32;++j)      // Shift and update
            poly[i-j]^=gf.Multiply(gen[32-j],fb);
        }
      }                               // Done encoding data bytes
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Output codeword: data bytes followed by parity bytes
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      o->resize(k+32,0);              // Resize output buffer
      std::memcpy(o->data(),dat,k);   // Copy data bytes
      std::memcpy(o->data()+k,poly,32);// Copy parity bytes
      if (lg)
        lg->Inf("RS Encode: k=%d n=%d",k,k+32);
      {
        char buf[512];
        int off=0;
        off+=std::snprintf(buf+off,sizeof(buf)-off," cw[0..7]=");
        for (int i=0;i<8&&i<k+32;i++)
          off+=std::snprintf(buf+off,sizeof(buf)-off," %02X",(*o)[i]);
        if (lg)
          lg->Inf("%s",buf);
      }
      {
        char buf[512];
        int off=0;
        off+=std::snprintf(buf+off,sizeof(buf)-off," cw[%d..%d]=",k+24,k+31);
        for (int i=k+24;i<k+32;i++)
          off+=std::snprintf(buf+off,sizeof(buf)-off," %02X",(*o)[i]);
        if (lg)
          lg->Inf("%s",buf);
      }
    }                                 // ~~~~~~~~~~ Encode ~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Encode but using vector input
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    inline void EncodeVec (
      const std::vector<uint8_t>* dat, // Input codeword
      std::vector<uint8_t>* const o)    // Corrected symbols buffer
    {                                  // ~~~~~~~~~~ EncodeVec ~~~~~~~~~~ //
      if (dat==nullptr||o==nullptr)    // Bad args?
        return;                        // Yes, nothing to do. Return.
      Encode(dat->data(),static_cast<int32_t>(dat->size()),o); // Call raw pointer version
    }                                  // ~~~~~~~~~~ EncodeVec ~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Decode RS(255,223) codeword into data bytes
    // Output status indicates success or failure
    // Output buffer expects at least 255 bytes to produce up to 223 
    // corrected symbols.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    inline RSStatus Decode (
      const uint8_t* const code,       // Input code word
      uint8_t*const o)                 // Decoded output
    {                                  // ~~~~~~~~~~ Decode ~~~~~~~~~~ //
      if (code==nullptr||o==nullptr)    // Bad args?
      {                                // Yes, return failure
        sto->corr=0;                   // No corrections
        sto->ok=false;                 // No success
        return*sto;                    // Return status
      }                                // Good args, proceed with decode
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Sanity checks
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      if (lg)
        lg->Inf("RS Decode: begin");   // Log decode start
      {                                // Log codeword snippets
        char buf[512];                 // Buffer for logging
        int off=0;                     // Offset into buffer
        off+=std::snprintf(buf+off,sizeof(buf)-off," code[0..7]=");
        for (int i=0;i<8;i++)
          off+=std::snprintf(buf+off,sizeof(buf)-off," %02X",code[i]);
        if (lg)
          lg->Inf("%s",buf);
      }
      {                                // Log codeword snippets
        char buf[256];
        std::snprintf(buf,sizeof(buf)," code[223..224]= %02X %02X",code[223],code[224]);
        if (lg)
          lg->Inf("%s",buf);
      }
      {
        char buf[512];
        int off=0;
        off+=std::snprintf(buf+off,sizeof(buf)-off," code[247..254]=");
        for (int i=247;i<255;i++)
          off+=std::snprintf(buf+off,sizeof(buf)-off," %02X",code[i]);
        if (lg)
          lg->Inf("%s",buf);
      }
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // 0) Build polynomial C(x): degrees 0..254 (low..high).
      // Data:    cw[0..222] = data bytes (degrees 0..222)
      // Parity:  cw[223..254] = parity bytes (degrees 223..254)
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      uint8_t cw[N];
      std::memcpy(cw,code,N);          // Direct copy, no reversal
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Layout convention for Horner evaluation:
      // We store coefficient for degree d at index idx = d, so that
      // Horner acc = acc*a + cw[idx] computes sum coeff[d] * a^d.
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Begin decode process, compute syndromes:
      // (1) Syndromes S_i = C(a^{i+112}), i=0..31
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      uint8_t S[2*T];
      bool azer=true;
      for (int i=0;i<2*T;++i)          // For each syndrome
      {                                // Compute S[i]=C(alpha^{112+i})
        S[i]=0;                        // Initialize for syndrome to 0
        for (int j=0;j<N;++j)          // For each coefficient
        {
          if (cw[j]!=0)
            S[i]^=gf.Multiply(cw[j],gf.Power(gf.alog[ROOT_START+i],j));
        }
        azer&=(S[i]==0);               // Determine if still all zero syndromes
        if (lg)
          lg->Deb(" RS Decode: S[%d]=%02X",i,S[i]);
      }                                // Done computing all syndromes
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // (1a) Check for all-zero syndromes -> no errors
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      if (azer)                        // All syndromes zero?
      {                                // Yes, no errors
        sto->corr=0;                   // No corrections needed
        sto->ok=true;                  // Decode successful
        std::memcpy(o,cw,K);           // Copy data bytes to output
        if (lg)
          lg->Inf(" RS Decode: all syndromes zero, no errors detected");
        return*sto;                    // Return status
      }                                // Errors detected, proceed to decoding
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Sanity check
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      if (lg)
        lg->Inf(" RS Decode: proceed to error correction");
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // (2) Berlekamp-Massey: LAMBDA(x)
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      uint8_t C[2*T+1]={1};            // Error locator polynomial
      uint8_t B[2*T+1]={1};            // Previous C(x)
      int L=0,m=1;                     // Current degree of C(x), correction factor
      uint8_t b=1;                     // Last discrepancy
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Main Berlekamp-Massey loop
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      for (int n=0;n<2*T;++n)          // For each syndrome index
      {                                // Compute discrepancy d
        uint8_t d=S[n];
        for (int i=1;i<=L;++i)
          if (n>=i)
            d^=gf.Multiply(C[i],S[n-i]);
        if (d==0)                      // No discrepancy?
        {                              // Yes
          ++m;                         // Increment m
          for (int i=2*T;i>0;--i)
            B[i]=B[i-1];               // Shift B(x)
          B[0]=0;
          continue;                    // Skip this syndrome
        }                              // Proceed with non-zero discrepancy
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // C(x) = C(x) - d/b * x^m * B(x)
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        uint8_t T[2*T+1];
        std::memcpy(T,C,sizeof(C));     // Copy C(x) to T(x)
        uint8_t coef=gf.Multiply(d,gf.Inverse(b));
        for (int j=0;j<=2*T;++j)
        {
          if (m+j<=2*T)
            C[m+j]^=gf.Multiply(coef,B[j]);
        }
        if (2*L<=n)                    // Time to update?
        {                              // Yes
          L=n+1-L;                     // Update degree L
          std::memcpy(B,T,sizeof(B));  // Update B(x)
          b=d;                         // Update b
          m=1;                         // Reset m
        }
        else                           // No update to B(x), L, b
        {
          ++m;                         // Increment m
          for (int i=2*T;i>0;--i)
            B[i]=B[i-1];               // Shift B(x)
          B[0]=0;
        }
      }                                // Done with all syndromes
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Sanity check
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      {
        char buf[1024];
        int off=0;
        off+=std::snprintf(buf+off,sizeof(buf)-off," RS BM: L=%d C[0..%d]=",L,L);
        for (int i=0;i<=L;i++)
          off+=std::snprintf(buf+off,sizeof(buf)-off," %02X",C[i]);
        if (lg)
          lg->Inf("%s",buf);
      }                                // Done sanity check
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Check L for validity
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      if (L==0||L>T)                   // Invalid number of errors?
      {                                // Yes, uncorrectable
        sto->corr=0;                   // No corrections
        sto->ok=false;                 // We are not OK
        if (lg)
          lg->Inf(" RS BM: L=%d (uncorrectable)",L);
        return*sto;                    // Return status
      }                                // Proceed with L valid
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // (3) Chien: find roots of ?(a^i), i=0..254
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      int ne=0;                        // Number of roots found
      int iroot[T],pos[T];             // Root indices, coefficient positions
      for (int i=0;i<255&&ne<T;++i)    // For each possible root
      {                                // Evaluate C(x)
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Evaluate at x = alpha^i. If there's an error at coefficient
        // j, LAMBDA(alpha^i) = 0 where j = (255-i) mod 255
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        uint8_t sum=C[0];
        uint8_t x=gf.alog[i];
        for (int j=1;j<=L;++j)
          sum^=gf.Multiply(C[j],gf.Power(x,j));
        if (sum==0)                    // Found a root?
        {                              // Yes
          iroot[ne]=i;                 // Save the root index
          pos[ne]=(255-i)%255;         // Save the coefficient position
          if (lg)
            lg->Inf(" RS Chien: root i=%3d -> coeff_index(pos)=%3d (x=a^{%d})",i,pos[ne],i);
          ++ne;                        // Increment number of roots found
        }                              // Done processing this possible root
        if ((i+1)%32==0||i==254)
          if (lg)
            lg->Deb(" RS Chien: searched up to i=%3d found ne=%d roots so far",i,ne);
      }                                // Done searching for roots
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Check ne for validity
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      if (ne==0||ne>T||ne!=L)          // Uncorrectable?
      {                                // Yes
        sto->corr=0;                   // No corrections
        sto->ok=false;                 // We are not OK
        if (lg)
          lg->Inf(" RS Chien: L=%d ne=%d (uncorrectable)",L,ne);
        return*sto;                    // Return status
      }                                // Proceed with ne valid
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // optional: enforce ne==L
      // if (ne != L) { sto->corr = 0; sto->ok = false; return *sto; }
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // (4) O(x) = [ S(x) * ?(x) ] mod x^{32}
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      uint8_t om[2*T]={0};             // Omega polynomial
      for (int i=0;i<2*T;++i)          // For each degree in Omega
      {                                // Compute Omega[i]
        om[i]=S[i];
        for (int j=1;j<=L;++j)
          if (i>=j)
            om[i]^=gf.Multiply(C[j],S[i-j]);
      }                                // Done computing all Omega[i]
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // (5) Forney corrections
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      for (int k=0;k<ne;++k)           // For each error found
      {                                // Compute error magnitude and location
        const int i=iroot[k];          // alpha^i was a root
        const int j=pos[k];            // coefficient index in cw[]
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // O(X^{-1}) / ?'(X^{-1})
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        uint8_t Xinv=gf.alog[(255-i)%255]; // Xinv = alpha^{255-i}
        uint8_t num=om[0];             // Omega evaluated at X^{-1}
        for (int t=1;t<2*T;++t)
          num^=gf.Multiply(om[t],gf.Power(Xinv,t));
        uint8_t den=1;                 // Derivative of error locator polynomial
        for (int m=0;m<ne;++m)
          if (m!=k)
            den=gf.Multiply(den,gf.Add(1,gf.Multiply(gf.alog[iroot[m]],Xinv)));
        if (den==0)                    // Derivative is zero?
        {                              // Yes, skip this error
          if (lg)
            lg->Inf(" RS Forney[%d]: i=%3d j=%3d Xinv=a^{%3d} (0x%02X) den=0 SKIP",k,i,j,(255-i)%255,Xinv);
          continue;                    // Skip this error
        }                              // Proceed with non-zero derivative
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Error magnitude: em = O(X^{-1}) / ?'(X^{-1})
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        uint8_t emag=gf.Multiply(num,gf.Inverse(den)); // Magnitude of the error
        uint8_t bef=cw[j];             // Before codeword
        cw[j]^=emag;                   // Correct codeword
        if (lg)
        {
          lg->Inf(" RS Forney[%d]: i=%3d j=%3d Xinv=a^{%3d} (0x%02X) d=0x%02X omx=0x%02X emag=0x%02X cw_before=0x%02X cw_after=0x%02X",
                  k,i,j,(255-i)%255,Xinv,num,om[0],emag,bef,cw[j]);
        }
      }                                // Done processing all errors
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // (6) Verify: recompute syndromes
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      bool vld=true;                   // Assume valid
      for (int i=0;i<2*T;++i)          // For each syndrome
      {                                // Recompute S[i]=C(alpha^{112+i})
        S[i]=0;
        for (int j=0;j<N;++j)          // For each coefficient
          if (cw[j]!=0)
            S[i]^=gf.Multiply(cw[j],gf.Power(gf.alog[ROOT_START+i],j));
        vld&=(S[i]==0);                // Valid if all syndromes are zero
        if (lg)
          lg->Deb(" RS Verify: S[%d]=%02X",i,S[i]);
      }                                // Done recomputing all syndromes
      sto->corr=vld?ne:0;              // Set corrections if valid
      sto->ok=vld;                     // Set success status
      if (vld)                         // Valid codeword?
        std::memcpy(o,cw,K);           // Copy data bytes to output
      if (lg)
        lg->Inf(" RS Verify: %s ne=%d corr=%d ok=%s",vld?"valid":"invalid",ne,sto->corr,vld?"true":"false");
      if (lg)
        lg->Inf(" RS Decode: end");    // Log decode end
      return*sto;                      // Return status
    }                                  // ~~~~~~~~~~ Decode ~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Decode but using vector input
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    inline RSStatus DecodeVec (
      const std::vector<uint8_t>* code,
      std::vector<uint8_t>* const o)
    {                                  // ~~~~~~~~~~ DecodeVec ~~~~~~~~~~ //
      if (code==nullptr||o==nullptr||code->size()<33||code->size()>N) // Bad args?
      {                                // Yes, return failure
        sto->corr=0;                   // No corrections
        sto->ok=false;                 // Decode failed
        return*sto;                    // Return status
      }                                // Good args, proceed with decode
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Ensure the output vector has space for 223 data bytes
      // and obtain a raw pointer
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      if (code->size()==N)
      {
        o->resize(K);                  // Resize output buffer
        return Decode(code->data(),o->data()); // Call raw pointer version
      }
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Support shortened codewords if needed.
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      int k=static_cast<int>(code->size())-32; // Number of data bytes
      if (k<=0||k>K)
      {
        sto->corr=0;                   // No corrections
        sto->ok=false;                 // Decode failed
        return*sto;                    // Return status
      }
      int nz=K-k;                      // Number of zero bytes to prefeed for shortening
      std::vector<uint8_t> full(N,0);  // Full codeword buffer
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Build full codeword with prefed zeroes
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      std::memcpy(full.data()+nz,code->data(),k); // Copy shortened codeword into full buffer
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Place parity after data bytes... canonical tail at full[223..254]
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      std::memcpy(full.data()+K,code->data()+k,32); // Copy parity bytes
      if (lg)
        lg->Inf("RS DecodeVec: reconstructed full-length from shortened n=%zu (k=%d,nz=%d)",code->size(),k,nz);
      o->resize(K);                    // Resize output buffer
      return Decode(full.data(),o->data()); // Call raw pointer version
    }                                  // ~~~~~~~~~~ DecodeVec ~~~~~~~~~~ //
  private:
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // GF(256) arithmetic tables and functions
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    GF256 gf;                         // GF(256) arithmetic tables
    RSStatus*sto{nullptr};            // Status object
    std::vector<uint8_t> gen;         // Generator polynomial coefficients
    // Logger for this component
    std::unique_ptr<logx::Logger> lg{};
  };
}

#endif // SDR_MDM_RS_API_GUARD