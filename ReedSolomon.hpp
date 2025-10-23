/* 
* *
* * Filename: ReedSolomon.hpp
* *
* * Description:
* *   CCSDS Reed-Solomon error correction encoder/decoder for RS(255,223) with
* *   support for both encoding and decoding operations. , t=16 over GF(256) with
* *   generator polynomial: G(x)= x^8 +x^4 +x^3+x^2 +1 (0x11D).
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
#include <cstdint>
#include <vector>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <memory>
#include "../logger/Logger.h"


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
    GF256 (void)                        // Constructor builds tables
    {
      uint16_t p=0x11D;                 // Primitive polynomial
      uint16_t x=1;                     // Start with alpha^0=1
      std::memset(alog,0,sizeof(alog)); // Clear anti-log table
      std::memset(log,0,sizeof(log));   // Clear log table
      for (uint16_t i=0;i<255;++i)      // For each exponent
      {                                 // Build tables
        alog[i]=static_cast<uint8_t>(x);// Store anti-log
        log[x]=i;                       // Store log
        x<<=1;                          // Multiply by alpha (x)
        if (x&0x100)                    // Overflow?
          x^=p;                         // Reduce modulo primitive polynomial
      }                                 // Done building tables
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Perform a symmetric extension of the anti-log table
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      for (uint16_t i=255;i<512;++i)    // Extend anti-log table with period 255
        alog[i]=alog[i-255];            // allow direct indexing without explicit mod
      log[0]=0;                         // Table log[0] is undefined, set to 0
    }                                   // End constructor
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // GF(256) arithmetic operations existing under the contour of the given primitive
    // polynomial x^8 +x^4 +x^3 +x^2 +1 (0x11D).
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    inline uint8_t Add (uint8_t a,uint8_t b) const { return a^b; } // GF(256) addition
    inline uint8_t Subtract (uint8_t a,uint8_t b) const { return a^b; } // GF(256) subtraction
    inline uint8_t Multiply (uint8_t a, uint8_t b) const // GF(256) multiplication
    {
      if (a==0||b==0)                   // Zero factor?
        return 0;                       // Yes, return zero
      uint16_t s = static_cast<uint16_t>(log[a]) + static_cast<uint16_t>(log[b]);
      if (s>=255) s-=255;               // Add exponents modulo 255
      return alog[s];                   // Lookup anti-log
    }                                   // End Multiply
    inline uint8_t Inverse (uint8_t a) const // GF(256) multiplicative inverse
    {
      if (a==0)                         // Zero has no inverse
        return 0;                       // Return zero
      return alog[255-log[a]];          // Inverse is alpha^{255 - log(a)}
    }                                   // ~~~~~~~~~~ Inverse ~~~~~~~~~~ //
    inline uint8_t Power (
      uint8_t a,                        // Base
      uint8_t e) const                  // Exponent
    {                                   // ~~~~~~~~~~ Power ~~~~~~~~~~ //
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
      RS255223 (void)                   // Constructor
      {
        // Initialize logger first so it's available for any early logging
        lg=logx::Logger::NewLogger();   // Create new logger instance
        Assemble();                     // Assemble generator polynomial
        sto=new RSStatus();             // Create status object
    // Note: Logging writes to stdout and to /home/ljt/Projects/SDR/src/logs/log.txt
      }                                 // End constructor
      ~RS255223 (void)
      {
        if (sto!=nullptr)               // Did we create a status object?
        {                               // Yes, delete it
          delete sto;                   // Delete status object
          sto=nullptr;                  // Remember we deleted it
        }                               // Done deleting status object
        if (lg)                         // If logger initialized
        {                  
          lg->Shutdown();               // Gracefully shutdown logger
          lg.reset();                   // Release resource
        }
      }
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Assemble generator polynomial G(x)
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      inline void Assemble (void)
      {                                 // ~~~~~~~~~ Assemble ~~~~~~~~~~ //
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // g(x)=PROD_{i=1}^32}{x-alpha^i}
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        gen.assign(33,0);               // Generator polynomial degree 32
        gen[0]=1;                       // Start with 1
        for (uint8_t i=1;i<=32;++i)     // For each factor (x + alpha^i)
        {
          uint8_t a=gf.alog[i];         // a = alpha^i
          for (int j=i;j>=1;--j)        // For each term in polynomial
            gen[j]=gf.Add(gen[j-1],gf.Multiply(a,gen[j]));
          gen[0]=gf.Multiply(a,gen[0]); // constant term accumulates product of roots
        }                               // Done assembling generator polynomial

        {
          char buf[1024];
          int off=0;
          off+=std::snprintf(buf+off,sizeof(buf)-off,"RS Assemble: gen[0..32]=");
          for (int i=0;i<=32;i++)
            off+=std::snprintf(buf+off,sizeof(buf)-off,"%s%02X",(i==0?"":" "),gen[i]);
          if (lg)
            lg->Inf("%s",buf);
        }
      }                                 // ~~~~~~~~~ Assemble ~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Encode data bytes into RS(255,223) codeword
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      inline void Encode (              // k<=223
        const uint8_t* const dat,       // Data bytes input buffer (length k)
        int32_t k,                      // Number of data bytes (k <= 223)
        std::vector<uint8_t>* const o)  // Output codeword buffer (length n=255)
      {                                 // ~~~~~~~~~~ Encode ~~~~~~~~~~ //
        if (dat==nullptr||o==nullptr)   // Bad args?
          return;                       // Yes, nothing to do. Return.
        if (k<=0||k>223)                // Invalid k?
          k=223;                        // Clamp to max 223
        uint8_t* p=new uint8_t[32];     // Parity bytes buffer
        std::memset(p,0,32);            // Clear all 32 parity bytes
        int32_t nz=223-k;               // Number of zero bytes to prefeed for shortening
        if (nz<0)                       // Negative?
          nz=0;                         // Clamp to zero
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Process data bytes
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        for (int32_t z=0;z<nz;z++)      // For the number of zeroes to prefeed
        {                               // Prefeed
          uint8_t fb=p[0];              // Get first parity byte (feedback)
          for (int32_t q=0;q<31;q++)    // Shift 0..30 using q+1
            p[q]=p[q+1]^gf.Multiply(fb,gen[q+1]);// Shift and update
          p[31]=gf.Multiply(fb,gen[32]);// Last parity byte
        }                               // Done prefeeding zeroes
        for (int32_t i=0;i<k;i++)       // For each data byte in input buffer
        {                               // Encode
          uint8_t fb=dat[i]^p[0];       // Compute feedback byte
          for (int32_t q=0;q<31;q++)    // Shift 0..30 using q+1
            p[q]=p[q+1]^gf.Multiply(fb,gen[q+1]);// Shift and update
          p[31]=gf.Multiply(fb,gen[32]);// Last parity byte
        }                               // Done encoding data bytes
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Output codeword: data bytes followed by parity bytes
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        o->resize(k+32,0);              // Resize output buffer
        std::memcpy(o->data(),dat,static_cast<size_t>(k));// Copy data bytes
        for (int32_t j=0;j<32;j++)      // For each parity byte
          (*o)[k+j]=p[31-j];            // Append parity bytes in reverse order

        if (lg)
          lg->Inf("RS Encode: k=%d n=%d",static_cast<int>(k),static_cast<int>(k+32));
        {
          char buf[512];                
          int off=0;
          off+=std::snprintf(buf+off,sizeof(buf)-off," cw[0..7]=");
          for (int i=0;i<8 && i<k+32;i++)
            off+=std::snprintf(buf+off,sizeof(buf)-off," %02X", (*o)[i]);
          if (lg)
            lg->Inf("%s", buf);
        }
        {
          char buf[512];
          int off=0;
          off+=std::snprintf(buf+off,sizeof(buf)-off," cw[%d..%d]=", (int)(k+32-8), (int)(k+32-1));
          for (int i=static_cast<int>(k)+24;i<static_cast<int>(k)+32;i++)
            off+=std::snprintf(buf+off,sizeof(buf)-off," %02X", (*o)[i]);
          if (lg)
            lg->Inf("%s", buf);
        }
        delete[] p;                     // Free parity bytes buffer
        p=nullptr;                      // Remember we deleted it.
      }                                 // ~~~~~~~~~~ Encode ~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Encode but using vector input
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      inline void EncodeVec (           // k<=223
        const std::vector<uint8_t>* dat, // Data bytes input buffer (length k)
        std::vector<uint8_t>* const o)  // Output codeword buffer (length n=255)
      {                                 // ~~~~~~~~~~ EncodeVec ~~~~~~~~~~ //
        if (dat==nullptr||o==nullptr)   // Bad args?
          return;                       // Yes, nothing to do. Return.
        Encode(dat->data(),static_cast<int32_t>(dat->size()),o);// Call vector version
      }                                 // ~~~~~~~~~~ EncodeVec ~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Decode RS(255,223) codeword into data bytes
      // Output status indicates success or failure
      // Output buffer expects at least 255 bytes to produce up to 223 
      // corrected symbols.
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      inline RSStatus Decode(           // n = 255
        const uint8_t* const code,      // Input codeword (length 255)
        uint8_t* const o)               // Output data buffer pointer (length >=223)
      {                                 // ~~~~~~~~~~ Decode ~~~~~~~~~~ //
        if (code==nullptr||o==nullptr)  // Bad args?
        {                               // Yes, return failure
          sto->corr=0;                  // No corrections
          sto->ok=false;                // No success
          return *sto;                  // Return status
        }                               // Good args, proceed with decode
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Sanity checks
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        if (lg) lg->Inf("RS Decode: begin");     // Log decode start
        {                               // Log codeword snippets
          char buf[512];                // Buffer for logging
          int off=0;                    // Offset into buffer
          off+=std::snprintf(buf+off,sizeof(buf)-off," code[0..7]=");
          for (int i=0;i<8;i++)         // First 8 bytes
            off+=std::snprintf(buf+off,sizeof(buf)-off," %02X",code[i]);
          if (lg)
            lg->Inf("%s",buf);         
        }                               
        {                               // Log codeword snippets
          char buf[256];                // Buffer for logging
          std::snprintf(buf,sizeof(buf)," code[223..224]= %02X %02X",code[223],code[224]);
          if (lg)
            lg->Inf("%s",buf);             
        }
        {
          char buf[512];                // Buffer for logging
          int off=0;                    // Offset into buffer
          off+=std::snprintf(buf+off,sizeof(buf)-off," code[247..254]=");
          for (int i=247;i<255;i++)
            off+=std::snprintf(buf+off,sizeof(buf)-off," %02X",code[i]);
          if (lg)
            lg->Inf("%s",buf);
        }    
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // 0) Build polynomial C(x): degrees 0..254 (low..high).
        // Parity:  cw[0..31] = r_0..r_31 (degrees 0..31, parity coefficients)
        // Data:   cw[32..254] = data bytes (degrees 32...254), code[i]->cw[32+i]
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        std::vector<uint8_t> cw(255);
        for (int j=0;j<32;++j)          // For each parity byte
        {
          cw[static_cast<size_t>(j)]=code[223+j];
          if (lg!=nullptr)
            lg->Inf(" RS Decode: cw[%d]=parity code[%d]=%02X",j,223+j,code[223+j]);
        }
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Layout convention for Horner evaluation:
        // We store coefficient for degree d at index idx = 254 - d, so that
        // Horner acc = acc*a + cw[idx] computes sum coeff[d] * a^d.
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        for (int i=0;i<223;++i)         // For each data byte...
        {
          cw[static_cast<size_t>(32+i)]=code[i]; // Copy data byte
          if ((i+1)%12==0&&lg!=nullptr)
            lg->Inf(" RS Decode: cw[%d]=data code[%d]=%02X",32+i,i,code[i]);
        }
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Clean codeword check using the encoder LFSR (authoritative)
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        const uint8_t* dat=cw.data()+32;// Data region lives at cw[32..254]
        uint8_t pcalc[32];              // Parity bit calculations here
        std::memset(pcalc,0,32);        // Zeroize our parity array
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Re-run the encoded LFSR over 223 data bytes
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        for (int i=0;i<223;++i)         // For our data's length
        {                               // Re-run LFSR encoder
          uint8_t fb=dat[i]^pcalc[0];   // Compute feedback byte
          for (int32_t q=0;q<31;++q)    // Shift 0..30 using q+1
            pcalc[q]=pcalc[q+1]^gf.Multiply(fb,gen[q+1]);// Shift and update
          pcalc[31]=gf.Multiply(fb,gen[32]);// Last parity byte
        }
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Compare true remainder coefficients r_j=pcalc[31-j]
        //  against received parity cw[0..31]
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        int pmat=-1;                    // Assume no match
        for (int j=0;j<32;++j)          // For each parity byte degree j
        {                               // Check for match against computed remainder r_j
          if (cw[static_cast<size_t>(j)]!=pcalc[31-j]) // Mismatch?
          {                             // Yes
            pmat=j;                     // Mark parity mismatch
            break;                      // No need to continue checking
          }                             // Done checking this byte
        }                               // Done checking all parity bytes
        if (lg)                         // Sanity check logging
        {
          lg->Inf("RS Decode: parity %s%s",
          (pmat==-1?"OK":"mismatch"),(pmat==-1?"":(std::string(" at r_")+std::to_string(pmat)).c_str()));
        }
        if (pmat==-1)                   // Parity matches?
        {                               // Yes, no errors
          sto->corr=0;                  // No corrections needed.
          sto->ok=true;                 // Decode successful
          if (o!=nullptr)               // Output buffer valid?
            std::memcpy(o,dat,223);     // Copy data bytes to output
          return *sto;                  // Return status
        }                               // Errors detected, proceed to decoding....
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Sanity check
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        if (lg)
          lg->Inf(" RS Decode: proceed to error correction");
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Begin decode process, compute syndromes:
        // (1) Syndromes S_i = C(a^{i+1}), i=0..31
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        uint8_t S[32];                  // Our syndromes s_i{0..31}
        bool azer=true;                 // Assume all zero syndromes
        for (int i=0;i<32;++i)          // For each syndrome
        {                               // Compute S[i]=C(alpha^{i+1})
          const uint8_t a=gf.alog[i+1]; // a = alpha^{i+1}
          uint8_t acc=0;                // Accumulator
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Evaluate C(a) using Horner's method
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          for (int j=254;j>=0;--j)      // For each coefficient degree from high to low
            acc=gf.Add(gf.Multiply(acc,a),cw[(size_t)j]);// Horner evaluation, cw[j]=coef of degree j
          S[i]=acc;                     // Store syndrome
          azer&=(acc==0);               // Determine if still all zero syndromes...
          if (lg)
            lg->Deb(" RS Decode: S[%d]=%02X",i,acc);
        }                               // Done computing all syndromes....
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // (1a) Check for all-zero syndromes -> no errors
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        if (azer)                       // All syndromes zero?
        {                               // Yes, no errors
          sto->corr=0;                  // No corrections needed
          sto->ok=true;                 // Decode successful
          if (o!=nullptr)               // Output buffer valid?
            std::memcpy(o,cw.data()+32,223);// Copy data bytes to output
          if (lg)
            lg->Inf(" RS Decode: all syndromes zero, no errors detected");
          return *sto;                  // Return status output
        }                               // Done with no errors.... Proceed to error correction....
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // (2) Berlekamp-Massey: LAMBDA(x)
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        uint8_t C[33];                  // Error locator polynomial
        uint8_t B[33];                  // Previous C(x)
        std::memset(C,0,33);            // Clear C(x)
        std::memset(B,0,33);            // Clear B(x)
        C[0]=1;                         // Initialize C(x)=1
        B[0]=1;                         // Initialize B(x)=1
        int L=0;                        // Current degree of C(x)
        int m=1;                        // Correction factor
        uint8_t b=1;                    // Last discrepancy
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Main Berlekamp-Massey loop
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        for (int n=0;n<32;++n)          // For each syndrome index
        {                               // Compute discrepancy d
          uint8_t d=S[n];               // Start with S[n]
          for (int i=1;i<=L&&i<=n;++i)  // For each term in error locator polynomial...
            if (C[i]&&S[n-i])           // Non-zero term?
              d^=gf.Multiply(C[i],S[n-i]);// Yes, update discrepancy....
          if (d==0)                     // Any discrepancy?
          {                             // No
            ++m;                        // Just increment m
            continue;                   // Skip this syndrome
          }                             // Done with zero discrepancy....
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // If we got here, there was a discrepancy d ... so we compute new error locator
          // polynomail C(x)
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          uint8_t T[33];                // Scratch copy of C(x)
          std::memcpy(T,C,33);          // Copy C(x) to T(x)
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // C(x) = C(x) - d/b * x^m * B(x)
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          const uint8_t coef=gf.Multiply(d,gf.Inverse(b));
          for (int j=0;j<=32;++j)       // For each term in the previous error locator...
          {                             // Update C(x) with shifted B(x)
            const int jm=j+m;           // Shift by m
            if (jm>32)                  // Out of range?
              break;                    // Yes, we are done...
            if (B[j])                   // Non-zero term?
              C[jm]^=gf.Multiply(coef,B[j]);// Update C(x) matrix.
          }                             // Done updating C(x)
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // If 2L <= n, update B(x), L, b, m
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          if (2*L<=n)                   // Time to update?
          {                             // Yes
            L=n+1-L;                    // Update degree L
            std::memcpy(B,T,33);        // Update B(x)
            b=d;                        // Update b
            m=1;                        // Reset m
          }                             // Done updating
          else                          // No update to B(x), L, b
            ++m;                        // Just increment m
        }                               // Done with all syndromes
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Sanity check
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //    
        {
          char buf[1024];               // Buffer for logging
          int off=0;                    // Offset into buffer
          off+=std::snprintf(buf+off,sizeof(buf)-off," RS BM: L=%d C[0..%d]=",L,L);
          for (int i=0;i<=L;i++)        // For each coefficient
            off+=std::snprintf(buf+off,sizeof(buf)-off," %02X",C[i]);
          if (lg)
            lg->Inf("%s",buf);
        }                               // Done sanity check
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Check L for validity
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        if (L==0||L>16)                 // Invalid number of errors? 
        {                               // Yes, uncorrectable
          sto->corr=0;                  // Say that 
          sto->ok=false;                // We are not OK.
          if (lg)
            lg->Inf(" RS BM: L=%d (uncorrectable); returned at Checking L < 0 || L > 16",L);
          return *sto;                  // Return status output
        }                               // Proceed with L valid
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // (3) Chien: find roots of ?(a^i), i=0..254
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        int ne=0;                       // Number of roots found
        int iroot[32];                  // Root indices
        int pos[32];                    // Coefficient positions
        for (int i=0;i<255;++i)         // For each possible root
        {
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Evaluate at x = alpha^{255-j}. If there's an error at coefficient
          // j, LAMDBA{(alpha^(255-j)} = 0 by construction
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          const uint8_t x=gf.alog[i];   // x = alpha^i
          uint8_t acc=0;                // Accumulator
          uint8_t xp=1;                 // x^0
          for (int d=0;d<=L;++d)        // For each degree in C(x)
          {                             // Evaluate C(x)
            if (C[d])                   // Non-zero coefficient?
              acc^=gf.Multiply(C[d],xp);// Yes, update accumulator
            xp=gf.Multiply(xp,x);       // Update x^d to x^{d+1}
          }                             // Done evaluating C(x)
          if (acc==0&&ne<32)            // Found a root?
          {
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Coefficient index in cw[] that this root points to:
            // j=(255-1) mod 255. Note j in 0..254
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            iroot[ne]=i;                // Save the root index
            pos[ne]=(255-i)%255;        // Save the coefficient position
            if (lg)
              lg->Inf(" RS Chien: root i=%3d -> coeff_index(pos)=%3d (x=a^{%d})",i,pos[ne],i);
            ++ne;                       // Increment number of roots found
          }                             // Done processing this possible root
        }                               // Done searching for roots
        if (lg)
          lg->Inf(" RS Chien: L=%d ne=%d",L,ne);
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Check ne for validity
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        if (ne==0||ne>16)               // Uncorrectable?
        {                               // Yes
          sto->corr=0;                  // Say that
          sto->ok=false;                // We are not OK.
          if (lg)
            lg->Inf(" RS Chien: L=%d ne=%d (uncorrectable); returned at checking ne==0||ne>16",L,ne);
          return *sto;                  // Return status output
        }                               // Proceed with L valid
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // optional: enforce ne==L
        // if (ne != L) { sto->corr = 0; sto->ok = false; return *sto; }
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // (4) O(x) = [ S(x) * ?(x) ] mod x^{32}
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        uint8_t om[32];                 // Omega polynomial
        std::memset(om,0,32);           // Clear Omega
        for (int i=0;i<32;++i)          // For each degree in Omega
        {                               // Compute Omega[i]
          uint8_t acc=0;                // Accumulator
          for (int j=0;j<=i&&j<32;++j)  // For each term in S(x)
          {                             // Multiply S[j] * C[i-j]
            const int k=i-j;            // Degree in C(x)
            if(k>L)                     // Out of range?
              continue;                 // Yes, skip
            if(S[j]&&C[k])              // Non-zero terms?
              acc^=gf.Multiply(S[j],C[k]);// Yes, update accumulator
          }                             // Done computing Omega[i]
          om[i]=acc;                    // Store Omega[i]
        }                               // Done computing all Omega[i]
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // (5) Forney corrections
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        for (int k=0;k<ne;++k)          // For each error found
        {                               // Compute error magnitude and location
          const int i=iroot[k];         // alpha^i was a root
          const int j=pos[k];           // coefficient index in cw()
          const uint8_t Xinv=gf.alog[(255-i)%255];// X_inv=a^{i}
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // ?'(X^{-1})
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          uint8_t d=0;                  // Derivative of error locator polynomial
          for (int t=1;t<=L;t+=2)       // For each odd term in C(x)
          {                             // Compute derivative
            if (C[t])                   // Non-zero term?
              d^=gf.Multiply(C[t],gf.Power(Xinv,t-1));// Yes, update derivative
          }                             // Done computing derivative
          if (d==0)                     // Derivative is zero?
          {                             // Yes, skip this error
            if (lg)
              lg->Inf(" RS Forney[%d]: i=%3d j=%3d Xinv=a^{%3d} (0x%02X) d=0x%02X SKIP",
                k,i,j,i,Xinv,d);
            continue;                   // Skip this error
          }                             // Proceed with non-zero derivative
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // O(X^{-1})
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          uint8_t omx=0;                // Omega evaluated at X^{-1}
          for (int t=0;t<32;++t)        // For each term in Omega
          {                             // Compute Omega(X^{-1})
            if (om[t])                  // Non-zero term?
              omx^=gf.Multiply(om[t],gf.Power(Xinv,t));// Yes, update Omega(X^{-1})
          }                             // Done computing Omega(X^{-1})
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Error magnitude: em = O(X^{-1}) / ?'(X^{-1})
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          const uint8_t emag=gf.Multiply(omx,gf.Inverse(d));// Magnitude of the error
          const uint8_t bef=cw[static_cast<size_t>(j)];// Before codeword
          cw[static_cast<size_t>(j)]^=emag;// Our code word
          if (lg)
          {
            lg->Inf(" RS Forney[%d]: i=%3d j=%3d Xinv=a^{%3d} (0x%02X) d=0x%02X omx=0x%02X emag=0x%02X cw_before=0x%02X cw_after=0x%02X",
              k,i,j,i,Xinv,d,omx,emag,bef,cw[static_cast<size_t>(j)]);            
          }
        }                               // Done processing all errors
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // (6) Verify: recompute syndromes
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        bool vld=true;                  // Assume valid
        for (int i=0;i<32;++i)          // For each syndrome
        {                               // Recompute S[i]=C(alpha^{i+1})
          const uint8_t a=gf.alog[i+1]; // a = alpha^{i+1}
          uint8_t acc=0;                // Accumulator for syndrome computation
          for (int j=254;j>=0;--j)      // For each coefficient degree from high to low
            acc=gf.Add(gf.Multiply(acc,a),cw[static_cast<size_t>(j)]);// Horner evaluation, cw[j]=coef of degree (254-j)
          vld&=(acc==0);                // Valid if all recomputed syndromes are zero
        }                               // Done recomputing all syndromes
        if (lg)
          lg->Inf(" RS Verify: val=%d ne=%d", static_cast<int>(vld),ne);
        if (vld)                        // Valid codeword after corrections?
        {                               // Yes, successful decode
          sto->corr=ne;                 // Number of corrections
          sto->ok=true;                 // Decode successful
          if (lg)
            lg->Inf(" RS Chien: L=%d ne=%d (corrected), corr=%d, ok=%s",L,ne,sto->corr,(sto->ok?"true":"false"));
          if (o!=nullptr)               // Output buffer valid?
            std::memcpy(o,cw.data()+32,223);// Copy data bytes to output
        }                               // Done with valid codeword
        else                            // No, decode failed
        {                               // Yes, unsuccessful decode
          sto->corr=0;                  // No corrections
          sto->ok=false;                // Decode failed
          if (lg)
            lg->Inf(" RS Chien: L=%d ne=%d (uncorrectable), corr=%d, ok=%s",L,ne,sto->corr,(sto->ok?"true":"false"));
        }                               // Return status output
        if (lg)
          lg->Inf(" RS Decode: end; returned here");       // Log decode end
        return *sto;                    // Return status output
      }                                 // ~~~~~~~~~~ Decode ~~~~~~~~~~ //
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Decode but using vector input
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      inline RSStatus DecodeVec (       // n = 255
        const std::vector<uint8_t>* code,// Input codeword (length 255)
        std::vector<uint8_t>* const o)  // Output data buffer pointer (expects >=255; returns 223 data bytes)
      {                                 // ~~~~~~~~~~ DecodeVec ~~~~~~~~~~ //
        if (code==nullptr||o==nullptr||code->size()!=255)// Bad args?
        {                               // Yes, return failure
          sto->corr=0;                  // No corrections
          sto->ok=false;                // Decode failed
          return *sto;                  // Return status
        }                               // Good args, proceed with decode
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Ensure the output vector has space for 223 data bytes
        // and obtain a raw pointer
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        o->resize(223);                 // Resize output buffer
        uint8_t* const p=o->data();     // Get raw output pointer
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Decode expects a uint8_t* const* for the output buffer pointer
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        return Decode(code->data(),p);  // Call raw pointer version
      }                                 // ~~~~~~~~~~ DecodeVec ~~~~~~~~~~ //
    private:
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // GF(256) arithmetic tables and functions
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      GF256 gf;                         // GF(256) arithmetic tables
      RSStatus* sto{nullptr};           // Status object
      std::vector<uint8_t> gen;         // Generator polynomial coefficients
      // Logger for this component
      std::unique_ptr<logx::Logger> lg{};

  };
}
