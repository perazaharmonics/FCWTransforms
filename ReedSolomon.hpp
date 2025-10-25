/* 
 * *
 * * Filename: ReedSolomon.hpp
 * *
 * * Description:
 * *   CCSDS Reed-Solomon error correction encoder/decoder for RS(255,223) with
 * *   support for both encoding and decoding operations. , t=16 over GF(256) with
 * *   generator polynomial: G(x)= x^8 +x^7 +x^2 + 1 (0x11d). Derived from the C# reference,
 * *   which in-itself is derived from Phil Karn's "General purpose Reed-Solomon decoder for
 * * 
 * * Encoding: 223 data bytes -> 32 parity bytes -> 255 byte codeword.
 * * Decoding: Correct up to 16 random symbol errors; erasureless path
 * * Shortening: k<=223 -> n=k+32 by prefeeding (223-k) zero bytes.
 * *
 * * Author:
 * *  JEP, J. Enrique Peraza
 * *
 * * References:
 * *  https://github.com/crozone/ReedSolomonCCSDS/tree/master
 * *  "General purpose Reed-Solomon decoder for 8-bit symbols or less", Copyright 2003 Phil Karn, KA9Q
 * *
 */
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <memory>
#include "../logger/Logger.h"

// Mutual-exclusion guard: ensure RS API types are defined only once
#ifndef SDR_MDM_RS_API_GUARD
#define SDR_MDM_RS_API_GUARD

namespace sdr::mdm
{
  namespace detail
  {
    struct RSTables
    {
     constexpr static int rb=8;         // Bits per symbol
     constexpr static int N=255;        // CW length or symbols per block (1<<rb)-1
     constexpr static int nroot=32;     // Number of roots = parity symbols
     constexpr static int nfr=112;      // First consecutive root, index from
     constexpr static int prim=11;      // Primitive element, index from
     constexpr static int iprim=116;    // I-th primitive root of unity, index from
     constexpr static int a0=N;         // Alpha^0
     // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
     // Alpha to octet table
     // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
     inline static constexpr uint8_t ato[256]=
     {
        0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x87,0x89,0x95,0xad,0xdd,0x3d,0x7a,0xf4,
        0x6f,0xde,0x3b,0x76,0xec,0x5f,0xbe,0xfb,0x71,0xe2,0x43,0x86,0x8b,0x91,0xa5,0xcd,
        0x1d,0x3a,0x74,0xe8,0x57,0xae,0xdb,0x31,0x62,0xc4,0x0f,0x1e,0x3c,0x78,0xf0,0x67,
        0xce,0x1b,0x36,0x6c,0xd8,0x37,0x6e,0xdc,0x3f,0x7e,0xfc,0x7f,0xfe,0x7b,0xf6,0x6b,
        0xd6,0x2b,0x56,0xac,0xdf,0x39,0x72,0xe4,0x4f,0x9e,0xbb,0xf1,0x65,0xca,0x13,0x26,
        0x4c,0x98,0xb7,0xe9,0x55,0xaa,0xd3,0x21,0x42,0x84,0x8f,0x99,0xb5,0xed,0x5d,0xba,
        0xf3,0x61,0xc2,0x03,0x06,0x0c,0x18,0x30,0x60,0xc0,0x07,0x0e,0x1c,0x38,0x70,0xe0,
        0x47,0x8e,0x9b,0xb1,0xe5,0x4d,0x9a,0xb3,0xe1,0x45,0x8a,0x93,0xa1,0xc5,0x0d,0x1a,
        0x34,0x68,0xd0,0x27,0x4e,0x9c,0xbf,0xf9,0x75,0xea,0x53,0xa6,0xcb,0x11,0x22,0x44,
        0x88,0x97,0xa9,0xd5,0x2d,0x5a,0xb4,0xef,0x59,0xb2,0xe3,0x41,0x82,0x83,0x81,0x85,
        0x8d,0x9d,0xbd,0xfd,0x7d,0xfa,0x73,0xe6,0x4b,0x96,0xab,0xd1,0x25,0x4a,0x94,0xaf,
        0xd9,0x35,0x6a,0xd4,0x2f,0x5e,0xbc,0xff,0x79,0xf2,0x63,0xc6,0x0b,0x16,0x2c,0x58,
        0xb0,0xe7,0x49,0x92,0xa3,0xc1,0x05,0x0a,0x14,0x28,0x50,0xa0,0xc7,0x09,0x12,0x24,
        0x48,0x90,0xa7,0xc9,0x15,0x2a,0x54,0xa8,0xd7,0x29,0x52,0xa4,0xcf,0x19,0x32,0x64,
        0xc8,0x17,0x2e,0x5c,0xb8,0xf7,0x69,0xd2,0x23,0x46,0x8c,0x9f,0xb9,0xf5,0x6d,0xda,
        0x33,0x66,0xcc,0x1f,0x3e,0x7c,0xf8,0x77,0xee,0x5b,0xb6,0xeb,0x51,0xa2,0xc3,0x00
      };
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Index of octet table
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      inline static constexpr uint8_t idxof[256]=
      {
        0xff,0x00,0x01,0x63,0x02,0xc6,0x64,0x6a,0x03,0xcd,0xc7,0xbc,0x65,0x7e,0x6b,0x2a,
        0x04,0x8d,0xce,0x4e,0xc8,0xd4,0xbd,0xe1,0x66,0xdd,0x7f,0x31,0x6c,0x20,0x2b,0xf3,
        0x05,0x57,0x8e,0xe8,0xcf,0xac,0x4f,0x83,0xc9,0xd9,0xd5,0x41,0xbe,0x94,0xe2,0xb4,
        0x67,0x27,0xde,0xf0,0x80,0xb1,0x32,0x35,0x6d,0x45,0x21,0x12,0x2c,0x0d,0xf4,0x38,
        0x06,0x9b,0x58,0x1a,0x8f,0x79,0xe9,0x70,0xd0,0xc2,0xad,0xa8,0x50,0x75,0x84,0x48,
        0xca,0xfc,0xda,0x8a,0xd6,0x54,0x42,0x24,0xbf,0x98,0x95,0xf9,0xe3,0x5e,0xb5,0x15,
        0x68,0x61,0x28,0xba,0xdf,0x4c,0xf1,0x2f,0x81,0xe6,0xb2,0x3f,0x33,0xee,0x36,0x10,
        0x6e,0x18,0x46,0xa6,0x22,0x88,0x13,0xf7,0x2d,0xb8,0x0e,0x3d,0xf5,0xa4,0x39,0x3b,
        0x07,0x9e,0x9c,0x9d,0x59,0x9f,0x1b,0x08,0x90,0x09,0x7a,0x1c,0xea,0xa0,0x71,0x5a,
        0xd1,0x1d,0xc3,0x7b,0xae,0x0a,0xa9,0x91,0x51,0x5b,0x76,0x72,0x85,0xa1,0x49,0xeb,
        0xcb,0x7c,0xfd,0xc4,0xdb,0x1e,0x8b,0xd2,0xd7,0x92,0x55,0xaa,0x43,0x0b,0x25,0xaf,
        0xc0,0x73,0x99,0x77,0x96,0x5c,0xfa,0x52,0xe4,0xec,0x5f,0x4a,0xb6,0xa2,0x16,0x86,
        0x69,0xc5,0x62,0xfe,0x29,0x7d,0xbb,0xcc,0xe0,0xd3,0x4d,0x8c,0xf2,0x1f,0x30,0xdc,
        0x82,0xab,0xe7,0x56,0xb3,0x93,0x40,0xd8,0x34,0xb0,0xef,0x26,0x37,0x0c,0x11,0x44,
        0x6f,0x78,0x19,0x9a,0x47,0x74,0xa7,0xc1,0x23,0x53,0x89,0xfb,0x14,0x5d,0xf8,0x97,
        0x2e,0x4b,0xb9,0x60,0x0f,0xed,0x3e,0xe5,0xf6,0x87,0xa5,0x17,0x3a,0xa3,0x3c,0xb7
      };
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Generator polynomial coefficients for RS(255,223)
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      inline static constexpr uint8_t genpol[33]=
      {
        0x00,0xf9,0x3b,0x42,0x04,0x2b,0x7e,0xfb,0x61,0x1e,0x03,0xd5,0x32,0x42,0xaa,0x05,
        0x18,0x05,0xaa,0x42,0x32,0xd5,0x03,0x1e,0x61,0xfb,0x7e,0x2b,0x04,0x42,0x3b,0xf9,
        0x00
      };
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Conversion table from Taylor to Dual Basis
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      inline static constexpr uint8_t tdb[256]=
      {
        0x00,0x7b,0xaf,0xd4,0x99,0xe2,0x36,0x4d,0xfa,0x81,0x55,0x2e,0x63,0x18,0xcc,0xb7,
        0x86,0xfd,0x29,0x52,0x1f,0x64,0xb0,0xcb,0x7c,0x07,0xd3,0xa8,0xe5,0x9e,0x4a,0x31,
        0xec,0x97,0x43,0x38,0x75,0x0e,0xda,0xa1,0x16,0x6d,0xb9,0xc2,0x8f,0xf4,0x20,0x5b,
        0x6a,0x11,0xc5,0xbe,0xf3,0x88,0x5c,0x27,0x90,0xeb,0x3f,0x44,0x09,0x72,0xa6,0xdd,
        0xef,0x94,0x40,0x3b,0x76,0x0d,0xd9,0xa2,0x15,0x6e,0xba,0xc1,0x8c,0xf7,0x23,0x58,
        0x69,0x12,0xc6,0xbd,0xf0,0x8b,0x5f,0x24,0x93,0xe8,0x3c,0x47,0x0a,0x71,0xa5,0xde,
        0x03,0x78,0xac,0xd7,0x9a,0xe1,0x35,0x4e,0xf9,0x82,0x56,0x2d,0x60,0x1b,0xcf,0xb4,
        0x85,0xfe,0x2a,0x51,0x1c,0x67,0xb3,0xc8,0x7f,0x04,0xd0,0xab,0xe6,0x9d,0x49,0x32,
        0x8d,0xf6,0x22,0x59,0x14,0x6f,0xbb,0xc0,0x77,0x0c,0xd8,0xa3,0xee,0x95,0x41,0x3a,
        0x0b,0x70,0xa4,0xdf,0x92,0xe9,0x3d,0x46,0xf1,0x8a,0x5e,0x25,0x68,0x13,0xc7,0xbc,
        0x61,0x1a,0xce,0xb5,0xf8,0x83,0x57,0x2c,0x9b,0xe0,0x34,0x4f,0x02,0x79,0xad,0xd6,
        0xe7,0x9c,0x48,0x33,0x7e,0x05,0xd1,0xaa,0x1d,0x66,0xb2,0xc9,0x84,0xff,0x2b,0x50,
        0x62,0x19,0xcd,0xb6,0xfb,0x80,0x54,0x2f,0x98,0xe3,0x37,0x4c,0x01,0x7a,0xae,0xd5,
        0xe4,0x9f,0x4b,0x30,0x7d,0x06,0xd2,0xa9,0x1e,0x65,0xb1,0xca,0x87,0xfc,0x28,0x53,
        0x8e,0xf5,0x21,0x5a,0x17,0x6c,0xb8,0xc3,0x74,0x0f,0xdb,0xa0,0xed,0x96,0x42,0x39,
        0x08,0x73,0xa7,0xdc,0x91,0xea,0x3e,0x45,0xf2,0x89,0x5d,0x26,0x6b,0x10,0xc4,0xbf
      };
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Conversion from Taylor to Conventional Basis
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      inline static constexpr uint8_t tcb[256]=
      {
        0x00,0xcc,0xac,0x60,0x79,0xb5,0xd5,0x19,0xf0,0x3c,0x5c,0x90,0x89,0x45,0x25,0xe9,
        0xfd,0x31,0x51,0x9d,0x84,0x48,0x28,0xe4,0x0d,0xc1,0xa1,0x6d,0x74,0xb8,0xd8,0x14,
        0x2e,0xe2,0x82,0x4e,0x57,0x9b,0xfb,0x37,0xde,0x12,0x72,0xbe,0xa7,0x6b,0x0b,0xc7,
        0xd3,0x1f,0x7f,0xb3,0xaa,0x66,0x06,0xca,0x23,0xef,0x8f,0x43,0x5a,0x96,0xf6,0x3a,
        0x42,0x8e,0xee,0x22,0x3b,0xf7,0x97,0x5b,0xb2,0x7e,0x1e,0xd2,0xcb,0x07,0x67,0xab,
        0xbf,0x73,0x13,0xdf,0xc6,0x0a,0x6a,0xa6,0x4f,0x83,0xe3,0x2f,0x36,0xfa,0x9a,0x56,
        0x6c,0xa0,0xc0,0x0c,0x15,0xd9,0xb9,0x75,0x9c,0x50,0x30,0xfc,0xe5,0x29,0x49,0x85,
        0x91,0x5d,0x3d,0xf1,0xe8,0x24,0x44,0x88,0x61,0xad,0xcd,0x01,0x18,0xd4,0xb4,0x78,
        0xc5,0x09,0x69,0xa5,0xbc,0x70,0x10,0xdc,0x35,0xf9,0x99,0x55,0x4c,0x80,0xe0,0x2c,
        0x38,0xf4,0x94,0x58,0x41,0x8d,0xed,0x21,0xc8,0x04,0x64,0xa8,0xb1,0x7d,0x1d,0xd1,
        0xeb,0x27,0x47,0x8b,0x92,0x5e,0x3e,0xf2,0x1b,0xd7,0xb7,0x7b,0x62,0xae,0xce,0x02,
        0x16,0xda,0xba,0x76,0x6f,0xa3,0xc3,0x0f,0xe6,0x2a,0x4a,0x86,0x9f,0x53,0x33,0xff,
        0x87,0x4b,0x2b,0xe7,0xfe,0x32,0x52,0x9e,0x77,0xbb,0xdb,0x17,0x0e,0xc2,0xa2,0x6e,
        0x7a,0xb6,0xd6,0x1a,0x03,0xcf,0xaf,0x63,0x8a,0x46,0x26,0xea,0xf3,0x3f,0x5f,0x93,
        0xa9,0x65,0x05,0xc9,0xd0,0x1c,0x7c,0xb0,0x59,0x95,0xf5,0x39,0x20,0xec,0x8c,0x40,
        0x54,0x98,0xf8,0x34,0x2d,0xe1,0x81,0x4d,0xa4,0x68,0x08,0xc4,0xdd,0x11,0x71,0xbd
      };
    }; // struct RSTables
    } // namespace detail
    struct RSStatus
    {
      int32_t corr{0};                    // Number of corrected symbols
      bool ok{false};                     // True if decode successful
      RSStatus (void)=default;            // Default constructor            
    };    
    class ReedSolomon
    {
        public:
          using T=detail::RSTables;
          static constexpr int N=T::N;  // Codeword length
          static constexpr int np=T::nroot; // Number of roots (parity symbols)
          static constexpr int K=T::N-T::nroot; // Data length
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Working state for one decode
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        struct CodecState
        {
          const uint8_t* in{nullptr};   // Input codeword
          uint8_t* o{nullptr};    // Output dataword
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Optional erasures
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          const int* eras{nullptr};     // Erasure positions
          int neras{0};                 // Number of erasures
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Stage buffers
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          uint8_t s[np]{};              // Syndromes
          uint8_t b[np+1]{};            // B polynomial
          uint8_t t[np+1]{};            // T polynomial
          uint8_t om[np+1]{};           // Omega polynomial
          uint8_t root[np]{};           // Error roots
          uint8_t reg[np+1]{};          // Lambda shift register
          uint8_t loc[np]{};            // Error locations
          uint8_t lambda[np+1]{};       // Lambda polynomial
          int synerr{0};                // Syndrome error flag
          int ordl{0};                  // Number of lambda coefficients (order)
          int ordom{0};                 // Number of omega coefficients (order)
          int nr{0};                    // Number of roots found
          bool forfail{false};          // True if Forney Correction fails.
        };  // struct CodecState
          ReedSolomon (void)
          {
            // Lazily create a logger so encode/decode paths emit to logs
            // Tests/exporters typically pass SDR env, so Logger resolves to $SDR/src/logs/log.txt
            lg = logx::Logger::NewLogger();
          }
          ~ReedSolomon (void)=default;  // Default destructor
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Encode/Decode functions
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          inline void Encode (
            uint8_t* dat,               // [Input,Output] dataword (223B); mutated if dbase=true
            size_t dlen,                // Length of dataword (>=223)
            uint8_t* par,               // Output parity symbols (32B)
            size_t plen,                // Length of parity symbols (>=32)
            bool dbase=false) const     // True if input data is in dual basis
          {                             // ~~~~~~~~~ Encode ~~~~~~~~~~ //
            if (dat==nullptr||par==nullptr)// Bad input args?
              return;                   // Early return, can't do much.
            if (dlen<static_cast<size_t>(K)||plen<static_cast<size_t>(np))
              return;                   // Early return, can't do much.
            if (lg!=nullptr)
              lg->Inf("RS Encode: Begin encoding %zu data bytes",dlen);
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Convert data from dual basis to conventional basis, if requested
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            if (dbase==true)            // Input data in dual basis?
            {                           // Yes, convert to conventional basis
              for (int i=0;i<K;++i)     // For each data byte
                dat[i]=T::tcb[dat[i]];  // Convert to conventional basis
            }
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Zeroize parity
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            std::memset(par,0,np);      // Clear parity bytes
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Shift register parity accumulation
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            for (int i=0;i<K;++i)       // For each data byte
            {
              const uint8_t fb=T::idxof[static_cast<uint8_t>(dat[i]^par[0])]; // Feedback byte
              if (fb!=T::a0)            // Non-zero feedback?
              {                         // Yes, update parity
                for (int j=1;j<np;++j)  // For each parity byte
                  par[j]^=T::ato[(fb+T::genpol[np-j])%T::N];// Update parity byte
                if (lg!=nullptr)
                  lg->Deb("RS Encode: Data byte %d, feedback=0x%02X",i,fb);
              }                         // Done updating parity
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              // Shift left by 1
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              std::memmove(&par[0],&par[1],static_cast<size_t>(np-1));// Shift left
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              // Inject new parity symbol
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              par[np-1]=(fb!=T::a0)?T::ato[(fb+T::genpol[0])%T::N]:0;// New parity byte
              if (lg!=nullptr)
                lg->Deb("RS Encode: Parity after data byte %d:",i);
            }                           // Done with shift register
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Convert parity to dual basis
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            if (dbase==true)            // Output parity in dual basis?
            {                           // Yes, convert to dual basis
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              // Convert each parity byte back to dual basis
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              for (int i=0;i<K;++i)     // For each data byte
                dat[i]=T::tdb[dat[i]];  // Convert to dual basis
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              // Convert parity bytes to dual basis
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              for (int i=0;i<np;++i)    // For each parity byte
                par[i]=T::tdb[par[i]];  // Convert to dual basis
            }                           // Done converting to dual basis
          }                             // ~~~~~~~~~ Encode ~~~~~~~~~~ //
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // Encode a 255-byte block (223 data + 32 parity) in-place
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          inline void EncodeBlock (
            uint8_t* bl,                // [In,Out] 255B buffer (first 223B = data)
            size_t len,                 // Length of buffer (>=255)
            bool dbase=false) const     // True if input data is in dual basis
          {                             // ~~~~~~~~~ EncodeBlock ~~~~~~~~~~ //
            if (bl==nullptr||len<static_cast<size_t>(N))// Bad args?
              return;                   // Early return, can't do much.
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Pointers to data and parity windows
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            uint8_t* d=bl;              // Data window
            uint8_t* p=bl+K;            // Parity window
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Call Encode function
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            Encode(d,K,p,np,dbase);     // Encode the block
          }                             // ~~~~~~~~~ EncodeBlock ~~~~~~~~~~ //
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // High-level decode function
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          inline RSStatus Decode (
            const uint8_t* cw255,       // Input codeword
            uint8_t* in223,             // Output dataword
            const int* eras=nullptr,    // Optional erasure positions
            int ner=0) const            // Number of erasures                
          {                             // ~~~~~~~~~ Decode ~~~~~~~~~ //
            RSStatus sto{};             // Status object to return
            if (cw255==nullptr||in223==nullptr)// Bad input args?
              return sto;               // Return with error status
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Work on a mutable copy of the block
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            uint8_t block[N];           // Mutable copy of codeword
            std::memcpy(block,cw255,N); // Copy input codeword
            CodecState cs{};            // Codec state
            cs.in=block;                // Set input codeword
            cs.o=in223;                 // Set output dataword
            cs.eras=eras;               // Set erasure positions
            cs.neras=ner;               // Set number of erasures
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // 1) Compute syndromes
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            if (!ComputeSyndromes(cs))  // Could we compute the syndromes?
            {                           // No, no errors detected.
              std::memcpy(cs.o,cs.in,K);// Copy data to output
              sto.ok=true;              // Set success status
              sto.corr=0;               // No corrections
              return sto;               // Early return.
            }                           // Done computing syndromes
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // 2) Berlekamp-Massey to compute error locator polynomial
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            if (!BerlekampMassey(cs))   // Could we compute the error locator?
            {                           // No, there was a decode failure.
              sto.ok=false;             // Decode failed.
              sto.corr=-1;              // Indicate failure with -1
              return sto;               // Early return.
            }                           // Done Berlekamp-Massey
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // 3) ChienSearch: Find roots of error locator polynomial
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            int n=ChienSearch(cs);      // Find roots
            if (n<0)                    // Any errors found?
            {
              sto.ok=false;             // Decode failed.
              sto.corr=-1;              // Indicate failure with -1
              return sto;               // Early return.
            }                           // Done with Chien Search
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // 4) Compute Omega polynomial: s(x)*lambda(x) mod x^np (index form)
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            ComputeOmega(cs);           // Compute Omega polynomial
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // 5) Forney algorithm to compute error magnitudes
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            ForneyCorrection(cs,block); // Correct the errors
            if (cs.forfail)             // Did Forney fail?
            {                           // Yes, indicate failure
              sto.ok=false;             // Decode failed.
              sto.corr=-1;              // Indicate failure with -1
              return sto;               // Early return.
            }                           // Done Forney correction
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // 6) Recompute syndromes to verify correction
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            if (!VerifyCorrections(cs,block)) // Were corrections successful?
            {                           // No, indicate failure
              sto.ok=false;             // Decode failed.
              sto.corr=-1;              // Indicate failure with -1
              return sto;               // Early return.
            }                           // Done verifying corrections
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Copy data to output
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            std::memcpy(cs.o,block,K);  // Copy data to output
            sto.ok=true;                // Set success status
            sto.corr=cs.nr;             // Set number of corrections
            return sto;                 // Return with success status
          }                             // ~~~~~~~~~ Decode ~~~~~~~~~ //
        protected:
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // 1) Compute syndromes
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          inline bool ComputeSyndromes (
            CodecState& cs) const       
          {                             // ~~~~~~~~~ ComputeSyndromes ~~~~~~~~~ //
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Form the syndromes in polynomial form
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            for (int i=0;i<np;++i)      // For each syndrome
              cs.s[i]=cs.in[0];         // Initialize to first byte
            for (int i=1;i<N;++i)       // For each byte in codeword
            {                           // and for...
              for (int j=0;j<np;++j)    // The number of roots (parity symbols)
              {                         // Compute syndrome j
                if (cs.s[j]==0)         // If current syndrome is zero
                  cs.s[j]=cs.in[i];     // Next byte is the new syndrome
                else                    // Else, non zero, compute syndrome
                  cs.s[j]=static_cast<uint8_t>(cs.in[i]^T::ato[(T::idxof[cs.s[j]]+(T::nfr+j)*T::prim)%T::N]);
                if (lg!=nullptr)
                  lg->Deb("RS Syn: After byte %d, syndrome %d=0x%02X",i,j,cs.s[j]);
              }                         // End for number of roots
            }                           // End for each byte in codeword
            cs.synerr=0;                // Clear syndrome error flag
            for (int i=0;i<np;++i)      // For the number of parity symbols (roots)
            {                           // Check each syndrome
              cs.synerr|=cs.s[i];       // Accumulate syndrome bytes
              cs.s[i]=T::idxof[cs.s[i]];// Convert to index form
              if (lg!=nullptr)
                lg->Deb("RS Syn: Syndrome %d=0x%02X (index form)",i,cs.s[i]);
            }                           // End for each parity symbol
            return (cs.synerr!=0);      // Return true if any syndrome non zero
          }                             // ~~~~~~~~~ ComputeSyndromes ~~~~~~~~~ //
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // 2) Berlekamp-Massey algorithm to compute error locator polynomial
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          inline bool BerlekampMassey (
            CodecState& cs) const
          {                             // ~~~~~~~~~ BerlekampMassey ~~~~~~~~~ //
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Lambda <- 1, all others 0
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            std::memset(cs.lambda,0,sizeof(cs.lambda));// Clear lambda
            cs.lambda[0]=1;             // Lambda_0=1
            if (lg!=nullptr)
              lg->Inf("RS BM: Begin Berlekamp-Massey algorithm");
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Fold erasure into Lambda if provided
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            if (cs.eras!=nullptr&&cs.neras>0)// Any erasures?
            {                           // Yes, fold into Lambda
              cs.lambda[1]=T::ato[(T::prim*(T::N-1-cs.eras[0]))%T::N];// First erasure
              for (int i=1;i<cs.neras;++i)// For the remaining erasure to process...
              {                         // Compute next erasure
                uint8_t u=static_cast<uint8_t>((T::prim*(T::N-1-cs.eras[i])) % T::N);
                for (int j=i+1;j>0;--j) // Starting from the highest degree coeff
                {
                  uint8_t tmp=T::idxof[cs.lambda[j-1]];// Get lambda j-1 in index form
                  if (tmp!=T::a0)       // Non-zero column?
                    cs.lambda[j]^=T::ato[(tmp+u)%T::N];// Update lambda j
                }                       // End for each lambda coeff
                if (lg!=nullptr)      // Logging enabled?
                {                     // Log current lambda polynomial
                  lg->Deb("RS BM: After erasure %d at pos %d, lambda=",i,cs.eras[i]);
                  for (int k=0;k<=np;++k)
                    lg->Deb("%02X ",cs.lambda[k]);
                }                     // End logging
              }                         // End for each erasure
            }                           // End if any erasures
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Update B(x) polynomial
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            for (int i=0;i<=np;++i)
              cs.b[i]=T::idxof[cs.lambda[i]];// B(x)=lambda(x) in index form
            int r=cs.neras;             // r=number of erasures
            int elf=cs.neras;           // elf=last failure position
            while (++r<=np)             // While we have more parity symbols than # of erasures
            {                           // Compute discrepancy in polynomial form
              uint8_t d=0;              // Discrepancy
              for (int i=0;i<r;++i)     // For each coeff up to r... 
                if ((cs.lambda[i]!=0)&&(cs.s[r-i-1]!=T::a0))// Did we find any non-zero terms?
                  d^=T::ato[(T::idxof[cs.lambda[i]]+cs.s[r-i-1])%T::N];// Yes, update discrepancy
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              // Convert discrepancy to index form
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              d=T::idxof[d];            // Discrepancy in index form
              if (d==T::a0)             // Did we find any discrepancy?
              {                         // No, shift B(x) <- x*B(x)
                std::memmove(&cs.b[1],&cs.b[0],np); // Shift B(x) coeffs
                cs.b[0]=T::a0;          // Set B_0=0
              }                         // End if discrepancy==N.
              else                      // Else, non-zero discrepancy
              {                         // Update T(x)
                // ~~~~~~~~~~~~~~~~~~~~ //
                // T(x)=Lambda(x)-d*x*B(x)
                // ~~~~~~~~~~~~~~~~~~~~ //
                cs.t[0]=cs.lambda[0];   // T_0=Lambda_0
                for (int i=0;i<np;++i)  // For each coeff up to # of parity symbols...
                {
                  if (cs.b[i]!=T::a0)   // Non-zero B_i?
                    cs.t[i+1]=static_cast<uint8_t>(cs.lambda[i+1]^T::ato[(d+cs.b[i])%T::N]);
                  else                  // Else, B_i==0
                    cs.t[i+1]=cs.lambda[i+1];// T_i+1=Lambda_i+1
                }                       // End for each coeff
                if ((2*elf)<=(r+cs.neras-1))// Is is a significant failure?
                {                       // Yes, update B(x)
                  elf=r+cs.neras-elf;   // Update elf
                  // ~~~~~~~~~~~~~~~~~~ //
                  // B(x) <- inv(d)*lambda(x)
                  // ~~~~~~~~~~~~~~~~~~ //
                  for (int i=0;i<=np;++i)
                    cs.b[i]=(cs.lambda[i]==0)?T::a0:static_cast<uint8_t>((T::idxof[cs.lambda[i]]-d+T::N)%T::N);
                  if (lg!=nullptr)
                    lg->Deb("RS BM: Significant failure at r=%d, updated B(x)",r);
                }                       // Done updating B(x)
                else                    // Else, non-significant failure
                {                       // Shift T(x) into B(x)
                  // ~~~~~~~~~~~~~~~~~~ //
                  // Shift B(x) <- x*B(x)
                  // ~~~~~~~~~~~~~~~~~~ //
                  std::memmove(&cs.b[1],&cs.b[0],np); // Shift B(x) coeffs
                  cs.b[0]=T::a0;        // Set B_0=0
                  if (lg!=nullptr)
                    lg->Deb("RS BM: Non-significant failure at r=%d, shifted B(x)",r);
                }                       // End else non-significant failure
                std::memcpy(cs.lambda,cs.t,sizeof(cs.lambda));// Copy T(x) to Lambda(x)
              }                         // End else non-zero discrepancy
              if (lg!=nullptr)// Logging enabled every 4 iterations
              {                         // Log current lambda polynomial
                lg->Deb("RS BM: r=%d lambda=",r);
                for (int i=0;i<=np;++i)
                  lg->Deb("%02X ",cs.lambda[i]);
              }                         // End logging
            }                           // End while r<=np
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // To index form; compute deg(lambda)
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            cs.ordl=0;                  // Clear lambda coeff order
            for (int i=0;i<=np;++i)     // For each lambda coeff
            {                           // Convert to index form
              cs.lambda[i]=T::idxof[cs.lambda[i]];// To index form
              if (cs.lambda[i]!=T::a0)  // Non-zero coeff?
                cs.ordl=i;              // Update order
              if (lg!=nullptr)          // Logging enabled?
                lg->Deb("RS BM: Final lambda[%d]=%02X",i,cs.lambda[i]);
            }                           // Done converting lambda coeffs to index form
            if (lg!=nullptr)
              lg->Inf("RS BM: Completed Berlekamp-Massey algorithm (success)");
            return true;                // Return success
          }                             // ~~~~~~~~~ BerlekampMassey ~~~~~~~~~ //
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // 3) Chien Search to find error locations
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          inline int ChienSearch (
            CodecState& cs) const
          {                             // ~~~~~~~~~~ ChienSearch ~~~~~~~~~~ //
            if (lg!=nullptr)
              lg->Inf("RS CS: Begin Chien's Search algorithm");
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // reg[1..] <- lambda[1..]
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            std::memset(cs.reg,0,sizeof(cs.reg));// Clear shift register
            std::memcpy(&cs.reg[1],&cs.lambda[1],np); // Load lambda coeffs
            cs.nr=0;                    // Clear number of roots found
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Begin search
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            for (int i=1,k=T::iprim-1;i<=T::N;++i,k=(k+T::iprim)%T::N)
            {
              uint8_t q=1;              // lambda(0) in poly domain contributes 1
              for (int j=cs.ordl;j>0;--j)// For each lambda coeff (skip j=0 per reference)
              {                         // Evaluate lambda at alpha^k
                if (cs.reg[j]!=T::a0)   // Non-zero coeff?
                {                       // Yes, update q
                  cs.reg[j]=static_cast<uint8_t>((cs.reg[j]+j)%T::N);// Shift: reg[j] += j
                  q^=T::ato[cs.reg[j]]; // Update q 
                }                       // End if non-zero coeff
                if (lg!=nullptr)
                  lg->Deb("RS CS: i=%d j=%d reg=%02X q=%02X",i,j,cs.reg[j],q);
              }                         // End for each lambda coeff
              if (q!=0)                 // Did we find a root?
                continue;               // No, continue search
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              // Found root; update locs
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              cs.root[cs.nr]=static_cast<uint8_t>(i); // Store root location
              cs.loc[cs.nr]=static_cast<uint8_t>(k); // Store error location
              if (lg!=nullptr)
                lg->Deb("RS CS: Found root %d at position %d",cs.nr,cs.root[cs.nr]);
              if (++cs.nr==cs.ordl)     // Found all roots?
                break;                  // Yes, exit search 
            }                           // End for each element in field
            if (lg!=nullptr)
              lg->Inf("RS CS: Completed Chien's Search algorithm, found %d roots",cs.nr);
            if (cs.ordl!=cs.nr)         // Found all roots?
              return -1;                // Return failure if not all roots found
            return cs.nr;               // Return number of roots found
          }                             // ~~~~~~~~~~ ChienSearch ~~~~~~~~~~ //
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // 4) Compute Omega polynomial
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          inline void ComputeOmega (
            CodecState& cs) const
          {                             // ~~~~~~~~~ ComputeOmega ~~~~~~~~~ //
            cs.ordom=0;                 // Clear omega order
            if (lg!=nullptr)
              lg->Inf("RS CO: Begin Omega polynomial computation");
            for (int i=0;i<np;++i)      // For each coeff up to # of parity symbols
            {                           // Compute omega coeff i
              uint8_t tmp{0};           // Temp accumulator
              int up=(cs.ordl<i)?cs.ordl:i;// Upper limit
              for (int j=up;j>=0;--j)   // Starting from the highest degree coeff
              {                         // Compute term j
                if ((cs.s[i-j]!=T::a0)&&(cs.lambda[j]!=T::a0))// Non-zero terms?
                {                       // Yes, update temp
                  tmp^=T::ato[(cs.s[i-j]+cs.lambda[j])%T::N];// Update temp
                  if (lg!=nullptr)
                    lg->Deb("RS CO: Omega update i=%d j=%d s=%02X lambda=%02X tmp=%02X",
                      i,j,cs.s[i-j],cs.lambda[j],tmp);                    
                }                       // End if non-zero terms
              }                         // End for each term
              if (tmp!=0)               // Non-zero omega coeff?
                cs.ordom=i;             // Update omega order
              cs.om[i]=T::idxof[tmp];   // Store omega coeff in index form
              if (lg!=nullptr)
                lg->Deb("RS CO: Omega[%d]=%02X",i,cs.om[i]);
            }                           // End for each coeff
            cs.om[np]=T::a0;            // Clear highest omega coeff
            if (lg!=nullptr)
              lg->Inf("RS CO: Completed Omega polynomial computation");
          }                             // ~~~~~~~~~ ComputeOmega ~~~~~~~~~ //
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // 5) Forney's algorithm to compute error magnitudes and correct
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          inline void ForneyCorrection (
            CodecState& cs,             // Codec state
            uint8_t* block) const       // Codeword buffer
          {                             // ~~~~~~~~ ForneyCorrection ~~~~~~~~ //
            if (block==nullptr)         // Valid codeword buffer?
              return;                   // No, return failure
            if (lg!=nullptr)
              lg->Inf("RS FC: Begin Forney's error correction");
            for (int j=cs.nr-1;j>=0;--j)// For each root found
            {                           // Compute error magnite
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              // num1 = omega(inv(X_l))
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              uint8_t num1{0};          // Numerator 1
              for (int i=cs.ordom;i>=0;--i)// For each omega coeff
              {                         // Compute term i
                if (cs.om[i]!=T::a0)    // Non-zero coeff?
                {                       // Yes, update num1
                  num1^=T::ato[(cs.om[i]+(i*cs.root[j]))%T::N];// Update num1
                  if (lg!=nullptr)
                    lg->Deb("RS FC: num1 update j=%d i=%d om=%02X num1=%02X",j,i,cs.om[i],num1);
                }                       // End if non-zero coeff   
              }                         // End for each omega coeff
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              // num2=inv(X_l)^(FCR-1)
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              uint8_t num2=T::ato[((cs.root[j]*(T::nfr-1))+T::N)%T::N];// num2=inv(X_l)^(FCR-1)
              if (lg!=nullptr)
                lg->Deb("RS FC: num2 for root %d = %02X",j,num2);
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              // den = lamdba'(inv(X_l)) using even terms of lambda (formal derivative)
              // ~~~~~~~~~~~~~~~~~~~~~~ //
              uint8_t den{0};           // Denominator
              int st=(cs.ordl<(np-1))?cs.ordl:(np-1);// Start point
              st&=~1;                   // Make even
              for (int i=st;i>=0;i-=2)  // For each even lambda coeff
              {                         // Compute denominator 
                if (cs.lambda[i+1]!=T::a0)// Non-zero coeff?
                {                       // Yes, update den
                  den^=T::ato[(cs.lambda[i+1]+(i*cs.root[j]))%T::N];// Update den
                  if (lg!=nullptr)
                    lg->Deb("RS FC: den update j=%d i=%d lambda=%02X den=%02X",j,i,cs.lambda[i+1],den);
                }                       // End if non-zero coeff
              }                         // End for each even lambda coeff
              if (den==0)               // Denominator zero?
              {                         // Yes, that's a Forney failure.
                cs.forfail=true;        // Set Forney failure flag
                if (lg!=nullptr)
                  lg->Err("RS FC: Forney correction failure at position %d (denominator zero)",cs.loc[j]);
                return;                 // Return failure
              }                         // End if denominator zero
              if (num1!=0)              // Numerator equal zero?
              {                         // No, compute error magnitude
               block[cs.loc[j]] ^= T::ato[(T::idxof[num1]+T::idxof[num2]+T::N-T::idxof[den])%T::N];
               if (lg!=nullptr)
                 lg->Deb("RS FC: Corrected error at position %d, magnitude %02X",
                   cs.loc[j],T::ato[(T::idxof[num1]+T::idxof[num2]+T::N-T::idxof[den])%T::N]);               
              }                         // End if numerator non-zero
            }                           // End for each root
            if (lg!=nullptr)
              lg->Inf("RS FC: Completed Forney's error correction");
          }                             // ~~~~~~~~ ForneyCorrection ~~~~~~~~ //
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          // 6) VerifyCorrections: recompute syndromes on corrected block,
          // expect all zero
          // ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
          inline bool VerifyCorrections (
            CodecState& cs,             // Codec state
            const uint8_t* block) const // Codeword buffer
          {                             // ~~~~~~~~ VerifyCorrections ~~~~~~~~ //
            if (lg!=nullptr)
              lg->Inf("RS VC: Begin verification of corrections");
            (void)cs; // silence unused warning in some builds
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            // Recompute syndromes
            // ~~~~~~~~~~~~~~~~~~~~~~~~ //
            int synerr{0};
            // Rebuild poly-form syndromes across full codeword like ComputeSyndromes
            uint8_t s2[np];             // New syndromes
            for (int i=0;i<np;++i)      // For each syndrome
              s2[i]=block[0];           // Initialize to first byte
            for (int j=1;j<N;++j)       // Iterate over the whole codeword
            {
              for (int i=0;i<np;++i)
              {
                if (s2[i]==0)
                  s2[i]=block[j];
                else
                  s2[i]=static_cast<uint8_t>(block[j]^T::ato[(T::idxof[s2[i]]+(T::nfr+i)*T::prim)%T::N]);
              }
            }
            for (int i=0;i<np;++i)
            {
              synerr|=s2[i];
              if (lg!=nullptr)
                lg->Deb("RS VC: Recomputed syndrome[%d]=%02X",i,s2[i]);
            }
            return synerr==0;           // Return true if all syndromes zero
          }                             // ~~~~~~~~ VerifyCorrections ~~~~~~~~ //
        private:
          std::unique_ptr<logx::Logger> lg{};
    };
  }


#endif // SDR_MDM_RS_API_GUARD
