/*
 * Copyright 2002-2020 Intel Corporation.
 * 
 * This software is provided to you as Sample Source Code as defined in the accompanying
 * End User License Agreement for the Intel(R) Software Development Products ("Agreement")
 * section 1.L.
 * 
 * This software and the related documents are provided as is, with no express or implied
 * warranties, other than those that are expressly stated in the License.
 */


#include "pin.H"
#include <iostream>
#include <fstream>
using std::cerr;
using std::endl;
using std::hex;
using std::ios;
using std::string;
using std::vector;
using std::pair;

/* ===================================================================== */
/* Names of malloc and free */
/* ===================================================================== */
// #if defined(TARGET_MAC)
// #define MALLOC "_malloc"
// #define FREE "_free"
// #else
// #define MALLOC "malloc"
// #define FREE "free"
// #endif

#define RRAM_MALLOC "rram_malloc"
#define RRAM_FREE "rram_free"
string RRAM_MALLOC_mangled = "";
string RRAM_FREE_mangled = "";

/* ===================================================================== */
/* Global Variables */
/* ===================================================================== */

std::ofstream TraceFile;

/* ===================================================================== */
/* Commandline Switches */
/* ===================================================================== */

KNOB< string > KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool", "o", "1.out", "specify trace file name");

/* ===================================================================== */

/* ===================================================================== */
/* Analysis routines                                                     */
/* ===================================================================== */
pair<ADDRINT, size_t> last_malloc_record;
vector<pair<ADDRINT, size_t> > all_malloc_record;

// Move from memory to register
ADDRINT DoLoad(REG reg, ADDRINT* addr)
{
    TraceFile << "Emulate loading from addr " << addr << " to " << REG_StringShort(reg) << endl;
    ADDRINT value;
    PIN_SafeCopy(&value, addr, sizeof(ADDRINT));
    return value;
}

VOID Arg1Before(CHAR* name, ADDRINT size)
{
    TraceFile << name << "(" << size << ")" << endl;
    last_malloc_record.first = size;
}

VOID MallocAfter(ADDRINT ret)
{
    TraceFile << "  returns " << ret << endl;
    last_malloc_record.second = ret;
    all_malloc_record.push_back(last_malloc_record);
}

/* ===================================================================== */
/* Instrumentation routines                                              */
/* ===================================================================== */
bool isRRAM_addr(ADDRINT addr)
{
    return true;
}

VOID EmulateLoad(INS ins, VOID* v)
{
    // Find the instructions that move a value from memory to a register
    if (INS_Opcode(ins) == XED_ICLASS_MOV && INS_IsMemoryRead(ins) && INS_OperandIsReg(ins, 0) && INS_OperandIsMemory(ins, 1))
    {
        // op0 <- *op1
        if (isRRAM_addr((ADDRINT) IARG_MEMORYREAD_EA)){
            INS_InsertCall(ins, IPOINT_BEFORE, AFUNPTR(DoLoad), IARG_UINT32, REG(INS_OperandReg(ins, 0)), IARG_MEMORYREAD_EA,
                       IARG_RETURN_REGS, INS_OperandReg(ins, 0), IARG_END);

            // Delete the instruction
            INS_Delete(ins);
        }
    }
}

void find_mangled_symbols(IMG img)
{
    std::cout << "----------------------" << std::endl;
    for (SYM sym = IMG_RegsymHead(img); SYM_Valid(sym); sym = SYM_Next(sym))
    {
        string s1 = SYM_Name(sym);
        if (s1.find(RRAM_MALLOC) != std::string::npos)
        {
            RRAM_MALLOC_mangled = s1;
        }
        if (s1.find(RRAM_FREE) != std::string::npos && s1.find("GLOBAL") == std::string::npos)
        {
            RRAM_FREE_mangled = s1;
        }
    }
    std::cout << RRAM_MALLOC_mangled << std::endl;
    std::cout << RRAM_FREE_mangled << std::endl;
}

VOID Image(IMG img, VOID* v)
{
    find_mangled_symbols(img);
    // Instrument the malloc() and free() functions.  Print the input argument
    // of each malloc() or free(), and the return value of malloc().
    //
    //  Find the malloc() function.
    RTN mallocRtn = RTN_FindByName(img, RRAM_MALLOC_mangled.c_str());
    if (RTN_Valid(mallocRtn))
    {
        RTN_Open(mallocRtn);
        // Instrument RRAM_MALLOC() to print the input argument value and the return value.
        RTN_InsertCall(mallocRtn, IPOINT_BEFORE, (AFUNPTR)Arg1Before, IARG_ADDRINT, RRAM_MALLOC, IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                       IARG_END);
        RTN_InsertCall(mallocRtn, IPOINT_AFTER, (AFUNPTR)MallocAfter, IARG_FUNCRET_EXITPOINT_VALUE, IARG_END);

        RTN_Close(mallocRtn);
    }

    // Find the free() function.
    RTN freeRtn = RTN_FindByName(img, RRAM_FREE_mangled.c_str());
    if (RTN_Valid(freeRtn))
    {
        RTN_Open(freeRtn);
        // Instrument free() to print the input argument value.
        RTN_InsertCall(freeRtn, IPOINT_BEFORE, (AFUNPTR)Arg1Before, IARG_ADDRINT, RRAM_FREE, IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                       IARG_END);
        RTN_Close(freeRtn);
    }
}

/* ===================================================================== */

VOID Fini(INT32 code, VOID* v) { TraceFile.close(); }

/* ===================================================================== */
/* Print Help Message                                                    */
/* ===================================================================== */

INT32 Usage()
{
    cerr << "This tool produces a trace of calls to rram_malloc." << endl;
    cerr << endl << KNOB_BASE::StringKnobSummary() << endl;
    return -1;
}

/* ===================================================================== */
/* Main                                                                  */
/* ===================================================================== */

int main(int argc, char* argv[])
{
    // Initialize pin & symbol manager
    PIN_InitSymbols();
    if (PIN_Init(argc, argv))
    {
        return Usage();
    }

    // Write to a file since cout and cerr maybe closed by the application
    TraceFile.open(KnobOutputFile.Value().c_str());
    TraceFile << hex;
    TraceFile.setf(ios::showbase);

    // Register Image to be called to instrument functions.
    IMG_AddInstrumentFunction(Image, 0);

    INS_AddInstrumentFunction(EmulateLoad, 0);
    PIN_AddFiniFunction(Fini, 0);

    // Never returns
    PIN_StartProgram();

    return 0;
}

/* ===================================================================== */
/* eof */
/* ===================================================================== */
