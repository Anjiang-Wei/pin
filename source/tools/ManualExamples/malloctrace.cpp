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
using std::map;

#define RRAM_MALLOC "rram_malloc"
string RRAM_MALLOC_mangled = "";
// #define RRAM_FREE "rram_free"
// string RRAM_FREE_mangled = "";

/* ===================================================================== */
/* Global Variables */
/* ===================================================================== */
VOID MallocBefore(CHAR* name, ADDRINT size);
VOID MallocAfter(ADDRINT ret);
bool isRRAM_addr(ADDRINT addr);
void store_data_reg(UINT8* dataptr, UINT8* addrptr, UINT32 size);
void load_data_reg(UINT8* dataptr, UINT8* addrptr, UINT32 size);
void store_data_bit(UINT8* dataptr, UINT8* addrptr, UINT32 size);
void load_data_bit(UINT8* dataptr, UINT8* addrptr, UINT32 size);
std::ofstream TraceFile;
KNOB< string > KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool", "o", "1.out", "specify trace file name");

/*---------------------------------------------------------------------- */

pair<ADDRINT, ADDRINT> last_malloc_record; // address, size
vector<pair<ADDRINT, ADDRINT> > all_malloc_record;
map<ADDRINT, UINT8> regmem; // regular memory, mapping from memory address to byte
map<ADDRINT, bool> bitmem; // bit-level memory, mapping from bit-level address (reg address * 8) to bit

void store_data_reg(UINT8* dataptr, UINT8* addrptr, UINT32 size)
{
    ADDRINT addr = (ADDRINT) addrptr;
    for (UINT32 i = 0; i < size; i++)
    {
        regmem[addr + i] = dataptr[i];
    }
}

void load_data_reg(UINT8* dataptr, UINT8* addrptr, UINT32 size)
{
    ADDRINT addr = (ADDRINT) addrptr;
    for (UINT32 i = 0; i < size; i++)
    {
        dataptr[i] = regmem[addr + i];
    }
}

void store_data_bit(UINT8* dataptr, UINT8* addrptr, UINT32 size)
{
    ADDRINT addr = (ADDRINT) addrptr;
    for (UINT32 i = 0; i < size; i++)
    {
        for (UINT32 j = 0; j < 8; j++)
        {
            bitmem[((addr + i) << 3) + j] = dataptr[i] & (1 << j);
        }
    }
}

void load_data_bit(UINT8* dataptr, UINT8* addrptr, UINT32 size)
{
    ADDRINT addr = ((ADDRINT) addrptr);
    for (UINT32 i = 0; i < size; i++)
    {
        dataptr[i] = 0;
        for (UINT32 j = 0; j < 8; j++)
        {
            dataptr[i] |= (bitmem[((addr + i) << 3) + j] & 1) << j;
        }
    }
}

VOID PIN_FAST_ANALYSIS_CALL MyLoad(ADDRINT* addrptr, UINT32 size)
{
	ADDRINT addr = (ADDRINT) addrptr;
    if (isRRAM_addr(addr))
    {
        UINT64 data;
	    PIN_SafeCopy(&data, (void *) addr, size);
        load_data_bit(reinterpret_cast<UINT8*>(&data), reinterpret_cast<UINT8*>(addrptr), size);
        TraceFile << "myload " << addrptr << " size " << size << " data " << data << std::endl;
        PIN_SafeCopy((void *) addr, &data, size);
    }
}

VOID PIN_FAST_ANALYSIS_CALL MyStore(ADDRINT * addrptr, UINT32 size)
{
    ADDRINT addr = (ADDRINT) addrptr;
	if (isRRAM_addr(addr))
    {
        UINT64 data;
	    PIN_SafeCopy(&data, (void *) addr, size);
        store_data_bit(reinterpret_cast<UINT8*>(&data), reinterpret_cast<UINT8*>(addrptr), size);
        TraceFile << "mystore " << addrptr << " size " << size << " data " << data << std::endl;
        PIN_SafeCopy((void *) addr, &data, size);
    }
}

VOID MallocBefore(CHAR* name, ADDRINT size)
{
    last_malloc_record.second = size;
}

VOID MallocAfter(ADDRINT ret)
{
    last_malloc_record.first = ret;
    TraceFile << last_malloc_record.first << " " << last_malloc_record.second << std::endl;
    all_malloc_record.push_back(last_malloc_record);
}

bool isRRAM_addr(ADDRINT addr)
{
    for (auto item:all_malloc_record)
    {
        if (item.first <= addr && addr < item.first + item.second)
        {
            return true;
        }
    }
    return false;
}

VOID InstrumentNormalInstruction(INS ins, VOID* v){
    UINT32 memOperands = INS_MemoryOperandCount(ins);
    // Iterate over each memory operand of the instruction.
    for (UINT32 memOp = 0; memOp < memOperands; memOp++)
    {
		UINT8 memOpSize = INS_MemoryOperandSize(ins, memOp);
		//found a memory read instance
        if (INS_MemoryOperandIsRead(ins, memOp))
        {
			INS_InsertPredicatedCall(
				ins, IPOINT_BEFORE, (AFUNPTR) MyLoad,
				IARG_MEMORYOP_EA, memOp,
				IARG_UINT32, memOpSize,
				IARG_END);
        }
        // Note that in some architectures a single memory operand can be 
        // both read and written (for instance incl (%eax) on IA-32)
        // In that case we instrument it once for read and once for write.
        if (INS_MemoryOperandIsWritten(ins, memOp) && INS_HasFallThrough(ins))
        {
           INS_InsertPredicatedCall(
                ins, IPOINT_AFTER, (AFUNPTR) MyStore,
                IARG_MEMORYOP_EA, memOp,
                IARG_UINT32, memOpSize,
                IARG_END);
        }
    }
}

void find_mangled_symbols(IMG img)
{
    for (SYM sym = IMG_RegsymHead(img); SYM_Valid(sym); sym = SYM_Next(sym))
    {
        string s1 = SYM_Name(sym);
        if (s1.find(RRAM_MALLOC) != std::string::npos && s1.find("GLOBAL") == std::string::npos)
        {
            RRAM_MALLOC_mangled = s1;
        }
        // if (s1.find(RRAM_FREE) != std::string::npos && s1.find("GLOBAL") == std::string::npos)
        // {
        //     RRAM_FREE_mangled = s1;
        // }
    }
    // std::cout << RRAM_MALLOC_mangled << std::endl;
    // std::cout << RRAM_FREE_mangled << std::endl;
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
        RTN_InsertCall(mallocRtn, IPOINT_BEFORE, (AFUNPTR)MallocBefore, IARG_ADDRINT, RRAM_MALLOC, IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                       IARG_END);
        RTN_InsertCall(mallocRtn, IPOINT_AFTER, (AFUNPTR)MallocAfter, IARG_FUNCRET_EXITPOINT_VALUE, IARG_END);
        RTN_Close(mallocRtn);
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

    IMG_AddInstrumentFunction(Image, 0);

    INS_AddInstrumentFunction(InstrumentNormalInstruction, 0);
    PIN_AddFiniFunction(Fini, 0);

    // Never returns
    PIN_StartProgram();

    return 0;
}

/* ===================================================================== */
/* eof */
/* ===================================================================== */
