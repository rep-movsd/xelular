#include "pch.h"
#include "xel.h"

// Global variables
#ifndef REGION_GLOBALS_

// Compile time constant to avoid ugly #ifdef in teh midst of code
#ifdef __CUDACC__
bool g_bOnCUDA = true;
#else
bool g_bOnCUDA = false;
#endif

int nGens = 0;

// The array of xels in the GPU memory
TXel2D *g_pXel2D;
bool TGLGroup::bOpen;

// The central point of the arena
TFXY g_ptCenter;

// Length of the major and minor diameters of the xel hexagons
__constant__ float g_fMajor, g_fMinor;

// Pointer to the XEL data on VRAM
__constant__ TPitchPtr g_pdXels;

// Pointer to the image force data on each pixel
__constant__ TPitchPtr g_pdImgForce;

// Pointer to the occupied list ( there are 10 groups of xels called "spreads")
__constant__ TPitchPtr g_pdOcc;

// Count of occupied xels in each "spread"
__device__ int g_pdCounts[10];

// Complete list of all occupied xels, used to regenerate g_pdOcc after every 
// generation
__constant__ TXY *g_pdOccAll;

__global__ void kUpdate();
__global__ void kDivideOccupiedSets();
__global__ void kClearOccupiedArr();
__device__  void kEvolve(TMany<6, TXel> &nears, TXel &xel);

// Most of our kernels work on an individual "spread" of xels
// A spread of xels is each of 9 sets of xels which do not have a commmon 
// neighbor

// The following macros are defined to help us call our kernels with a constant 
// parameter (spread) , to avoid extra symbol passing between CPU and GPU
// It is similar to a loop unrolling macro

#define LAUNCH_9(X) \
    KERNEL_1D_GRID_1D(X##0, nBlocks[0], nThreads[0])(); \
    KERNEL_1D_GRID_1D(X##1, nBlocks[1], nThreads[1])(); \
    KERNEL_1D_GRID_1D(X##2, nBlocks[2], nThreads[2])(); \
    KERNEL_1D_GRID_1D(X##3, nBlocks[3], nThreads[3])(); \
    KERNEL_1D_GRID_1D(X##4, nBlocks[4], nThreads[4])(); \
    KERNEL_1D_GRID_1D(X##5, nBlocks[5], nThreads[5])(); \
    KERNEL_1D_GRID_1D(X##6, nBlocks[6], nThreads[6])(); \
    KERNEL_1D_GRID_1D(X##7, nBlocks[7], nThreads[7])(); \
    KERNEL_1D_GRID_1D(X##8, nBlocks[8], nThreads[8])();

#define DECL_9(X) \
    X(0);\
    X(1);\
    X(2);\
    X(3);\
    X(4);\
    X(5);\
    X(6);\
    X(7);\
    X(8);

__device__ void kMovement(int i);
__device__ void kInterpolate(int i);
__device__ void kMergeOccupiedSets(int i);

#define DECL_MOVEMENT_KERNEL(X)         __global__ void kMovement##X() {kMovement(X);}
#define DECL_INTERPOLATE_KERNEL(X)      __global__ void kInterpolate##X() {kInterpolate(X);}
#define DECL_MERGE_KERNEL(X)            __global__ void kMergeOccupiedSets##X() {kMergeOccupiedSets(X);}
#define DECL_MIGRATE_OCC_KERNEL(X)      __global__ void kMigratePointsOcc##X() {kMigratePoints(X, true);}

DECL_9(DECL_MOVEMENT_KERNEL);
DECL_9(DECL_INTERPOLATE_KERNEL);
DECL_9(DECL_MERGE_KERNEL);

// These two source files contain kernels for xel movement and 
// xel initialization / image force
#include "xelmovement.cc"
#include "xelimg.cc"

#endif 

//////////////////////////////////////////////////////////////////////////
// Profiler class
// To use, declare an object of this type at the top of any block, passing a
// name to identify this particular block.
// The object will log the time it stays alive (until the block scope)
// The elapsed time and count is added to a map under the name specified
// The macro PROFILE_IT logs under the current function name, use at the 
// top of any function to be profiled.
// PROFILE_BLOCK is to be used for anonymous blocks, specify a string 
// identifier for the block
// At the end of the run the stats are dumped to the specified file or stderr
// Dumped files can be loaded at startup to accumulate stats over runs
//////////////////////////////////////////////////////////////////////////

#ifndef REGION_PROFILER_
#define DO_PROFILE 1
#ifdef DO_PROFILE
#define PROFILE_IT TProfile _profile(__FUNCTION__)
#define PROFILE_BLOCK(X) TProfile _profileblock(X);

class TProfile
{
    // The file in which results are dumped
    static string m_sFile;

    // counts and cumulative time for each profiled block/function
    static map<string, int> ms_nFnCounts;
    static map<string, int> ms_nFnTimes;

    // Timestamp 
    clock_t m_nStartTick;

    // Block ID string
    const char* m_pszFnName;

public:
    // In the constructor, increment the hit count and save the timestamp
    TProfile(const char *pszFnName): m_pszFnName(pszFnName)
    {
        ms_nFnCounts[m_pszFnName]++;
        m_nStartTick = clock();
    }
    //////////////////////////////////////////////////////////////////////////

    // In the destructor, increment the cumulative time for this ID
    ~TProfile()
    {
        ms_nFnTimes[m_pszFnName] += (clock() - m_nStartTick);
    }
    //////////////////////////////////////////////////////////////////////////

    // Dumps stats to the file
    static void dump()
    {
        map<string, int>::iterator ci = ms_nFnCounts.begin();
        map<string, int>::iterator ti = ms_nFnTimes.begin();

        if(!m_sFile.empty())
        {
            ofstream ofs(m_sFile.c_str());
            ofs << "Function Name\tCall count\tTotal Function Time\tAverage function time" << endl;
            for(; ti != ms_nFnTimes.end(); ++ti, ++ci )
            { 
                // if(ci->second > 1 && ti->second)
                ofs << ti->first << "\t" << ci->second << "\t" << ti->second << "\t" << (double(ti->second) / ci->second) << endl;
            }
        }
        else
        {
            cout << CLOCKS_PER_SEC << " clocks per second" << endl;
            for(; ti != ms_nFnTimes.end(); ++ti, ++ci )
            {
                int nCalls = ci->second;
                double fTotal = (double)ti->second / (CLOCKS_PER_SEC/1000);
                cout << setw(24) << ti->first << " " << setw(12) << nCalls << " " << setw(12) << fTotal << "ms " << setw(12) << (fTotal / nCalls) << "ms " << endl;
            }
        }

    }
    //////////////////////////////////////////////////////////////////////////

    // Loads statistics from a previously dumped file, to allow multiple runs 
    // to be timed and coalesced
    static void load(const char *pszFile)
    {
        m_sFile = pszFile;
        ifstream ifs(pszFile);

        if(ifs.is_open())
        {
            string s, sName;
            int iCount, iTime;
            double dummy;
            getline(ifs, s);
            while(getline(ifs, s))
            {
                istringstream iss(s);
                iss >> sName;
                iss >> iCount;
                iss >> iTime;
                iss >> dummy;

                ms_nFnCounts[sName] = iCount;
                ms_nFnTimes[sName] = iTime;
            }
        }
    }
    //////////////////////////////////////////////////////////////////////////
};
//////////////////////////////////////////////////////////////////////////

// Helper class to dump results automatically on program shutdown
class TProfileHelper
{
public:
    ~TProfileHelper() 
    { 
        TProfile::dump(); 
    }
};
//////////////////////////////////////////////////////////////////////////

// Statics for TProfile
map<string, int> TProfile::ms_nFnCounts;
map<string, int> TProfile::ms_nFnTimes;
string TProfile::m_sFile;
TProfileHelper p;


#else
#define PROFILE_IT ;
#define PROFILE_BLOCK(X) X;
#endif // DO_PROFILE
#endif 

// Main program
#ifndef REGION_MAIN_

#ifdef USEGL
//////////////////////////////////////////////////////////////////////////
// Open GL handling functions
// All of them call the main class, whose pointer is in g_pXel2D
//////////////////////////////////////////////////////////////////////////

void oglDraw()
{
    g_pXel2D->oglDraw();
}
//////////////////////////////////////////////////////////////////////////

void oglMouse(int button, int state, int x, int y)
{
    g_pXel2D->oglMouse(button, state, x, y);
}
//////////////////////////////////////////////////////////////////////////

void oglMouseMove(int x, int y)
{
    g_pXel2D->oglMouseMove(x, y);
}
//////////////////////////////////////////////////////////////////////////

void oglKeyboard(byte ch, int x, int y)
{
    g_pXel2D->oglKeyboard(ch, x, y);
}
//////////////////////////////////////////////////////////////////////////

void exitOpenGL()
{
    glutDisplayFunc(0);
    glutMouseFunc(0);
    glutPassiveMotionFunc(0);
}
//////////////////////////////////////////////////////////////////////////

// Calculates certain dimensions required for calculations and display
void calcWindowMetrics(TXel2D &xels, int w, int h)
{
    // A xel width is twice the minor radius
    xels.m_fXelWidth = xels.m_fMinor * 2.0f;

    // A row of xels has height 1.5 times the major radius
    xels.m_fXelHeight = xels.m_fMajor * 1.5f;

    // We keep 1/4 xel width padding on the left and right + 300px for the help text
    xels.m_fPadL = xels.m_fMinor / 2.0f;
    xels.m_fPadR = xels.m_fMinor;
    xels.m_fTxtWidth = 300;

    xels.m_fWidth = xels.m_fXelWidth * w;

    xels.m_fPadV = xels.m_fMajor * 0.5f;
    xels.m_fHeight = xels.m_fXelHeight * h + (h%2 ? 0 : xels.m_fMajor) ;

    xels.m_iLeft = xels.m_fPadL;
    xels.m_iTop = xels.m_fPadV;
    xels.m_iRight = xels.m_fWidth + xels.m_fPadL + xels.m_fPadR;
    xels.m_iBottom = xels.m_fHeight + xels.m_fPadV;
}

// Initialize the glut window
void initOpenGL(TXel2D &xels, int w, int h) 
{
    // set RGBA mode with float and depth buffers
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(xels.m_fWidth + xels.m_fPadR + xels.m_fPadL + 
        xels.m_fTxtWidth, xels.m_fHeight + xels.m_fPadV * 2);
    glutCreateWindow("Xels 2d");

    glutSetCursor(GLUT_CURSOR_FULL_CROSSHAIR);
    gluOrtho2D(-xels.m_fPadL, xels.m_fPadR + xels.m_fWidth + xels.m_fTxtWidth, 
        xels.m_fHeight + xels.m_fPadV, -xels.m_fPadV);

    // Enable smoothing and blending
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Set handlers
    glutDisplayFunc(oglDraw);
    glutMouseFunc(oglMouse);
    glutKeyboardFunc(oglKeyboard);
    glutPassiveMotionFunc(oglMouseMove);
}
//////////////////////////////////////////////////////////////////////////
#endif // USEGL
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// main program functions
//////////////////////////////////////////////////////////////////////////


// Parse command line args
void parseArgs(int argc, char ** argv, TArgs &args) 
{
    // Get any command line options
    FOR(i, argc)
    {
        // every  option starts with one letter followed by an '='
        string s(argv[i]);
        if(s.length() > 2 && s[1] == '=')
        {
            istringstream iss(s.substr(2));

            switch(tolower(s[0]))
            {
                case 'i':
                    iss >> args.sInFile;
                break;

                case 'o':
                    iss >> args.sOutFile;
                break;

                // d=dimension
                case 'd':
                    iss >> args.w;
                    args.h = args.w;
                break;

                // n=cellsize
                case 'n':
                    iss >> args.nEdgePels;
                break;

                // Image xel threshold
                case 't':
                    iss >> args.fImgThreshold;
                break;

                // No of times to convolve
                case 'c':
                    iss >> args.nBlurCount;
                break;

                // generations to run in batch mode
                case 'g':
                    iss >> args.nGens;
                break;

                // Profile file
                case 'p':
                {
                    string s;
                    iss >> s;
                    TProfile::load(s.c_str());
                    break;
                }

                // Load BMP file for Xel pressure
                case 'b':
                    iss >> args.sImageFile;
                break;

                case 'h':
                    iss >> args.sHoleImageFile;
                break;

                // repeating for benchmarking
                case 'r':
                    iss >> args.nReps;
                break;
            }
        }
    }
}
//////////////////////////////////////////////////////////////////////////

// Initializes xels from image, xel file or empty
void initXels(TXel2D &xels, const TArgs &args)
{
    if(args.sImageFile.empty())
    {
        if(!args.sInFile.empty())   // Load from file
        {
            cerr << "Loading xel points file : " << args.sInFile << endl;
            xels.load(args.sInFile);
        }
        else                        // Init empty xels only if no input image is provided
        {
            xels.init(args.w, args.h, args.nEdgePels);
        }
    }
    else // Load the image and init xels
    {
        cerr << "Loading image file : " << args.sImageFile << endl;
        if(!args.sHoleImageFile.empty())
            cerr << "Loading holes file : " << args.sHoleImageFile << endl;
        xels.loadImage(args);
    }

    cerr << xels.dims().width << " x " << xels.dims().height << " grid" << endl;

    xels.copyXelDataToDevice();
}
//////////////////////////////////////////////////////////////////////////

// Launches the openGL display
void runDisplayMode(int argc, char **argv, TXel2D &xels, TArgs &args)
{
#ifdef USEGL
    glutInit(&argc, argv);
    g_pXel2D = &xels;
    initOpenGL(xels, xels.dims().width, xels.dims().height);
    glutMainLoop();
    exitOpenGL();
#else

    cerr << "OpenGL display disabled" << endl;

#endif // USEGL
}

/////////////////////////////////////////////////////////////////////////

// Runs a certain number of generations and save results to file
void runBatchMode(TXel2D &xels, const TArgs &args)
{
    cerr << "Evolving : " << args.nGens << " generations, " << args.nReps << " repetitions" << endl;

    clock_t start = clock();
    int j = 0;
    do
    {
        FOR(i, args.nGens)
        {
            nGens = i;
            xels.evolve(); 
        }

        xels.copyXelDataToHost();
        j++;

        if(j < args.nReps)    // reinitialize
            initXels(xels, args);

    }
    while(j < args.nReps);

    clock_t stop = clock();

    cerr << endl << "Done in : " << (stop - start) << " clock ticks" << endl;
    cerr << "Saving file : " << args.sOutFile << endl;
    xels.save(args.sOutFile);
}
//////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        TXel2D xels;
        TArgs args;
        parseArgs(argc, argv, args);
        initXels(xels, args);

        if(args.sOutFile.empty())
        {
            runDisplayMode(argc, argv, xels, args);
        }
        else
        {
            runBatchMode(xels, args);
        }
    }
    catch(string &s)
    {
        cerr << "Fatal error : " << s << endl;
    }
    catch(TCudaException &e)
    {
        cerr << e.message() << endl;
    }

    catch(...)
    {
        cerr << "OS exception" << endl;
    }

    return 0;
}
#endif 

// Utility functions
#ifndef REGION_UTILS_

// Gets the 6 neighboring XEL coordinates
GPU_CPU TXY getNearXels(TXY& pt, int i)
{
    int iIsOdd = pt.y % 2;
    int iIsEven = 1 - iIsOdd;
    TXY ret;

    switch(i)
    {
    case 0 : ret.x = pt.x + 1;         ret.y = pt.y;      break;
    case 1 : ret.x = pt.x + iIsOdd;    ret.y = pt.y + 1;  break;
    case 2 : ret.x = pt.x - iIsEven;   ret.y = pt.y + 1;  break;
    case 3 : ret.x = pt.x - 1;         ret.y = pt.y;      break;
    case 4 : ret.x = pt.x - iIsEven;   ret.y = pt.y - 1;  break;
    case 5 : ret.x = pt.x + iIsOdd;    ret.y = pt.y - 1;  break;
    default : ret = pt;
    }

    return ret;
}
//////////////////////////////////////////////////////////////////////////

// Get neighbours upto 2 xels away
// GPU_CPU TXY getNearXels2(TXY& pt, int i)
// {
//     int iIsOdd = pt.y % 2;
//     int iIsEven = 1 - iIsOdd;
//     TXY ret;
// 
//     switch(i)
//     {
//     case 0 : ret.x = pt.x + 1;         ret.y = pt.y;      break;
//     case 1 : ret.x = pt.x + iIsOdd;    ret.y = pt.y + 1;  break;
//     case 2 : ret.x = pt.x - iIsEven;   ret.y = pt.y + 1;  break;
//     case 3 : ret.x = pt.x - 1;         ret.y = pt.y;      break;
//     case 4 : ret.x = pt.x - iIsEven;   ret.y = pt.y - 1;  break;
//     case 5 : ret.x = pt.x + iIsOdd;    ret.y = pt.y - 1;  break;
//     
//     case 6 : ret.x = pt.x + 2;          ret.y = pt.y;      break;   
//     case 7 : ret.x = pt.x + iIsOdd + 1; ret.y = pt.y + 1;  break;
//     case 8 : ret.x = pt.x - iIsEven;
//     case 9 : 
// 
// 
//     }
// 
//     return ret;
// }
//////////////////////////////////////////////////////////////////////////


GPU_CPU TXY getNearXelsSafe(TXY& pt, int i, int minx, int miny, int maxx, int maxy)
{
    TXY ret = getNearXels(pt, i);

    if(ret.x < minx) ret.x = minx;
    if(ret.x > maxx) ret.x = maxx;

    if(ret.y < miny) ret.y = miny;
    if(ret.y > maxy) ret.y = maxy;

    return ret;
}
//////////////////////////////////////////////////////////////////////////

// gets the i, j, k floating point co ordinates for the given point and xel size
GPU_CPU void storeIJK(TFXY ptf, float fMin, float fMaj, float &i, float &j, float &k)
{
    // Scale by minor radius
    ptf.op<opdiv>(fMin, fMaj);

    i = ptf.x;
    j = ptf.x / 2.0f + ptf.y - 0.5f;
    k = -(ptf.x / 2.0f - ptf.y - 0.5f);
}
//////////////////////////////////////////////////////////////////////////

// gets the box co ordinate p, q given the i, j, k
GPU_CPU void storePQ(float i, float j, float k, int &p, int &q)
{
    float ii = floor(i), jj = floor(j), kk = floor(k);
    q = (int)floor((jj + kk) / 3.0f);  
    p = (int)floor((ii - (q % 2)) / 2.0f); 
}
//////////////////////////////////////////////////////////////////////////

// Gets the row and column of the xel within which the point x, y lies
GPU_CPU  TXY getPQ(const TFXY &ptf, float fMin, float fMaj)
{
    TXY ptRet;
    float i, j, k;
    storeIJK(ptf, fMin, fMaj, i, j, k);
    storePQ(i, j, k, ptRet.x, ptRet.y);
    return ptRet;
}
/////////////////////////////////////////////////////////////////////////

// Gets the center point of a xel
GPU_CPU TFXY getXelCenterXY(TXY &pt, float fMin, float fMaj)
{
    TFXY ptfRet;
    ptfRet.x = (pt.x * 2.0f + pt.y % 2) * fMin;
    ptfRet.y = pt.y * 1.5f * fMaj;
    ptfRet.op<opadd>(fMin, fMaj);
    return ptfRet;
}
//////////////////////////////////////////////////////////////////////////
#endif

// Xel2D class
#ifndef REGION_XEL2D_

TXel2D::TXel2D() : m_bEvolving(), m_bShowImage(), m_bShowPoints(true), 
m_bShowForces(), m_bShowXels(), m_bShowGVF()
{
}
//////////////////////////////////////////////////////////////////////////

template<class T> void copyArrayPtrToDevConst(const T &data, T &devConstPtr)
{
     g_e = cudaMemcpyToSymbol((const char*)&devConstPtr, &data, sizeof(T), 0, cudaMemcpyHostToDevice);
}
//////////////////////////////////////////////////////////////////////////


// constructor takes width, height and xel major radius
void TXel2D::init(int w, int h, int nEdgePels)
{
    PROFILE_IT;
    g_e.setContext(__FUNCTION__);

    m_nGens = 0;
    m_fMajor = float(nEdgePels);
    m_fMinor = m_fMajor * SIN60;

    calcWindowMetrics(*this, w, h);

    // Xels data on CPU and GPU
    m_CPUXels.setDims(w, h);
    m_GPUXels.setDims(w, h);

    // Occupied points divided into 9 groups
    m_CPUOcc.setDims(w * h, 9);
    m_GPUOcc.setDims(w * h, 9);

    // Set of all occupied points
    m_GPUOccAll.setDims(w * h, 1);

    // Image forces and gradient vector flow force (one vector for each x, y pixel)
    if(m_Img.width() > 1)
    {
        m_GPUImgForce.setDims(m_Img.width(), m_Img.height());
    }
    else
    {
        m_GPUImgForce.setDims(m_iRight - m_iLeft, m_iBottom - m_iTop);
    }

    // set of points in the 9 groups maintained on CPU as a std::set
    FOR(i, 9) 
        m_PointSet[i].clear();


    g_e = cudaMemcpyToSymbol((const char*)&g_fMinor, &m_fMinor, sizeof(float), 0, cudaMemcpyHostToDevice);
    g_e = cudaMemcpyToSymbol((const char*)&g_fMajor, &m_fMajor, sizeof(float), 0, cudaMemcpyHostToDevice);
    
    m_pArr = T2DView<TXel>(m_CPUXels.data());

    if(g_bOnCUDA)
    {
        copyArrayPtrToDevConst(m_GPUXels.data(), g_pdXels);

        TPitchPtr pdOcc = m_GPUOcc.data();
        g_e = cudaMemcpyToSymbol((const char*)&g_pdOcc, &pdOcc, sizeof(TPitchPtr), 0, cudaMemcpyHostToDevice);

        TXY *pdOccAll = m_GPUOccAll.ptr();
        g_e = cudaMemcpyToSymbol((const char*)&g_pdOccAll, &pdOccAll, sizeof(TXY*), 0, cudaMemcpyHostToDevice);
    }
    else
    {
        g_pdXels = m_CPUXels.data();
        g_pdOcc = m_GPUOcc.data();
        g_pdOccAll = m_GPUOccAll.ptr();
        g_pdImgForce = m_GPUImgForce.data();
    }

    TXY ptTmp(w, h);
    g_ptCenter = getXelCenterXY(ptTmp, g_fMinor, g_fMajor) / 2;
}
//////////////////////////////////////////////////////////////////////////

void TXel2D::loadImage(const TArgs &args)
{
    PROFILE_IT;

    m_sImgName = args.sImageFile.substr(0, args.sImageFile.find_last_of('.'));
    m_Img.load(args.sImageFile.c_str(), true);
    if(!args.sHoleImageFile.empty())
        m_ImgHole.load(args.sHoleImageFile.c_str(), true);

    m_nBlurCount = args.nBlurCount;
    m_fImgThreshold = args.fImgThreshold;

    // calculate hex grid size and initialize
    TFXY ptfImgSize(m_Img.width(), m_Img.height());

    float fMajor = float(args.nEdgePels);
    float fMinor = fMajor * SIN60;

    TXY ptMax = getPQ(ptfImgSize, fMinor, fMajor) + TXY(2, 2);
    init(ptMax.x, ptMax.y, args.nEdgePels);

    TXelImg xi(*this);
    xi.loadImage();
}
//////////////////////////////////////////////////////////////////////////

// Load xel data from a file
void TXel2D::load(const string &sFile)
{
    PROFILE_IT;
    ifstream ifs(sFile.c_str());
    string sLine;

    if(!ifs.is_open())
        throw string("Cannot open file '") + sFile + string("'");

    // Read dimensions and xel display size
    int w, h, n;
    getline(ifs, sLine);
    istringstream iss(sLine);
    iss >> w;
    iss >> h;
    iss >> n;

    // Resize the array, then fill the co ord values
    init(w, h, n);

    // Read data
    n = 0;
    while(getline(ifs, sLine))  // Read each line, containing state, x, y
    {
        // read an entry
        istringstream iss(sLine);
        TFXY ptf;
        int state;
        iss >> state >> ptf.x >> ptf.y;
        setPoint(state, ptf);

        if(n++ == (w * h)) 
        {
            cerr << "Warning : More points than xels!!!" << endl;
        }
    }
}
//////////////////////////////////////////////////////////////////////////

// Save data to a file
void TXel2D::save(const string &sFile)
{
    PROFILE_IT;
    ofstream ofs(sFile.c_str());

    // write dimensions and xel display size
    ofs << m_CPUXels.width() << " " << m_CPUXels.height() << " " << int(m_fMajor) << endl;

    FOR(n, 9)
    {
        FOR_EACH(set<TXY>, i, m_PointSet[n])
        {
            TXel &xel = m_pArr[*i];
            ofs << xel.state0 << " " << xel.pt0.x << " " << xel.pt0.y << endl;
        }
    }
}
//////////////////////////////////////////////////////////////////////////

//Evolve xel points to new positions
void TXel2D::evolve()
{
    PROFILE_IT;
    ++m_nGens;

    int n = nGens / 10;
    if(n > 9) n = 9;

//    m_nBlurIndex = 9 - n;

    int nThreads[9], nBlocks[9], piCounts[10], nTotalOccupied = 0;

    {
        PROFILE_BLOCK("copyCounts()");
        cudaMemcpyFromSymbol(piCounts, (const char*)g_pdCounts, sizeof(int) * 10, 0, cudaMemcpyDeviceToHost);
    }

    // Get block and thread counts
    FOR(i, 9)
    {
        getBlockGridSizes(piCounts[i], nBlocks[i], nThreads[i]);
        nTotalOccupied += piCounts[i];
    }

    // Movement
    doMovement(nBlocks, nThreads);

    // Post movement interpolation
    interpolate(nBlocks, nThreads);

    // merge all the 9 occupied sets into m_pdOccAll, adding every xel and its new interpolated points
    mergeOccupiedSets(nBlocks, nThreads);

    // Need to calculate nTotalOccupied again
    nTotalOccupied = 0;
    {
        PROFILE_BLOCK("copyCounts()");
        cudaMemcpyFromSymbol(piCounts, (const char*)g_pdCounts, sizeof(int) * 10, 0, cudaMemcpyDeviceToHost);
        nTotalOccupied = piCounts[9];
    }

    // divide the m_pdOccAll xel co ords into the the 9 sets m_pOcc
    divideOccupiedSets(nTotalOccupied);

    // Update XEL states on GPU
    updateXelStates(nTotalOccupied);
}
//////////////////////////////////////////////////////////////////////////

#ifdef USEGL

//////////////////////////////////////////////////////////////////////////
// opengl callbacks
//////////////////////////////////////////////////////////////////////////

void TXel2D::oglDraw()
{
    // Draw the hexagons (2 sides of the hexagon are paralell to Y axis )
    // hexY[0], hexY[0] is the vertex at 8 o' clock
    static float f = m_fMajor / 2.0f;
    static float hexX[6] = {0, 0, m_fMinor, m_fMinor * 2.0f, m_fMinor * 2.0f, m_fMinor};
    static float hexY[6] = {m_fMajor - f, m_fMajor + f, m_fMajor * 2.0f, m_fMajor + f, m_fMajor - f, 0};
    float ex, ey; // represents the coordinate with the lowest x, y along the edge of the xels

    // Wipe the canvas
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    //T2DView<TFXY> img(m_bShowGVF ? m_GPUImgGVFForce.data() : m_GPUImgForce.data());
    T2DView<TFXY> img(m_GPUImgForce.data());

    if(!m_bEvolving)
    {
        glPolygonMode(GL_FRONT, GL_FILL);
        LOOP_YX(m_CPUXels.width(), m_CPUXels.height())
        {
            ex = (x * 2.0f + y % 2) * m_fMinor;
            ey = y * 1.5f * m_fMajor;

            TXel &xel = m_pArr[y][x];

            //glLineWidth(xel.state ? 2 : 0.6);
            if(m_bShowImage)
            {
                glPolygonMode(GL_FRONT, GL_FILL);
                TGLGroup gl1(GL_POLYGON, 1, 1, 1, xel.fIntensity * 0.5);
                FOR(i, 6)
                    glVertex2f(ex + hexX[i], ey + hexY[i]);
            }
            else //if(xel.state0)
            {
                //glPolygonMode(GL_FRONT , xel.state0 ? GL_FILL : GL_LINE);
                glPolygonMode(GL_FRONT , GL_LINE);
                TGLGroup gl(GL_POLYGON, xel.state0 ? 0 : 1, 1, 0, xel.state0 ? 0.4 : 0.2);

                FOR(i, 6)
                    glVertex2f(ex + hexX[i], ey + hexY[i]);
            }

            if(m_bShowForces)
            {
                if(ex < m_Img.width() && ey < m_Img.height() && ex > 0 && ey > 0)
                {
                    TFXY pt(ex, ey), pt1 = pt;
                    float fr = 0.1;
                    
                    FOR(i, 20)
                    {
                        if(pt.x >= m_Img.width() || pt.x < 0) break;
                        if(pt.y >= m_Img.height() || pt.y < 0) break;

                        TGLGroup gl1(GL_LINES, 1.0 - fr, fr, 0, 0.75);
                        gl1.vertex(pt);
                        pt1 = pt + img[pt];

                        gl1.vertex(pt1);
                        pt = pt1;
                        fr += (0.9 / 20);
                    }
                }
            }
        }
    }
    else
    {
        glColor3f(1, 1, 1);
        TGLText t(m_iRight + 10, 10, 15);
        char s[256];
        sprintf(s, "Generation: %d", m_nGens);
        t.writeLn(s);
    }

    // Draw the points with 4 pixels size
    if(m_bShowPoints)
    {
        LOOP_YX(m_CPUXels.width(), m_CPUXels.height())
        {
            TXel &xel = m_pArr[y][x];
            glPointSize(3);
            if(xel.state0 != xsEmpty)
            {
                int n = xel.iChain;
                if(n == -1)
                {
                    n = 1;
                    FOR(j, 6)
                    {
                        TXY &pt = getNearXels(TXY(x, y), j);
                        if(m_pArr[pt].state0 != xsEmpty)
                            n++;
                    }
                }
                else
                {
                    if((n % 8) == 0)
                        n++;
                }

                TGLGroup gl(GL_POINTS, (n & 4) >> 2, (n & 2 ) >> 1, n & 1);
                gl.vertex(xel.pt0);
            }
        }
    }

    if(!m_bEvolving)
    {
        int x, y;
        {

            TGLGroup g(GL_LINES, 1, 1, 1);
            glVertex2i(m_iRight, 0);
            glVertex2i(m_iRight, m_iBottom);
        }

        {
            glColor3f(1, 1, 1);
            TGLText t(m_iRight + 10, 10, 15);

            t.writeLn("Keyboard interface");
            t.writeLn("------------------");
            t.writeLn("E : evolve 1 gen");
            t.writeLn("0-9 : evolve 2^N gens");
            t.writeLn("F : Show/Hide image data");
            t.writeLn("P : Show/Hide xel points");
            t.writeLn("V : Show/Hide GVF data");
            t.writeLn("I : Show/Hide img force");
            t.writeLn("S : Save points to xel.dat");
            t.writeLn("L : Load points from xel.dat");
            t.writeLn("C : Detect chains");
            t.writeLn("Z : Save chains to xelchains.txt");
            t.writeLn("X : Load chains from xelchains_in.txt");

            t.writeLn();
            t.writeLn();

            t.writeLn("Comand Line Options");
            t.writeLn("------------------");
            t.writeLn("i=<infile> : Load xel file");
            t.writeLn("o=<outfile> : Save xel file");
            t.writeLn("g=<gens> : No. of gens");
            t.writeLn("b=<bmpfile> : Load image file");
            t.writeLn("--------------------");
            t.writeLn("b & i are mutually exclusive");
            t.writeLn("o sets batch mode(no GUI)");

            t.writeLn("--------------------");
            t.writeLn("0 1 2 3 4 5 6");


            x = t.x;
            y = t.y;
        }

        glPointSize(14.0);
        FOR(k, 7)
        {
            int n = k + 1;
            TGLGroup gl(GL_POINTS, (n & 4) >> 2, (n & 2 ) >> 1, n & 1);
            
            gl.vertex(TFXY( 4 + x + k * 16, y));
        }
     }

    glutSwapBuffers();
}
/////////////////////////////////////////////////////////////////////////

void TXel2D::oglMouse(int button, int state, int x, int y)
{
    if(state == GLUT_DOWN && x < m_iRight)
    {
        // Adjust co ordinate based on window padding
        TFXY ptf(x, y);
        ptf.op<opadd>(-m_iLeft, -m_iTop);

        TXY pt;
        switch(button)
        {
        case GLUT_LEFT_BUTTON: pt = setPoint(xsMovable, ptf); break;
        case GLUT_RIGHT_BUTTON: pt = setPoint(xsFixed, ptf); break;
        default: pt = setPoint(xsEmpty, ptf); break;
        }

        glutPostRedisplay();
    }
}
/////////////////////////////////////////////////////////////////////////

void TXel2D::oglMouseMove(int x, int y)
{
    if(x < m_iRight)
    {
        // Adjust co ordinate based on window padding
        TFXY ptf(x, y);
        ptf.op<opadd>(-m_iLeft, -m_iTop);

        mx = ptf.x;
        my = ptf.y;
        float mi, mj, mk;

        storeIJK(ptf, m_fMinor, m_fMajor, mi, mj, mk);
        storePQ(mi, mj, mk, rx, ry);

        glutPostRedisplay();

    }
}
/////////////////////////////////////////////////////////////////////////

void TXel2D::oglKeyboard(byte ch, int x, int y)
{
    if(ch == 'c' || ch == 'C')
    {
        m_XelSet.clear();
        m_XelNMap.clear();
        cleanChains(m_XelSet, m_XelNMap);  
        glutPostRedisplay();
    }

    if(ch == 'x' || ch == 'X')
    {
        m_XelSet.clear();
        m_XelNMap.clear();
        cleanChains(m_XelSet, m_XelNMap);  

        TXelChains chains = getXelChains(m_XelNMap);
        saveChains(chains);
        glutPostRedisplay();
    }

    if(ch == 'z' || ch == 'Z')
    {
        m_XelSet.clear();
        m_XelNMap.clear();
        cleanChains(m_XelSet, m_XelNMap);  

        TXelChains chains = getXelChains(m_XelNMap);
        loadChains(chains, m_XelSet);
        glutPostRedisplay();
    }

    if(ch == 'q' || ch == 'Q')
        exit(1);

    if(ch == 'i' || ch == 'I')
    {
        m_bShowImage = !m_bShowImage;
        glutPostRedisplay();
    }

    if(ch == 'F' || ch == 'f')
    {
        m_bShowForces = !m_bShowForces;
        glutPostRedisplay();
    }

//     if(ch == 'V' || ch == 'v')
//     {
//         m_bShowGVF = !m_bShowGVF;
//         glutPostRedisplay();
//     }
    
    if(ch == 'p' || ch == 'P')
    {
        m_bShowPoints = !m_bShowPoints;
        glutPostRedisplay();
    }

    if(ch == 'E' || ch == 'e')
    {
        copyXelDataToDevice();
        evolve(); 
        copyXelDataToHost();
        glutPostRedisplay();
    }

    if(ch >= '0' && ch <='9')
    {
        int i = 2 << (int(ch - '0') - 1);
        m_bEvolving = true;
        copyXelDataToDevice();
        FOR(j, i)
        {
            evolve(); 
            ++nGens;
            copyXelDataToHost();
            oglDraw();
        }


        m_bEvolving = false;
        glutPostRedisplay();
        m_nGens = 0;
    }

    try
    {
        if(ch == 's' || ch == 'S')
        {
            char szFile[64] = {0};
            sprintf(szFile, ".\\xel%d.dat", nGens);
            save(szFile);
        }

        if(ch == 'l' || ch == 'L')
        {
            load(string(".\\xel.dat"));
            glutPostRedisplay();
        }
    }
    catch(string &s)
    {
        cerr << s << endl;
    }
}
//////////////////////////////////////////////////////////////////////////
#endif // USEGL
//////////////////////////////////////////////////////////////////////////

void TXel2D::setXel(int state, const TXY &ptXel, const TFXY &ptfPoint)
{
    TXel &xel = m_pArr[ptXel];
    xel.state0 = xel.state1 = state;

    // Which of the 9 m_PointSet will this go in?
    TXY ptMod(ptXel);
    ptMod.op<opmod>(3, 3);
    int n = ptMod.x + ptMod.y * 3;

    if(state) // If point added then set the x, y for the xel and add to occupied set
    {
        xel.pt0 = xel.pt1 = ptfPoint;
        m_PointSet[n].insert(ptXel);
    }
    else 
    {
        // Clear cell and remove from occupied set
        xel.pt0.x = xel.pt1.x = 0;
        xel.pt0.y = xel.pt1.y = 0;
        xel.state0 = xel.state1 = xsEmpty;
        m_PointSet[n].erase(ptXel);
    }
}
//////////////////////////////////////////////////////////////////////////

TXY TXel2D::setPoint(int state, const TFXY &ptf)
{
    // get the XEL within which ptf lies
    TXY ptXel = getPQ(ptf, m_fMinor, m_fMajor);
    setXel(state, ptXel, ptf);
    return ptXel;
}
//////////////////////////////////////////////////////////////////////////

void TXel2D::copyXelDataToDevice()
{
    PROFILE_IT;
    g_e.setContext(__FUNCTION__);

    // Copy Xel data to GPU
    m_GPUXels = m_CPUXels; 

    // Copy occupied xels info to GPU
    {
        int piCounts[10];
        FOR(n, 9)
        {
            set<TXY> &occ = m_PointSet[n];
            copy(occ.begin(), occ.end(), &m_CPUOcc.p2d()[n][0]);
            piCounts[n] = occ.size();
        }

        // Copy occupied xels co-ords and counts to device
        m_GPUOcc = m_CPUOcc;
        cudaMemcpyToSymbol((const char*)g_pdCounts, piCounts, 10 * sizeof(int), 0, cudaMemcpyHostToDevice);
    }
}
//////////////////////////////////////////////////////////////////////////

void TXel2D::copyXelDataToHost()
{
    PROFILE_IT;
    g_e.setContext(__FUNCTION__);

    // On non-GPU version of code m_CPUXels already has the current data so no need to copy
    if(g_bOnCUDA)
        m_CPUXels = m_GPUXels;

    // copy occupied set data from GPU to CPU
    m_CPUOcc = m_GPUOcc;

    int piCounts[10];
    cudaMemcpyFromSymbol(piCounts, (const char*)g_pdCounts, sizeof(int) * 10, 0, cudaMemcpyDeviceToHost);

    FOR(n, 9)
    {
        set<TXY> &occ = m_PointSet[n];
        occ = set<TXY>();

        TXY *begi = &m_CPUOcc.p2d()[n][0];
        TXY *endi = begi + piCounts[n];
        occ.insert(begi, endi);

    }
}
//////////////////////////////////////////////////////////////////////////

void TXel2D::doMovement(int *nBlocks, int *nThreads)
{
    PROFILE_BLOCK("Movement");
    LAUNCH_9(kMovement);
    g_e = cudaThreadSynchronize();
}
//////////////////////////////////////////////////////////////////////////

void TXel2D::interpolate(int *nBlocks, int *nThreads)
{
    PROFILE_BLOCK("Interpolation");
    LAUNCH_9(kInterpolate);
    g_e = cudaThreadSynchronize();
}
//////////////////////////////////////////////////////////////////////////

void TXel2D::updateXelStates(int nTotalOccupied)
{
    PROFILE_BLOCK("Update");
    int nBlocks, nThreads;
    getBlockGridSizes(nTotalOccupied, nBlocks, nThreads);

    KERNEL_1D_GRID_1D(kUpdate, nBlocks, nThreads)();
    CHECK_CUDA("kUpdate");
    g_e = cudaThreadSynchronize();
}
//////////////////////////////////////////////////////////////////////////

void TXel2D::mergeOccupiedSets(int *nBlocks, int *nThreads)
{
    {
        PROFILE_BLOCK("Merge");
        LAUNCH_9(kMergeOccupiedSets);
        g_e = cudaThreadSynchronize();
    }

    {
        PROFILE_BLOCK("Clear");
        // clear the 9 elemnts of the counts array
        KERNEL_1D(kClearOccupiedArr, 9)();
        g_e = cudaThreadSynchronize();
        CHECK_CUDA(__FUNCTION__);
    }
}
//////////////////////////////////////////////////////////////////////////

void TXel2D::divideOccupiedSets(int nTotalOccupied)
{
    PROFILE_BLOCK("Divide");

    int nBlocks, nThreads;
    getBlockGridSizes(nTotalOccupied, nBlocks, nThreads);
    KERNEL_1D_GRID_1D(kDivideOccupiedSets, nBlocks, nThreads)();
    g_e = cudaThreadSynchronize();
    CHECK_CUDA("kDivideOccupiedSets");
}
//////////////////////////////////////////////////////////////////////////

void TXel2D::getXelsAndNears(TXelSet &xels, TXelNMap &nodes)
{
    // Get the live neighbors of all the live points
    TXY ptThis;
    LOOP_PT_YX(ptThis, m_CPUXels.width(), m_CPUXels.height())
    {
        TXel &xel = m_pArr[ptThis];
        if(xel.state0)
        {
            xels.insert(ptThis);
            FOR(j, 6)
            {
                TXY &pt = getNearXels(ptThis, j);
                if(m_pArr[pt].state0)
                {
                    nodes[ptThis].insert(pt);
                }
            }
        }
    }
}
//////////////////////////////////////////////////////////////////////////

bool TXel2D::isTriadXelRedundant(TXelNMap &nodes, TXY &pt0, int j)
{
    //Find the jth triad of xels with pt0 in it, 
    TXY &pt1 = getNearXels(pt0, j);
    TXY &pt2 = getNearXels(pt0, (j + 1) % 6);

    // pt0, pt1 and pt2 are a triangle, check if triad is populated
    if(nodes.count(pt1) && nodes.count(pt2))
    {
        // We can throw away pt0 if n-count of pt0 is 2 or else if
        // all neighbors of pt0 are neighbors of pt0|pt1 
        if(nodes[pt0].size() != 2) 
        {
            FOR_EACH(TXelSet, ni, nodes[pt0])
            {
                const TXY &pt = *ni;
                if(pt != pt1 && pt != pt2 && !nodes[pt1].count(pt) && !nodes[pt2].count(pt))
                {
                    return false;
                }
            }
        }

        return true;
    }

    return false;
}
//////////////////////////////////////////////////////////////////////////

void TXel2D::removeRedundantXels(TXelSet &xels, TXelNMap &nodes)
{
    // Find triads of xels where one of them is redundant
    FOR_EACH(TXelSet, xi, xels)
    {
        TXY pt0 = *xi;
        FOR(j, 6)
        {
            if(isTriadXelRedundant(nodes, pt0, j))
            {
                setXel(xsEmpty, pt0, TFXY());
                nodes.erase(pt0);
                FOR_EACH(TXelNMap, i, nodes)
                {
                    i->second.erase(pt0);
                }
                break;
            }
        }
    }
}
//////////////////////////////////////////////////////////////////////////

// Fixes the xel chains so that the neighbor counts are consistent 
// All chains' xels must have 2 neighbours, and all triple points' xels 3
void TXel2D::cleanChains(TXelSet &xels, TXelNMap &nodes)
{
    getXelsAndNears(xels, nodes);
    removeRedundantXels(xels, nodes);
}
///////////////////////////////t///////////////////////////////////////////

// Finds a neighbour of the xel j, which has only 2 neighbours, if any and
// stores in "link"
bool TXel2D::getLinkNeighbour(TXelNMap &nodes, TXelClass &xc, TXY &j, TXY &link)
{
    FOR_EACH(TXelSet, ni, nodes[j])
    {
        // does ni have 2 neighbors? then we are done, return true
        link = *ni;
        if(xc.hasNears(link, 2))
            return true;
    }
    return false;
}
//////////////////////////////////////////////////////////////////////////

// Finds a neighbour of the xel "link", which has 1 or 2 neighbours, if any 
// and stores it back in "link"
bool TXel2D::findNextLink(TXelNMap &nodes, TXelClass &xc, TXY &link)
{
    FOR_EACH(TXelSet, ni, nodes[link])
    {
        if(xc.hasNears(*ni, 2) || xc.hasNears(*ni, 1))
        {
            link = *ni;
            return true;
        }
    }

    return false;
}
//////////////////////////////////////////////////////////////////////////

// Walks through the network starting from the node "link", adding to
// chains[n] all those xels who have 1 or 2 neighbours
void TXel2D::buildChain(TXelChains &chains, int &n, TXY &link, TXelClass &xc, TXelNMap &nodes)
{
    do 
    {
        // Here link is a link in the current chain so add it 
        chains[n].push_back(link);
        xc.cls[2].erase(link);
        xc.cls[1].erase(link);
    } 
    while(findNextLink(nodes, xc, link));
    ++n;
}
//////////////////////////////////////////////////////////////////////////

TXelChains TXel2D::getXelChains(TXelNMap &nodes)
{
    // Classify xels with 0, 1, 2, 3 neighbors
    TXelClass xc;

    FOR_EACH(TXelNMap, ni, nodes)
    {
        int n = ni->second.size(); 
        assert(n <= 3);
        xc.cls[n].insert(ni->first);
    }

    // We treat triple points as well as ends as the start of chains
    TXelSet junctions;
    junctions.insert(xc.cls[3].begin(), xc.cls[3].end());
    junctions.insert(xc.cls[1].begin(), xc.cls[1].end());

    TXY junction, link;
    TXelChains chains;
    int n = 0;

    // Repeat until all 2 neighbored xels are gone
    while(xc.cls[2].size())
    {
        bool bFound = false;

        // Find a neighbour of this junction that is a link
        FOR_EACH(set<TXY>, i, junctions)
        {
            junction = *i;
            if(bFound = getLinkNeighbour(nodes, xc, junction, link))
                break;
        }

        assert(bFound || !"No start link found, but there is at least one link!");

        // If junction has only 1 neigbour it's part of the chain
        if(xc.hasNears(junction, 1))
        {
            chains[n].push_back(junction);
            junctions.erase(junction);
            xc.cls[1].erase(junction);
        }

        buildChain(chains, n, link, xc, nodes);
    }

    FOR_EACH(TXelChains, ci, chains)
    {
        vector<TXY> &v = ci->second;
        for(size_t i = 0; i < v.size(); ++i)
        {
            //m_pArr[v[i]].iChain = ci->first;
        }
    }

    return chains;

}
//////////////////////////////////////////////////////////////////////////

void TXel2D::saveChains(TXelChains &chains)
{
    ofstream ofs("xelchains.txt");

    FOR_EACH(TXelChains, ci, chains)
    {
        ofs << ci->first;
        vector<TXY> &v = ci->second;

        for(size_t i = 0; i < v.size(); ++i)
        {
            ofs << "\t" << m_pArr[v[i]].fIntensity;
        }

        ofs << endl;
    }
}
//////////////////////////////////////////////////////////////////////////

void TXel2D::loadChains(TXelChains &chains, TXelSet &xels)
{
    ifstream ifs("xelchains_in.txt");

    set<int> loaded;
    string sLine;
    while(getline(ifs, sLine))
    {
        istringstream iss(sLine);
        int n;
        iss >> n;
        loaded.insert(n);
    }

    // Now everything chains thats not in loaded must be emptied
    FOR_EACH(TXelChains, i, chains)
    {
        if(!loaded.count(i->first))
        {
            FOR_EACH(vector<TXY>, ii, i->second)
            {
                setXel(xsEmpty, *ii, TFXY());
            }
        }
    }

}
//////////////////////////////////////////////////////////////////////////

#endif

// CUDA kernel and device functions
#ifndef REGION_CUDA_

// Simple selection sort because we have very few items ( also removes duplicates )
__device__ int kSortUniq(float *p, int count)
{
    int nLeast = 0;
    int nBase = 0;
    float infinity = 99999999999999999.0f;
    FOR(n, count)
    {
        float least = infinity;

        // Find the lowest element after index n
        for(int i = n ; i < count; ++i)
        {
            if(least > p[i])
            {
                least = p[i];
                nLeast = i;
            }
            else if(fabs(least - p[i]) < 0.0001) // duplicate element, make sure its never selected again as the lowest
            {
                p[i] = infinity;
            }
        }

        if(least != infinity)
        {
            // swap element with the one in the base slot and increment base index
            p[nLeast] = p[nBase];       
            p[nBase] = least;
            ++nBase;
        }
    }

    return nBase;
}
//////////////////////////////////////////////////////////////////////////

// Gets the distances on the line p1 -> p2 ( scaled 0 to 1) at which it crosses xel planes and stores points 
// midway between those in pOutS ( these are in-xel points )
// Number of values in pOutS are returned in count (it cannot exceed 6 if p1 and p2 are in adjacent xels) 
__device__ void kGetPlaneAndXelOffsets(float p1[3], float p2[3], float *pOutS, int &count)
{
    __shared__ float t[100];

    // Get points on xel plane boundaries into t
    count = 0;
    FOR(m, 3)
    {
        int l1 = (int)floor(p1[m]);
        int l2 = (int)floor(p2[m]);

        assert(count < 100);

        for(int l = min(l1, l2) + 1; l <= max(l1, l2); l++, count++)
        {
            t[count] = (l - p1[m]) / (p2[m] - p1[m]);
        }
    }

    // Now sort and de-duplicate all the values in t
    count = kSortUniq(t, count);

    // Get the points in between the elements of t ( which will be in-xel points ), store in pOutS
    float s = 0;
    FOR(n, count-1)
    {
        s = t[n] + (t[n+1] - t[n]) / 2.0f;
        pOutS[n] = s;
    }

    count--; // pOutS has 1 element fewer than t
}
//////////////////////////////////////////////////////////////////////////

// Interpolate points for neighboring cells
// pXels is the main array of xels, pOccupied is the 9 row array of non-empty xel co ords
__device__ void kInterpolate(int y)
{
    int *pOccupiedCount = g_pdCounts;
    int x;
    getLinearThreadIndex(x);
    if(x >= pOccupiedCount[y]) return;

    TPitchPtr pXels = g_pdXels;
    TPitchPtr pOccupied = g_pdOcc;

    TXY ptThis = pOccupied.p2d<TXY>()[y][x];
    T2DView<TXel> pArr(pXels);
    TXel &xel = pArr[ptThis];
    TFXY ptfXel = xel.pt1;

    // Get i, j, k for this xels new point position
    float a[3];
    storeIJK(ptfXel, g_fMinor, g_fMajor, a[0], a[1], a[2]);

    int nNew = 0;

    FOR(n, 3)
    {
        TXY ptOther = getNearXels(ptThis, n);
        TXel &other = pArr[ptOther];
        if(other.state0) // there was a coordinate in the neighboring xel
        {
            TFXY ptfOther = other.pt1;

            // Calculate t and s where 
            // t has all the distances along the line(scaled 0 to 1) which cross xel planes 
            // s has values halfway between values in t ( in xel points )
            float s[100];
            float b[3];
            int count;

            // Get i, j, k for other xels new point position
            storeIJK(ptfOther, g_fMinor, g_fMajor, b[0], b[1], b[2]);
            kGetPlaneAndXelOffsets(a, b, s, count);

            // Now for every s, get the xel it's within 
            FOR(m, count)
            {
                // get the xel row and column for the point at s[m]
                TXY pt;
                float i = a[0] + s[m] * (b[0] - a[0]);
                float j = a[1] + s[m] * (b[1] - a[1]);
                float k = a[2] + s[m] * (b[2] - a[2]);
                storePQ(i, j, k, pt.x, pt.y);

                // Store the interpolated point if its future state is xsEmpty
                TXel x = pArr[pt];
                if(x.state1 == xsEmpty)
                {
                    TFXY ptfNew = (ptfOther - ptfXel) * s[m] + ptfXel; 
                    x.pt1 = ptfNew;
                    x.state1 = xsMovable;
                    pArr[pt] = x;

                    // Store this PQ in ptI so we can add it to the occupied set later
                    xel.ptI[nNew] = pt;
                    nNew++;
                }
            }
        }

        assert(nNew < 6);
        xel.ptI.size = nNew;
    }

    // Set the total counter to 0 
    if(x == 0)
        g_pdCounts[9] = 0;
}

//////////////////////////////////////////////////////////////////////////

__device__ bool kAddToOccupiedAllIfNot(TXel& xel, TXY &pt, TMany<64, TXY> &pToAdd, int &index)
{
    int iChosen = atomicAdd(&xel.iChosen, 1);
    if(!iChosen)
    {
        pToAdd[index++] = pt;
        return true;
    }
    return false;
}
//////////////////////////////////////////////////////////////////////////

// Add each xel and its interpolated points into pOccupiedAll
__device__ void kMergeOccupiedSets(int y)
{
    int *pOccupiedCount = g_pdCounts;
    int x;
    getLinearThreadIndex(x);

    if(x >= pOccupiedCount[y]) return;

    TPitchPtr pXels = g_pdXels;
    TXY *pOccupiedAll = g_pdOccAll;

    TXY ptThis = g_pdOcc.p2d<TXY>()[y][x];
    T2DView<TXel> pArr(pXels);
    TXel &xel = pArr[ptThis];
    TXel tmp;
    tmp = xel;


    TMany<60, TXY> &ptInterpolated = xel.ptI;
    TXel &xelDest = pArr[tmp.ptNew];
    int numInterpolated = xel.ptI.size; 
    int isLive = tmp.state1 > 0;
    int hasMoved = tmp.ptNew != ptThis;
    int n = numInterpolated + isLive + hasMoved;

    if(!isLive)
    {
        xel.state0 = xsEmpty;
    }

    if(n)
    {
        // Add all new points to a temporary array
        TMany<64, TXY> ptToAdd;
        ptToAdd.size = 0;

        // Add interpolated points
        for(int m = 0; m < numInterpolated; ++m)
        {
            TXY pt = ptInterpolated[m];
            kAddToOccupiedAllIfNot(pArr[pt], pt, ptToAdd, ptToAdd.size);
        }

        // Add this point
        if(isLive)
        {
            //cerr /**/ << ptThis.x << "," << ptThis.y << " ";
            kAddToOccupiedAllIfNot(xel, ptThis, ptToAdd, ptToAdd.size);
        }

        if(hasMoved) // If it reached another xel then add that xel to occupied set
        {
            //cerr /**/ << xel.ptNew.x << "," << xel.ptNew.y << " ";
            kAddToOccupiedAllIfNot(xelDest, tmp.ptNew, ptToAdd, ptToAdd.size);
        }

        // Now ptToAdd has the points we need to actually add
        if(ptToAdd.size)
        {
            int index = atomicAdd(&pOccupiedCount[9], ptToAdd.size);
            for(int i = 0; i < ptToAdd.size; ++i, ++index)
            {
                pOccupiedAll[index] = ptToAdd[i];
            }
        }
    }
}
//////////////////////////////////////////////////////////////////////////

// Evolves the xel points based on kEvolveRuleFast defined in xelrule.cc
__device__ void kMovement(int y)
{
    int *pOccupiedCount = g_pdCounts;
    int x;
    getLinearThreadIndex(x);
    if(x >= pOccupiedCount[y]) return;

    TPitchPtr pXels = g_pdXels;
    TPitchPtr pOccupied = g_pdOcc;

    TXY ptThis = pOccupied.p2d<TXY>()[y][x];

    // Get the new xel point 
    T2DView<TXel> pArr(pXels);
    TXel &xel = pArr[ptThis];

    // get neighbors
    int n = 0;
    TMany<6, TXel> nears;
    FOR(i, 6)
    {
        TXY ptOther = getNearXels(ptThis, i);
        TXel &other = pArr[ptOther];
        if(other.state0)
        {
            nears[n] = other;
            ++n;
        }
    }

    nears.size = n;

    TXel xelNew = xel;
    kEvolve(nears, xelNew);
    xelNew.ptNew = getPQ(xelNew.pt1, g_fMinor, g_fMajor);
    xel = xelNew;

    if(xelNew.ptNew != ptThis) // moving into another xel
    {
        xel.state1 = xsEmpty;

        int xd = abs(xelNew.ptNew.x - ptThis.x);
        int yd = abs(xelNew.ptNew.y - ptThis.y);
        if(xd > 1 || yd > 1)
            assert(false);

        // Change state of dest xel to movable if its empty
        // No thread contention here because the dest xel is not processed in this batch 
        // of kernel threads ( for y are all same ). Dest xel y is different
        TXel &dest = pArr[xelNew.ptNew];
        if(dest.state0 == xsEmpty)
        {
            dest.state1 = xsMovable;
            dest.pt1 = xelNew.pt1;
            dest.ptNew = xelNew.ptNew;
        }
    }
}
//////////////////////////////////////////////////////////////////////////

// Moves xels to a new xel, optionally into occupied once
// __device__ void kMigratePoints(int y, bool bMoveIntoOccupied)
// {
//     int *pOccupiedCount = g_pdCounts;
//     int x = getLinearThreadIndex();
//     if(x >= pOccupiedCount[y]) return;
// 
//     TPitchPtr pXels = g_pdXels;
//     TPitchPtr pOccupied = g_pdOcc;
// 
//     TXY ptThis = pOccupied.p2d<TXY>()[y][x];
// 
//     // Get the new xel point 
//     T2DView<TXel> pArr(pXels);
//     TXel xel = pArr[ptThis];
// 
//     if(xel.ptNew != ptThis) // moving into another xel
//     {
//         TXel dest = pArr[xel.ptNew];
//         if(dest.state0 == xsEmpty)
//         {
//             xel.state1 = xsEmpty;
//             dest.state1 = xsMovable;
//             dest.pt1 = xel.pt1;
//             dest.ptNew = xel.ptNew;
//             pArr[xel.ptNew] = dest;
//         }
//         else if(bMoveIntoOccupied)
//         {
//             xel.state1 = xsEmpty;
//         }
//     }
// }
//////////////////////////////////////////////////////////////////////////


// Update xel states
__global__ void kUpdate()
{
    int *pOccupiedCount = g_pdCounts;
    int n;
    getLinearThreadIndex(n);
    if(n >= pOccupiedCount[9]) return;

    // Get the new xel point 
    T2DView<TXel> pArr(g_pdXels);
    TXel &xel = pArr[g_pdOccAll[n]];

    // Set the old state and x,y to the new state 
    xel.pt0 = xel.pt1;
    xel.state0 = xel.state1;
    xel.iChosen = 0;
}
//////////////////////////////////////////////////////////////////////////

__global__ void kClearOccupiedArr()
{
    int *pOccupiedCount = g_pdCounts;
    int n;
    getLinearThreadIndex(n);
    if(n > 8) return;
    pOccupiedCount[n] = 0;
}

//////////////////////////////////////////////////////////////////////////

// Takes all the elements in pOccupiedAll and puts it into the corresponding row in pOccupied
__global__ void kDivideOccupiedSets()
{
    int *pOccupiedCount = g_pdCounts;
    int n;
    getLinearThreadIndex(n);
    if(n >= pOccupiedCount[9]) return;

    TXY *pOccupiedAll = g_pdOccAll;
    TXY pt = pOccupiedAll[n];
    TXY ptMod(pt);
    ptMod.op<opmod>(3, 3);
    int y = ptMod.y * 3 + ptMod.x;
    int x = atomicAdd(&pOccupiedCount[y], 1);

    T2DView<TXY> pArr(g_pdOcc);
    pArr[y][x] = pt;
}
//////////////////////////////////////////////////////////////////////////


#endif
