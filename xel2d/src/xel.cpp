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

TXel2D *g_pXels;
bool TGLGroup::bOpen;

TFXY g_ptCenter;
__constant__ float g_fMajor, g_fMinor;
__constant__ TPitchPtr g_pdXels;
__constant__ TPitchPtr g_pdOcc;
__device__ int g_pdCounts[10];
__constant__ TXY *g_pdOccAll;

__global__ void kUpdate();
__global__ void kDivideOccupiedSets();
__global__ void kClearOccupiedArr();
__device__  void kEvolve(TMany<6, TXel> &nears, TXel &xel);

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
//__device__ void kMigratePoints(int i, bool bMoveIntoOccupied);

#define DECL_MOVEMENT_KERNEL(X)         __global__ void kMovement##X() {kMovement(X);}
#define DECL_INTERPOLATE_KERNEL(X)      __global__ void kInterpolate##X() {kInterpolate(X);}
#define DECL_MERGE_KERNEL(X)            __global__ void kMergeOccupiedSets##X() {kMergeOccupiedSets(X);}
//#define DECL_MIGRATE_EMPTY_KERNEL(X)    __global__ void kMigratePointsEmpty##X() {kMigratePoints(X, false);}
#define DECL_MIGRATE_OCC_KERNEL(X)      __global__ void kMigratePointsOcc##X() {kMigratePoints(X, true);}

DECL_9(DECL_MOVEMENT_KERNEL);
DECL_9(DECL_INTERPOLATE_KERNEL);
DECL_9(DECL_MERGE_KERNEL);
//DECL_9(DECL_MIGRATE_EMPTY_KERNEL);
//DECL_9(DECL_MIGRATE_OCC_KERNEL);


#include "xelmovement.cc"
#include "xelimg.cc"

#endif 

// Profiler class
#ifndef REGION_PROFILER_
#define DO_PROFILE 1
#ifdef DO_PROFILE
#define PROFILE_IT TProfile _profile(__FUNCTION__)
#define PROFILE_BLOCK(X) TProfile _profileblock(X);

class TProfile
{
    static string m_sFile;
    static map<string, int> ms_nFnCounts;
    static map<string, int> ms_nFnTimes;

    clock_t m_nStartTick;
    const char* m_pszFnName;

public:
    TProfile(const char *pszFnName): m_pszFnName(pszFnName)
    {
        ms_nFnCounts[m_pszFnName]++;
        m_nStartTick = clock();
    }
    //////////////////////////////////////////////////////////////////////////


    ~TProfile()
    {
        ms_nFnTimes[m_pszFnName] += (clock() - m_nStartTick);
    }
    //////////////////////////////////////////////////////////////////////////


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
                //           if(ci->second > 1 && ti->second)
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

class TProfileHelper
{
public:
    ~TProfileHelper() 
    { 
        TProfile::dump(); 
    }
};
//////////////////////////////////////////////////////////////////////////

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
//////////////////////////////////////////////////////////////////////////


void oglDraw()
{
    g_pXels->oglDraw();
}
//////////////////////////////////////////////////////////////////////////

void oglMouse(int button, int state, int x, int y)
{
    g_pXels->oglMouse(button, state, x, y);
}
//////////////////////////////////////////////////////////////////////////

void oglMouseMove(int x, int y)
{
    g_pXels->oglMouseMove(x, y);
}
//////////////////////////////////////////////////////////////////////////

void oglKeyboard(byte ch, int x, int y)
{
    g_pXels->oglKeyboard(ch, x, y);
}
//////////////////////////////////////////////////////////////////////////

void exitOpenGL()
{
    glutDisplayFunc(0);
    glutMouseFunc(0);
    glutPassiveMotionFunc(0);
}
//////////////////////////////////////////////////////////////////////////

void initOpenGL(TXel2D &xels, int w, int h) 
{
    float fXelWidth = xels.m_fMinor * 2.0f;
    float fXelHeight = xels.m_fMajor * 1.5f;

    float fPadL = xels.m_fMinor / 2.0f;
    float fPadR = xels.m_fMinor * 1.5f + 250;
    float fWidth = fXelWidth * w;

    float fPadV = xels.m_fMajor * 0.5f;
    float fHeight = fXelHeight * h + (h%2 ? 0 : xels.m_fMajor) ;

    xels.m_iLeft = fPadL;
    xels.m_iTop = fPadV;
    xels.m_iRight = fWidth + fPadL;
    xels.m_iBottom = fHeight + fPadV;

    // set RGBA mode with float and depth buffers
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(fWidth + fPadR + fPadL, fHeight + fPadV * 2);
    glutCreateWindow("Curve Dynamics - 2D");

    glutSetCursor(GLUT_CURSOR_FULL_CROSSHAIR);

    //glutSetCursor(GLUT_CURSOR_CROSSHAIR);
    gluOrtho2D(-fPadL, fPadR + fWidth, fHeight + fPadV, -fPadV);

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

struct TArgs
{
    int w, h;
    int nEdgePels, nGens, nReps;
    string sInFile, sOutFile, sImageFile;

    TArgs() : w(16), h(16), nEdgePels(32), nGens(500), nReps(1)
    {
    }
};
//////////////////////////////////////////////////////////////////////////

// Parse command line args
void parseArgs(int argc, char ** argv, TArgs &args) 
{
    // Get any command line options
    FOR(i, argc)
    {
        string s(argv[i]);
        if(s.length() > 2 && s[1] == '=')
        {
            istringstream iss(s.substr(2));

            switch(tolower(s[0]))
            {
                // i=file
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

                // generatiosn to run in batch mode
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
        xels.loadImage(args.sImageFile, args.nEdgePels);
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
    g_pXels = &xels;
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

    //     catch(...)
    //     {
    //         cerr << "OS exception" << endl;
    //     }

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

TXel2D::TXel2D() :m_bEvolving(), m_bShowImage(), m_bShowPoints(true), m_bShowForces(), m_bShowXels(true)
{
}
//////////////////////////////////////////////////////////////////////////

// constructor takes width, height and xel major radius
void TXel2D::init(int w, int h, int nEdgePels)
{
    PROFILE_IT;
    g_e.setContext(__FUNCTION__);

    // Xels data on CPU and GPU
    m_CPUXels.setDims(w, h);
    m_GPUXels.setDims(w, h);

    // Occupied points divided into 9 groups
    m_CPUOcc.setDims(w * h, 9);
    m_GPUOcc.setDims(w * h, 9);

    // Set of all occupied points
    m_GPUOccAll.setDims(w * h, 1);

    // set of points in the 9 groups maintained on CPU as a std::set
    FOR(i, 9) 
        m_PointSet[i].clear();

    m_nGens = 0;
    m_fMajor = float(nEdgePels);
    m_fMinor = m_fMajor * SIN60;

    g_e = cudaMemcpyToSymbol((const char*)&g_fMinor, &m_fMinor, sizeof(float), 0, cudaMemcpyHostToDevice);
    g_e = cudaMemcpyToSymbol((const char*)&g_fMajor, &m_fMajor, sizeof(float), 0, cudaMemcpyHostToDevice);

    m_pArr = T2DView<TXel>(m_CPUXels.data());

    if(g_bOnCUDA)
    {
        TPitchPtr pdXels = m_GPUXels.data();
        g_e = cudaMemcpyToSymbol((const char*)&g_pdXels, &pdXels, sizeof(TPitchPtr), 0, cudaMemcpyHostToDevice);

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
    }

    TXY ptTmp(w, h);
    g_ptCenter = getXelCenterXY(ptTmp, g_fMinor, g_fMajor) / 2;
}
//////////////////////////////////////////////////////////////////////////

void TXel2D::loadImage(const string &sImageFile, int nEdgePels)
{
    PROFILE_IT;

    m_Img.load(sImageFile.c_str(), true);

    // calculate hex grid size and initialize
    TFXY ptfImgSize(m_Img.width(), m_Img.height());

    float fMajor = float(nEdgePels);
    float fMinor = fMajor * SIN60;

    TXY ptMax = getPQ(ptfImgSize, fMinor, fMajor) + TXY(2, 2);
    init(ptMax.x, ptMax.y, nEdgePels);

    TXelImg xi;
    xi.loadImage(*this);
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
                glPolygonMode(GL_FRONT , xel.state0 ? GL_FILL : GL_LINE);
                TGLGroup gl(GL_POLYGON, xel.state0 ? 0 : 1, 1, 0, xel.state0 ? 0.4 : 0.2);

//                 if(rx == x && ry == y)
//                     glPolygonMode(GL_FRONT, GL_FILL);
//                 else
//                     glPolygonMode(GL_FRONT, GL_LINE);
//                TGLGroup gl(GL_POLYGON, 1, 1, 0, 0.2);
                FOR(i, 6)
                    glVertex2f(ex + hexX[i], ey + hexY[i]);
            }
        }

//         char s[256];
//         sprintf(s, "%d, %d -> %f %f", rx, ry, mx, my);
//         glColor3f(1, 1, 1);
//         TGLText t(mx + 10, my - 15, 15);
//         t.writeLn(s);
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
            glPointSize(2);
            if(xel.state0 != xEmpty)
            {
                float green = xel.state0 == xFixed;
                switch(xel.state0)
                {
                    case xFixed: green += 0.5;
                    case xTixed: green += 0.5;
                }

                TGLGroup gl(GL_POINTS, 1, green, 0);
                gl.vertex(xel.pt0);
            }
        }
    }


    if(!m_bEvolving)
    {
        {

            TGLGroup g(GL_LINES, 1, 1, 1);
            glVertex2i(m_iRight, 0);
            glVertex2i(m_iRight, m_iBottom);
        }

        glColor3f(1, 1, 1);
        TGLText t(m_iRight + 10, 10, 15);
      
        t.writeLn("Keyboard interface");
        t.writeLn("------------------");
        t.writeLn("E : evolve 1 gen");
        t.writeLn("0-9 : evolve 2^N gens");
        t.writeLn("F : Show/Hide image data");
        t.writeLn("P : Show/Hide xel points");
        t.writeLn("S : Save points to xel.dat");
        t.writeLn("L : Load points from xel.dat");

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
        case GLUT_LEFT_BUTTON: pt = setPoint(xMovable, ptf); break;
        case GLUT_RIGHT_BUTTON: pt = setPoint(xFixed, ptf); break;
        default: pt = setPoint(xEmpty, ptf); break;
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
    if(ch == 'f' || ch == 'F')
    {
        m_bShowImage = !m_bShowImage;
        glutPostRedisplay();
    }

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
            save(string(".\\xel.dat"));
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
        xel.state0 = xel.state1 = xEmpty;
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
        //int *piCounts = m_pCounts.ptr();
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

void TXel2D::doMigrateXels(int *nBlocks, int *nThreads)
{
    //     {
    //         PROFILE_BLOCK("MigrateEmpty");
    //         LAUNCH_9(kMigratePointsEmpty);
    //         g_e = cudaThreadSynchronize();
    //     }
    //     {
    //         PROFILE_BLOCK("MigrateOccupied");
    //         LAUNCH_9(kMigratePointsOcc);
    //         g_e = cudaThreadSynchronize();
    //     }
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

                // Store the interpolated point if its future state is xEmpty
                TXel x = pArr[pt];
                if(x.state1 == xEmpty)
                {
                    TFXY ptfNew = (ptfOther - ptfXel) * s[m] + ptfXel; 
                    x.pt1 = ptfNew;
                    x.state1 = xMovable;
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
        xel.state0 = xEmpty;
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
        xel.state1 = xEmpty;

        int xd = abs(xelNew.ptNew.x - ptThis.x);
        int yd = abs(xelNew.ptNew.y - ptThis.y);
        if(xd > 1 || yd > 1)
            assert(false);

        // Change state of dest xel to movable if its empty
        // No thread contention here because the dest xel is not processed in this batch 
        // of kernel threads ( for y are all same ). Dest xel y is different
        TXel &dest = pArr[xelNew.ptNew];
        if(dest.state0 == xEmpty)
        {
            dest.state1 = xMovable;
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
//         if(dest.state0 == xEmpty)
//         {
//             xel.state1 = xEmpty;
//             dest.state1 = xMovable;
//             dest.pt1 = xel.pt1;
//             dest.ptNew = xel.ptNew;
//             pArr[xel.ptNew] = dest;
//         }
//         else if(bMoveIntoOccupied)
//         {
//             xel.state1 = xEmpty;
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
