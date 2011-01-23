#ifndef __XEL_H__
#define __XEL_H__

enum TXelState {xsEmpty, xsMovable, xsFixed};

// A xel node, containing upto 6 connections to other xels, 
typedef set<TXY> TXelSet;

// A xel and its neighbours
typedef map<TXY, TXelSet> TXelNMap;

typedef map<int, vector<TXY> > TXelChains;

struct TXelClass
{
    TXelSet cls[4];
    int hasNears(const TXY &pt, int ncount)
    {
        assert(ncount <=3);
        return cls[ncount].count(pt);
    }
};

// Structure encapsulating arguments
struct TArgs
{
    int w, h;
    int nEdgePels, nGens, nReps, nBlurCount;
    string sInFile, sOutFile, sImageFile, sHoleImageFile;
    float fImgThreshold;

    TArgs() : w(16), h(16), nEdgePels(32), nGens(500), nReps(1), fImgThreshold(0.01), nBlurCount(7)
    {
    }
};
//////////////////////////////////////////////////////////////////////////


struct TXel
{
    int state0, state1;            // current and new state
    TFXY pt0, pt1;                 // X Y of the current and next point in this XEL 
    TMany<60, TXY> ptI;            // P, Q of new xels which have got interpolated
    TXY ptNew;                     // P, Q of xels into which this point has gone
    float fIntensity;              // Image intensity
    int iChosen;               
    int iImgForce;
    int iChain;

    DECL_CUDA_COPY_CTOR(TXel);

    GPU_CPU TXel() 
    {
        state0 = state1 = xsEmpty;
        ptI.size = 0;
        iChosen = 0;
        iImgForce = -1;
        iChain = -1;

        fIntensity = 0;
    }
    //////////////////////////////////////////////////////////////////////////
    
};
//////////////////////////////////////////////////////////////////////////

class TXel2D
{
    friend struct TXelImg;

    TCPUArray<TXel> m_CPUXels;
    TGPUArray<TXel> m_GPUXels;
    TGPUArray<TFXY> m_GPUImgForce;
    //TGPUArray<TFXY> m_GPUImgGVFForce;

    set<TXY> m_PointSet[9];
    TGPUArray<TXY> m_GPUOcc;
    TGPUArray<TXY> m_GPUOccAll;
    TCPUArray<TXY> m_CPUOcc;

    TBitmap<float> m_Img;
    TBitmap<float> m_ImgHole;

    // The last set of occupied xels
    set<TXY> m_lastOccupied;

    // CUDA blocks and threads count 
    int m_nPoints, m_nGrids;

    bool m_bEvolving, m_bShowImage, m_bShowPoints, m_bShowForces, m_bShowXels;
    bool m_bShowGVF;

    int m_nBlurCount;
    float m_fImgThreshold;

    TXelSet m_XelSet;
    TXelNMap m_XelNMap;

    // Set a xels state given point co ordinate
    TXY setPoint(int state, const TFXY &pt);

    // Sets a xels stae given xel location, point is absolute X, Y
    void setXel(int state, const TXY &ptXel, const TFXY &ptfPoint);

    // Updates the set tracking the occupied xels after interpolation
    void updateOccupiedSet();

    //////////////////////////////////////////////////////////////////////////
    // Following methods are for extracting xel network topology
    //////////////////////////////////////////////////////////////////////////
    
    // Fixes the xel chains so that the neighbor counts are consistent 
    // All chains' xels must have 2 neighbours, and all triple points' xels 3
    // cleanChains() depends on the other three here
    void cleanChains(TXelSet &xels, TXelNMap &nodes);
    void removeRedundantXels(TXelSet &xels, TXelNMap &nodes);
    void getXelsAndNears( TXelSet &xels, TXelNMap &nodes);
    bool isTriadXelRedundant(TXelNMap &nodes, TXY &pt0, int j);
    
    // Identifies the distinct chains of xels for detecting features like
    // "sunken bridges" or "whiskers" 
    // getXelChains() uses the other three methods
    TXelChains getXelChains(TXelNMap &nodes);
    bool getLinkNeighbour(TXelNMap &nodes, TXelClass &xc, TXY &j, TXY &link);
    bool findNextLink(TXelNMap &nodes, TXelClass &xc, TXY &link);
    void buildChain(TXelChains &chains, int &n, TXY &link, TXelClass &xc, TXelNMap &nodes);

    // Saves the network xel intensities to a file xelchains.txt
    // Format is in the form of lines : "N I I I I ..." N is chain index
    void saveChains(TXelChains &chains);

    // Loads chains data from a file xelchains_in.txt, 
    // It removes all chain indices that are not present in the file
    void loadChains(TXelChains &chains, TXelSet &xels);


public:

    float m_fMajor, m_fMinor;
    int m_iTop, m_iLeft, m_iRight, m_iBottom;
    T2DView<TXel> m_pArr;
    int m_nGens;
    string m_sImgName;

    //int mi, mj;
    double mx, my;
    int rx, ry;

    // Various values needed for display calculations;
    float m_fXelWidth, m_fXelHeight;
    float m_fPadL, m_fPadR, m_fPadV, m_fTxtWidth;
    float m_fWidth, m_fHeight;


    TXel2D();
    void load(const string &sFile);
    void save(const string &sFile);
    void init(int w, int h, int nEdgePels);
    void setFriction(const string &sImageFile);
    void loadImage(const TArgs &args);
    void evolve();

    void divideOccupiedSets(int nTotalOccupied);
    void mergeOccupiedSets(int *nBlocks, int *nThreads);
    void updateXelStates(int nTotalOccupied);
    void interpolate(int *nBlocks, int *nThreads);
    void doMovement(int *nBlocks, int *nThreads);
    void doMigrateXels(int *nBlocks, int *nThreads);

    const cudaExtent &dims() { return m_CPUXels.dims(); }
    void copyXelDataToDevice();
    void copyXelDataToHost();

    //////////////////////////////////////////////////////////////////////////
    // OpenGL handlers
    //////////////////////////////////////////////////////////////////////////
    void oglDraw();

    void drawHexagon( dword x, dword y, float r, float g, float b, float a, float hexX, float hexY );
    void oglMouse(int button, int state, int x, int y);
    void oglMouseMove(int x, int y);
    void oglKeyboard(byte ch, int x, int y);
};
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// OpenGL RAII helper ( handles unclosed blocks automatically)
//////////////////////////////////////////////////////////////////////////
#ifdef USEGL

class TGLGroup
{
    static bool bOpen;

public:    
    TGLGroup(GLenum mode, GLfloat r = -1, GLfloat g = -1, GLfloat b = -1, GLfloat a = 1) 
    { 
        if(bOpen)
            glEnd();

        if(r >= 0)
            glColor4f(r, g, b, a);

        glBegin(mode);
        bOpen = true; 
    }
    //////////////////////////////////////////////////////////////////////////

    ~TGLGroup() 
    { 
        if(bOpen)
        {
            glEnd();
            bOpen = false; 
        }
    }
    //////////////////////////////////////////////////////////////////////////

    void vertex(TFXY &p)
    {
        glVertex2d(p.x, p.y);
    }
};

class TGLText
{
    public:

        int x, y, h;

        TGLText(int x, int y, int h) : x(x), y(y), h(h)
        {
            glRasterPos2i(x, y);
        }

        void write(const char *s)
        {
            while(*s)
                glutBitmapCharacter(GLUT_BITMAP_8_BY_13, *s++);
        }

        void writeLn(const char *s = 0)
        {
            if(s)
                write(s);
            y += h;
            glRasterPos2i(x, y);
        }
};

#else // DUMMY 

typedef unsigned int GLenum;
typedef float GLfloat;

class TGLGroup
{
    static bool bOpen;

public:    
    TGLGroup(GLenum mode, GLfloat r = -1, GLfloat g = -1, GLfloat b = -1, GLfloat a = 1) 
    { 
    }
    //////////////////////////////////////////////////////////////////////////

    ~TGLGroup() 
    { 
    }
    //////////////////////////////////////////////////////////////////////////

    void vertex(TFXY &p)
    {
    }
};



#endif // USEGL


#endif //__XEL_H__
