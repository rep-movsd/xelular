#ifndef __XEL_H__
#define __XEL_H__

enum TXelState {xEmpty, xMovable, xFixed, xTixed};

struct TXel
{
    int state0, state1;            // current and new state
    TFXY pt0, pt1;                 // X Y of the current and next point in this XEL 
    TMany<6, TXY> ptI;             // P, Q of new xels which have got interpolated
    TXY ptNew;                     // P, Q of xels into which this point has gone
    float fIntensity;              // Image "friction"
    int iChosen;               
    int iImgForce;
    TFXY ptfImgForce;
    
    DECL_CUDA_COPY_CTOR(TXel);

    GPU_CPU TXel()
    {
        state0 = state1 = xEmpty;
        ptI.size = 0;
        fIntensity = 0.0;
        iChosen = 0;
        iImgForce = -1;
    }
    //////////////////////////////////////////////////////////////////////////
};
//////////////////////////////////////////////////////////////////////////

class TXel2D
{
    friend struct TXelImg;

    TCPUArray<TXel> m_CPUXels;
    TGPUArray<TXel> m_GPUXels;

    set<TXY> m_PointSet[9];
    TGPUArray<TXY> m_GPUOcc;
    TGPUArray<TXY> m_GPUOccAll;
    TCPUArray<TXY> m_CPUOcc;

    TBitmap<byte> m_Img;

    // The last set of occupied xels
    set<TXY> m_lastOccupied;

    // CUDA blocks and threads count 
    int m_nPoints, m_nGrids;

    bool m_bEvolving, m_bShowImage, m_bShowPoints, m_bShowForces, m_bShowXels;
    

    // Set a xels state given point co ordinate
    TXY setPoint(int state, const TFXY &pt);

    // Sets a xels stae given xel location, point is absolute X, Y
    void setXel(int state, const TXY &ptXel, const TFXY &ptfPoint);

    // Updates the set tracking the occupied xels after interpolation
    void updateOccupiedSet();

public:

    float m_fMajor, m_fMinor;
    int m_iTop, m_iLeft, m_iRight, m_iBottom;
    T2DArr<TXel> m_pArr;
    int m_nGens;

    //int mi, mj;
    double mx, my;
    int rx, ry;

    TXel2D();
    void load(const string &sFile);
    void save(const string &sFile);
    void init(int w, int h, int nEdgePels);
    void setFriction(const string &sImageFile);
    void loadImage(const string &sImageFile, int nEdgePels);
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
    int x, y, h;
    public:
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
