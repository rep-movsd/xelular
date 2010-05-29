//////////////////////////////////////////////////////////////////////////
GPU_CPU TXY getNearXels(TXY& pt, int i);
GPU_CPU TXY getNearXelsSafe(TXY& pt, int i);
GPU_CPU void storeIJK(TFXY ptf, float fMin, float fMaj, float &i, float &j, float &k);
GPU_CPU void storePQ(float i, float j, float k, int &p, int &q);
GPU_CPU TXY getPQ(const TFXY &ptf, float fMin, float fMaj);
GPU_CPU TFXY getXelCenterXY(TXY &pt, float fMin, float fMaj);
//////////////////////////////////////////////////////////////////////////

// Assign any xel with intensity > 0.5 as a fixed point
__global__ void kAssignFixedPoints()
{
    TPitchPtr pXels = g_pdXels;
    TXY ptThis;
    getThreadXY(ptThis, pXels.w, pXels.h);

    T2DView<TXel> pArr(pXels);

    if(ptThis.x > 2 && ptThis.x < pXels.w - 2 && ptThis.y > 2 && ptThis.y < pXels.h - 2)
    {
        TXel &xel = pArr[ptThis];
        if(xel.fIntensity > 0.1f)
        {
            xel.state0 = /*xel.fIntensity > 0.5f ? xFixed : */xMovable;
            xel.pt0 = getXelCenterXY(ptThis, g_fMinor, g_fMajor);
            xel.iImgForce = 0;
        }
    }
}
//////////////////////////////////////////////////////////////////////////

__global__ void kAssignForce(int iForceIndex)
{
    TPitchPtr pXels = g_pdXels;
    TXY ptThis;
    getThreadXY(ptThis, pXels.w, pXels.h);

    T2DView<TXel> pArr(pXels);
    if(ptThis.x > 2 && ptThis.x < pXels.w - 2 && ptThis.y > 2 && ptThis.y < pXels.h - 2)
    {
        TXel &xel = pArr[ptThis];
        if(xel.iImgForce == -1)
        {
            FOR(i, 6)
            {
                TXel &other = pArr[getNearXels(ptThis, i)];

                if(other.iImgForce == iForceIndex)
                {
                    xel.iImgForce = iForceIndex + 1;
                    pArr[ptThis].pt0 = getXelCenterXY(ptThis, g_fMinor, g_fMajor);
                    assert(xel.state0 == xEmpty);
                    xel.state0 = xMovable;
                    break;
                }
            }
        }
    }
}
//////////////////////////////////////////////////////////////////////////

__global__ void kAssignForceVector()
{
    TPitchPtr pXels = g_pdXels;
    TXY ptThis;
    getThreadXY(ptThis, pXels.w, pXels.h);

    T2DView<TXel> pArr(pXels);
    if(ptThis.x > 2 && ptThis.x < pXels.w - 2 && ptThis.y > 2 && ptThis.y < pXels.h - 2)
    {
        TXel &xel = pArr[ptThis];
        int iImgForce = xel.iImgForce;
        if(iImgForce != -1)
        {
            TFXY ptfTotal;
            FOR(i, 6)
            {
                TXY ptOther = getNearXels(ptThis, i);
                TXel &other = pArr[ptOther];

                if(other.state0)
                {
                    TFXY ptfUnit(ptOther - ptThis);
                    int iNetForce = (iImgForce - other.iImgForce);
                    ptfUnit *= iNetForce;
                    ptfTotal += ptfUnit;
                }
            }

           // ptfTotal.op<opmul>(g_fMinor, g_fMajor * 1.5);
            xel.ptfImgForce = ptfTotal / 4.0f;
        }
    }
}
//////////////////////////////////////////////////////////////////////////

struct TXelImg
{
    void loadImage(TXel2D &xels)
    {
        // Simply fill each xel with the scaled intensity of the image

        TBitmap<byte> &bmp = xels.m_Img;
        T2DView<byte> pBmp(bmp.data());
        int w = xels.m_CPUXels.width(), h = xels.m_CPUXels.height();
        float fMaj = xels.m_fMajor, fMin = xels.m_fMinor;
        T2DView<TXel> &pArr = xels.m_pArr;

        // The friction is proportional to the average image intensity in the xel
        LOOP_YX(bmp.width(), bmp.height())
        {
            TFXY ptf(x, y);
            TXY pt(ptf);
            TXY pq = getPQ(ptf, fMin, fMaj);

            if(pq.x >= 2 && pq.y >= 2 && pq.x < (w - 2) && pq.y < (h - 2))
            {
                pArr[pq].fIntensity += (float(pBmp[pt]) / 255.0f);
                pArr[pq].fIntensity /= 2.0f;
            }
        }

        // Calculate image forces in CUDA
        if(g_bOnCUDA)
            xels.m_GPUXels = xels.m_CPUXels;

        int nBlocks, nThreads;
        getBlockGridSizes(w * h, nBlocks, nThreads);
        KERNEL_1D_GRID_1D(kAssignFixedPoints, nBlocks, nThreads)();

        // Assign image force indexes
        FOR(iImgForce, 4)
        {
            KERNEL_1D_GRID_1D(kAssignForce, nBlocks, nThreads)(iImgForce);
        }

        // Assign image force vectors
        KERNEL_1D_GRID_1D(kAssignForceVector, nBlocks, nThreads)();

        if(g_bOnCUDA)
            xels.m_CPUXels = xels.m_GPUXels;

        float maxForceX = 0, maxForceY = 0;
        float minForceX = 999999, minForceY = 999999;
        LOOP_YX(xels.m_CPUXels.width(), xels.m_CPUXels.height())
        {
            TXY pt(x, y);
            TXel &xel = pArr[pt];
            if(xel.state0)
            {
                xels.setXel(xel.state0, pt, getXelCenterXY(pt, g_fMinor, g_fMajor));
                if(maxForceX < xel.ptfImgForce.x) maxForceX = xel.ptfImgForce.x;
                if(maxForceY < xel.ptfImgForce.y) maxForceY = xel.ptfImgForce.y;

                if(minForceX > xel.ptfImgForce.x) minForceX = xel.ptfImgForce.x;
                if(minForceY > xel.ptfImgForce.y) minForceY = xel.ptfImgForce.y;
            }
        }

        if(g_bOnCUDA)
            xels.m_GPUXels = xels.m_CPUXels;

        cerr << "max and min forces " << endl;
        cerr << minForceX << "," << minForceY << " -> " << maxForceX << "," << maxForceY << endl;
    }
};
//////////////////////////////////////////////////////////////////////////
