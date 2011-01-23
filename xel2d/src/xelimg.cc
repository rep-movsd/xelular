
//////////////////////////////////////////////////////////////////////////
GPU_CPU TXY getNearXels(TXY& pt, int i);
GPU_CPU TXY getNearXelsSafe(TXY& pt, int i);
GPU_CPU void storeIJK(TFXY ptf, float fMin, float fMaj, float &i, float &j, float &k);
GPU_CPU void storePQ(float i, float j, float k, int &p, int &q);
GPU_CPU TXY getPQ(const TFXY &ptf, float fMin, float fMaj);
GPU_CPU TFXY getXelCenterXY(TXY &pt, float fMin, float fMaj);
//////////////////////////////////////////////////////////////////////////

// Assign any xel with intensity > Z as a fixed point
GPU void kAssignPoints(float fImgThreshold)
{
    TPitchPtr pXels = g_pdXels;
    TXY ptThis;
    getThreadXY(ptThis, pXels.w, pXels.h);

    T2DView<TXel> pArr(pXels);

    if(ptThis.x > 2 && ptThis.x < pXels.w - 2 && ptThis.y > 2 && ptThis.y < pXels.h - 2)
    {
        TXel &xel = pArr[ptThis];
        if(xel.fIntensity > fImgThreshold)
        {
            xel.state0 = xsMovable;
            xel.pt0 = getXelCenterXY(ptThis, g_fMinor, g_fMajor);
            xel.iImgForce = 0;
        }
    }
}
//////////////////////////////////////////////////////////////////////////

// Any point which has at least 1 neighbour greater than it is a hole
GPU void kAssignPoints1(float fEpsilon)
{
    TPitchPtr pXels = g_pdXels;
    TXY ptThis;
    getThreadXY(ptThis, pXels.w, pXels.h);

    T2DView<TXel> pArr(pXels);

    if(ptThis.x > 2 && ptThis.x < pXels.w - 2 && ptThis.y > 2 && ptThis.y < pXels.h - 2)
    {
        TXel &xel = pArr[ptThis];
        xel.state0 = xsMovable;
        xel.pt0 = getXelCenterXY(ptThis, g_fMinor, g_fMajor);
        xel.iImgForce = 0;

        for(int i = 0; i < 6; ++i)
        {
            const TXY &neigh = getNearXels(ptThis, i);
            TXel &other = pArr[neigh];

            if(other.fIntensity - xel.fIntensity > fEpsilon)
            {
                xel.state0 = xsEmpty;
                break;
            }
        }

    }
}
//////////////////////////////////////////////////////////////////////////


// calculates out(x) = in(x + 1) - in(x - 1) and
//  out(y) = out(y) = in(y + 1) - in(y - 1)
GPU void kGetGradientVector(TPitchPtr imgIn, TPitchPtr xyOut)
{
    TXY ptThis;
    getThreadXY(ptThis, xyOut.w, xyOut.h);
    
    T2DView<TFXY> pOut(xyOut);
    
    T2DDefView<float> pIn(imgIn, 0, imgIn.w, 0, imgIn.h, 0);

    TXY ptLeft = ptThis, ptRight = ptThis;
    TXY ptTop = ptThis, ptBottom = ptThis;
    
    ptLeft.x--;
    ptRight.x++;
    ptTop.y--;
    ptBottom.y++;

    pOut[ptThis].x = (pIn[ptRight] - pIn[ptLeft]);
    pOut[ptThis].y = (pIn[ptBottom] - pIn[ptTop]);
}
//////////////////////////////////////////////////////////////////////////

// GPU void kGetBC(TPitchPtr imgGradIn, TPitchPtr imgGradOut)
// {
//     TXY ptThis;
//     getThreadXY(ptThis, imgGradIn.w, imgGradIn.h);
// 
//     T2DDefView<TFXY> pImgIn(imgGradIn, 0, imgGradIn.w, 0, imgGradIn.h);
//     T2DView<TFXY> pImgOut(imgGradOut);
// 
//     float gx = pImgIn[ptThis].x, gy = pImgIn[ptThis].y;
//     float B = gx * gx + gy * gy;
//     float C1 = B * gx;
//     float C2 = B * gy;
// 
//     float dt, mu;
//     dt = 0.025;
//     mu = 10;
// 
//     TXY ptLeft = ptThis, ptRight = ptThis;
//     TXY ptTop = ptThis, ptBottom = ptThis;
// 
//     ptLeft.x--;
//     ptRight.x++;
//     ptTop.y--;
//     ptBottom.y++;
// 
//     TFXY ptSum = pImgIn[ptLeft]+ pImgIn[ptRight] + pImgIn[ptTop] + pImgIn[ptBottom];
//     
//     pImgOut[ptThis].x = (1 - B * dt) * gx + mu*dt*(ptSum.x - 4 * gx) + C1 * dt;
//     pImgOut[ptThis].y = (1 - B * dt) * gy + mu*dt*(ptSum.y - 4 * gy) + C2 * dt;
// }
// //////////////////////////////////////////////////////////////////////////

struct TXelImg
{
    TXel2D &xels;

    // Less blurred is for image forces, more blurred is starting point for gradient vector flow force
    TBitmap<float> bmpLowBlur, bmpHighBlur;
    TGPUArray<float> gpuFilter;
    //TGPUArray<TFXY> imgForceHighBlur;
    int nBlocks, nThreads;

    TXelImg(TXel2D &x) : xels(x){}

    // Generate a 1D separable gaussian kernel
    void createGaussKernel()
    {
        const int FILTER_SIZE = 31;
        float filter_coeff[2][FILTER_SIZE];        
        getGaussianKernel(FILTER_SIZE / 2, filter_coeff[0]);
        getGaussianKernel(FILTER_SIZE / 2, filter_coeff[1]);

        // Put the filter into a GPU array
        TCPUArray<float> cpuFilter;
        cpuFilter.assign2dArr<2, FILTER_SIZE>(filter_coeff);
        gpuFilter = cpuFilter;
    }
    //////////////////////////////////////////////////////////////////////////

    // Create less blurred and more blurred versions of the image
    void createBlurredImgs()
    {
        TGPUArray<float> src(xels.m_Img), dst(xels.m_Img.dims()), tmp(xels.m_Img.dims());
        FOR(i, xels.m_nBlurCount)
        {
            applyImageFilterFast(src, dst, tmp, gpuFilter);
            src = dst;
        }

        bmpHighBlur = dst;
        bmpLowBlur = dst;
    }
    //////////////////////////////////////////////////////////////////////////

    // Calculates the image force vectors and each xels intensity
    void calcImgForce()
    {
        int w = xels.m_CPUXels.width(), h = xels.m_CPUXels.height();
        float fMaj = xels.m_fMajor, fMin = xels.m_fMinor;
        T2DView<TXel> &pArr = xels.m_pArr;

        // Create the image gradient function of the low blurred image and high blurred image
        int nBlocks, nThreads;
        getBlockGridSizes(bmpLowBlur.width() * bmpLowBlur.height(), nBlocks, nThreads);
        KERNEL_1D_GRID_1D(kGetGradientVector, nBlocks, nThreads)(bmpHighBlur.data(), xels.m_GPUImgForce.data());


        // Get average image intensity in each xel for the low blur image
        // TODO CUDAize this
        T2DView<float> pBmp(bmpLowBlur.data());
        LOOP_YX(bmpLowBlur.width(), bmpLowBlur.height())
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

    }
    //////////////////////////////////////////////////////////////////////////

    void setInitialPoints()
    {
        if(g_bOnCUDA)
            xels.m_GPUXels = xels.m_CPUXels;
       
        KERNEL_1D_GRID_1D(kAssignPoints, nBlocks, nThreads)(xels.m_fImgThreshold);
        //KERNEL_1D_GRID_1D(kAssignPoints, nBlocks, nThreads)();

        if(g_bOnCUDA)
            xels.m_CPUXels = xels.m_GPUXels;

        int w = xels.m_CPUXels.width(), h = xels.m_CPUXels.height();
        LOOP_YX(w, h)
        {
            TXY pt(x, y);
            TXel &xel = xels.m_pArr[pt];
            if(xel.state0)
                xels.setXel(xel.state0, pt, getXelCenterXY(pt, g_fMinor, g_fMajor));
        }

        if(xels.m_ImgHole.width())
        {
            T2DView<float> pBmp(xels.m_ImgHole.data());
            LOOP_YX(xels.m_ImgHole.width(), xels.m_ImgHole.height())
            {
                TFXY ptf(x, y);
                TXY pt(ptf);
                TXY pq = getPQ(ptf, xels.m_fMinor, xels.m_fMajor);
                if(pq.x >= 2 && pq.y >= 2 && pq.x < (w - 2) && pq.y < (h - 2))
                {
                    if(pBmp[pt] == 0)
                    {
                        xels.setXel(xsEmpty, pq, TFXY());
                    }
                }
            }
        }


        if(g_bOnCUDA)
            xels.m_GPUXels = xels.m_CPUXels;
    }
    //////////////////////////////////////////////////////////////////////////

//     void calcGradientVectorFlow()
//     {
//         //xels.m_GPUImgGVFForce.setDims(bmpHighBlur.dims());
// //         FOR(i, 5)
// //         {
// //             KERNEL_1D_GRID_1D(kGetBC, nBlocks, nThreads)(imgForceHighBlur.data(), xels.m_GPUImgGVFForce.data());
// //             imgForceHighBlur = xels.m_GPUImgGVFForce;
// //         }
//     }
    //////////////////////////////////////////////////////////////////////////
    
    void loadImage()
    {
        getBlockGridSizes(xels.m_Img.width() * xels.m_Img.width(), nBlocks, nThreads);

        createGaussKernel();
        createBlurredImgs();
        calcImgForce();
        setInitialPoints();

        xels.m_Img = bmpHighBlur;
    }
};
//////////////////////////////////////////////////////////////////////////

