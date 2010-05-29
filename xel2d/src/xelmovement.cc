GPU TFXY kGetIntrinsicForce(TMany<6, TXel> &nears, TXel &xel)
{
    TFXY ptfMean;
    TMany<6, TFXY> dists;

    // Get mean point
    FOR(i, nears.size) ptfMean += nears[i].pt0;
    ptfMean += xel.pt0;
    ptfMean /= nears.size + 1;

    // Get the distances from the mean point
    FOR(i, nears.size)
        dists[i] = (nears[i].pt0 - ptfMean);

    // calculate A = SIGMA(X^2), B = SIGMA(XY), C = SIGMA(Y^2)
    float A = 0, B = 0, C = 0;
    for(int i = 0; i < nears.size; ++i)
    {
        A += dists[i].x * dists[i].x;
        B += dists[i].x * dists[i].y;
        C += dists[i].y * dists[i].y;
    }

    float a, b, c, d;
    float e = ((A + C) - sqrt((A - C) * (A - C) + 4.0f * B * B)) / 2.0f;

    a = ((e - C) / (B + (B == 0))) * (B != 0);
    b = (B != 0);

    //adding values for B==0
    a += (B == 0) * (A > C);
    b += (B == 0) * (A < B);

    float len = sqrt(a * a + b * b);
    a /= (len + (len == 0));
    b /= (len + (len == 0));

    c = a * ptfMean.x + b * ptfMean.y;
    d = a * xel.pt0.x + b * xel.pt0.y - c;

    TFXY ptfIntrinsic(a * d * -0.5f, b * d * -0.5f);

    return ptfIntrinsic;
}

//Calculates the new point for the given xel based on the inertial tensor rule
GPU void kEvolve(TMany<6, TXel> &nears, TXel &xel)
{
    TFXY old = xel.pt0;
    xel.pt1 = xel.pt0;

    if(nears.size && xel.state0 == xMovable)
    {
        TFXY ptfIntrinsic = kGetIntrinsicForce(nears, xel);
        //TFXY ptfImgForce = xel.ptfImgForce * g_fMinor;

        //Take only radial component of image force
        // as f = (f.v / v.v) * v

//         float vv = ptfIntrinsic.dot(ptfIntrinsic) + 0.1f;
//         float fv = ptfImgForce.dot(ptfIntrinsic);
//         ptfImgForce = ptfIntrinsic * (fv / vv);

        TFXY ptfTotal = ptfIntrinsic;// * 0.1f;// + ptfImgForce * 0.1f;
        xel.pt1 += ptfTotal;
    }

    assert(old == xel.pt0);


}
//////////////////////////////////////////////////////////////////////////

