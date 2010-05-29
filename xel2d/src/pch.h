#ifndef __PCH_H__
#define __PCH_H__


#include <ctime>
#include <cmath>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <set>
using namespace std;

#include "cudalib.h"

#ifdef USEGL
    #define GLUT_DISABLE_ATEXIT_HACK 1
    #include <glut.h>
    #ifdef _MSC_VER
        #include <windows.h>
    #endif
#endif


#endif //__PCH_H__
