#pragma once
#include <iostream>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "Vector3Int.cpp"
#include <cuda.h>

class Kernel
{
public:
	void recalccc(double* x, double* y, Vector3Int* triangles, int pointLength, int theLength);
	void deactivate(double* x, double* y, int detailx, int detaily, Vector3Int* triangles, int* workingIndex, int pointLength, int length);
};

