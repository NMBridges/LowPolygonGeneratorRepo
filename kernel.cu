// Created by Nolan Bridges on 11/15/20.
// Copyright © 2020 NiMBLe Interactive. All rights reserved.

#include "kernel.h"

__global__ void recalculateTriangles(double* x, double* y, uint16_t* g_tri1, uint16_t* g_tri2, uint16_t* g_tri3, int* theLength)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	

	if (i < theLength[0])
	{
		double p1x = x[(int)g_tri1[i]];
		double p1y = y[(int)g_tri1[i]];
		double p2x = x[(int)g_tri2[i]];
		double p2y = y[(int)g_tri2[i]];
		double p3x = x[(int)g_tri3[i]];
		double p3y = y[(int)g_tri3[i]];

		double s1 = sqrt((p2x - p3x) * (p2x - p3x) + (p2y - p3y) * (p2y - p3y));
		double s2 = sqrt((p1x - p3x) * (p1x - p3x) + (p1y - p3y) * (p1y - p3y));
		double s3 = sqrt((p1x - p2x) * (p1x - p2x) + (p1y - p2y) * (p1y - p2y));

		double cenX = (s1 * p1x + s2 * p2x + s3 * p3x) / (s1 + s2 + s3);
		double cenY = (s1 * p1y + s2 * p2y + s3 * p3y) / (s1 + s2 + s3);

		double tempDeg = atan((p1y - cenY) / (p1x - cenX));
		if (p1x < cenX)
		{
			tempDeg += 3.1415926535;
		}
		else if (p1y < cenY)
		{
			tempDeg += 6.283185307;
		}
		double aDeg = tempDeg;

		tempDeg = atan((p2y - cenY) / (p2x - cenX));
		if (p2x < cenX)
		{
			tempDeg += 3.1415926535;
		}
		else if (p2y < cenY)
		{
			tempDeg += 6.283185307;
		}
		double bDeg = tempDeg;

		tempDeg = atan((p3y - cenY) / (p3x - cenX));
		if (p3x < cenX)
		{
			tempDeg += 3.1415926535;
		}
		else if (p3y < cenY)
		{
			tempDeg += 6.283185307;
		}
		double cDeg = tempDeg;

		double aPoints = 0.0;
		double bPoints = 0.0;
		double cPoints = 0.0;

		if (aDeg <= bDeg)
		{
			aPoints += 0.5;
			if (aDeg < bDeg)
			{
				aPoints += 0.5;
			}
		}
		if (aDeg <= cDeg)
		{
			aPoints += 0.5;
			if (aDeg < cDeg)
			{
				aPoints += 0.5;
			}
		}
		if (bDeg <= aDeg)
		{
			bPoints += 0.5;
			if (bDeg < aDeg)
			{
				bPoints += 0.5;
			}
		}
		if (bDeg <= cDeg)
		{
			bPoints += 0.5;
			if (bDeg < cDeg)
			{
				bPoints += 0.5;
			}
		}
		if (cDeg <= aDeg)
		{
			cPoints += 0.5;
			if (cDeg < aDeg)
			{
				cPoints += 0.5;
			}
		}
		if (cDeg <= bDeg)
		{
			cPoints += 0.5;
			if (cDeg < bDeg)
			{
				cPoints += 0.5;
			}
		}
		if (aPoints >= bPoints && aPoints >= cPoints)
		{
			if (bPoints > cPoints)
			{
				// 1 2 3
				uint16_t one = g_tri1[i];
				uint16_t two = g_tri2[i];
				uint16_t three = g_tri3[i];
				g_tri1[i] = one;
				g_tri2[i] = two;
				g_tri3[i] = three;
			}
			else
			{
				// 1 3 2
				uint16_t one = g_tri1[i];
				uint16_t two = g_tri3[i];
				uint16_t three = g_tri2[i];
				g_tri1[i] = one;
				g_tri2[i] = two;
				g_tri3[i] = three;
			}
		}
		else if (bPoints >= aPoints && bPoints >= cPoints)
		{
			if (aPoints > cPoints)
			{
				// 2 1 3
				uint16_t one = g_tri2[i];
				uint16_t two = g_tri1[i];
				uint16_t three = g_tri3[i];
				g_tri1[i] = one;
				g_tri2[i] = two;
				g_tri3[i] = three;
			}
			else
			{
				// 2 3 1
				uint16_t one = g_tri2[i];
				uint16_t two = g_tri3[i];
				uint16_t three = g_tri1[i];
				g_tri1[i] = one;
				g_tri2[i] = two;
				g_tri3[i] = three;
			}
		}
		else if (cPoints >= aPoints && cPoints >= bPoints)
		{
			if (aPoints > bPoints)
			{
				// 3 1 2
				uint16_t one = g_tri3[i];
				uint16_t two = g_tri1[i];
				uint16_t three = g_tri2[i];
				g_tri1[i] = one;
				g_tri2[i] = two;
				g_tri3[i] = three;
			}
			else
			{
				// 3 2 1
				uint16_t one = g_tri3[i];
				uint16_t two = g_tri2[i];
				uint16_t three = g_tri1[i];
				g_tri1[i] = one;
				g_tri2[i] = two;
				g_tri3[i] = three;
			}
		}
	}
}

__global__ void deactivateTris(double* x, double* y, uint16_t* g_tri1, uint16_t* g_tri2, uint16_t* g_tri3, int* g_workingIndex, int* g_length)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int q = 0; q < g_length[1]; q++)
	{
		if ((int)g_tri1[q] != i && (int)g_tri2[q] != i && (int)g_tri3[q] != i)
		{
			int ind1 = (int)g_tri1[q];
			int ind2 = (int)g_tri2[q];
			int ind3 = (int)g_tri3[q];

			double ax = x[ind1] - x[i];
			double ay = y[ind1] - y[i];
			double az = (x[ind1] - x[i]) * (x[ind1] - x[i]) + (y[ind1] - y[i]) * (y[ind1] - y[i]);
			double bx = x[ind2] - x[i];
			double by = y[ind2] - y[i];
			double bz = (x[ind2] - x[i]) * (x[ind2] - x[i]) + (y[ind2] - y[i]) * (y[ind2] - y[i]);
			double cx = x[ind3] - x[i];
			double cy = y[ind3] - y[i];
			double cz = (x[ind3] - x[i]) * (x[ind3] - x[i]) + (y[ind3] - y[i]) * (y[ind3] - y[i]);

			double i = (by * cz - bz * cy) * ax;
			double j = (bx * cz - bz * cx) * ay;
			double k = (bx * cy - by * cx) * az;

			if (i - j + k > 0)
			{
				g_workingIndex[q] = 0;
			}
		}
	}
}

void Kernel::deactivate(double* x, double* y, int detailx, int detaily, Vector3Int* triangles, int* workingIndex, int pointLength, int length)
{
	double *g_x, *g_y;
	int* lengthArray = new int[2]{ pointLength, length };
	int* g_workingIndex;
	int* g_length = lengthArray;
	uint16_t* tri1 = new uint16_t[length];
	uint16_t* tri2 = new uint16_t[length];
	uint16_t* tri3 = new uint16_t[length];
	uint16_t* g_tri1;
	uint16_t* g_tri2;
	uint16_t* g_tri3;
	for (int q = 0; q < length; q++)
	{
		tri1[q] = triangles[q].x;
		tri2[q] = triangles[q].y;
		tri3[q] = triangles[q].z;
	}

	if (cudaMalloc(&g_x, sizeof(double) * pointLength) != cudaSuccess)
	{
		std::cout << "Failed to load x list to variable" << std::endl;
		return;
	}

	if (cudaMalloc(&g_y, sizeof(double) * pointLength) != cudaSuccess)
	{
		std::cout << "Failed to load y list to variable" << std::endl;
		cudaFree(g_x);
		return;
	}

	if (cudaMalloc(&g_tri1, sizeof(uint16_t) * length) != cudaSuccess)
	{
		std::cout << "Failed to load triangle list 1 to variable" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		return;
	}

	if (cudaMalloc(&g_tri2, sizeof(uint16_t) * length) != cudaSuccess)
	{
		std::cout << "Failed to load triangle list 2 to variable" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		return;
	}

	if (cudaMalloc(&g_tri3, sizeof(uint16_t) * length) != cudaSuccess)
	{
		std::cout << "Failed to load triangle list 3 to variable" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		return;
	}

	if (cudaMalloc(&g_length, sizeof(int) * 2) != cudaSuccess)
	{
		std::cout << "Failed to load length to variable" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		cudaFree(g_tri3);
		return;
	}

	if (cudaMalloc(&g_workingIndex, sizeof(int) * length) != cudaSuccess)
	{
		std::cout << "Failed to load length to variable" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		cudaFree(g_tri3);
		cudaFree(g_length);
		return;
	}

	if (cudaMemcpy(g_x, x, sizeof(double) * pointLength, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cout << "Failed to copy x list to GPU" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		cudaFree(g_tri3);
		cudaFree(g_length);
		cudaFree(g_workingIndex);
		return;
	}

	if (cudaMemcpy(g_y, y, sizeof(double) * pointLength, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cout << "Failed to copy y list to GPU" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		cudaFree(g_tri3);
		cudaFree(g_length);
		cudaFree(g_workingIndex);
		return;
	}

	if (cudaMemcpy(g_tri1, tri1, sizeof(uint16_t) * length, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cout << "Failed to copy triangle list to GPU" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		cudaFree(g_tri3);
		cudaFree(g_length);
		cudaFree(g_workingIndex);
		return;
	}

	if (cudaMemcpy(g_tri2, tri2, sizeof(uint16_t) * length, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cout << "Failed to copy triangle list to GPU" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		cudaFree(g_tri3);
		cudaFree(g_length);
		cudaFree(g_workingIndex);
		return;
	}

	if (cudaMemcpy(g_tri3, tri3, sizeof(uint16_t) * length, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cout << "Failed to copy triangle list to GPU" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		cudaFree(g_tri3);
		cudaFree(g_length);
		cudaFree(g_workingIndex);
		return;
	}

	if (cudaMemcpy(g_length, lengthArray, sizeof(int) * 2, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cout << "Failed to copy length to GPU" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		cudaFree(g_tri3);
		cudaFree(g_length);
		cudaFree(g_workingIndex);
		return;
	}

	if (cudaMemcpy(g_workingIndex, workingIndex, sizeof(int) * length, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cout << "Failed to copy length to GPU" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		cudaFree(g_tri3);
		cudaFree(g_length);
		cudaFree(g_workingIndex);
		return;
	}

	int aLength = pointLength - 2;
	int bLength = pointLength - 1;
	int cLength = pointLength;
	if (aLength % 2 == 0)
	{
		aLength /= 2;
	}
	else if (bLength % 2 == 0)
	{
		bLength /= 2;
	}
	else if (cLength % 2 == 0)
	{
		cLength /= 2;
	}
	if (aLength % 3 == 0)
	{
		aLength /= 3;
	}
	else if (bLength % 3 == 0)
	{
		bLength /= 3;
	}
	else if (cLength % 3 == 0)
	{
		cLength /= 3;
	}

	aLength *= bLength;

	deactivateTris<<<detailx, detaily>>>(g_x, g_y, g_tri1, g_tri2, g_tri3, g_workingIndex, g_length);

	cudaDeviceSynchronize();

	if (cudaMemcpy(workingIndex, g_workingIndex, sizeof(int) * length, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		std::cout << "Failed to copy back triangle list 1" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		cudaFree(g_tri3);
		cudaFree(g_length);
		return;
	}

	cudaFree(g_x);
	cudaFree(g_y);
	cudaFree(g_tri1);
	cudaFree(g_tri2);
	cudaFree(g_tri3);
	cudaFree(g_length);

	std::cout << "Finished deactivating triangles" << std::endl;
}

void Kernel::recalccc(double* x, double* y, Vector3Int* triangles, int pointLength, int length)
{
	double *g_x, *g_y;
	int* lengthArray = new int[1]{ length };
	int* g_length = lengthArray;
	uint16_t* tri1 = new uint16_t[length];
	uint16_t* tri2 = new uint16_t[length];
	uint16_t* tri3 = new uint16_t[length];
	uint16_t* g_tri1;
	uint16_t* g_tri2;
	uint16_t* g_tri3;
	for (int q = 0; q < length; q++)
	{
		tri1[q] = triangles[q].x;
		tri2[q] = triangles[q].y;
		tri3[q] = triangles[q].z;
	}

	if (cudaMalloc(&g_x, sizeof(double) * pointLength) != cudaSuccess)
	{
		std::cout << "Failed to load x list to variable" << std::endl;
		return;
	}

	if (cudaMalloc(&g_y, sizeof(double) * pointLength) != cudaSuccess)
	{
		std::cout << "Failed to load y list to variable" << std::endl;
		cudaFree(g_x);
		return;
	}

	if (cudaMalloc(&g_tri1, sizeof(uint16_t) * length) != cudaSuccess)
	{
		std::cout << "Failed to load triangle list 1 to variable" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		return;
	}

	if (cudaMalloc(&g_tri2, sizeof(uint16_t) * length) != cudaSuccess)
	{
		std::cout << "Failed to load triangle list 2 to variable" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		return;
	}

	if (cudaMalloc(&g_tri3, sizeof(uint16_t) * length) != cudaSuccess)
	{
		std::cout << "Failed to load triangle list 3 to variable" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		return;
	}

	if (cudaMalloc(&g_length, sizeof(int)) != cudaSuccess)
	{
		std::cout << "Failed to load length to variable" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		cudaFree(g_tri3);
		return;
	}

	if (cudaMemcpy(g_x, x, sizeof(double) * pointLength, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cout << "Failed to copy x list to GPU" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		cudaFree(g_tri3);
		cudaFree(g_length);
		return;
	}

	if (cudaMemcpy(g_y, y, sizeof(double) * pointLength, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cout << "Failed to copy y list to GPU" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		cudaFree(g_tri3);
		cudaFree(g_length);
		return;
	}

	if (cudaMemcpy(g_tri1, tri1, sizeof(uint16_t) * length, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cout << "Failed to copy triangle list to GPU" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		cudaFree(g_tri3);
		cudaFree(g_length);
		return;
	}

	if (cudaMemcpy(g_tri2, tri2, sizeof(uint16_t) * length, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cout << "Failed to copy triangle list to GPU" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		cudaFree(g_tri3);
		cudaFree(g_length);
		return;
	}

	if (cudaMemcpy(g_tri3, tri3, sizeof(uint16_t) * length, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cout << "Failed to copy triangle list to GPU" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		cudaFree(g_tri3);
		cudaFree(g_length);
		return;
	}

	if (cudaMemcpy(g_length, lengthArray, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cout << "Failed to copy length to GPU" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		cudaFree(g_tri3);
		cudaFree(g_length);
		return;
	}

	int aLength = pointLength - 2;
	int bLength = pointLength - 1;
	int cLength = pointLength;
	if (aLength % 2 == 0)
	{
		aLength /= 2;
	}
	else if (bLength % 2 == 0)
	{
		bLength /= 2;
	}
	else if (cLength % 2 == 0)
	{
		cLength /= 2;
	}
	if (aLength % 3 == 0)
	{
		aLength /= 3;
	}
	else if (bLength % 3 == 0)
	{
		bLength /= 3;
	}
	else if (cLength % 3 == 0)
	{
		cLength /= 3;
	}

	aLength *= bLength;

	recalculateTriangles<<<aLength, cLength>>>(g_x, g_y, g_tri1, g_tri2, g_tri3, g_length);

	cudaDeviceSynchronize();

	if (cudaMemcpy(tri1, g_tri1, sizeof(uint16_t) * length, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		std::cout << "Failed to copy back triangle list 1" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		cudaFree(g_tri3);
		cudaFree(g_length);
		return;
	}

	if (cudaMemcpy(tri2, g_tri2, sizeof(uint16_t) * length, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		std::cout << "Failed to copy back triangle list 2" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		cudaFree(g_tri3);
		cudaFree(g_length);
		return;
	}

	if (cudaMemcpy(tri3, g_tri3, sizeof(uint16_t) * length, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		std::cout << "Failed to copy back triangle list 3" << std::endl;
		cudaFree(g_x);
		cudaFree(g_y);
		cudaFree(g_tri1);
		cudaFree(g_tri2);
		cudaFree(g_tri3);
		cudaFree(g_length);
		return;
	}

	for (int q = 0; q < length; q++)
	{
		triangles[q].x = tri1[q];
		triangles[q].y = tri2[q];
		triangles[q].z = tri3[q];
	}

	cudaFree(g_x);
	cudaFree(g_y);
	cudaFree(g_tri1);
	cudaFree(g_tri2);
	cudaFree(g_tri3);
	cudaFree(g_length);

	std::cout << "Finished reordering triangles" << std::endl;
}