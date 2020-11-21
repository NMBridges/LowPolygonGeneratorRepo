#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GPU_Acceleration.h"

__global__ void GPU_Acceleration_FindWorkingTriangles(unsigned char* Image, int Channels);

void GPU_Acceleration(unsigned char* Input_Image, int Height, int Width, int Channels)
{
	unsigned char* Dev_Input_Image = NULL;

	// allocate the memory in GPU

	cudaMalloc((void**)&Dev_Input_Image, Height * Width * Channels);

	// copy data from CPU to GPU

	cudaMemcpy(Dev_Input_Image, Input_Image, Height * Width * Channels, cudaMemcpyHostToDevice);

	dim3(Width, Height);
	GPU_Acceleration << <Grid_Image, 1 >> > (Dev_Input_Image, Channels);

	// copy processed data back to CPU from GPU

	cudaMemcpy(Input_Image, Dev_Input_Image, Height * Width * Channels, cudaMemcpyDeviceToHost);

	// free GPU memory

	cudaFree(Dev_Input_Image);
}

/*void GPU_Acceleration_Triangles(Vector3Int* triangles, double* x, double* y, int length, int pointLength)
{

	std::cout << "LENGTH OF X AND Y: " << pointLength << std::endl;

	// creates triangles for every combination of three points

	int triCoun = 0;
	for (int i = 0; i < pointLength - 2; i++)
	{
		for (int j = i + 1; j < pointLength - 1; j++)
		{
			for (int k = j + 1; k < pointLength; k++)
			{
				triangles[triCoun] = Vector3Int(i, j, k);
				//std::cout << triCoun << ": " << triangles[triCoun].toString() << std::endl;
				triCoun++;
			}
		}
	}


	// MAKE EVERYTHING IN ONE FUNCTION AND ONLY RETURN THE FINAL TRIANGLE LIST



	// orders the points in each triangle set in counterclockwise order

	std::thread p1(&PointCreator::recalcTriangles, 0, (int)floor(length / 8), x, y, std::ref(triangles));
	std::thread p2(&PointCreator::recalcTriangles, (int)floor(length / 8), (int)floor(length / 4), x, y, std::ref(triangles));
	std::thread p3(&PointCreator::recalcTriangles, (int)floor(length / 4), (int)floor(length * 3 / 8), x, y, std::ref(triangles));
	std::thread p4(&PointCreator::recalcTriangles, (int)floor(length * 3 / 8), (int)floor(length / 2), x, y, std::ref(triangles));
	std::thread p5(&PointCreator::recalcTriangles, (int)floor(length / 2), (int)floor(length * 5 / 8), x, y, std::ref(triangles));
	std::thread p6(&PointCreator::recalcTriangles, (int)floor(length * 5 / 8), (int)floor(length * 3 / 4), x, y, std::ref(triangles));
	std::thread p7(&PointCreator::recalcTriangles, (int)floor(length * 3 / 4), (int)floor(length * 7 / 8), x, y, std::ref(triangles));
	std::thread p8(&PointCreator::recalcTriangles, (int)floor(length * 7 / 8), length, x, y, std::ref(triangles));

	p1.join();
	p2.join();
	p3.join();
	p4.join();
	p5.join();
	p6.join();
	p7.join();
	p8.join();

	// instantiate list that keeps track of whether each triangle has other points inside it (1 = good, 0 = bad)

	workingIndex = new int[length];
	for (int i = 0; i < length; i++)
	{
		workingIndex[i] = 1;
	}

	// goes through every point and checks if it is inside any triangles; then deactivates that triangle

	std::thread p9(&PointCreator::deactivateTriangles, 0, (int)floor(pointLength / 8), length, x, y, std::ref(workingIndex), std::ref(triangles));
	std::thread p10(&PointCreator::deactivateTriangles, (int)floor(pointLength / 8), (int)floor(pointLength / 4), length, x, y, std::ref(workingIndex), std::ref(triangles));
	std::thread p11(&PointCreator::deactivateTriangles, (int)floor(pointLength / 4), (int)floor(pointLength * 3 / 8), length, x, y, std::ref(workingIndex), std::ref(triangles));
	std::thread p12(&PointCreator::deactivateTriangles, (int)floor(pointLength * 3 / 8), (int)floor(pointLength / 2), length, x, y, std::ref(workingIndex), std::ref(triangles));
	std::thread p13(&PointCreator::deactivateTriangles, (int)floor(pointLength / 2), (int)floor(pointLength * 5 / 8), length, x, y, std::ref(workingIndex), std::ref(triangles));
	std::thread p14(&PointCreator::deactivateTriangles, (int)floor(pointLength * 5 / 8), (int)floor(pointLength * 3 / 4), length, x, y, std::ref(workingIndex), std::ref(triangles));
	std::thread p15(&PointCreator::deactivateTriangles, (int)floor(pointLength * 3 / 4), (int)floor(pointLength * 7 / 8), length, x, y, std::ref(workingIndex), std::ref(triangles));
	std::thread p16(&PointCreator::deactivateTriangles, (int)floor(pointLength * 7 / 8), pointLength, length, x, y, std::ref(workingIndex), std::ref(triangles));

	p9.join();
	p10.join();
	p11.join();
	p12.join();
	p13.join();
	p14.join();
	p15.join();
	p16.join();
}*/

__global__ void GPU_Acceleration_FindWorkingTriangles(unsigned char* Image, int Channels)
{
	
}