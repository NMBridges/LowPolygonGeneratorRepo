#include <iostream>
#include "Matrix.cpp"
#include <SDL.h>
#include <array>
#include <SDL_image.h>
#include <thread>
#include <vector>
#include "kernel.h"

class PointCreator
{
private:

	static void deactivateTriangles(int start, int end, int length, double* x, double* y, int* workingIndex, Vector3Int* triangles)
	{
		for (int i = start; i < end; i++)
		{
			for (int q = 0; q < length; q++)
			{
				if (triangles[q].x != i && triangles[q].y != i && triangles[q].z != i)
				{
					int ind1 = triangles[q].x;
					int ind2 = triangles[q].y;
					int ind3 = triangles[q].z;

					Vector2 vec1 = Vector2(x[ind1], y[ind1]);
					Vector2 vec2 = Vector2(x[ind2], y[ind2]);
					Vector2 vec3 = Vector2(x[ind3], y[ind3]);
					Vector2 vec4 = Vector2(x[i], y[i]);
					if (isDInside(vec1, vec2, vec3, vec4))
					{
						workingIndex[q] = 0;
					}
				}
			}
		}
	}

	static void recalcTriangles(int start, int end, double* x, double* y, Vector3Int* triangles)
	{
		for (int i = start; i < end; i++)
		{
			double p1x = x[triangles[i].x];
			double p1y = y[triangles[i].x];
			double p2x = x[triangles[i].y];
			double p2y = y[triangles[i].y];
			double p3x = x[triangles[i].z];
			double p3y = y[triangles[i].z];

			double s1 = sqrt((p2x - p3x) * (p2x - p3x) + (p2y - p3y) * (p2y - p3y));
			double s2 = sqrt((p1x - p3x) * (p1x - p3x) + (p1y - p3y) * (p1y - p3y));
			double s3 = sqrt((p1x - p2x) * (p1x - p2x) + (p1y - p2y) * (p1y - p2y));

			double cenX = (s1 * p1x + s2 * p2x + s3 * p3x) / (s1 + s2 + s3);
			double cenY = (s1 * p1y + s2 * p2y + s3 * p3y) / (s1 + s2 + s3);

			double aDeg = degrees(cenX, cenY, p1x, p1y);
			double bDeg = degrees(cenX, cenY, p2x, p2y);
			double cDeg = degrees(cenX, cenY, p3x, p3y);

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
					int one = triangles[i].x;
					int two = triangles[i].y;
					int three = triangles[i].z;
					triangles[i] = Vector3Int(one, two, three);
				}
				else
				{
					// 1 3 2
					int one = triangles[i].x;
					int two = triangles[i].z;
					int three = triangles[i].y;
					triangles[i] = Vector3Int(one, two, three);
				}
			}
			else if (bPoints >= aPoints && bPoints >= cPoints)
			{
				if (aPoints > cPoints)
				{
					// 2 1 3
					int one = triangles[i].y;
					int two = triangles[i].x;
					int three = triangles[i].z;
					triangles[i] = Vector3Int(one, two, three);
				}
				else
				{
					// 2 3 1
					int one = triangles[i].y;
					int two = triangles[i].z;
					int three = triangles[i].x;
					triangles[i] = Vector3Int(one, two, three);
				}
			}
			else if (cPoints >= aPoints && cPoints >= bPoints)
			{
				if (aPoints > bPoints)
				{
					// 3 1 2
					int one = triangles[i].z;
					int two = triangles[i].x;
					int three = triangles[i].y;
					triangles[i] = Vector3Int(one, two, three);
				}
				else
				{
					// 3 2 1
					int one = triangles[i].z;
					int two = triangles[i].y;
					int three = triangles[i].x;
					triangles[i] = Vector3Int(one, two, three);
				}
			}
		}
	}

	static double degrees(double x1, double y1, double x2, double y2)
	{
		double tempDeg = atan((y2 - y1) / (x2 - x1));
		if (x2 < x1)
		{
			tempDeg += M_PI;
		}
		else if (y2 < y1)
		{
			tempDeg += 2 * M_PI;
		}
		return tempDeg;
	}

	static bool isDInside(Vector2 a, Vector2 b, Vector2 c, Vector2 d)
	{
		Vector3 matrixTop = Vector3(a.x - d.x, a.y - d.y, (a.x - d.x) * (a.x - d.x) + (a.y - d.y) * (a.y - d.y));
		Vector3 matrixMid = Vector3(b.x - d.x, b.y - d.y, (b.x - d.x) * (b.x - d.x) + (b.y - d.y) * (b.y - d.y));
		Vector3 matrixBot = Vector3(c.x - d.x, c.y - d.y, (c.x - d.x) * (c.x - d.x) + (c.y - d.y) * (c.y - d.y));
		double determ = Matrix(matrixTop, matrixMid, matrixBot).determinant();
		return (determ > 0);
	}

	static void drawColor(int start, int end, int windowWidth, int windowHeight, double* x, double* y, int usedLength, SDL_Renderer* rend, Vector3Int* usedTriangles, Vector4* colors, Vector3* pointColors)
	{
		for (int ey = start; ey < end; ey += 1)
		{
			for (int ex = 0; ex < windowWidth; ex += 1)
			{
				for (int i = 0; i < usedLength; i++)
				{
					int x1 = (int)(x[usedTriangles[i].x] * windowWidth);
					int y1 = (int)(y[usedTriangles[i].x] * windowHeight);
					int x2 = (int)(x[usedTriangles[i].y] * windowWidth);
					int y2 = (int)(y[usedTriangles[i].y] * windowHeight);
					int x3 = (int)(x[usedTriangles[i].z] * windowWidth);
					int y3 = (int)(y[usedTriangles[i].z] * windowHeight);

					if (isInTriangle(Vector2(ex, ey), Vector2(x1, y1), Vector2(x2, y2), Vector2(x3, y3)))
					{
						int b = (int)colors[i].z;
						int g = (int)colors[i].y;
						int r = (int)colors[i].x;

						pointColors[ex + ey * windowWidth] = Vector3(r, g, b);
					}
				}
			}
			std::cout << "2nd: " << ey << std::endl;
		}
	}
	
	static void scanColor(int start, int end, int windowWidth, int windowHeight, double* x, double* y, int usedLength, SDL_Surface* imageSurf, Vector3Int* usedTriangles, Vector4* colors)
	{
		for (int ey = start; ey < end; ey += 4)
		{
			for (int ex = 0; ex < windowWidth; ex += 4)
			{
				for (int i = 0; i < usedLength; i++)
				{
					int x1 = (int)(x[usedTriangles[i].x] * windowWidth);
					int y1 = (int)(y[usedTriangles[i].x] * windowHeight);
					int x2 = (int)(x[usedTriangles[i].y] * windowWidth);
					int y2 = (int)(y[usedTriangles[i].y] * windowHeight);
					int x3 = (int)(x[usedTriangles[i].z] * windowWidth);
					int y3 = (int)(y[usedTriangles[i].z] * windowHeight);

					if (isInTriangle(Vector2(ex, ey), Vector2(x1, y1), Vector2(x2, y2), Vector2(x3, y3)))
					{
						Uint32 data = getPixel(imageSurf, (int)((double)ex / (double)windowWidth * (double)imageSurf->w), (int)((double)ey / (double)windowHeight * (double)imageSurf->h));

						double count = colors[i].w;
						double totalr = colors[i].x * count;
						double totalg = colors[i].y * count;
						double totalb = colors[i].z * count;

						int b = (data >> 16) & 255;
						int g = (data >> 8) & 255;
						int r = data & 255;

						totalr += r;
						totalg += g;
						totalb += b;

						count += 1;
						totalr /= count;
						totalg /= count;
						totalb /= count;

						colors[i] = Vector4(totalr, totalg, totalb, count);
					}
				}
			}

			std::cout << ey << std::endl;
		}
	}

	static double sign(Vector2 a, Vector2 b, Vector2 c)
	{
		return (a.x - c.x) * (b.y - c.y) - (b.x - c.x) * (a.y - c.y);
	}

	static bool isInTriangle(Vector2 p, Vector2 a, Vector2 b, Vector2 c)
	{
		double d1 = sign(p, a, b);
		double d2 = sign(p, b, c);
		double d3 = sign(p, c, a);

		bool hasANegative = (d1 < 0) || (d2 < 0) || (d3 < 0);
		bool hasAPositive = (d1 > 0) || (d2 > 0) || (d3 > 0);

		return !(hasANegative && hasANegative);
	}

	static Uint32 getPixel(SDL_Surface* surface, int x, int y)
	{
		int bpp = surface->format->BytesPerPixel;

		Uint8* p = (Uint8*)surface->pixels + y * surface->pitch + x * bpp;

		switch (bpp)
		{
		case 1:
			return *p;
			break;

		case 2:
			return *(Uint16*)p;
			break;

		case 3:
			if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
				return p[0] << 16 | p[1] << 8 | p[2];
			else
				return p[0] | p[1] << 8 | p[2] << 16;
			break;

		case 4:
			return *(Uint32*)p;
			break;

		default:
			return 0;
		}
	}

	int factorial(int o)
	{
		int p = 1;
		for (int q = 1; q <= o; q++)
		{
			p *= q;
		}
		return p;
	}

	int factOverFact(int o, int m)
	{
		int p = 1;
		for (int q = m + 1; q <= o; q++)
		{
			p *= q;
		}
		return p;
	}

	int xdetail;
	int ydetail;
	int windowWidth;
	int windowHeight;
	int detailSquared;
	double* x;
	double* y;
	int length;
	int usedLength;
	int seed;
	Vector3Int* triangles;
	Vector3Int* usedTriangles;
	int* workingIndex;
	SDL_Texture* imageTex;
	SDL_Surface* imageSurf;
	Vector4* colors;

public:

	PointCreator(int width, int height, int xdet, int ydet, int seedin, SDL_Renderer *rend)
	{
		xdetail = xdet;
		ydetail = ydet;
		detailSquared = xdetail * ydetail;
		x = new double[detailSquared];
		y = new double[detailSquared];
		length = factOverFact(detailSquared, detailSquared - 3) / factorial(3);
		seed = seedin;
		triangles = new Vector3Int[length];
		workingIndex = new int[length];

		// create texture

		imageSurf = IMG_Load("assets/arnav.png");
		if (!imageSurf)
		{
			std::cout << "Failed to make surface " << std::endl;
		}
		imageTex = SDL_CreateTextureFromSurface(rend, imageSurf);
		int flags = IMG_INIT_JPG | IMG_INIT_PNG;
		if (IMG_Init(flags) != flags)
		{
			std::cout << "failed to initialize image " << std::endl;
		}
		colors = new Vector4[imageSurf->w * imageSurf->h];
		colors[0] = Vector4(0.0, 0.0, 0.0, 0.0);

		windowHeight = height;
		windowWidth = (int)(height * returnRatio());
	}

	void createPoints(int xdetailLevel, int ydetailLevel)
	{
		srand(time_t(seed));

		for (int i = 0; i < ydetailLevel; i++)
		{
			for (int q = 0; q < xdetailLevel; q++)
			{
				int item = (int)(i * xdetailLevel + q);
				double perc1 = q / ((double)xdetailLevel - 1.0);
				double perc2 = i / ((double)ydetailLevel - 1.0);
				x[item] = perc1;
				y[item] = perc2;
			}
		}
	}

	double returnRatio()
	{
		return (double)imageSurf->w / (double)imageSurf->h;
	}

	void jitterPoints(int xdetailLevel, int ydetailLevel, double amount)
	{

		for (int i = 0; i < ydetailLevel; i++)
		{
			for (int q = 0; q < xdetailLevel; q++)
			{
				int item = (int)(i * xdetailLevel + q);
				double perc1 = (q == 0 || q == xdetailLevel - 1) ? 0 : ((double)std::rand() / RAND_MAX - 0.5) * amount * 0.5 / ((double)xdetailLevel - 1.0);
				double perc2 = (i == 0 || i == ydetailLevel - 1) ? 0 : ((double)std::rand() / RAND_MAX - 0.5) * amount * 0.5 / ((double)ydetailLevel - 1.0);
				x[item] = x[item] + perc1;
				y[item] = y[item] + perc2;
				if (x[item] < 0.0)
				{
					x[item] = 0.0;
				}
				if (y[item] < 0.0)
				{
					y[item] = 0.0;
				}
				if (x[item] > 1.0)
				{
					x[item] = 1.0;
				}
				if (y[item] > 1.0)
				{
					y[item] = 1.0;
				}
			}
		}
	}

	void delegateColors(SDL_Renderer* rend)
	{
		// loops through every pixel, checks every triangle if it is in it, then adds that point to the triangle's average list

		std::thread t1(&PointCreator::scanColor, 0, (windowHeight / 8), windowWidth, windowHeight, x, y, usedLength, imageSurf, usedTriangles, std::ref(colors));
		std::thread t2(&PointCreator::scanColor, (windowHeight / 8), (windowHeight / 4), windowWidth, windowHeight, x, y, usedLength, imageSurf, usedTriangles, std::ref(colors));
		std::thread t3(&PointCreator::scanColor, (windowHeight / 4), (windowHeight * 3 / 8), windowWidth, windowHeight, x, y, usedLength, imageSurf, usedTriangles, std::ref(colors));
		std::thread t4(&PointCreator::scanColor, (windowHeight * 3 / 8), (windowHeight / 2), windowWidth, windowHeight, x, y, usedLength, imageSurf, usedTriangles, std::ref(colors));
		std::thread t5(&PointCreator::scanColor, (windowHeight / 2), (windowHeight * 5 / 8), windowWidth, windowHeight, x, y, usedLength, imageSurf, usedTriangles, std::ref(colors));
		std::thread t6(&PointCreator::scanColor, (windowHeight * 5 / 8), (windowHeight * 3 / 4), windowWidth, windowHeight, x, y, usedLength, imageSurf, usedTriangles, std::ref(colors));
		std::thread t7(&PointCreator::scanColor, (windowHeight * 3 / 4), (windowHeight * 7 / 8), windowWidth, windowHeight, x, y, usedLength, imageSurf, usedTriangles, std::ref(colors));
		std::thread t8(&PointCreator::scanColor, (windowHeight * 7 / 8), (windowHeight), windowWidth, windowHeight, x, y, usedLength, imageSurf, usedTriangles, std::ref(colors));

		t1.join();
		t2.join();
		t3.join();
		t4.join();
		t5.join();
		t6.join();
		t7.join();
		t8.join();

		// loops through every pixel and draws the averaged color

		Vector3* pointColors = new Vector3[windowWidth * windowHeight];

		std::thread t9(&PointCreator::drawColor, 0, (windowHeight / 8), windowWidth, windowHeight, x, y, usedLength, rend, usedTriangles, colors, std::ref(pointColors));
		std::thread t10(&PointCreator::drawColor, (windowHeight / 8), (windowHeight / 4), windowWidth, windowHeight, x, y, usedLength, rend, usedTriangles, colors, std::ref(pointColors));
		std::thread t11(&PointCreator::drawColor, (windowHeight / 4), (windowHeight * 3 / 8), windowWidth, windowHeight, x, y, usedLength, rend, usedTriangles, colors, std::ref(pointColors));
		std::thread t12(&PointCreator::drawColor, (windowHeight * 3 / 8), (windowHeight / 2), windowWidth, windowHeight, x, y, usedLength, rend, usedTriangles, colors, std::ref(pointColors));
		std::thread t13(&PointCreator::drawColor, (windowHeight / 2), (windowHeight * 5 / 8), windowWidth, windowHeight, x, y, usedLength, rend, usedTriangles, colors, std::ref(pointColors));
		std::thread t14(&PointCreator::drawColor, (windowHeight * 5 / 8), (windowHeight * 3 / 4), windowWidth, windowHeight, x, y, usedLength, rend, usedTriangles, colors, std::ref(pointColors));
		std::thread t15(&PointCreator::drawColor, (windowHeight * 3 / 4), (windowHeight * 7 / 8), windowWidth, windowHeight, x, y, usedLength, rend, usedTriangles, colors, std::ref(pointColors));
		std::thread t16(&PointCreator::drawColor, (windowHeight * 7 / 8), (windowHeight), windowWidth, windowHeight, x, y, usedLength, rend, usedTriangles, colors, std::ref(pointColors));

		t9.join();
		t10.join();
		t11.join();
		t12.join();
		t13.join();
		t14.join();
		t15.join();
		t16.join();

		int counter = 0;
		for (int hori = 0; hori < windowWidth; hori++)
		{
			for (int vert = 0; vert < windowHeight; vert++)
			{
				Vector3 cOLOR = pointColors[hori + vert * windowWidth];
				SDL_SetRenderDrawColor(rend, (int)cOLOR.x, (int)cOLOR.y, (int)cOLOR.z, 255);
				SDL_RenderDrawPoint(rend, hori, vert);
				counter++;
			}
		}

		std::cout << "Finished delegating colors" << std::endl;
	}

	void createTriangles(SDL_Renderer *rend)
	{
		int pointLength = detailSquared;

		std::cout << "LENGTH OF X AND Y: " << pointLength << std::endl;

		// creates triangles for every combination of three points. Eventually accelerate with the GPU

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
		
		// orders the points in each triangle set in counterclockwise order
		
		Kernel kernelCaller = Kernel();
		kernelCaller.recalccc(x, y, std::ref(triangles), pointLength, length);

		// instantiate list that keeps track of whether each triangle has other points inside it (1 = good, 0 = bad)

		workingIndex = new int[length];
		for (int i = 0; i < length; i++)
		{
			workingIndex[i] = 1;
		}

		// goes through every point and checks if it is inside any triangles; then deactivates that triangle

		kernelCaller.deactivate(x, y, xdetail, ydetail, triangles, std::ref(workingIndex), pointLength, length);
	}

	void drawTriangles(SDL_Renderer *rend)
	{
		SDL_RenderCopy(rend, imageTex, NULL, NULL);
		SDL_RenderPresent(rend);
		//SDL_SetRenderDrawColor(rend, 255, 0, 0, 255);
		
		std::cout << length << std::endl;

		usedLength = 0;

		for (int i = 0; i < length; i++)
		{
			if (workingIndex[i] == 1)
			{
				usedLength++;
				/*int x1 = (int)(x[triangles[i].x] * windowWidth);
				int y1 = (int)(y[triangles[i].x] * windowHeight);
				int x2 = (int)(x[triangles[i].y] * windowWidth);
				int y2 = (int)(y[triangles[i].y] * windowHeight);
				int x3 = (int)(x[triangles[i].z] * windowWidth);
				int y3 = (int)(y[triangles[i].z] * windowHeight);
				SDL_RenderDrawLine(rend, x1, y1, x2, y2);
				SDL_RenderDrawLine(rend, x1, y1, x3, y3);
				SDL_RenderDrawLine(rend, x3, y3, x2, y2);*/
			}
		}

		usedTriangles = new Vector3Int[usedLength];
		int cou = 0;
		for (int i = 0; i < length; i++)
		{
			if (workingIndex[i] == 1)
			{
				usedTriangles[cou] = triangles[i];
				cou++;
			}
		}

		delete[] triangles;

		std::cout << "got here" << std::endl;


	}
};