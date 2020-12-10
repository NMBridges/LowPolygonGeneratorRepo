// Created by Nolan Bridges on 11/15/20.
// Copyright © 2020 NiMBLe Interactive. All rights reserved.

#include <iostream>
#include "PointCreator.cpp"

int main(int argc, char *argv[])
{
	if (SDL_Init(SDL_INIT_EVERYTHING) < 0)
	{
		std::cout << "SDL couldn't initialize" << std::endl;
	}
	else
	{
		int qualityX = 38;
		int qualityY = 30;
		int winHei = 800;
		int seed = 5.1;
		int colorQuality = 1;
		std::string imageTitle = "camera3";
		std::string imageExtension = ".jpg";
		SDL_Window *window = SDL_CreateWindow("LowPolyGenerator", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, winHei, winHei, SDL_WINDOW_SHOWN);
		SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);

		SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
		SDL_RenderClear(renderer);

		PointCreator pcTEMP = PointCreator(1000, winHei, qualityX, qualityY, seed, colorQuality, imageTitle, imageExtension, renderer, false);
		PointCreator pc = PointCreator((int)(winHei * pcTEMP.returnRatio()), winHei, qualityX, qualityY, seed, colorQuality, imageTitle, imageExtension, renderer, true);

		SDL_SetWindowSize(window, (int)(winHei * pc.returnRatio()), winHei);
		SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
		SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
		SDL_RenderPresent(renderer);
		
		int first = SDL_GetTicks();
		pc.createPoints(qualityX, qualityY);
		pc.jitterPoints(qualityX, qualityY, 1.5);
		pc.createTriangles(renderer);
		pc.drawTriangles(renderer);
		pc.delegateColors(renderer);
		SDL_RenderPresent(renderer);
		pc.saveImage(renderer, window);
		std::cout << "Final time: " << (((double)(SDL_GetTicks() - first) / 1000.0)) << std::endl;
		SDL_Delay(150000);
	}

	return EXIT_SUCCESS;
}