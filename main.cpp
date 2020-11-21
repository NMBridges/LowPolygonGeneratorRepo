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
		int qualityX = 46;
		int qualityY = 26;
		int winHei = 800;
		SDL_Window *window = SDL_CreateWindow("LowPolyGenerator", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, winHei, winHei, SDL_WINDOW_SHOWN);
		SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);

		SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
		SDL_RenderClear(renderer);

		PointCreator pcTEMP = PointCreator(1000, winHei, qualityX, qualityY, renderer);
		PointCreator pc = PointCreator((int)(winHei * pcTEMP.returnRatio()), winHei, qualityX, qualityY, renderer);

		SDL_SetWindowSize(window, (int)(winHei * pc.returnRatio()), winHei);
		SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
		SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
		SDL_RenderPresent(renderer);

		pc.createPoints(qualityX, qualityY);
		pc.jitterPoints(qualityX, qualityY, 1.5);
		pc.drawPoints(renderer);
		pc.createTriangles(renderer);
		pc.drawTriangles(renderer);
		pc.delegateColors(renderer);
		SDL_RenderPresent(renderer);
		SDL_Delay(150000);
	}

	return EXIT_SUCCESS;
}