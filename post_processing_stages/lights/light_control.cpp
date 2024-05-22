// light_control.cpp
#include "light_control.hpp"
#include <iostream>
#include <csignal>
#include <iostream>
#include <pigpio.h>
const int LED = 21;

void lightOn()
{
	std::cout << "Light is turned ON" << std::endl;
	// Add your hardware-specific code here to turn the light on
	if (gpioInitialise() == PI_INIT_FAILED) {
      		std::cout << "ERROR: Failed to initialize the GPIO interface." << std::endl;
      	return 1;
   	}
	gpioSetMode(LED, PI_OUTPUT);
	gpioWrite(LED, PI_HIGH);
}

void lightOff()
{
	std::cout << "Light is turned OFF" << std::endl;
	// Add your hardware-specific code here to turn the light off
	gpioSetMode(LED, PI_OUTPUT);
	gpioWrite(LED, PI_LOW);
}
