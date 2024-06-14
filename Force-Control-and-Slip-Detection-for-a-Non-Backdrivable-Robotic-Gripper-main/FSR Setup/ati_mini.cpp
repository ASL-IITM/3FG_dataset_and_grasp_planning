/*! \file x_examples.cpp
*
* Examples main(). Calls example code.
*
*/
#include "x_examples.h"
#include <stdio.h>
#include <stdlib.h>
#include "atidaq/ftconfig.h"
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cstdio>
#include <ctime>
#include <chrono>

#include <termios.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <vector>


using namespace std;

ofstream outputFile;
ofstream fs;

GCon g = 0; //var used to refer to a unique connection. A valid connection is nonzero.
int Galil_connect();
int Galil_disconnect();
void read_analog_inputs(float analog_inputs[]);
int convert_analog_to_FT(float SampleBias[6], float analog_in[], float FT[]);
int run = 1;
string ext = ".csv";
string filename;

string FormatDateTime(chrono::system_clock::time_point tp) {
    stringstream ss;
    auto t = chrono::system_clock::to_time_t(tp);
    auto tp2 = chrono::system_clock::from_time_t(t);
    if (tp2 > tp)
        t = chrono::system_clock::to_time_t(tp - chrono::seconds(1));
    ss  << put_time(localtime(&t), "%Y-%m-%d_%T");
        // << "." << setfill('0') << setw(3)
        // << (chrono::duration_cast<chrono::milliseconds>(
        //    tp.time_since_epoch()).count() % 1000);
    return ss.str();
}

string FormatTime(chrono::system_clock::time_point tp) {
    stringstream ss;
    auto t = chrono::system_clock::to_time_t(tp);
    auto tp2 = chrono::system_clock::from_time_t(t);
    if (tp2 > tp)
        t = chrono::system_clock::to_time_t(tp - chrono::seconds(1));
    ss  << put_time(localtime(&t), "%T")
        << "." << setfill('0') << setw(3)
        << (chrono::duration_cast<chrono::milliseconds>(
           tp.time_since_epoch()).count() % 1000);
    return ss.str();
}
string FormatDate(chrono::system_clock::time_point tp) {
    stringstream ss;
    auto t = chrono::system_clock::to_time_t(tp);
    auto tp2 = chrono::system_clock::from_time_t(t);
    if (tp2 > tp)
        t = chrono::system_clock::to_time_t(tp - chrono::seconds(1));
    ss  << put_time(localtime(&t), "%Y-%m-%d");
    return ss.str();
}
string CurrentTimeStr() {
    return FormatTime(chrono::system_clock::now());
}

int config_COM_port(int device, struct termios tty){
	if(tcgetattr(device, &tty) != 0) {
        printf("Error %i from tcgetattr: %s\n", errno);
        return 1;
    }

    tty.c_cflag &= ~PARENB; // Clear parity bit, disabling parity (most common)
    tty.c_cflag &= ~CSTOPB; // Clear stop field, only one stop bit used in communication (most common)
    tty.c_cflag &= ~CSIZE; // Clear all bits that set the data size 
    tty.c_cflag |= CS8; // 8 bits per byte (most common)
    tty.c_cflag &= ~CRTSCTS; // Disable RTS/CTS hardware flow control (most common)
    tty.c_cflag |= CREAD | CLOCAL; // Turn on READ & ignore ctrl lines (CLOCAL = 1)

    tty.c_lflag &= ~ICANON;
    tty.c_lflag &= ~ECHO; // Disable echo
    tty.c_lflag &= ~ECHOE; // Disable erasure
    tty.c_lflag &= ~ECHONL; // Disable new-line echo
    tty.c_lflag &= ~ISIG; // Disable interpretation of INTR, QUIT and SUSP
    tty.c_iflag &= ~(IXON | IXOFF | IXANY); // Turn off s/w flow ctrl
    tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL); // Disable any special handling of received bytes

    tty.c_oflag &= ~OPOST; // Prevent special interpretation of output bytes (e.g. newline chars)
    tty.c_oflag &= ~ONLCR; // Prevent conversion of newline to carriage return/line feed
    // tty.c_oflag &= ~OXTABS; // Prevent conversion of tabs to spaces (NOT PRESENT ON LINUX)
    // tty.c_oflag &= ~ONOEOT; // Prevent removal of C-d chars (0x004) in output (NOT PRESENT ON LINUX)

    tty.c_cc[VTIME] = 10;    // Wait for up to 1s (10 deciseconds), returning as soon as any data is received.
    tty.c_cc[VMIN] = 0;

    // Set in/out baud rate to be 9600
    cfsetispeed(&tty, B115200);
    // Save tty settings, also checking for error
    if (tcsetattr(device, TCSANOW, &tty) != 0) {
        printf("Error %i from tcsetattr: %s\n", errno);
        return 1;
    }
}

int read_com_port(int device, string port, struct termios tty, int num_bytes, char read_buf [16],string response, bool dataPopulated = false) {
	num_bytes = read(device, read_buf, sizeof(read_buf));
	if (num_bytes < 0) {
		printf("Error reading: %s", strerror(errno));
		return 1;
	} 
	for (int i = 0; i < sizeof(read_buf); i++){
		while (read_buf[i] == '<')
		{
			response = {read_buf[i+1] , read_buf[i+2] , read_buf[i+3] , read_buf[i+4]};
			dataPopulated = true;
			break;
		}
		if (dataPopulated) 
			break;
	}
	cout << "\n Response is: ------------------------------------- " << response << endl;
	close(device);
}
int main()
{
	Galil_connect();

	float analog_inputs[6];
	float bias[6]={0,0,0,0,0,0};
	// float bias[6];
	float FT[6];
	char interrupt;
	int choice;
	string ang;
	string port = "/dev/ttyUSB0";
	int n;
	int num_bytes; 
	int device;
	char read_buf [16];
	string response;
    bool dataPopulated = false;    char buffer[5];
    struct termios config;
	string curTime;

	// int 
	struct termios tty;

    // Read in existing settings, and handle any error
    

	while(run) {
		// curTime = CurrentTimeStr();
		// cout << "time " <<  curTime << endl;

		cout << "\n \n \n Please enter your choice:\n 1: Bias the sensor\n 2: Record 1000 data points. \n 11: Bias the sensor with COM. \n 3: (COM)run 1000 data points to set force.\n 4: (COM)Record 100 data points.\n 0 Exit: \n";
		cin >> choice;
		switch( choice)
		{	
			case 0:
				run = 0;
				break;
			case 1:
				read_analog_inputs(bias);
				cout << "read bias";
				printf("\nBias reading:\n");
				for (int i=0;i<6;i++)
					printf("%9.6f ",bias[i]);
				// read_com_port(device,port, tty, num_bytes,read_buf,response, dataPopulated);				
				break;
			case 11:
				device = open(port.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
				config_COM_port(device, tty);
				read_analog_inputs(bias);
				cout << "read bias";
				printf("\nBias reading:\n");
				for (int i=0;i<6;i++)
					printf("%9.6f ",bias[i]);
				num_bytes = read(device, read_buf, sizeof(read_buf));
				if (num_bytes < 0) {
					printf("Error reading: %s", strerror(errno));
					return 1;
				} 
				for (int i = 0; i < sizeof(read_buf); i++){
					while (read_buf[i] == '<')
					{
						response = {read_buf[i+1] , read_buf[i+2] , read_buf[i+3] , read_buf[i+4]};
						dataPopulated = true;
						break;
					}
					if (dataPopulated) 
						break;
				}
				cout << "\n Response is: ------------------------------------- " << response << endl;
				close(device);				
				break;
			case 2:
					cout << "\n Enter the corresponding angle value \n";
					cin >> ang;
					curTime = FormatDateTime(chrono::system_clock::now());
				
					filename = ang + "_" + curTime + ext;
					outputFile.open(filename);
					outputFile << "No" << "," << "time" << "," << "Fx" <<  "," << "Fy" <<  "," << "Fz" <<  "," << "Tx" <<  "," << "Ty" <<  "," << "Tz"  << "," << "fsr" << endl;
					/* interrupt = getchar();
					if (interrupt == 27){
						break;
					} */
					for (int run = 0; run<1000; run++){
						// device = open(port.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
						// config_COM_port(device, tty);

						read_analog_inputs(analog_inputs);
				
						curTime = FormatTime(chrono::system_clock::now());
						outputFile << run + 1  << "," << curTime << ",";
						convert_analog_to_FT(bias, analog_inputs, FT);
						outputFile << fixed << setprecision(6) << FT[0] << "," << FT[1] << "," << FT[2] << FT[3] << "," << FT[4] << "," << FT[5] << endl;
						// for(int i=0;i<6;i++)
						// {
						// 	//cout << "AN[" << i+1 << "]=" << analog_inputs[i] << " , " ;
						// 	outputFile << analog_inputs[0] << "," << analog_inputs[1] << "," << analog_inputs[2] << analog_inputs[3] << "," << analog_inputs[4] << "," << analog_inputs[5] << endl;
						// }
					}
					outputFile.close(); 
					break;
			case 3:
					// interrupt = getchar();
					// if (interrupt == 27){
					// 	break;
					// }
					for (int run = 0; run<1000; run++){
						
						device = open(port.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
						config_COM_port(device, tty);

						read_analog_inputs(analog_inputs);
						convert_analog_to_FT(bias, analog_inputs, FT);
						// for(int i=0;i<6;i++)
						// {
						// 	//cout << "AN[" << i+1 << "]=" << analog_inputs[i] << " , " ;
						// 	outputFile << analog_inputs[0] << "," << analog_inputs[1] << "," << analog_inputs[2] << analog_inputs[3] << "," << analog_inputs[4] << "," << analog_inputs[5] << endl;
						// }

						num_bytes = read(device, read_buf, sizeof(read_buf));
						if (num_bytes < 0) {
							printf("Error reading: %s", strerror(errno));
							return 1;
						} 
						for (int i = 0; i < sizeof(read_buf); i++){
							// cout <<read_buf[i] << endl;
							while (read_buf[i] == '<')
							{
								response = {read_buf[i+1] , read_buf[i+2] , read_buf[i+3] , read_buf[i+4]};
								dataPopulated = true;
								break;
							}
							if (dataPopulated) 
								break;
						}
						cout << "Response is: ----------------------------------------------------------------------  " << response << endl;						
						close(device);
						// interrupt = getchar();
						// if (interrupt == 27){
						// 	break;
						// }
					}
					break;
			case 4:
					cout << "\n Enter the corresponding mN value \n";
					cin >> ang;
					curTime = FormatDateTime(chrono::system_clock::now());
					filename = ang + "_" + curTime + ext;
					outputFile.open(filename);
					outputFile << "No" << "," << "time" << "," << "Fx" <<  "," << "Fy" <<  "," << "Fz" <<  "," << "Tx" <<  "," << "Ty" <<  "," << "Tz"  << "," << "fsr" << endl;
					/* interrupt = getchar();
					if (interrupt == 27){
						break;
					} */
					for (int run = 0; run<100; run++){
						device = open(port.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
						config_COM_port(device, tty);

						read_analog_inputs(analog_inputs);
						curTime = FormatTime(chrono::system_clock::now());
						outputFile << run + 1  << "," << curTime << ",";
						convert_analog_to_FT(bias, analog_inputs, FT);
						outputFile << fixed << setprecision(6) << FT[0] << "," << FT[1] << "," << -(FT[2]) << ","  << FT[3] << "," << FT[4] << "," << FT[5] << ",";
						// for(int i=0;i<6;i++)
						// {
						// 	//cout << "AN[" << i+1 << "]=" << analog_inputs[i] << " , " ;
						// 	outputFile << analog_inputs[0] << "," << analog_inputs[1] << "," << analog_inputs[2] << analog_inputs[3] << "," << analog_inputs[4] << "," << analog_inputs[5] << endl;
						// }
						
						num_bytes = read(device, read_buf, sizeof(read_buf));
						if (num_bytes < 0) {
							printf("Error reading: %s", strerror(errno));
							return 1;
						} 
						for (int i = 0; i < sizeof(read_buf); i++){
							while (read_buf[i] == '<')
							{
								response = {read_buf[i+1] , read_buf[i+2] , read_buf[i+3] , read_buf[i+4]};
								dataPopulated = true;
								break;
							}
							if (dataPopulated) 
								break;
						}
						cout << "Response is: ----------------------------------------------------------------------  " << response << endl;						
						outputFile << response << endl;
						close(device);
					}
					outputFile.close(); 
					break;
			default:
				break;
		}
		choice = 0;
	}
		
		/* read_analog_inputs(analog_inputs);
		for(int i=0;i<6;i++)
		{
			//cout << "AN[" << i+1 << "]=" << analog_inputs[i] << " , " ;
		}
//		cout<<endl;
		convert_analog_to_FT(bias, analog_inputs, FT); */

	Galil_disconnect();
}

int Galil_connect()
{ 
	int buf_size = G_SMALL_BUFFER;
	char buf[G_SMALL_BUFFER]; //traffic buffer
	


	try
	{
	
		x_e(GVersion(buf, sizeof(buf))); //library version
		cout << "Library versions: " << buf << "\n";
		cout << "Connecting to the controller\n";

		//Basic connections
		 (GOpen("192.168.1.100 --subscribe ALL", &g)); //connect and assign a value to g. 
		//x_e(GOpen("/dev/galilpci0 --subscribe ALL", &g)); 
		//x_e(GOpen("COM1 --baud 115200 --subscribe ALL", &g));
		
		x_e(GInfo(g, buf, sizeof(buf))); //grab connection string


		return 1;

	}//try
	catch (exception& e)
	{
		cerr << "Unexpected exception... Kaboom. " << e.what() << endl;
		if (g) GClose(g); g = 0; //close g
	}
	catch (...)
	{
		cout << "Unexpected error... Kaboom." << endl;
		if (g) GClose(g); g = 0; //close g
	}
	return 0;

}
int Galil_disconnect()
{

		if (g) x_e(GClose(g)); g = 0; //close g 
		return 1;
		
}
void read_analog_inputs(float analog_inputs[])
{
	char buf[100]; //traffic buffer
	double val;
	for(int i=0;i<6;i++)
	{
		sprintf(buf, "MG @AN[%d]", i+1);
		//cout << buf << " " ;
		x_e(GCmdD(g, buf, &val));
		analog_inputs[i] = (float)val;
		//cout << "AN[" << i+1 << "]=" << analog_inputs[i] << " , " ;
	}
}






//This is ATI source code

int convert_analog_to_FT(float SampleBias[6], float SampleReading[6], float FT[6]) {


	Calibration *cal;		// struct containing calibration information
	unsigned short i;       // loop variable used to print results
	short sts;              // return value from functions

	// In this sample application, readings have been hard-coded for demonstration.
	// A working application would retrieve these vectors from a data acquisition system.
	// PLEASE NOTE:  There are 7 elements in the bias and reading arrays.  The first 6 are
	//	the gage values you would retrieve from the transducer.  The seventh represents the 
	//	thermistor gage, which is only meaningful if your sensor uses software temperature
	//	compensation.  If your sensor uses hardware temperature compensation (all sensors
	//	sold after mid-2003 do), the last value is meaningless, and you can just use a 6 element
	//	array.
//	cout<< "Please enter to get the current values as Bias" <<endl;
	// float SampleBias[6]={0.037200,0.140700,-0.220000,-0.212400,0.363800,-0.161100};
	// float SampleReading[6]={-3.2863,0.3875,-3.4877,0.4043,-3.9341,0.5474};

	// This sample transform includes a translation along the Z-axis and a rotation about the X-axis.
//	float SampleTT[6]={0,0,20,45,0,0};

	// create Calibration struct
	cal=createCalibration("/home/inspire_01/Downloads/ATI_Galil/FT43673/FT43673.cal",1);
	if (cal==NULL) {
		printf("\nSpecified calibration could not be loaded.\n");
		scanf(".");
		return 0;
	}
	
	// Set force units.
	// This step is optional; by default, the units are inherited from the calibration file.
	sts=SetForceUnits(cal,"N");
	switch (sts) {
		case 0: break;	// successful completion
		case 1: printf("Invalid Calibration struct"); return 0;
		case 2: printf("Invalid force units"); return 0;
		default: printf("Unknown error"); return 0;
	}


	// Set torque units.
	// This step is optional; by default, the units are inherited from the calibration file.
	sts=SetTorqueUnits(cal,"N-m");
	switch (sts) {
		case 0: break;	// successful completion
		case 1: printf("Invalid Calibration struct"); return 0;
		case 2: printf("Invalid torque units"); return 0;
		default: printf("Unknown error"); return 0;
	}


	// store an unloaded measurement; this removes the effect of tooling weight
	Bias(cal,SampleBias);

	// convert a loaded measurement into forces and torques
	ConvertToFT(cal,SampleReading,FT);
	
	
	// print results
	printf("\nBias reading:\n");
	for (i=0;i<6;i++)
		printf("%9.6f ",SampleBias[i]);
	printf("\nMeasurement:\n");
	for (i=0;i<6;i++)
		printf("%9.6f ",SampleReading[i]);
	printf("\nResult:---------------------------------------------------\t");
	for (i=0;i<6;i++)
		printf("%9.6f ",FT[i]);
		// if (i != 5){
		// 	outputFile << FT[0] << "," << FT[1] << "," << FT[2] << FT[3] << "," << FT[4] << "," << FT[5] << endl;
		// } else {}
	// cout << run << endl;
	// outputFile << fixed << setprecision(6) << FT[0] << "," << FT[1] << "," << FT[2] << FT[3] << "," << FT[4] << "," << FT[5] << ",";
	// free memory allocated to Calibration structure
	destroyCalibration(cal);
	printf("\n");

	return 0;
}

