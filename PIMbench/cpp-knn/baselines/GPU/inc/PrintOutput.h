#ifndef PRINT_OUTPUT_H
#define PRINT_OUTPUT_H

double gigaProcessedInSec;
double timeInMsec=1;
double  sizeInGBytes=0;
double  outPutSizeInGBytes=0;
int nIter=1;
bool noWarmUp=false;

//--------------------------------
#define MultiLineExternalVariablesMacros \
extern double gigaProcessedInSec; \
extern double timeInMsec;\
extern double  sizeInGBytes;\
extern double  outPutSizeInGBytes;\
extern int nIter;\
extern bool noWarmUp;


void printOutput(){
		//Anay change in the format of the output should also change the format of the parser script
		/*
        printf(
            "GBInputPerSecond = %.10f \n ",
            gigaProcessedInSec);
            */
	   printf("nIter = %d \n",nIter);

        printf(
             "TimeOfProcessINmsec = %.10f \n ",
             timeInMsec);
        printf(
            "GBInputSize = %.10f \n ",
            sizeInGBytes);
        printf(
                   "GBOutputSize = %.10f \n ",
                    outPutSizeInGBytes);
        printf(
            "GBOutputPerSecond = %.10f \n ",
            (outPutSizeInGBytes)/(timeInMsec*1.0e-3));

        printf(
            "GBInputPerSecond = %.10f \n ",
            (sizeInGBytes)/(timeInMsec*1.0e-3));
}
#endif
