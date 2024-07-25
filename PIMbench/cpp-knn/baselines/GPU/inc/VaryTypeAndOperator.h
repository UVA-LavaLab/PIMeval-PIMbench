#ifndef VARY_TYPE_AND_OPERATOR_H
#define VARY_TYPE_AND_OPERATOR_H



	#ifdef OPERATION


		#ifdef AND_OP
		#define OPERATOR &
		#endif

		#ifdef XOR_OP
		#define OPERATOR ^
		#endif

		#ifdef MUL_OP
		#define OPERATOR *
		#endif

		#ifdef DIV_OP
		#define OPERATOR /
		#endif

		#ifdef MOD_OP
		#define OPERATOR %
		#endif

		#ifdef ADD_OP
		#define OPERATOR +
		#endif


	#else
	#define OPERATOR  *
	#endif
//---------------------------------------------------------------
	#ifdef PRIMITVE_TYPE
		#define NumberOfOperands 3
	#else
		#define NumberOfOperands 2
	#endif
//--------------------------------------------------------------
//PART SIZE DETERMINES the
#ifdef TYPEOFVAR
	#ifdef  DOUBLE_T
	  typedef double     NumericT;
		 #define  PART_SIZE  (SHARED_MEM_SIZE/(NumberOfOperands*8))
	#endif
	#ifdef FLOAT_T
	  typedef float     NumericT;
	 #define  PART_SIZE  (SHARED_MEM_SIZE/(NumberOfOperands*4))
	#endif
	#ifdef CHAR_T
	  typedef char     NumericT;
	 #define  PART_SIZE  (SHARED_MEM_SIZE/(NumberOfOperands))
	#endif
	#ifdef SHORT_INT_T
	  typedef short int      NumericT;
	 #define  PART_SIZE  (SHARED_MEM_SIZE/(NumberOfOperands*2))
	#endif
	#ifdef LONG_INT_T
	  typedef long int      NumericT;
	 #define  PART_SIZE  (SHARED_MEM_SIZE/(NumberOfOperands*8))
	#endif
	#ifdef INT_T
	 #define  PART_SIZE  (SHARED_MEM_SIZE/(NumberOfOperands*4))
	  typedef int      NumericT;
	#endif

	#else
	  typedef float       NumericT;
	  #define  PART_SIZE  (SHARED_MEM_SIZE/(NumberOfOperands*4))  //three array and each element four bytes
	#endif
#endif
