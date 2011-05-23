#include "itkImage.h"
#include "itkImageIOBase.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "itkCastImageFilter.h"
#include "itkGradientMagnitudeRecursiveGaussianImageFilter.h"
#include "itkSigmoidImageFilter.h"
#include "itkFastMarchingImageFilter.h"
#include "itkGeodesicActiveContourLevelSetImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkSimilarityIndexImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include <itkSmoothingRecursiveGaussianImageFilter.h>




const unsigned int maxDimension = 3;

template<class TInputPixel, unsigned int VDimension> 
int Execute(int argc, char* argv[]);


// Get the PixelType, the ComponentType and the number of dimensions 
// from the fileName 
void GetImageType (std::string							fileName, 
									 itk::ImageIOBase::IOPixelType&		pixelType, 
									 itk::ImageIOBase::IOComponentType&	componentType,
									 unsigned int&						noOfDimensions) 
{ 
	typedef itk::Image<unsigned char, maxDimension>		ImageType; 
	typedef itk::ImageFileReader<ImageType>				ReaderType;
	
	ReaderType::Pointer  reader = ReaderType::New(); 
	reader->SetFileName( fileName ); 
	reader->UpdateOutputInformation(); 
	
	pixelType = reader->GetImageIO()->GetPixelType(); 
	componentType = reader->GetImageIO()->GetComponentType(); 
	noOfDimensions = reader->GetImageIO()->GetNumberOfDimensions();
} 




void Usage(char* argv[])
{
	std::cerr << "Usage: " << std::endl
	<< argv[0] << std::endl
	<< " <input image file>" << std::endl
	<< " sigmoid alpha" << std::endl
	<< " sigmoid beta" << std::endl
	<< " Seed position X " << std::endl
	<< " Seed position Y " << std::endl
	<< " Seed position Z " << std::endl
	<< " Negative distance from the seed " << std::endl
	<< " Propagation Scaling " << std::endl
	<< " Curvature Scaling " << std::endl
	<< " Advection Scaling " << std::endl
	<< " Derivative Sigma " << std::endl
	<< " Max RMS " << std::endl
	<< " Max number of iterations " << std::endl
	<< " Grad Magnitude FileName " << std::endl
	<< " Output FileName " << std::endl
	<< " Initial FileName " << std::endl;
	std::cerr << "------------------ EXAMPLE CALL ------------------" << std::endl;
	std::cerr << argv[0] << " "
	"/users/feth/Documents/Work/Data/Sinergia/Alan/CroppedSoma/100916RMS04TTX_17-29-17_PMT_TP07.ome.nrrd 51 28 21 -8.0 1.0 0.1 0.5 0.0 0.03 200 /users/feth/Documents/Work/Data/Sinergia/Alan/CroppedSoma/100916RMS04TTX_17-29-17_PMT_TP07.omeGradMagnitue.nrrd /users/feth/Documents/Work/Data/Sinergia/Alan/CroppedSoma/100916RMS04TTX_17-29-17_PMT_TP07.omeOut.nrrd /users/feth/Documents/Work/Data/Sinergia/Alan/CroppedSoma/100916RMS04TTX_17-29-17_PMT_TP07.omeInitial.nrrd"
	<< std::endl << std::endl;
}




int main(int argc, char* argv[])
{
	
	if(argc < 17)
	{
		Usage(argv);		
		return EXIT_FAILURE;
	}
	
	itk::ImageIOBase::IOPixelType		pixelType; 
	itk::ImageIOBase::IOComponentType	componentType; 
	unsigned int						noOfDimensions;
	
	try 
	{ 
		GetImageType(argv[1], pixelType, componentType, noOfDimensions); 
		
		
		switch( noOfDimensions ) 
		{
			case 2: 
				switch (componentType) 
			{ 
				case itk::ImageIOBase::UCHAR: 
					return Execute<unsigned char, 2>( argc, argv); 
					break; 
					//				case itk::ImageIOBase::CHAR: 
					//					return Execute<char, 2>( argc, argv ); 
					//					break; 
					//				case itk::ImageIOBase::USHORT: 
					//					return Execute<unsigned short, 2>( argc, argv ); 
					//					break; 
					//				case itk::ImageIOBase::SHORT: 
					//					return Execute<short, 2>( argc, argv ); 
					//					break; 
					//				case itk::ImageIOBase::UINT: 
					//					return Execute<unsigned int, 2>( argc, argv ); 
					//					break; 
					//				case itk::ImageIOBase::INT: 
					//					return Execute<int, 2>( argc, argv ); 
					//					break; 
					//				case itk::ImageIOBase::ULONG: 
					//					return Execute<unsigned long, 2>( argc, argv ); 
					//					break; 
					//				case itk::ImageIOBase::LONG: 
					//					return Execute<long, 2>( argc, argv ); 
					//					break; 
					//				case itk::ImageIOBase::DOUBLE: 
					//					return Execute<double, 2>( argc, argv ); 
					//					break; 
				case itk::ImageIOBase::FLOAT: 
					return Execute<float, 2>( argc, argv ); 
					break; 
					
				case itk::ImageIOBase::UNKNOWNCOMPONENTTYPE: 
				default: 
					std::cout << "Unknown pixel component type" << std::endl; 
					break; 
			} 
				break;
			case 3: 
				switch (componentType) 
			{ 
				case itk::ImageIOBase::UCHAR: 
					return Execute<unsigned char, 3>( argc, argv); 
					break; 
				case itk::ImageIOBase::CHAR: 
					return Execute<char, 3>( argc, argv ); 
					break; 
					//				case itk::ImageIOBase::USHORT: 
					//					return Execute<unsigned short, 3>( argc, argv ); 
					//					break; 
					//				case itk::ImageIOBase::SHORT: 
					//					return Execute<short, 3>( argc, argv ); 
					//					break; 
					//				case itk::ImageIOBase::UINT: 
					//					return Execute<unsigned int, 3>( argc, argv ); 
					//					break; 
					//				case itk::ImageIOBase::INT: 
					//					return Execute<int, 3>( argc, argv ); 
					//					break; 
					//				case itk::ImageIOBase::ULONG: 
					//					return Execute<unsigned long, 3>( argc, argv ); 
					//					break; 
					//				case itk::ImageIOBase::LONG: 
					//					return Execute<long, 3>( argc, argv ); 
					//					break;
					//				case itk::ImageIOBase::DOUBLE: 
					//					return Execute<double, 3>( argc, argv ); 
					//					break; 
				case itk::ImageIOBase::FLOAT: 
					return Execute<float, 3>( argc, argv ); 
					break; 
				case itk::ImageIOBase::UNKNOWNCOMPONENTTYPE: 
				default: 
					std::cout << "Unknown pixel component type" << std::endl; 
					break; 
			} 
				break;
			default: 
				std::cout << "Only dimensions 2D and 3D are supported currently. "  
				<< "Add the routines to extend it." << std::endl; 
				break; 
		}
	} 
	catch( itk::ExceptionObject &excep) 
	{ 
		std::cerr << argv[0] << ": exception caught !" << std::endl; 
		std::cerr << excep << std::endl; 
		return EXIT_FAILURE; 
	} 
	return EXIT_SUCCESS;
	
}



// Main code goes here! 
template<class TInputPixel, unsigned int VDimension> 
int Execute(int argc, char* argv[])
{
	// Define the dimension of the images
	const unsigned int Dimension = VDimension;
	
	// Typedefs
	typedef TInputPixel																			InputPixelType;
	typedef itk::Image<InputPixelType,Dimension>						InputImageType;
	
	
	typedef float																						OutputPixelType;
	typedef itk::Image<OutputPixelType,Dimension>						OutputImageType;
	typedef OutputImageType																	InternalImageType;
	
	typedef itk::ImageFileReader<InputImageType>						FileReaderType;
	typedef itk::ImageFileWriter<OutputImageType>						FileWriterType;
	
	// Parse the input arguments.
	unsigned int argumentOffset = 1;
	std::string inputImageFilePath = argv[argumentOffset++];
	
	typename FileReaderType::Pointer imageReader = FileReaderType::New();
	imageReader->SetFileName(inputImageFilePath);
	try
	{
		imageReader->Update();
	}
	catch (itk::ExceptionObject &ex)
	{
		std::cout << ex << std::endl;
		return EXIT_FAILURE;
	}
	
	//std::cout << "reader:" << imageReader->GetOutput()->GetSpacing() << std::endl;
	
	
	/**
   * Create an edge potential map.
   * First compute the image gradient magnitude using a derivative of gaussian filter.
   * Then apply a sigmoid function to the gradient magnitude.
   */
  typedef itk::CastImageFilter< InputImageType, InternalImageType > CastFilterType;
  typename CastFilterType::Pointer caster = CastFilterType::New();
  caster->SetInput( imageReader->GetOutput() );
	//std::cout << "caster:" << caster->GetOutput()->GetSpacing() << std::endl;
	
	typedef itk::SmoothingRecursiveGaussianImageFilter
	< InternalImageType, InternalImageType > SmoothingImageType;
	
	
	typename SmoothingImageType::Pointer smooth = SmoothingImageType::New();
	smooth->SetInput( caster->GetOutput() );
	smooth->SetSigma(1.0);
	//std::cout << "smooth:" << smooth->GetOutput()->GetSpacing() << std::endl;
	
	
	typedef itk::SigmoidImageFilter< InternalImageType, InternalImageType >
	SigmoidFilterType;
  typename SigmoidFilterType::Pointer sigmoid = SigmoidFilterType::New();
  sigmoid->SetOutputMinimum( 0.0 );
  sigmoid->SetOutputMaximum( 1.0 );
  sigmoid->SetAlpha( atof( argv[argumentOffset++] ) );
  sigmoid->SetBeta( atof( argv[argumentOffset++] ) );
  sigmoid->SetInput( smooth->GetOutput() );

	//std::cout << "sigmoid:" << sigmoid->GetOutput()->GetSpacing() << std::endl;
	
	/**
   * Create an initial level.
   * Use fast marching to create an signed distance from a seed point.
   */
  typedef itk::FastMarchingImageFilter<InternalImageType> FastMarchingFilterType;
  typename FastMarchingFilterType::Pointer fastMarching = FastMarchingFilterType::New();
	
  typedef typename FastMarchingFilterType::NodeContainer NodeContainer;
  typedef typename FastMarchingFilterType::NodeType      NodeType;
	
  typename NodeContainer::Pointer seeds = NodeContainer::New();
	
  // Choose an initial contour that overlaps the square to be segmented.
  typename InternalImageType::IndexType seedPosition;
  seedPosition[0] = atoi( argv[argumentOffset++] );
  seedPosition[1] = atoi( argv[argumentOffset++] );
	seedPosition[2] = atoi( argv[argumentOffset++] );
	
  NodeType node;
  node.SetValue( atof( argv[argumentOffset++] ) );
  node.SetIndex( seedPosition );
	
  seeds->Initialize();
  seeds->InsertElement( 0, node );
	
  fastMarching->SetTrialPoints( seeds );
  fastMarching->SetSpeedConstant( 1.0 );
  fastMarching->SetOutputSize( imageReader->GetOutput()->GetBufferedRegion().GetSize() );
	fastMarching->SetOutputSpacing( imageReader->GetOutput()->GetSpacing() );
//	fastMarching->Update();
//	std::cout << "Fast Marching:" << fastMarching->GetOutput()->GetSpacing() << std::endl;
	/**
   * Set up and run the shape detection filter
   */
  typedef itk::GeodesicActiveContourLevelSetImageFilter<
	InternalImageType, InternalImageType > ShapeDetectionFilterType;
  
  typename ShapeDetectionFilterType::Pointer shapeDetection = ShapeDetectionFilterType::New();
	
	// set the initial level set
  shapeDetection->SetInput( fastMarching->GetOutput() );
	
  // set the edge potential image
  shapeDetection->SetFeatureImage( sigmoid->GetOutput() );
	
  // set the weights between the propagation, curvature and advection terms
  shapeDetection->SetPropagationScaling( atof( argv[argumentOffset++] ) );
  shapeDetection->SetCurvatureScaling(   atof( argv[argumentOffset++] ) );
  shapeDetection->SetAdvectionScaling(   atof( argv[argumentOffset++] ) );
	
  // use finite differences instead of derivative of Gaussian to build advection
  shapeDetection->SetDerivativeSigma( atof( argv[argumentOffset++] ) );
	
  // set the convergence criteria
  shapeDetection->SetMaximumRMSError( atof( argv[argumentOffset++] ) );
  shapeDetection->SetNumberOfIterations( atoi( argv[argumentOffset++] ) );
//	shapeDetection->Update();
	/**
   * Threshold the output level set to display the final contour.
   */
  typedef itk::BinaryThresholdImageFilter< InternalImageType, InternalImageType >
	ThresholdFilterType;
  typename ThresholdFilterType::Pointer thresholder = ThresholdFilterType::New();
	
  thresholder->SetInput( shapeDetection->GetOutput() );
  thresholder->SetLowerThreshold( -1e+10 );
  thresholder->SetUpperThreshold( 0.0 );
  thresholder->SetOutsideValue( 0 );
  thresholder->SetInsideValue( 255 );
	/** 
   * Uncomment to write out image files.
   */
	

	 typedef itk::ImageFileWriter< InternalImageType > WriterType;
	 typename WriterType::Pointer writer = WriterType::New();
	 std::string gradMagnitudeFileName = argv[argumentOffset++];
	 writer->SetFileName( gradMagnitudeFileName );
	 writer->SetInput( sigmoid->GetOutput() );
	 writer->Update();
	 
	 writer->SetInput( thresholder->GetOutput() );
	 std::string outputFileName = argv[argumentOffset++];
	 writer->SetFileName( outputFileName );
	 writer->Update();
	 
	 thresholder->SetInput( fastMarching->GetOutput() );
	 writer->SetInput( thresholder->GetOutput() );
	 std::string initialLevelSetFileName = argv[argumentOffset++];
	 writer->SetFileName( initialLevelSetFileName );
	 writer->Update();

	
	return EXIT_SUCCESS;
}
