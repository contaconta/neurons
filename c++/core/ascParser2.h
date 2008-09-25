 /** Parses an asc file and stores it into the neuron structure.
 * German Gonzalez - german.gonzalez@epfl.ch
 * November 2006
 */

#ifndef ASCPARSER2_H_
#define ASCPARSER2_H_

class Neuron;
class NeuronContour;
class NeuronPoint;
class NeuronSegment;
class NeuronMarker;
class NeuronColor;

#include "neseg.h"

#include "Neuron.h"
#include "utils.h"
using namespace std;


/** Parses an ASC files and store the information in a Neuron structure.
 *
 * ============= ASC PROTOTYPE =======================
 *
 * ; comments about the version and so on....
 * (ImageCoords .... blah blah blah ) ; End of ImageCoords
 *
 *
 * ("SectionContour_1" 
 *    .... blah blah blah (this section will be ignored
 *   points
 *   last_point Closure point
 * ) ; End of contour
 *
 *
 *
 * ("CellBody")
 *   [Color Section]
 *   (CellBody)
 *   (X1  Y1  Z1  W1) ; NumberofSomething, NumberOfPoint
 *   (X2  Y2  Z2  W2) ; NumberofSomething, NumberOfPoint
 *   ...
 *   (Xn  Yn  Zn  Wn) ; NumberofSomething, NumberOfPoint
 * ) ; End of contour
 *
 * //As many axons as needed
 * ( [Color Section]
 *   (Axon)
 *   (X0  Y0  Z0  W0) ; Root
 *   [point section]     x 1
 *	 [marker section]    x [0..N]
 *   [subbranch section] x [0..1]
 *   [type of ending]    x [0..1]
 * ) ; End of tree
 *
 * //As many dendrites as needed
 * ( [Color Section]
 *   (Dendrite)
 *   (X0  Y0  Z0  W0) ; Root
 *   [point section]     x 1
 *	 [marker section]    x [0..N]
 *   [subbranch section] x [0..1]
 *   [type of ending]    x [0..1]
 * ) ; End of tree
 *
 *
 * V3 introduces new things in the axon / dendrite section:  
 *
 * //As many axons as needed
 * ( [Color Section] ; [n1,n2]
 *   (Axon)
 *   (X0  Y0  Z0  W0) ; Root
 *   [point section]     x 1
 *	 [marker section]    x [0..N]
 *   [subbranch section] x [0..1]
 *   [type of ending]    x [0..1]
 * ) ; End of tree
 *
 * //As many dendrites as needed
 * ( [Color Section] ; [n1,n2]
 *   (Dendrite)
 *   (X0  Y0  Z0  W0) ; Root
 *   [point section]     x 1
 *	 [marker section]    x [0..N]
 *   [subbranch section] x [0..1]
 *   [type of ending]    x [0..1]
 * ) ; End of tree
 *
 * //New in V3
 * //As many apicals as needed
 * ( [Color Section] ; [n1,n2]
 *   (Apical)
 *   (X0  Y0  Z0  W0) ; Root
 *   [point section]     x 1
 *	 [marker section]    x [0..N]
 *   [subbranch section] x [0..1]
 *   [type of ending]    x [0..1]
 * ) ; End of tree
 *
 *
 * ------- POINT SECTION ----------
 *   (X1  Y1  Z1  W1) ; NumberOfPoint, Recursive code
 *   (X2  Y2  Z2  W2) ; NumberOfPoint
 *   ...
 *   (Xn  Yn  Zn  Wn) ; NumberOfPoint
 *
 *  V3 of the ASC files introduced a new format:
 *   (X1  Y1  Z1  W1) ; RecursiveCode, NumberOfPoint
 *   (X2  Y2  Z2  W2) ; RecursiveCode, NumberOfPoint
 *   ...
 *   (Xn  Yn  Zn  Wn) ; RecursiveCode, NumberOfPoint
 *
 *
 * ------ SUBBRANCH SECTION ------
 * (
 *   [point section]
 *   [marker section]
 *   [subbrach section]
 *   [type of ending]
 * |
 *   [point section]
 *   [marker section]
 *   [subbrach section]
 *   [type of ending]
 * ) ; End of split
 *
 *
 * ------ MARKER SECTION --------
 * (TypeofMarker
 *   [Color Section]
 *   (Name "name")
 *   (Function)     //Optional
 *   [Point section without recursive code]
 * ) ; End of markers
 *
 * // V3 Marker section
 *
 * (TypeofMarker ; [n1,n2]
 *   [Color Section]
 *   (Name "name")
 *   (Function)     //Optional
 *   [Point section without recursive code]
 * ) ; End of markers

 * //As many markers as necessary
 *
 * ------ COLOR SECTION ------
 * (Color NAMEOFCOLOR)
 * or
 * (Color RGB (R, G, B))
 *
 * ------ ENDING ------------
 * Normal
 * or
 * Incomplete
 *
 * ======= CONVENCTIONS =================
 * All the methods should parse their section and leave the tokenizer
 * at the first parenthesis or word of the following section, so that when
 * doing file >> string, string will be a "("
 *
 * The beggining of the methods should check wether the section exists or not,
 * it is the method responsibility
 *
 *  The tokens will be defined as
 *  ( ) | < > " ;
 *
 *
 */

class ascParser2
{

public:

  string lastToken;

  typedef enum {INTRO,
                IMAGECOORDS,
                THUMBNAIL,
                CELLBODY,
                AXON,
                DENDRITE,
                ERROR,
                SECTIONS,
                APICAL,
                FONT,
                ASCEOF} sectionNames;

  /** Gets the next string from the file and compares it with s*/
  bool getAndCheck(string s);

  /**  Peeks the next token*/
  string peekNextToken();

  /**Compares the string with the next token of the file without getting it */
  bool checkNextToken(string s);

  /** Eats until the token given*/
  void eatUntil(string s);

  /** Eats until the token given. Leaves the pointer at the beginning of the token*/
  void eatUntilWithReturn(string s);

  /** Transforms s to lower case*/
  void toLower(string* s);

  /** Gets the next token */
  string getNextToken();

  /** Returns true if it is one char token*/
  bool isOneCharToken(string s);

  /** Deals with errors.*/
  void error(string s);

  /** Translates a string into a color vector.*/
  vector< float > color2rgb(string colorstr);

  /** Returns the section name.
   * The pseudocode will be something like:
   *
   * check for the (
   * check for the next token
   * if(it is a string)
   *   if it is ImageCoords ->
   * 			put the string back to the file
   * 			put the ( back to the file
   * 			return imagecoords
   *   if it is thumbnail
   * 			put the string back to the file
   * 			put the ( back to the file
   * 			return thumbnail
   *   if it is "CellBody"
   * 			put the string back to the file
   * 			put the ( back to the file
   * 			return CellBody
   * if(it is a "(")
   *   get a color section
   *   check for the (
   *   get a string
   *   if it is Axon ->
   * 		 put the Axon back
   *       put the color section back
   *       put the ( back
   *       return Axon
   *   if it is Dendrite
   * 		 put the Dendrite back
   *       put the color section back
   *       put the ( back
   *       return Dendrite
   * else
   *   return -1; //error
   *
   */
  int get_section_name();


  /** Puts the tokenizer in the next section.
   * Pseudocode:
   * read until find a ")". Check for sublevels of parenthesis
   * check for ; End of section
   * get the next token and check that is a "("
   */
  void skip_until_next_section();



  /** Parses all the file at the highest level.
   *
   * elliminate the comments at the start of the file
   *
   * while(!eof)
   * {
   * int sectionName = get_section_name();
   * swich(type)
   *   case cellbody
   *      neuron.soma = parseCellBodySetion();
   *   case axon
   *      neuron.axon.push_back(parseAxonOrDendrite());
   *   case dendrite
   *      neuron.dendrites.push_back(parseAxonOrDendrite());
   *   case ImageCoords
   * 		parseImageCoords();
   *   ...
   *
   * }
   */
  bool parseMainSection(Neuron* neuron);

  /** Parses the information section of the file. The information will be stored in a
   * string info. */
  bool parseInfoSection();



  /* Gets rid of the information contained in the imageCoords section
   * Pseudocode:
   * read until: )
   * read: ; End of ImageCoords (if exists)
   * read next token and check that it is "("
   */
  bool parseImageCoords();

  /* Gets rid of the Sections section.
   * Pseudocode:
   * read: (
   * skip until )
   * return
   */
  bool parseSectionSection();


  /* Gets rid of the information contained in the imageCoords section
   * Pseudocode:
   * read until: )
   * read: ; End of Thumbnail (if exists)
   * read next token and check that it is "("
   */
  bool parseThumbnail();

  /* Stores the cellbody info in a contour.
   *
   * Create a Contour.
   * read: "CellBody"
   * parseColorSection
   * read (CellBody)
   * while( get_next_token_is a ( )
   * {
   *   read (
   *   read X Y Z W
   *   read )
   *   read ;
   *   read firstNonSenseNumber
   *   read ,
   *   read point order number
   *   attach the point to the contour
   * }
   * read ;
   * read "End of contour"
   */
  bool parseCellBodySection(NeuronContour& contout);

  /** Pseudocode.
   *  read (
   *  readColorSection
   *  read (
   *  read axon/dendrite
   *  read )
   *  read ( X Y Z W ) ; Root
   *  attach to the segment root
   *  parsepointsectio()
   *  parseSplitSection
   *  parseMarkersSection()
   *  parse end section
   */
  bool parseAxonOrDendrite(NeuronSegment* segment, int nAxonDendrite = 0, bool AxonDendrite = true);

  /** Pseudocode.
   *  read (
   *  read Color
   *  read type of color
   *  if type = RGB
   *     read (R,G,B)
   *  read )
   *  get next token
   */
  bool parseColorSection(NeuronColor& retcolor);

  /**Pseudocode.
   *
   * while(next Token is a parenthesis and and
   *       next one is a digit)
   * {
   *      Get the Spines of the asc file and return it in &spines
   *      // -- old -- skip also the <(   ... )> ; spine
   * 	X = token
   *  read Y Z W ) ; numberOfPoint
   * }
   *
   * Put back the parenthesis(if there is)
   * and the previous token
   *
   *
   *
   */
  bool   parsePointSection(vector< NeuronPoint >& points, vector< NeuronPoint >& spines,  NeuronSegment* parent = NULL, bool addToAllPointsVector = true);

  /**
   *  * (
   *   [point section]
   *   [subbrach section]    //Error, there should be a conmutation there!!!! First parse markers and then subbranch
   *   [marker section]
   *   [type of ending]
   * |
   *   [point section]
   *   [subbrach section]
   *   [marker section]
   *   [type of ending]
   * ) ; End of split
   *
   *
   */
  bool parseSplitSection(NeuronSegment* parent, NeuronColor color, string name);


  /** Gets all the markers in the section
   * while(Read '(' && nextIsNotAOneCharToken)
   * Get the marker type
   * [Color section]
   * Read (
   * Read name
   * Read NAME
   * Read )
   * [point section]
   * Read ')'
   * Read ; End of Markers
   * Attach the marker to the markers vector
   */
  bool parseMarkerSection(vector< NeuronMarker >& pepe);

  /** Gets the projection matrix stored in the file matrixForNeuron.txt in the same directory as the asc file.
   * It is stored as a matrix for OpenGl, the order of the elements can be found in:
   * http://www.mevis.de/opengl/glLoadMatrix.html . It is loaded when an ascParser is called.
   * */
  bool parseProjectionMatrix(Neuron* neuron);


  //string parseEndingSection();
  string fileInfo;

  std::ifstream file;

  /** Neuron where all the information is being stored.*/
  Neuron* neuron;

  string oneCharTokens;

  ascParser2(string filename = "", string oneCharTokens = "()|<>;,[]");

  virtual ~ascParser2();

  /** Parses the file the ascParser has been created with and stores it in the neuron*/
  bool parseFile(Neuron* neuron);

  /** Parses filename into neuron*/
  bool parseFile(Neuron* neuron, string filename);

  void printNextToken(int i=1);

  string filename;

  /** Saves the neuron into the specified filename. */
  bool saveNeuron(Neuron* neuron, string filename);

  /** Saves a color section
   * @param color The NeuronColor that is to be saved
   * @param writer The Ofstream writer
   * @param n_blancs the number of spaces to be written before the information*/
  void writeColorSection(NeuronColor &color, std::ofstream &writer, int n_blancs);

  void writeAxonSection(NeuronSegment* axon, std::ofstream& writer);

  void writeSegmentSection(NeuronSegment* segment, NeuronSegment* parent, std::ofstream& writer, int n_blancs, int nKid);

  void writeMarkersSection(NeuronMarker &marker, std::ofstream& writer, int n_blancs);

  void writeDendriteSection(NeuronSegment* axon, std::ofstream& writer);


};

#endif /*ASCPARSER2_H_*/
