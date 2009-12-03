#include "ascParser2.h"

ascParser2::ascParser2(string filename, string oneCharTokens)
{
  this->filename = filename;
  this->oneCharTokens = oneCharTokens;
  if(filename != "")
    file.open(filename.c_str());
}


ascParser2::~ascParser2()
{
  if(file.is_open())
    file.close();
}


bool ascParser2::checkNextToken(string s)
{
  ios::pos_type startPosition = file.tellg();

  string token = getNextToken();
  if(s == token)
    {
      file.seekg(startPosition);
      return true;
    }
  else
    {
      file.seekg(startPosition);
      return false;
    }
}

string ascParser2::peekNextToken()
{
  ios::pos_type startPosition = file.tellg();
  string s = getNextToken();
  file.seekg(startPosition);
  return s;

}


bool ascParser2::getAndCheck(string s)
{
  string gotten = getNextToken();
  return s == gotten;
}


bool ascParser2::isOneCharToken(string s)
{
  return oneCharTokens.find(s)!=string::npos;
}


void ascParser2::eatUntil(string s)
{
  string token = getNextToken();
  while(token != s)
    {
      token = getNextToken();
    }

}

void ascParser2::eatUntilWithReturn(string s)
{
  ios::pos_type startPosition = file.tellg();
  string token = getNextToken();
  while((token != s))
    {
      startPosition = file.tellg();
      token = getNextToken();
      if(file.eof())
        break;
    }
  if(!file.eof())
    file.seekg(startPosition);
}

//Returns the next token, either string or oneCharToken
//In lower case
string ascParser2::getNextToken()
{

  char ch;
  ch = file.get();

  //elliminate the blank spaces and open line
  //and gets the next characters
  while((ch==' ') || (ch=='\n') || (ch == '\t') || (ch == '\r')  )
    {
      ch = file.get();
      if(file.eof()) return "";
    }


  //If it is a one char token, return it
  if(oneCharTokens.find(ch)!=string::npos)
    {
      string s = "";
      s = s + ch;
#ifdef debug
      //std::cout << s << std::endl;
#endif
      lastToken = s;
      return s;
    }


  else if(ch == 34) //34 = \"
    {
      string s = "\"";
      ch = file.get();
      while(ch != 34)
  	{
          s = s + ch;
          ch = file.get();
  	}
      s = s + ch;
      toLower(&s);
#ifdef debug
      std::cout << s << std::endl;
#endif
      lastToken = s;
      return s;
    }

  else
    {
      //Read until next one char token or the end of file or blank space
      string s;
      s = s + ch;
      ch = file.get();
      while( (oneCharTokens.find(ch)==string::npos) &&
             (!file.eof())							 &&
             (ch != ' ')							 &&
             (ch != '\n')							 &&
             (ch != '\t')							 &&
             (ch != '\r')
             )
        {
          s = s + ch;
          ch = file.get();
          if(file.eof()) return "";
        }
      file.putback(ch);
      toLower(&s);
#ifdef debug
      //std::cout << s << std::endl;
#endif
      lastToken = s;
      return s;
    }
}


void ascParser2::printNextToken(int i)
{
  ios::pos_type startPosition = file.tellg();
  for(int j = 0; j < i; j++)
    std::cout << getNextToken() <<  std::endl;
  file.seekg(startPosition);

}

void ascParser2::toLower(string* s)
{
  //transform(s->begin(), s->end(), s->begin(), (int(*)(int))tolower);
  for(int i = 0; i < s->length(); i++)
    {
      if( isupper((*s)[i]))
        {
          (*s)[i] = tolower((*s)[i]);
        }

    }
}

void ascParser2::error(string s)
{
  std::cout << "--- Printing the tokens before the error" << std::endl;
  std::cout << lastToken << std::endl;
  printNextToken(10);
  std::cout << s << std::endl;
}

vector< float > ascParser2::color2rgb(string colorstr)
{

  vector< float > color(3);

  if(colorstr == "red")
    {
      color[0]=1;
      color[1]=0;
      color[2]=0;
    }
  else if(colorstr == "moneygreen")
    {
      color[0]=0.235;
      color[1]=0.70;
      color[2]=0.443;
    }
  else if(colorstr == "white")
    {
      color[0]=1;
      color[1]=1;
      color[2]=1;
    }
  else if(colorstr == "magenta")
    {
      color[0]=1;
      color[1]=0;
      color[2]=1;
    }
  else if(colorstr == "cyan")
    {
      color[0]=0;
      color[1]=1;
      color[2]=1;
    }
  else if(colorstr == "darkyellow")
    {
      color[0]=0.545;
      color[1]=0.545;
      color[2]=0;
    }
  else if(colorstr == "darkmagenta")
    {
      color[0]=0.545;
      color[1]=0.0;
      color[2]=0.545;
    }
  else if(colorstr == "mediumgray")
    {
      color[0]=0.509;
      color[1]=0.509;
      color[2]=0.509;
    }
  else if(colorstr == "darkcyan")
    {
      color[0]=0.0;
      color[1]=0.545;
      color[2]=0.545;
    }
  else if(colorstr == "green")
    {
      color[0]=0;
      color[1]=1;
      color[2]=0;
    }
  else if(colorstr == "yellow")
    {
      color[0]=1;
      color[1]=1;
      color[2]=0;
    }
  else if(colorstr == "skyblue")
    {
      color[0]=0.52;
      color[1]=0.807;
      color[2]=1;
    }
  else if(colorstr == "darkred")
    {
      color[0]=0.545;
      color[1]=0;
      color[2]=0;
    }
  else
    {
      color[0]=1;
      color[1]=1;
      color[2]=1;
    }

  return color;

}


bool ascParser2::parseColorSection(NeuronColor &retColor)
{

  if(!getAndCheck("("))
    error("parseColorSection called and no '('");

  if(!getAndCheck("color"))
    error("parseColorSection called when there is no 'color'");

  string s = getNextToken();
  if(s=="rgb")
    {
      if(!getAndCheck("("))
        error("parseColorSection error in RGB");

      float R,G,B;

      file >> R;
      if(!getAndCheck(","))
        error("parseColorSection error in RGB");

      file >> G;
      if(!getAndCheck(","))
        error("parseColorSection error in RGB");

      file >> B;
      if(!getAndCheck(")"))
        error("parseColorSection error in RGB");

      if(!getAndCheck(")"))
        error("parseColorSection error in RGB");
      retColor =  NeuronColor(R/255,G/255,B/255);
      return true;
    }
  else
    {
      vector< float > color = color2rgb(s);
      if(!getAndCheck(")"))
        error("parseColorSection no ')'");
      retColor = NeuronColor(color);
      return true;
    }

  return false;
}

int ascParser2::get_section_name()
{
  //Gets the pointer to the part of the file where it is called
  ios::pos_type startPosition = file.tellg();

  string s = getNextToken();

  if(s == ""){
    return ascParser2::ASCEOF;
  }

  if( s == ";")
    {
      file.putback(';');
#ifdef debug
      std::cout << "get_section_name() returns: intro" << std::endl;
#endif
      return ascParser2::INTRO;
    }

  else if (s == "(")
    {

      ios::pos_type positionAfterParenthesis = file.tellg();
      string nextToken = getNextToken();
      //std::cout << "nextToken in get_section_name is: " << nextToken << std::endl;

      if(nextToken == "imagecoords"){
        file.seekg(startPosition);
#ifdef debug
        std::cout << "get_section_name() returns: imagecoords" << std::endl;
#endif
        return ascParser2::IMAGECOORDS;
      }

      if(nextToken == "thumbnail"){
        file.seekg(startPosition);
#ifdef debug
        std::cout << "get_section_name() returns: thumbnail" << std::endl;
#endif
        return ascParser2::THUMBNAIL;
      }

      if( (nextToken == "\"cellbody\"") ||
          (nextToken == "cellbody")
          )
	{
          file.seekg(startPosition);
#ifdef debug
          std::cout << "get_section_name() returns: cellbody" << std::endl;
#endif
          return ascParser2::CELLBODY;
	}

      if(nextToken == "sections") {
        file.seekg(startPosition);
#ifdef debug
        std::cout << "get_section_name() returns: sections" << std::endl;
#endif
        return ascParser2::SECTIONS;
      }

      if(nextToken == "(")
	{
          //Gets the color of the section
          file.seekg(positionAfterParenthesis);
          NeuronColor dummy = NeuronColor();
          parseColorSection(dummy);

          /** Chechs for '; [n1,n2] HERE*/
          string kk2 = peekNextToken();
          if(kk2==";"){
            kk2 = getNextToken(); //;
            if(!getAndCheck("[")) error("parseAxonOrDendrite no '[' after color and ;");
            kk2 = getNextToken();
            if(!getAndCheck(",")) error("parseAxonOrDendrite no ',' after color and ;");
            kk2 = getNextToken();
            if(!getAndCheck("]")) error("parseAxonOrDendrite no ']' after color and ;");
          }

          if(!getAndCheck("("))
            error("get_section_name() no '(' after Color Section");

          //Gets the name of the sextion
          nextToken = getNextToken();
          //std::cout << "Next token with the axon/dendrite: " << nextToken << std::endl;

          if(nextToken == "axon")
            {
              file.seekg(startPosition);
#ifdef debug
              std::cout << "get_section_name() returns: axon" << std::endl;
#endif
              return ascParser2::AXON;
            }
          else if(nextToken == "dendrite")
            {
              file.seekg(startPosition);
#ifdef debug
              std::cout << "get_section_name() returns: dendrite" << std::endl;
#endif
              return ascParser2::DENDRITE;
            }
          else if(nextToken == "apical")
            {
              file.seekg(startPosition);
// #ifdef debug
//               std::cout << "get_section_name() returns: apical" << std::endl;
// #endif
              return ascParser2::APICAL;
            }
          else if(nextToken == "font")
            {
              file.seekg(startPosition);
// #ifdef debug
//               std::cout << "get_section_name() returns: apical" << std::endl;
// #endif
              return ascParser2::FONT;
            }

          else
            {
              string s = "get_section_name() I do not know what is a ";
              s = s + nextToken;
//               error(s);
              file.seekg(startPosition);
              return ascParser2::ERRORN;
            }
	}

      string s2 = "get_section_name() I do not know what is a tt ";
      s2 = s2 + nextToken;
      error(s2);
      file.seekg(startPosition);
      return ascParser2::ERRORN;
    }

  else
    {
//       string s = "get_section_name() I do not know what is a ";
//       s = s + token;
//       error(s);
      file.seekg(startPosition);
      return ascParser2::ERRORN;
    }

}


bool ascParser2::parseMainSection(Neuron* neuron)
{

  int nDendrites = 0;
  int nAxons = 0;

  while(!file.eof())
    {
      int sn = get_section_name();
      NeuronContour cont = NeuronContour();
      NeuronSegment* seg = new NeuronSegment();

      switch(sn)
	{
	case ascParser2::INTRO:
          std::cout << "Parsing intro section" << std::endl;
          parseInfoSection();
          break;

	case ascParser2::IMAGECOORDS:
          std::cout << "Parsing imagecoords section" << std::endl;
          parseImageCoords();
          break;

	case ascParser2::THUMBNAIL:
          std::cout << "Parsing thumbnail section" << std::endl;
          parseThumbnail();
          break;

	case ascParser2::CELLBODY:
          std::cout << "Parsing cellbody section" << std::endl;
          if(!parseCellBodySection(cont))
            error("parseMainSection getting the cellbody section");
          printf("Cellbody section parsed\n");
          neuron->soma = cont;
          break;

	case ascParser2::AXON:
          std::cout << "Parsing axon section" << std::endl;
          if(!parseAxonOrDendrite(seg,nAxons++,true))
            error("parseMainSection error with the Axon");
          neuron->axon.push_back(seg);
          break;

	case ascParser2::APICAL:
          std::cout << "Parsing apical section" << std::endl;
          if(!parseAxonOrDendrite(seg,nDendrites++,true))
            error("parseMainSection error with the Axon");
          neuron->dendrites.push_back(seg);
          break;

	case ascParser2::DENDRITE:
          std::cout << "Parsing dendrite section" << std::endl;
          if(!parseAxonOrDendrite(seg,nDendrites++,false))
            error("parseMainSection error with the Dendrite");
          neuron->dendrites.push_back(seg);
          break;

        case ascParser2::SECTIONS:
          std::cout << "Skipping the \"Sections\" section\n" << std::endl;
          parseSectionSection();
          break;
        case ascParser2::FONT:
          std::cout << "Skipping the \"Font\" section\n" << std::endl;
          skip_until_next_section();
          break;

        case ascParser2::ASCEOF:
          std::cout << "Eof reached\n";
          return true;

	default:
          error("parseMainSection does not recognize the next section");
          skip_until_next_section();
          break;
	}
    }
  return false;
}


void ascParser2::skip_until_next_section()
{
  int level = 0;
  string token;
  while(true){
    token = getNextToken();
    if( token == "(" )
      level++;
    if(token == ")")
      level--;
//     std::cout << "Level: " << level << " " <<  token << std::endl;
    if( ((token == ")") && (level==0))
        || file.eof()) break;
  }
  eatUntilWithReturn("(");
//   printf("Next tokens: \n");
//   printNextToken(10);
//   exit(0);q
}

bool ascParser2::parseInfoSection()
{
  if(!getAndCheck(";")) error("parseInfoSection() called without a ';'");
  fileInfo = "";
  char ch;
  ch = file.get();
  while( (ch!='(') || file.eof() )
    {
      fileInfo = fileInfo + ch;
      ch = file.get();
    }
  if(file.eof())
    {
      error("parseInfo has reached the end of the file");
      return false;
    }
  else
    {
      file.putback('(');
      return true;
    }

}

bool ascParser2::parseSectionSection(){
  if(!getAndCheck("(")) {
    error("Error while getting '(' in the Section section");
    return false;
  }
  if(!getAndCheck("sections")){
    error("Error while getting 'section' in the section section");
    return false;
  }
  if(!getAndCheck(")")){
    error("Error while getting ')' in the section section");
    return false;
  }
  return true;
}

bool ascParser2::parseImageCoords()
{
  //Check the begginning
  if(!getAndCheck("(")) error("Error while getting '(' in the imagecoords");
  if(!getAndCheck("imagecoords")) error("Error while getting 'imagecoords' in the imagecoords");

  //Skip until the end
  while(!getAndCheck(")"))
    ;

  string s = getNextToken();

  if(s == "(")
    {
      file.putback('(');
      return true;
    }
  else if(s == ";")
    {
      if(!getAndCheck("end")) error("Error while getting 'end' in the imagecoords");
      if(!getAndCheck("of")) error("Error while getting 'of' in the imagecoords");
      if(!getAndCheck("imagecoords")) error("Error while getting 'imagecoords' in the imagecoords");
      if(getAndCheck("("))
        {
          file.putback('(');
          return true;
        }
      else if (file.eof()) return true;
      else
        {
          error("Bad ending of the imagecoords section");
          return false;
        }
    }
  else if(file.eof()) return true;
  else
    {
      error("Error at the ending or the imagecoords section");
      return false;
    }
}

bool ascParser2::parseThumbnail()
{
  //Check the begginning
  if(!getAndCheck("(")) error("Error while getting '(' in the Thumbnail");
  if(!getAndCheck("thumbnail")) error("Error while getting 'thumbnail' in the Thumbnail");

  //Skip until the end
  while(!getAndCheck(")"))
    ;

  string s = getNextToken();

  if(s == "(")
    {
      file.putback('(');
      return true;
    }
  else if(s == ";")
    {
      if(!getAndCheck("end")) error("Error while getting 'end' in the Thumbnail");
      if(!getAndCheck("of")) error("Error while getting 'of' in the Thumbnail");
      if(!getAndCheck("thumbnail")) error("Error while getting 'thumbnail' in the Thumbnail");
      if(getAndCheck("("))
        {
          file.putback('(');
          return true;
        }
      else if (file.eof()) return true;
      else
        {
          error("Bad ending of the thumbnail section");
          return false;
        }
    }
  else if(file.eof()) return true;
  else
    {
      error("Error at the ending or the thumbnail section");
      return false;
    }
}

bool ascParser2::parseCellBodySection(NeuronContour& contour)
{
  if(!getAndCheck("(")) error("parseCellBodySection called where there is no (");

  string name = getNextToken();

  NeuronColor bodyColor = NeuronColor();
  if(!parseColorSection(bodyColor))
    error("parseCellBodySection: error with the color");

  contour.color = bodyColor;

  if(!getAndCheck("(")) error("parseCellBodySection: error after the color");

  if(!getAndCheck("cellbody"))
    error("parseCellBodySection: can't find 'cellbody' token");

  if(!getAndCheck(")")) error("parseCellBodySection: can't find ')' after cellbody");

  vector< NeuronPoint > points;
  vector< NeuronPoint > spines;

  if(!parsePointSection(points, spines, NULL, false))
    error("parseCellBodySection: error while parsing the points");

  contour.points = points;

  if(!getAndCheck(")")) error("parseCellBodySection: no ')' after point section");

  string s = getNextToken();

  if(s == "(")
    {
      file.putback('(');
      return true;
    }
  else if(s == ";")
    {
      if(!getAndCheck("end"))     error("Error while getting 'end' in the cellbody section");
      if(!getAndCheck("of"))      error("Error while getting 'of' in the cellbody");
      if(!getAndCheck("contour")) error("Error while getting 'contour' in the cellbody");
      if(getAndCheck("("))
        {
          file.putback('(');
          return true;
        }
      else if (file.eof()) return true;
      else
        {
          error("Bad ending of the cellbody section");
          return false;
        }
    }
  else if(file.eof()) return true;
  else
    {
      error("Error at the ending or the cellbody section");
      return false;
    }
}


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
bool ascParser2::parseAxonOrDendrite(NeuronSegment* segment, int nAxonDendrite, bool AxonDendrite)
{

  // Generates the name of the axonOrDendrite
 
  char buff[512];

  if (AxonDendrite) sprintf(buff, "a-%02i",nAxonDendrite);
  else 			  sprintf(buff, "d-%02i",nAxonDendrite);


  segment->name = buff;

  std::cout << "AxonOrDendrite= " << buff << std::endl;

  ios::pos_type beforeCalled = file.tellg();
  if(!getAndCheck("(")) error("parseAxonOrDendrite called whithout '('");
  NeuronColor color = NeuronColor();
  if(!parseColorSection(color))
    {
      error("parseAxonOrDendrite can't parse the color properly");
      file.seekg(beforeCalled);
      return false;
    }
  segment->color = color;

  /** Chechs for '; [n1,n2] HERE*/
  string kk2 = peekNextToken();
  if(kk2==";"){
    kk2 = getNextToken(); //;
    if(!getAndCheck("[")) error("parseAxonOrDendrite no '[' after color and ;");
    kk2 = getNextToken();
    if(!getAndCheck(",")) error("parseAxonOrDendrite no ',' after color and ;");
    kk2 = getNextToken();
    if(!getAndCheck("]")) error("parseAxonOrDendrite no ']' after color and ;");
  }


  if(!getAndCheck("(")) error("parseAxonOrDendrite no '(' after color");

  string s = getNextToken();
  if( (s!="dendrite") && (s!="axon") && (s!="apical") )
    error("parseAxonOrDendrite: can't find the axon / dendrite token");

  if(!getAndCheck(")"))
    error("parseAxonOrDendrite: can't find ')' after the axon / dendrite token");

  if(!getAndCheck("(")) error("parseAxonOrDendrite: can't find '(' before the points section");
  float x = 0, y = 0, z = 0, w = 0;
  file >> x;
  file >> y;
  file >> z;
  file >> w;
  if(!(getAndCheck(")") && getAndCheck(";") && getAndCheck("root")))
    error("parseAxonOrDendrite: can't get the Root point");

  NeuronPoint root = NeuronPoint(x,y,z,w);
  segment->root = root;
  segment->parent = NULL;

  vector< NeuronPoint > points;
  vector< NeuronPoint > spines;
  if(!parsePointSection(points, spines, segment))
    error("parseAxonOrDendrite: parsing the points section");

  segment->points = points;
  segment->spines = spines;

  vector< NeuronMarker > markers;
  if(!parseMarkerSection(markers))
    error("parseAxonOrDendrite: parsing the marker section");
  segment->markers = markers;

  //error("Printing the error before the split of the section");
  //exit(0);

//   vector< NeuronSegment > childs;
//   segment.childs = childs;
  if(!parseSplitSection(segment, color, string(buff)))
    error("parseAxonOrDendrite: parsing the split section");


  string nextTok = getNextToken();

  if(!isOneCharToken(nextTok))
    {
      segment->ending = nextTok;
      nextTok = getNextToken();
    }

  if(!(nextTok == ")"))
    error("parseAxonOrDendrite: no ')' after the markers");

  if(!(getAndCheck(";") &&
       getAndCheck("end") &&
       getAndCheck("of") &&
       getAndCheck("tree")))
    error("parseAxonOrDendrite: bad reading 'end of tree'");

  return true;

}

bool ascParser2::parsePointSection
(vector< NeuronPoint >& points,
 vector< NeuronPoint >& spines,
 NeuronSegment* segmentParent,
 bool includeInAllPointsVector)
{

  float x=0, y=0, z=0, w=0;
  string noSenseNumber;
  int pointNumber = 1;
  int spineNumber = 1;

  ios::pos_type beforeBreakingToken;
  ios::pos_type beforeParenthesis;
  int i = 0;

  //int pointNumber = 0; // number of the point in the segment


  while( checkNextToken("(") || checkNextToken("<") )
    {
      beforeParenthesis = file.tellg();
      //puts the '(' or '<' in s
      string s = getNextToken();

      //If the oppening token of the point is a '<' skip the point
      if(s == "<")
        {
          if(!getAndCheck("("))
            error("parsePointSection: no \"(\" after the \"<\" of the Spine\n");
          //                  NeuronPoint spine;
          file >> x;
          file >> y;
          file >> z;
          file >> w;
          NeuronPoint spine = NeuronPoint(x,y,z,w, 0, 0);
          if(!getAndCheck(")"))
            error("parsePointSection: no \")\" after the points of the Spine\n");
          if(!getAndCheck(">"))
            error("parsePointSection: no \">\" after the \")\" of the Spine\n");
          if(!getAndCheck(";")) error("parsePointSection: error after '>'");
          if(!getAndCheck("spine")) error("parsePointSection: can't get spine");
          spines.push_back(spine);
          continue;
        }


      //If after the '(' the next token is a oneCharToken or a name, return
      s = getNextToken();
      if( isOneCharToken(s.c_str()) ||
          ((s.data()[0] >= 'A') && (s.data()[0] <= 'z'))
          ) //if it is not a number
        {
          file.seekg(beforeParenthesis);
          return true;
        }

      //read the coordinates
      x =  atof(s.c_str());
      file >> y;
      file >> z;
      file >> w;

      if(!getAndCheck(")"))
        error("parsePointSection: no ')' after the coordinates");
      if(!getAndCheck(";"))
        error("parsePointSection: no ';' after the coordinates");

      noSenseNumber =  getNextToken();
//       std::cout << "noSenseNumber = " << noSenseNumber << std::endl;
//       exit(0);

      NeuronPoint parsedPoint = NeuronPoint();
      if(checkNextToken(","))
        {
          string s = getNextToken(); //remove the ','
          string posibleNumber =  getNextToken();
          //file >> pointNumber;
          if(posibleNumber.data()[0] == 'R' || posibleNumber.data()[0] == 'r')
            {
              //This is a really bad method to get the last branch of the name of the root point, but it works
              int posibleNumberEndingStart = posibleNumber.size()>=2?posibleNumber.size()-2:1;
              string toAttachToName = posibleNumber.substr(posibleNumberEndingStart,posibleNumber.size());
              pointNumber = 1;
//               if(segmentParent != NULL){
//                 segmentParent->name = segmentParent->name + toAttachToName;
                //std::cout << " Segment name = " << segmentParent->name << std::endl;
//               }
            }
          else{
            pointNumber = atoi(posibleNumber.c_str());
          }
          if(noSenseNumber.data()[0] == 'R' || noSenseNumber.data()[0] == 'r'){
            noSenseNumber = "0";
          }

          parsedPoint = NeuronPoint(x,y,z,w,atoi(noSenseNumber.c_str()), pointNumber);
          char buff[64]; sprintf(buff, "-p%02i", pointNumber);
          if(segmentParent!=NULL){
            parsedPoint.name = segmentParent->name + buff;
            parsedPoint.parent = segmentParent;
          }
          else
            parsedPoint.name = buff;
        }
      else
        {
          parsedPoint = NeuronPoint(x,y,z,w,0, atoi(noSenseNumber.c_str())); //in this case noSenseNumber is pointNumber
          char buff[64]; sprintf(buff, "-p%02i", atoi(noSenseNumber.c_str()));
          if(segmentParent!=NULL){
            parsedPoint.name = segmentParent->name + buff;
            parsedPoint.parent = segmentParent;
          }
          else
            parsedPoint.name = buff;
        }

      points.push_back(parsedPoint);
      if(includeInAllPointsVector)
        neuron->allPointsVector.push_back((points[points.size()-1]));


      //std::cout << neuron->allPointsVector.size() << std::endl;
      //neuron->allPointsVector[neuron->allPointsVector.size()-1]->print();

#ifdef debug
      std::cout << "point: " << x << " " << y << " " << z << " " << w << " " << noSenseNumber << " " << pointNumber << std::endl;
      std::cout << "point: " << parsedPoint.coords[0] << " " << parsedPoint.coords[1] << " " << parsedPoint.coords[2] << " " << parsedPoint.coords[3] << " " << noSenseNumber << " " << parsedPoint.pointNumber << " " << parsedPoint.name << std::endl;

#endif
    }

  //printNextToken();
  //if(!checkNextToken(")"))
  //	error("parsePointSection: the points does not end with a ')'");

  return true;

}


/**
 *  * (
 *   [point section]
 *   [subbrach section]
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

bool ascParser2::parseSplitSection
(
 NeuronSegment* parent,
 NeuronColor color,
 string name)
{

  ios::pos_type startOfAll = file.tellg();

  if(!(getAndCheck("(") && getAndCheck("(")))
    {
      //error("parseSplitSection called without ( ( ");
      file.seekg(startOfAll);
      return true;
    }

//   vector< NeuronSegment > split;
  parent->childs.push_back(new NeuronSegment());
  parent->childs[0]->color = color;
  parent->childs[0]->name  = name + "-1";
  parent->childs[0]->parent = parent;

  file.putback('(');
  vector< NeuronPoint > points;
  vector< NeuronPoint > spines;
  if(!parsePointSection(points, spines, parent->childs[0]))
    {
      error("parseSplitSection error parsing the first set of points");
      file.seekg(startOfAll);
      return false;
    }

  parent->childs[0]->points = points;
  parent->childs[0]->spines = spines;

  vector< NeuronMarker > markers1stChild;
  parseMarkerSection(markers1stChild);
  parent->childs[0]->markers = markers1stChild;
  parseSplitSection(parent->childs[0], color, parent->childs[0]->name);


  if( !checkNextToken("|") &&
      !checkNextToken(")")
      )
    parent->childs[0]->ending = getNextToken();

  //In case it is a split with only one branch (unprobable but possible)
  if(checkNextToken(")"))
    {
      getNextToken();
      if(!(getAndCheck(";") &&
           getAndCheck("end") &&
           getAndCheck("of") &&
           getAndCheck("split")))
        error("parseSplitSection: bad reading 'end of split'");

//       childs = split;

      return true;
    }

  //Finds all the children of the split
  int nChildren = 1;

  while(checkNextToken("|")){
    if(!getAndCheck("|"))
      error("parseSplitSection can't read '|'");

    char buff3[512];
    sprintf(buff3,"-%i",nChildren+1);

    parent->childs.push_back(new NeuronSegment());
    parent->childs[nChildren]->color = color;
    parent->childs[nChildren]->name  = name+buff3;
    parent->childs[nChildren]->parent = parent;

    points.resize(0);
    spines.resize(0);
    if(!parsePointSection(points, spines, parent->childs[nChildren]))
      error("parseSplitSection error parsing the second set of points");

    parent->childs[nChildren]->points = points;
    parent->childs[nChildren]->spines = spines;

    vector< NeuronMarker > markers2ndChild;
    parseMarkerSection(markers2ndChild);
    parent->childs[nChildren]->markers = markers2ndChild;

    vector< NeuronSegment > split2ndChild;
    parseSplitSection(parent->childs[nChildren], color, parent->childs[nChildren]->name);

    if(!checkNextToken(")") &&
       !checkNextToken("|") )
      parent->childs[nChildren]->ending = getNextToken();
    nChildren++;
  }

  if(!getAndCheck(")"))
    error("parseSplitSection can't read ')' at the end of the split");

  if(!(getAndCheck(";") && getAndCheck("end") && getAndCheck("of") && getAndCheck("split")))
    error("parseSplitSection: bad reading 'end of split'");

  return true;
}


bool ascParser2::parseMarkerSection(vector< NeuronMarker >& markers)
{
  //To prevent that it is called when there are no markers:
  ios::pos_type startOfAll = file.tellg();
  if( getAndCheck("(") )
    {
      string s = getNextToken();
      if( !((s.data()[0]>='A') && (s.data()[0]<='z'))
          )
        {
          file.seekg(startOfAll);
          return true;
        }
      else
        file.seekg(startOfAll);
    }


  string token = peekNextToken();
  while(  (getAndCheck("("))			       &&
          (!isOneCharToken(peekNextToken()))  &&
          (peekNextToken() != "\"cellbody\"")
          )
    {
      string type = getNextToken();

      string kk2 = peekNextToken();
      if(kk2==";"){
        kk2 = getNextToken(); //;
        if(!getAndCheck("[")) error("parseAxonOrDendrite no '[' after color and ;");
        kk2 = getNextToken();
        if(!getAndCheck(",")) error("parseAxonOrDendrite no ',' after color and ;");
        kk2 = getNextToken();
        if(!getAndCheck("]")) error("parseAxonOrDendrite no ']' after color and ;");
      }

      NeuronColor color = NeuronColor();
      if(!parseColorSection( color )) error("parseMarkerSection: reading the color");

      if(!getAndCheck("(")) error("parseMarkerSection: reading '(' at the name of marker");
      if(!getAndCheck("name"))
        error("parseMarkerSection: reading 'name' at the name of marker");

      string name = getNextToken();
      if(!getAndCheck(")")) error("parseMarkerSection: reading ')' at the name of marker");

      //In case there is another name, like Varicosity, check for it.
      ios::pos_type startOfPoints = file.tellg();
      if(!getAndCheck("(")) error("parseMarkerSection: no '(' after name section");
      string s = getNextToken();
      if( (s.data()[0] > 'A') && (s.data()[0] < 'z') )
        {
          if(!getAndCheck(")")) error("parseMarkerSection: no ')' after Variscosity section");
        }
      else
        {
          file.seekg(startOfPoints);
        }

      vector< NeuronPoint > points;
      vector< NeuronPoint > spines;
      if(!parsePointSection( points, spines, NULL, false )) error("parseMarkerSection: reading the points");
      if(!getAndCheck(")")) error("parseMarkerSection: reading ')' at the end of marker");
      if(!getAndCheck(";")) error("parseMarkerSection: reading ';' at the end of marker");
      if(!getAndCheck("end")) error("parseMarkerSection: reading 'end' at the end of marker");
      if(!getAndCheck("of")) error("parseMarkerSection: reading 'of' at the end of marker");
      if(!getAndCheck("markers")) error("parseMarkerSection: reading 'markers' at the end of marker");

      markers.push_back(NeuronMarker(name, type, points, color));
    }

  if(     (isOneCharToken(peekNextToken()))  ||
          (peekNextToken() == "\"cellbody\"")
          )
    file.putback('(');

  return true;

}


bool ascParser2::parseProjectionMatrix(Neuron* neuron)
{
  int found;
  found = filename.find_last_of("/");
  if(found == string::npos)
    found = -1;
  string matrixName = filename.substr(0,found+1)
    + "matrixForNeuron.txt";

  std::ifstream readMatrix(matrixName.c_str());

  if(!readMatrix.good())
    {
      printf("Error while loading the projection matrix in the neuron: %s. The files will be created.\n", matrixName.c_str());
      readMatrix.close();
      std::ofstream out1(matrixName.c_str());
      int found;
      found = filename.find_last_of("/");
      if(found == string::npos)
        found = -1;
      string matrixNameInv = filename.substr(0,found+1)
        + "matrixForNeuronInv.txt";
      std::ofstream out2(matrixNameInv.c_str());
      for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
          if(i==j){
            out1 << "1.0 ";
            out2 << "1.0 ";
          }
          else{
            out1 << "0.0 ";
            out2 << "0.0 ";
          }
        }
        out1 << std::endl;
        out2 << std::endl;
      }
      out1.close();
      out2.close();
      return false;
    }

  neuron->projectionMatrix.resize(16);

  for(int i = 0; i < 16; i++)
    neuron->projectionMatrix[i] = 0;

  std::cout << "Size of the projection matrix: " << neuron->projectionMatrix.size() << std::endl;


  readMatrix >> neuron->projectionMatrix[0];
  readMatrix >> neuron->projectionMatrix[4];
  readMatrix >> neuron->projectionMatrix[8];
  readMatrix >> neuron->projectionMatrix[12];
  readMatrix >> neuron->projectionMatrix[1];
  readMatrix >> neuron->projectionMatrix[5];
  readMatrix >> neuron->projectionMatrix[9];
  readMatrix >> neuron->projectionMatrix[13];
  readMatrix >> neuron->projectionMatrix[2];
  readMatrix >> neuron->projectionMatrix[6];
  readMatrix >> neuron->projectionMatrix[10];
  readMatrix >> neuron->projectionMatrix[14];
  readMatrix >> neuron->projectionMatrix[3];
  readMatrix >> neuron->projectionMatrix[7];
  readMatrix >> neuron->projectionMatrix[11];
  readMatrix >> neuron->projectionMatrix[15];

  readMatrix.close();


  //Now reads the inverse matrix
  string matrixNameInv = filename.substr(0,filename.find_last_of("/")) + "/matrixForNeuronInv.txt";

  readMatrix.open(matrixNameInv.c_str());

  readMatrix >> neuron->projectionMatrixInv[0];
  readMatrix >> neuron->projectionMatrixInv[4];
  readMatrix >> neuron->projectionMatrixInv[8];
  readMatrix >> neuron->projectionMatrixInv[12];
  readMatrix >> neuron->projectionMatrixInv[1];
  readMatrix >> neuron->projectionMatrixInv[5];
  readMatrix >> neuron->projectionMatrixInv[9];
  readMatrix >> neuron->projectionMatrixInv[13];
  readMatrix >> neuron->projectionMatrixInv[2];
  readMatrix >> neuron->projectionMatrixInv[6];
  readMatrix >> neuron->projectionMatrixInv[10];
  readMatrix >> neuron->projectionMatrixInv[14];
  readMatrix >> neuron->projectionMatrixInv[3];
  readMatrix >> neuron->projectionMatrixInv[7];
  readMatrix >> neuron->projectionMatrixInv[11];
  readMatrix >> neuron->projectionMatrixInv[15];


  readMatrix.close();



#ifdef debug
  printf("----------------------- Printing the projection matrix -------------------\n");

  for(int i = 0; i < 16; i++)
    printf("%i -> %f\n", i, neuron->projectionMatrix[i]);

  printf("----------------------- Printing the projection matrix  inverse ----------\n");

  for(int i = 0; i < 16; i++)
    printf("%i -> %f\n", i, neuron->projectionMatrixInv[i]);


#endif
  return true;
}

bool ascParser2::parseFile(Neuron* neuron)
{
  this->neuron = neuron;
  bool pp = parseMainSection(neuron);
  bool pp2 = parseProjectionMatrix(neuron);
  return ( pp && pp2 );
}

bool ascParser2::parseFile(Neuron* neuron, string filename)
{
  this->neuron = neuron;
  if(filename == "") return false;
  file.open(filename.c_str());
  if(file.fail())
    {
      std::cout << "AscParser can not read the file: " << filename << std::endl;
      return false;
    }
  bool pms = parseMainSection(neuron);
  bool ppms = parseProjectionMatrix(neuron);
  return pms & ppms;
}


void ascParser2::writeColorSection(NeuronColor &color, std::ofstream& writer, int n_blancs)
{
  for(int i = 0; i < n_blancs; i++)
    writer << " ";
  writer << "(Color RGB (" << floor(color.coords[0]*255) << "," << floor(color.coords[1]*255) << ","
         << floor(color.coords[2]*255) << "))" << std::endl;
}


void ascParser2::writeAxonSection(NeuronSegment* axon, std::ofstream& writer)
{
  writer << "( ";
  writeColorSection(axon->color, writer, 0);
  writer << "  (Axon)" << std::endl;

  writer << "  ( " << axon->root.coords[0] << " " << axon->root.coords[1] << " "
         << axon->root.coords[2] << " " << axon->root.coords[3] << " ) ; Root \n";
  writer << "  ( " << axon->points[0].coords[0] << " " << axon->points[0].coords[1] << " "
         << axon->points[0].coords[2] << " " << axon->points[0].coords[3] << " ) ; 1, R \n";

  for(int i = 1; i < axon->points.size(); i++)
    {
      writer << "( " << axon->points[i].coords[0] << " " << axon->points[i].coords[1] << " "
             << axon->points[i].coords[2] << " " << axon->points[i].coords[3] << " ) ; " << i + 1 << "\n";

      for(int k = 0; k < axon->spines.size(); k++)
        {
          if( (i+1) == axon->spines[k].pointNumber)
            {
              writer << "  <( " << axon->spines[k].coords[0] << " " << axon->spines[k].coords[1] << " " << axon->spines[k].coords[2]
                     << " " << axon->spines[k].coords[3] << " )>  ; Spine\n";
            }
        }
    }

  for(int i = 0; i < axon->markers.size(); i++)
    writeMarkersSection(axon->markers[i], writer, 2);

  if(axon->childs.size() != 0){
    writer << "  (\n";
    writeSegmentSection(axon->childs[0], axon, writer, 4,1);
    for(int j = 1; j < axon->childs.size()-1; j++){
      writer << "|\n";
      writeSegmentSection(axon->childs[j], axon, writer, 4,1);
    }
    writer << "|\n";
    writeSegmentSection(axon->childs[axon->childs.size()-1],
                        axon, writer, 4,1);
    writer << "  ) ; End of split\n";
  }
  else
    {
      writer << "  " << axon->ending << std::endl;
    }
  writer << ") ; End of tree\n \n";

}


void ascParser2::writeDendriteSection(NeuronSegment* dendrite, std::ofstream& writer)
{
  writer << "( ";
  writeColorSection(dendrite->color, writer, 0);
  writer << "  (Dendrite)" << std::endl;

  writer << "  ( " << dendrite->root.coords[0] << " " << dendrite->root.coords[1] << " "
         << dendrite->root.coords[2] << " " << dendrite->root.coords[3] << " ) ; Root \n";
  writer << "  ( " << dendrite->points[0].coords[0] << " " << dendrite->points[0].coords[1] << " "
         << dendrite->points[0].coords[2] << " " << dendrite->points[0].coords[3] << " ) ; 1, R \n";

  for(int i = 1; i < dendrite->points.size(); i++)
    {
      writer << "( " << dendrite->points[i].coords[0] << " " << dendrite->points[i].coords[1] << " "
             << dendrite->points[i].coords[2] << " " << dendrite->points[i].coords[3] << " ) ; " << i + 1 << "\n";

      for(int k = 0; k < dendrite->spines.size(); k++)
        {
          if( (i+1) == dendrite->spines[k].pointNumber)
            {
              writer << "  <( " << dendrite->spines[k].coords[0] << " " << dendrite->spines[k].coords[1] << " " << dendrite->spines[k].coords[2]
                     << " " << dendrite->spines[k].coords[3] << " )>  ; Spine\n";
            }
        }
    }


  for(int i = 0; i < dendrite->markers.size(); i++)
    writeMarkersSection(dendrite->markers[i], writer, 2);

  if(dendrite->childs.size() != 0){
    writer << "  (\n";
    writeSegmentSection(dendrite->childs[0], dendrite, writer, 4,1);
    for(int j = 1; j < dendrite->childs.size()-1; j++){
      writer << "|\n";
      writeSegmentSection(dendrite->childs[j], dendrite, writer, 4,1);
    }
    writer << "|\n";
    writeSegmentSection(dendrite->childs[dendrite->childs.size()-1],
                        dendrite, writer, 4,1);
    writer << "  ) ; End of split\n";
  }
  else
    {
      writer << "  " << dendrite->ending << std::endl;
    }
  writer << ") ; End of tree\n \n";
}



void ascParser2::writeSegmentSection
(NeuronSegment* segment,
 NeuronSegment* parent,
 std::ofstream& writer,
 int n_blancs, int nKid)
{
  for(int i = 0; i < n_blancs; i++) writer << " " ;

  string segment_name = "";
  if(parent->name.size() >= 4){
    string segment_name = parent->name.substr(4);
  }
  segment_name = "R" + segment_name;

  writer << "( " << segment->points[0].coords[0] << " " << segment->points[0].coords[1] << " "
         << segment->points[0].coords[2] << " " << segment->points[0].coords[3] <<  " ) ; 1, " << segment_name << "-" << nKid <<  "\n";

  for(int i = 1; i < segment->points.size(); i++)
    {
      for(int j = 0; j < n_blancs; j++) writer << " " ;
      writer << "( " << segment->points[i].coords[0] << " " << segment->points[i].coords[1] << " "
             << segment->points[i].coords[2] << " " << segment->points[i].coords[3] << " ) ; " << i + 1 << "\n";

      for(int k = 0; k < segment->spines.size(); k++)
        {
          if(segment->points[i].pointNumber == segment->spines[k].pointNumber)
            {
              for(int j = 0; j < n_blancs; j++) writer << " " ;
              writer << "<( " << segment->spines[k].coords[0] << " " << segment->spines[k].coords[1] << " " << segment->spines[k].coords[2]
                     << " " << segment->spines[k].coords[3] << " )>  ; Spine\n";
            }
        }
    }

  for(int i = 0; i < segment->markers.size(); i++)
    writeMarkersSection(segment->markers[i], writer, n_blancs);

  if(segment->childs.size() != 0){
    for(int i = 0; i < n_blancs; i++) writer << " " ;
    writer << "(\n";
    writeSegmentSection(segment->childs[0], segment, writer, n_blancs + 2,1);
    for(int j = 1; j < segment->childs.size()-1; j++){
      for(int k = 0; k < n_blancs; k++) writer << " " ;
      writer << "|\n";
      writeSegmentSection(segment->childs[j], segment, writer, n_blancs + 2,1);
    }
    for(int k = 0; k < n_blancs; k++) writer << " " ;
    writer << "|\n";
    writeSegmentSection(segment->childs[segment->childs.size()-1],
                        segment, writer, n_blancs + 2,1);
    for(int k = 0; k < n_blancs; k++) writer << " " ;
    writer << ") ; End of split\n";
  }
  else
    {
      for(int i = 0; i < n_blancs; i++) writer << " " ;
      writer << segment->ending << std::endl;
    }
}

void ascParser2::writeMarkersSection(NeuronMarker &marker, std::ofstream& writer, int n_blancs)
{
  writer << std::endl;
  for(int i = 0; i < n_blancs; i++) writer << " " ;
  writer << "(" << marker.type << std::endl;
  writeColorSection(marker.color, writer, n_blancs+1);
  for(int i = 0; i < n_blancs+1; i++) writer << " " ;
  writer << "(Name " << marker.name << " )\n";

  if(marker.name.find("varicosity",0)!= string::npos){
    for(int i = 0; i < n_blancs+1; i++) writer << " " ;
    writer << "(Varicosity)\n";
  }

  for(int i = 0; i < marker.points.size(); i++)
    {
      for(int j = 0; j < n_blancs + 1; j++) writer << " " ;
      writer << "( " << marker.points[i].coords[0] << " " << marker.points[i].coords[1] << " "
             << marker.points[i].coords[2] << " " << marker.points[i].coords[3] << " ) ; " << i + 1 << "\n";
    }

  for(int i = 0; i < n_blancs; i++) writer << " " ;
  writer << ") ; End of markers\n";
}


bool ascParser2::saveNeuron(Neuron* neuron, string filename)
{
  std::ofstream writer(filename.c_str());

  if(!writer.good())
    return false;

  writer << "; ASC file created with ascParser2\n";
  writer << "(ImageCoords The ImageCoords section I found that has no sense, but it probably has... research more throught the internet?\n";
  writer << ") ; End of ImageCoords\n" << std::endl;
  writer.precision(4);
  writer << std::fixed;

  //Saves the soma
  writer << "(\"CellBody\"\n";
  writeColorSection(neuron->soma.color, writer, 2);
  writer << "  (CellBody)\n";
  for(int i = 0; i < neuron->soma.points.size(); i++)
    writer << "  ( " << neuron->soma.points[i].coords[0] << " " <<
      neuron->soma.points[i].coords[1] << " " <<
      neuron->soma.points[i].coords[2] << " " <<
      neuron->soma.points[i].coords[3] << " " <<
      " ) ; 1, " << i + 1 << std::endl;
  writer << ") ; End of contour\n" << std::endl;

  for(int i = 0; i < neuron->axon.size(); i++)
    writeAxonSection(neuron->axon[i], writer);

  for(int i = 0; i < neuron->dendrites.size(); i++)
    writeDendriteSection(neuron->dendrites[i], writer);

  writer.close();

  //Saves the matrices
  string matrixName = filename.substr(0,filename.find_last_of("/")) + "/matrixForNeuron.txt";
  writer.open(matrixName.c_str());
  if(!writer.good()){
    printf("AscParser2:: error saving the matrix\n");
    return false;
  }

  writer << neuron->projectionMatrix[0] << " ";
  writer << neuron->projectionMatrix[4] << " ";
  writer << neuron->projectionMatrix[8] << " ";
  writer << neuron->projectionMatrix[12];
  writer << std::endl;
  writer << neuron->projectionMatrix[1] << " ";
  writer << neuron->projectionMatrix[5] << " ";
  writer << neuron->projectionMatrix[9] << " ";
  writer << neuron->projectionMatrix[13];
  writer << std::endl;
  writer << neuron->projectionMatrix[2] << " ";
  writer << neuron->projectionMatrix[6] << " ";
  writer << neuron->projectionMatrix[10] << " ";
  writer << neuron->projectionMatrix[14];
  writer << std::endl;
  writer << neuron->projectionMatrix[3] << " ";
  writer << neuron->projectionMatrix[7] << " ";
  writer << neuron->projectionMatrix[11] << " ";
  writer << neuron->projectionMatrix[15];
  writer << std::endl;
  writer.close();

  matrixName = filename.substr(0,filename.find_last_of("/")) + "/matrixForNeuronInv.txt";
  writer.open(matrixName.c_str());
  if(!writer.good()){
    printf("AscParser2:: error saving the matrixInv\n");
    return false;
  }

  writer << neuron->projectionMatrixInv[0] << " ";
  writer << neuron->projectionMatrixInv[4] << " ";
  writer << neuron->projectionMatrixInv[8] << " ";
  writer << neuron->projectionMatrixInv[12];
  writer << std::endl;
  writer << neuron->projectionMatrixInv[1] << " ";
  writer << neuron->projectionMatrixInv[5] << " ";
  writer << neuron->projectionMatrixInv[9] << " ";
  writer << neuron->projectionMatrixInv[13];
  writer << std::endl;
  writer << neuron->projectionMatrixInv[2] << " ";
  writer << neuron->projectionMatrixInv[6] << " ";
  writer << neuron->projectionMatrixInv[10] << " ";
  writer << neuron->projectionMatrixInv[14];
  writer << std::endl;
  writer << neuron->projectionMatrixInv[3] << " ";
  writer << neuron->projectionMatrixInv[7] << " ";
  writer << neuron->projectionMatrixInv[11] << " ";
  writer << neuron->projectionMatrixInv[15];
  writer << std::endl;
  writer.close();

  return true;
}

