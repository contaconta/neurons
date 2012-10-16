/*
  Standalone c++ code to clean bibtex files.

  Compile with:

  g++ bibfile_cleaner.cpp -o bibfile_cleaner

  Usage:

  bibfile_cleaner [--short | -s] input_file.bib [output_file.bib]

*/


#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <string.h>
using namespace std;

////////////////////////////////////////////////////////////////////////////////

const char * do_not_capitalize_these_words[] = {"of", "and", "the", "on", "from",
						"into", "for", "in", "by", "http",
						"as", "an", "to", "or", "a", "at",
						"through", "with", "is",  "also",
						"via",
						0};

// journal fields that contain the string on the left side will be replaced by the string on the right side:
const char * strings_for_journals[] = {
  "Pattern Analysis and Machine Intelligence", "PAMI",
  "Pattern Anal. Mach. Intell.",  "PAMI",
  "PAMI", "PAMI",
  "International Journal of Computer Vision", "IJCV",
  "Int. J. Comput. Vision", "IJCV",
  "IJCV", "IJCV",
  "Computer Vision and Image Understanding", "CVIU",
  "Comput. Vis. Image Underst.", "CVIU",
  "CVIU", "CVIU",
  "Pattern Recognition", "PR",
  "Journal of Machine Learning Research", "JMLR",
  "JMLR", "JMLR",
  0, 0};

// booktitle fields that contain the string on the left side will be replaced by the string on the right side:
const char * strings_for_conferences[] = {
  "International Conference on Computer Vision", "ICCV",
  "ICCV", "ICCV",
  "Computer Vision and Pattern Recognition", "CVPR",
  "CVPR", "CVPR",
  "Asian Conference on Computer Vision", "ACCV",
  "ACCV", "ACCV",
  "Neural Information Processing Systems", "NIPS",
  "NIPS", "NIPS",
  "International Conference on Image Processing", "ICIP",
  "ICIP", "ICIP",
  "European Conference on Computer Vision", "ECCV",
  "ECCV", "ECCV",
  "International Symposium on Mixed and Augmented Reality", "ISMAR",
  "ISMAR", "ISMAR",
  "British Machine Vision Conference", "BMVC",
  "BMVC", "BMVC",
  0, 0};

// the strings on the left side that appear in the author fields will be replaced by the string on the right:
const char * important_authors_with_a_weird_name[] =  {
  "Lecun", "LeCun",
  "van Gool", "{Van~Gool}",
  "Van~Gool", "{Van~Gool}",
  "{{Van~Gool}}", "{Van~Gool}", // that's a dirty trick to avoid Van~Gool -> {Van~Gool} -> {{Van~Gool}} -> ..
  0, 0};

// the strings on the left that appear in the title fields will be replaced by the string on the right:
const char * special_words_in_titles[] =  {
  "3d",   "3D",
  "3-d",  "3D",
  "3--d", "3D",
  "3-D",  "3D",
  "2d",   "2D",
  "2-d",  "2D",
  "2--d", "2D",
  "2-D",  "2D",
  "Pde",  "PDE",
  "Svm",  "SVM",
  "Dof",  "DoF",
  ": a",  ": A",
  ": o",  ": O",
  ": w",  ": W",
  0, 0};

// month fields that contain the string on the left side will be replaced by the string on the right side:
const char * months[] = {
  "Jan", "\"January\"",
  "Feb", "\"February\"",
  "Fev", "\"February\"",
  "Mar", "\"March\"",
  "Apr", "\"April\"",
  "Avr", "\"April\"",
  "May", "\"May\"",
  "Mai", "\"May\"",
  "Jun", "\"June\"",
  "Juin", "\"June\"",
  "Jul", "\"July\"",
  "Aug", "\"August\"",
  "Sep", "\"September\"",
  "Oct", "\"October\"",
  "Nov", "\"November\"",
  "Dec", "\"December\"",
  0, 0};

////////////////////////////////////////////////////////////////////////////////

class bibfile_parser
{
public:
  bibfile_parser() { }
  ~bibfile_parser() { end_parsing(); }

  void start_parsing(string filename) {
    bibfile.open(filename.c_str());
    if (!bibfile.good()) {
      cerr << endl << " Can't read file " << filename << "." << endl;
      exit(0);
    }
    line_number = 1;
    last_non_blank_character_read = 0;
    reached_end_of_file = false;
  }

  void end_parsing(void) { if (bibfile.is_open()) bibfile.close(); }

  int get_line_number(void) { return line_number; }
  char get_last_non_blank_character_read(void) { return last_non_blank_character_read; }

  char get_very_next_character(void);
  char get_next_character(void);
  char get_next_non_blank_character(void);
  bool reached_end_of_file;
  int line_number;

  string read_type(void);
  string read_key(void);
  string read_field_name(void);
  string read_string_name(void);
  string read_field_string(bool & end_of_entry);

private:
  ifstream bibfile;
  bool cesure, endline;
  char last_non_blank_character_read;
};

class bibtex_entry
{
public:
  bibtex_entry() { }

  string type, key, author, title; // all types
  string booktitle; // inproceedings
  string journal, volume, number; // article
  string publisher, editor; // book, inbook
  string report_type, institution; //techreport
  string school; // phdthesis
  string chapter; // inbook
  string note, url; // misc
  string pages, month, year; // all types
  
  string string_name, string_value; // for @string entries
};

////////////////////////////////////////////////////////////////////////////////

bibfile_parser parser;
ofstream output_file;
bool short_output = false;

void error(string err)
{
  cerr << endl << "Line " << parser.get_line_number() << ": ";
  cerr << err << endl;
  output_file.close();

  exit(0);
}

////////////////////////////////////////////////////////////////////////////////

char bibfile_parser::get_very_next_character(void)
{
  char c = char(bibfile.get());
  if (bibfile.eof()) {
    reached_end_of_file = true;
    return 0;
  }
  if (!bibfile.good()) {
    cerr << endl << "error reading input bibfile" << endl;
    exit(0);
  }
  return  c;
}

char bibfile_parser::get_next_character(void)
{
  static char pc = ' ';
  char c = ' ';
  int state = 0;
  bool ok = false;

  do {
    c = get_very_next_character();

    if (reached_end_of_file) return 0;

    // state machine.
    // Checks for end of lines (either 13+10, 10, or 13 codes),
    // skips comments that start by % up to the end of the line.
    // boolean endline is used to set boolean cesure, which is used to add spaces correctly in fields
    switch(state) {
    case 0:
      if (c == 13) {
	state = 1;  endline = true;  line_number++;
      } else if (c == 10) {
	state = 0;  endline = true;  line_number++;
      } else if (c == '%' && pc != '\\') {
	state = 2;  endline = true;
      } else
	ok = true;
      break;
    case 1:
      if (c == 10) {
	state = 0;
      } else if (c == 13) {
	state = 0;  line_number++;
      } else if (c == '%' && pc != '\\') {
	state = 2;
      } else
	ok = true;
      break;
    case 2:
      if (c == 13 || c == 10) {
	state = 3;  line_number++;
      } else
	state = 2;
      break;
    case 3:
      if (c == 13) {
	state = 0;  line_number++;
      } else if (c == 10) {
	state = 0;
      } else if (c == '%' && pc != '\\') {
	state = 2;
      } else
	ok = true;
    }
    pc = c;
  } while (!ok);

  return c;
}


char bibfile_parser::get_next_non_blank_character(void)
{
  char c;

  // boolean endline is used to set boolean cesure, which is used to add spaces correctly in fields
  cesure = endline = false;
  do {
    c = get_next_character();

    if (reached_end_of_file) return 0;

    if (endline || c == ' ' || c == '\t') {
      cesure = true;
    }
  } while(c == ' ' || c == '\t');
  last_non_blank_character_read = c;

  return c;
}

// Reads and returns the entry type (inproceedings, article, ... including comment) in lower case:
string bibfile_parser::read_type(void)
{
  char arobase;
  
  do {
    arobase = get_next_non_blank_character();
    if (reached_end_of_file) return "";
  } while (arobase != '@');

  string type("");

  char c;
  do {
    //     c = get_very_next_character();
    c = get_next_non_blank_character();

    if (reached_end_of_file) error("Reached end of file while reading type of entry");

    if (isalpha(c)) {
      type += tolower(c);
    } else if (c == '{') {
      return type;
    } else {
      cerr << endl << "error when reading type of entry around line " << line_number << endl;
      exit(0);
    }

    if (type.compare("comment") == 0) return "comment";
  } while(true);

  return "";
}

// Reads and returns the entry key:
string bibfile_parser::read_key(void)
{
  string key("");

  char c = get_next_non_blank_character();

  if (reached_end_of_file) error("Reached end of file while reading key");

  do {
    if (isalnum(c) || c == '-' || c == ':' || c == '_') {
      key += c;
    } else if (c == ',') {
      key[0] = toupper(key[0]);
      return key;
    } else {
      cerr << endl 
	   << "error when reading key around line " << line_number << endl;
      cerr << "not supported character: " << c << " ascii: " << int(c) << endl;
      exit(0);
    }
    //     c = get_very_next_character();
    c = get_next_non_blank_character();
    if (reached_end_of_file) error("Reached end of file while reading key");
  } while(true);

  return "";
}

// Reads and returns field name (title, author, ...) in lower case.
// Returns stop if encounters a '}':
string bibfile_parser::read_field_name(void)
{
  string field_name("");

  char c = get_next_non_blank_character();

  if (reached_end_of_file) error("Reached end of file while reading field name");

  if (c == '}') { // there was a comma at the end of the last line
    return "stop";
  }

  do {
    if (isalnum(c) || c == '_' || c == '-' ) {
      field_name += tolower(c);
    } else if (c == '=') {
      return field_name;
    } else if (c == ' ' || c == '\t') {
    } else {
      cerr << endl << "error when reading field name around line " << line_number << endl;
      exit(0);
    }
    c = get_very_next_character();
    if (reached_end_of_file) error("Reached end of file while reading field name");
  } while(true);

  return "";
}

// Reads and returns string name:
string bibfile_parser::read_string_name(void)
{
  string string_name("");

  char c = get_next_non_blank_character();

  if (reached_end_of_file) error("Reached end of file while reading string name");

  if (c == '}') { // there was a comma at the end of the last line
    return "stop";
  }

  do {
    if (isalnum(c) || c == '_' || c == '-' ) {
      string_name += c;
    } else if (c == '=') {
      return string_name;
    } else if (c == ' ' || c == '\t') {
    } else {
      cerr << endl << "error when reading string name around line " << line_number << endl;
      exit(0);
    }
    c = get_very_next_character();
    if (reached_end_of_file) error("Reached end of file while reading string name");
  } while(true);

  return "";
}

// Reads and returns field. This one is tricky because the syntax is very permissive.
// Must be able to read:
// "Towards Urban {3D} Reconstruction From Video", or
// {{Dense Disparity Map Estimation Respecting Image Discontinuities: A {PDE} and Scale-Space Based Approach}}, or
// "H. Hirschm{\"{u}}ller", or
// {{\v S}ochman, Jan and Matas, Ji{\v r}{\' \i}}, ...

string bibfile_parser::read_field_string(bool & end_of_entry)
{
  string field_string("");

  char c;
  int level = 0;
  bool inside_quotes = false;

  end_of_entry = false;

  const int starting_line_number = get_line_number();
  do {
    c = get_next_non_blank_character();
    if (reached_end_of_file) {
      cout << "Line " << starting_line_number
	   << ": started reading "
	   << field_string.substr(0,100) << "..." << endl;
      error("Reached end of file while reading field");
    }
    
    if (c == '{') {
      if (cesure && field_string.length() != 0) field_string += ' ';
      level++;
      field_string += c;
    } else if (c == '}') {
      if (level == 0) {
	end_of_entry = true;
	return field_string;
      }
      level--;
      field_string += c;
    } else if (c == '"') {
      if (field_string.length() == 0) {
	inside_quotes = true;
	field_string += c;
      } else if (field_string[field_string.length() - 1] == '\\')
	field_string += c;
      else {
	inside_quotes = false;
	field_string += c;
      }
    } else if (c == ',') {
      if (level == 0 && !inside_quotes) {
	return field_string;
      } else {
	field_string += c;
      }
    } else if (cesure && field_string.length() != 0) {
      field_string += ' ';
      field_string += c;
    } else {
      field_string += c;
    }
  } while(true);

  return "";
}

////////////////////////////////////////////////////////////////////////////////

void print_field(ostream & o, const string & fn, const string & fs)
{
  if (fs.length() != 0) {
    o << "," << endl;
    o << "  " << fn << " = " << fs;
  }
}

ostream & operator<<(ostream & o, bibtex_entry & entry)
{
  if (entry.type.compare("string") == 0) {
    o << "@string{ " << entry.string_name << " = " << entry.string_value << " }" << endl;
    return o;
  }

  if (short_output) {
    o << "@" << entry.type << "{" << entry.key;
    print_field(o, "author", entry.author);
    print_field(o, "title", entry.title);

    if (entry.type.compare("inproceedings") == 0) {
      if (entry.booktitle[0] != '{' && entry.booktitle[0] != '"' && entry.year.length() == 4) {
	string booktitle_year = "\"";
	booktitle_year += entry.booktitle;
	booktitle_year += '\'';
	booktitle_year += entry.year[2];
	booktitle_year += entry.year[3];
	booktitle_year += '\"';
	print_field(o, "booktitle", booktitle_year);
      } else {
	print_field(o, "booktitle", entry.booktitle);
	print_field(o, "year", entry.year);
      }
    } else if (entry.type.compare("article") == 0) {
      if (entry.journal[0] != '{' && entry.journal[0] != '"') {
	string short_journal = "\"";
	short_journal += entry.journal;
	short_journal += '\"';
	print_field(o, "journal", short_journal);
	print_field(o, "volume",  entry.volume);
	print_field(o, "number",  entry.number);
	print_field(o, "pages",   entry.pages);
      } else
	print_field(o, "journal", entry.journal);
      print_field(o, "year", entry.year);
    } else if (entry.type.compare("book") == 0) {
      print_field(o, "year", entry.year);
    } else if (entry.type.compare("techreport") == 0) {
      print_field(o, "institution", entry.institution);
      print_field(o, "year", entry.year);
    } else if (entry.type.compare("incollection") == 0) {
      print_field(o, "booktitle", entry.booktitle);
      print_field(o, "publisher", entry.publisher);
      print_field(o, "year", entry.year);
    } else if (entry.type.compare("phdthesis") == 0) {
      print_field(o, "year", entry.year);
    } else if (entry.type.compare("inbook") == 0) {
      print_field(o, "chapter", entry.chapter);
      print_field(o, "year", entry.year);
    } else if (entry.type.compare("misc") == 0) {
      print_field(o, "note", entry.note);
      print_field(o, "url", entry.url);
    }

    o << endl << "}" << endl;
  } else {
    o << "@" << entry.type << "{" << entry.key;
    print_field(o, "author", entry.author);
    print_field(o, "title", entry.title);

    if (entry.type.compare("inproceedings") == 0) {
      print_field(o, "booktitle", entry.booktitle);
    } else if (entry.type.compare("article") == 0) {
      print_field(o, "journal", entry.journal);
      print_field(o, "volume",  entry.volume);
      print_field(o, "number",  entry.number);
    } else if (entry.type.compare("book") == 0) {
      print_field(o, "publisher", entry.publisher);
      print_field(o, "editor", entry.editor);
    } else if (entry.type.compare("techreport") == 0) {
      print_field(o, "type", entry.report_type);
      print_field(o, "institution", entry.institution);
    } else if (entry.type.compare("incollection") == 0) {
      print_field(o, "booktitle", entry.booktitle);
      print_field(o, "publisher", entry.publisher);
      print_field(o, "editor", entry.editor);
    } else if (entry.type.compare("phdthesis") == 0) {
      print_field(o, "school", entry.school);
    } else if (entry.type.compare("inbook") == 0) {
      print_field(o, "chapter", entry.chapter);
      print_field(o, "publisher", entry.publisher);
      print_field(o, "editor", entry.editor);
    } else if (entry.type.compare("misc") == 0) {
      print_field(o, "note", entry.note);
      print_field(o, "url", entry.url);
    }

    print_field(o, "pages", entry.pages);
    print_field(o, "month", entry.month);
    print_field(o, "year", entry.year);

    o << endl << "}" << endl;
  }

  return o;
}

////////////////////////////////////////////////////////////////////////////////

// Remove { } or " " or spaces at the beginning or end of S.
string trim(string & S)
{
  string S2 = S;
  bool change;
  do {
    change = false;

    if (S2[0] == '{' && S2[S2.length()-1] == '}') {
      // Not elegant but should take care of
      // {\v S}ochman, Jan and Matas, Ji{\v r}{\' \i} :
      for(int i = 1; i < S2.length(); i++) {
	if (S2[i] == '{') {
	  S2 = S2.substr(1, S2.length()-2);
	  change = true;
	  break;
	} else if (S2[i] == '}') {
	  if (i == S2.length() - 1) {
	    S2 = S2.substr(1, S2.length()-2);
	    change = true;
	    break;
	  } else break;
	}
      }
    }

    if (S2[0] == '"' && S2[S2.length()-1] == '"') {
      S2 = S2.substr(1, S2.length()-2);
      change = true;
    }

    if (S2[0] == ' ') {
      S2 = S2.substr(1, S2.length()-1);
      change = true;
    }

    if (S2[S2.length() - 1] == ' ') {
      S2 = S2.substr(0, S2.length()-2);
      change = true;
    }
  } while(change);

  return S2;
}


// Change
//  Fredrik Kahl and Richard Hartley   into
//  F. Kahl and R. Hartley
// or
//  Bujnak, M. and Kukelova, Z. and Pajdla, T.   into
//  M. Bujnak and Z. Kukelova and T. Pajdla
//
// Still fails on examples like:
//  Bajcsy, R.K. and Lieberman, L.I.  or
//  Q.-T. Luong and O. Faugeras
// because of the two initials or the -
//
// If it cannot parse the input, it returns it cowardly instead of changing anything.

string try_to_clean_author_initials_and_stuff(string initial)
{
  string final, name;
  int state = 0;

  for(int i = 0; i < initial.length(); i++) {
    char c = initial[i];
    //     cout << c << " " << i << " " << state << endl;
    switch(state) {
    case 0:
      if (c == ' ') {
	state = 0;
      } else if (isalpha(c)) {
	state = 1;  name = "";  name += toupper(c);
      } else {
	return initial;
      }
      break;
    case 1:
      if (c == '.') {
	state = 9;  final += name;  final += ". ";
      } else if (isalpha(c)) {
	state = 2;  name += tolower(c);
      } else {
	return initial;
      }
      break;
    case 2:
      if (c == ' ') {
	state = 3;  final += name[0];  final += ". ";  name = "";
      } else if (isalpha(c)) {
	state = 2;  name += tolower(c);
      } else if (c == ',') {
	state = 11;
      } else {
	return initial;
      }
      break;
    case 3:
      if (c == ' ') {
	state = 3;
      } else if (isalpha(c)) {
	state = 4;  final += toupper(c);
      } else {
	return initial;
      }
      break;
    case 4:
      if (c == ' ') {
	state = 5;
      } else if (isalpha(c) || c == '-') {
	state = 4;  final += tolower(c);
      } else {
	return initial;
      }
      break;
    case 5:
      if (c == ' ') {
	state = 5;
      } else if (c == 'a') {
	state = 6;
      } else {
	return initial;
      }
      break;
    case 6:
      if (c == 'n') {
	state = 7;
      } else {
	return initial;
      }
      break;
    case 7:
      if (c == 'd') {
	state = 8;
      } else {
	return initial;
      }
      break;
    case 8:
      if (c == ' ') {
	state = 0;  final += " and ";
      } else {
	return initial;
      }
      break;
    case 9:
      if (c == ' ') {
	state = 9;
      } else if (isalpha(c)) {
	state = 10;  final += toupper(c);
      } else {
	return initial;
      }
      break;
    case 10:
      if (c == ' ') {
	state = 5;
      } else if (isalpha(c)) {
	state = 10;  final += tolower(c);
      } else {
	return initial;
      }
      break;
    case 11:
      if (c == ' ') {
	state = 11;
      } else if (isalpha(c)) {
	state = 12;  final += toupper(c);  final += ". ";  final += name;
      } else {
	return initial;
      }
      break;
    case 12:
      if (c == '.') {
	state = 5;
      } else {
	return initial;
      }
      break;
    default:
      return initial;
    } // switch
  } // for

  if (state == 0 || state == 4 || state == 5 || state == 10)
    return final;
  else
    return initial;
}

// Add { } around the authors names:
string clean_author(string author)
{
  string A = trim(author);


  string final_author = "{";
  string A2 = try_to_clean_author_initials_and_stuff(A);
  //   cout << A << " -> " << A2 << endl;
  final_author += A2;
  final_author += "}";
  
  for(int i = 0; important_authors_with_a_weird_name[2 * i] != 0; i++) {
    int found = final_author.find(important_authors_with_a_weird_name[2 * i]);

    if (found != string::npos)
      final_author.replace(found,
			   strlen(important_authors_with_a_weird_name[2 * i]),
			   important_authors_with_a_weird_name[2 * i + 1]);
  }

  return final_author;
}

bool should_be_capitalized(string & W)
{
  if (! islower(W[0]))
    return false;

  for(int i = 0; do_not_capitalize_these_words[i] != 0; i++)
    if (strcmp(W.c_str(), do_not_capitalize_these_words[i]) == 0)
      return false;

  return true;
}

// Add {{ }} around the title and capitalize properly:
string clean_title(string title)
{
  string T = trim(title);

  stringstream ss(T);
  string one_word;

  string final_title = "{{";

  bool first_word = true, stop_changing_case = false;

  while (ss >> one_word) {
    string new_word("");
    for(int i = 0; i < one_word.length(); i++) {
      if (one_word[i] == '{') {
	new_word += "{";
	stop_changing_case = true;
      } else if (one_word[i] == '}') {
	new_word += "}";
	stop_changing_case = false;
      } else if (isupper(one_word[i]) && !stop_changing_case) {
	new_word += tolower(one_word[i]);
      } else
	new_word += one_word[i];
    }

    // Capitalize words that are not in the do_not_capitalize_these_words array and starting by http :
    if (first_word || should_be_capitalized(new_word)) {
      if (new_word.substr(0, 4).compare("http") != 0)
	new_word[0] = toupper(new_word[0]);
    }
    
    // For words with a '-' in them:
    for(int i = 1; i < new_word.length(); i++)
      if (islower(new_word[i]) && new_word[i-1] == '-')
	new_word[i] = toupper(new_word[i]);

    final_title += new_word;
    final_title += " ";
    first_word = false;
  }

  for(int i = 0; special_words_in_titles[2 * i] != 0; i++) {
    int found = final_title.find(special_words_in_titles[2 * i]);

    if (found != string::npos)
      final_title.replace(found,
			  strlen(special_words_in_titles[2 * i]),
			  special_words_in_titles[2 * i + 1]);
  }

  // Remove last space:
  final_title.erase(final_title.length() - 1);

  if (final_title[final_title.length() - 1] == '.')
    final_title.erase(final_title.length() - 1);
  
  final_title += "}}";

  return final_title;
}

string lower_case(const char * s)
{
  string result;

  for(int i = 0; i < strlen(s); i++)
    result += tolower(s[i]);

  return result;
}

// Replace by standard string if needed:
string clean_journal(string journal)
{
  string lower_case_journal = lower_case(journal.c_str());
  
  for(int i = 0; strings_for_journals[2 * i] != 0; i++) {
    string lower_case_key = lower_case(strings_for_journals[2 * i]);
    if (lower_case_journal.find(lower_case_key) != string::npos)
      return strings_for_journals[2 * i + 1];
  }

  return journal;
}

// Replace by standard string if needed:
string clean_booktitle(string booktitle)
{
  string lower_case_booktitle = lower_case(booktitle.c_str());

  for(int i = 0; strings_for_conferences[2 * i] != 0; i++) {
    string lower_case_key = lower_case(strings_for_conferences[2 * i]);
    if (lower_case_booktitle.find(lower_case_key) != string::npos)
      return strings_for_conferences[2 * i + 1];
  }

  return booktitle;
}

// Add " " around the volume value:
string clean_volume(string volume)
{
  string vol = "\"";
  vol += trim(volume);
  vol += "\"";
  return vol;
}

// Add " " around the number value:
string clean_number(string number)
{
  string num = "\"";
  num += trim(number);
  num += "\"";
  return num;
}

// Add " " around the pages value, and replace - by --:
string clean_pages(string pages)
{
  string p2 = trim(pages);

  string pp = "\"";
  bool separator_added = false;
  for(int i = 0; i < p2.length(); i++)
    if (p2[i] == '-') {
      if (!separator_added) {
	pp += "--";
	separator_added = true;
      }
    } else if (isdigit(p2[i]))
      pp += p2[i];

  pp += "\"";

  return pp;
}

string clean_year(string year)
{
  return trim(year);
}

// Replace by the standard string if needed:
string clean_month(string month)
{
  string lower_case_month = lower_case(month.c_str());

  for(int i = 0; months[2 * i] != 0; i++) {
    string lower_case_key = lower_case(months[2 * i]);
    if (lower_case_month.find(lower_case_key) != string::npos)
      return months[2 * i + 1];
  }

  return month;
}

// Reads and returns the next entry:
bibtex_entry read_entry(bibfile_parser & parser)
{
  bibtex_entry entry;

  do {
    entry.type = parser.read_type();
    if (parser.reached_end_of_file) return entry;

    if (entry.type.compare("comment") == 0) {
      char c;
      bool ok = false;
      do {
	c = parser.get_very_next_character();
	if (parser.reached_end_of_file) return entry;

	if (c == 13 || c == 10) {
	  parser.line_number++;
	  ok = true;
	}
      } while (!ok);
    }
  } while(entry.type.compare("comment") == 0);

  if (entry.type.compare("string") == 0) {
    entry.string_name = parser.read_string_name();
    bool end_of_entry = false;
    entry.string_value = parser.read_field_string(end_of_entry);

    return entry;
  }

  entry.key = parser.read_key();

  bool end_of_entry = false;

  do {
    string field_name = parser.read_field_name();
    if (field_name.compare("stop") == 0) {
      end_of_entry = true;
    } else {
      string field_string = parser.read_field_string(end_of_entry);

      if (field_name.compare("author") == 0) {
	entry.author = clean_author(field_string);
      } else if (field_name.compare("title") == 0) {
	entry.title = clean_title(field_string);
      } else if (field_name.compare("booktitle") == 0) {
	entry.booktitle = clean_booktitle(field_string);
      } else if (field_name.compare("journal") == 0) {
	if (entry.type.compare("article") == 0 && field_string.find("onference") != string::npos) {
	  // huho
	  entry.type = "inproceedings";
	  entry.booktitle = clean_booktitle(field_string);
	} else
	  entry.journal = clean_journal(field_string);
      } else if (field_name.compare("volume") == 0) {
	entry.volume = clean_volume(field_string);
      } else if (field_name.compare("number") == 0) {
	entry.number = clean_number(field_string);
      } else if (field_name.compare("issue") == 0) {
	entry.number = clean_number(field_string);
      } else if (field_name.compare("numero") == 0) {
	entry.number = clean_number(field_string);
      } else if (field_name.compare("pages") == 0) {
	entry.pages = clean_pages(field_string);
      } else if (field_name.compare("month") == 0) {
	entry.month = clean_month(field_string);
      } else if (field_name.compare("year") == 0) {
	entry.year = clean_year(field_string);
      } else if (field_name.compare("publisher") == 0) {
	entry.publisher = field_string;
      } else if (field_name.compare("type") == 0) {
	entry.report_type = field_string;
      } else if (field_name.compare("institution") == 0) {
	entry.institution = field_string;
      } else if (field_name.compare("school") == 0) {
	entry.school = field_string;
      } else if (field_name.compare("chapter") == 0) {
	entry.chapter = clean_title(field_string);
      } else if (field_name.compare("editor") == 0) {
	entry.editor = field_string;
      } else if (field_name.compare("url") == 0) {
	entry.url = field_string;
      } else if (field_name.compare("note") == 0) {
	entry.note = field_string;
      } else if (field_name.compare("abstract") == 0) {
      } else if (field_name.compare("address") == 0) {
      } else if (field_name.compare("keywords") == 0) {
      } else if (field_name.compare("bibsource") == 0) {
      } else if (field_name.compare("isbn") == 0) {
      } else if (field_name.compare("key") == 0) {
      } else if (field_name.compare("issn") == 0) {
      } else if (field_name.compare("doi") == 0) {
      } else if (field_name.compare("series") == 0) {
      } else {
	// 	cerr << "field " << field_name << " ignored line " << parser.get_line_number() << endl;
      }
      end_of_entry = parser.get_last_non_blank_character_read() == '}';
    }
  } while(!end_of_entry);

  return entry;
}

////////////////////////////////////////////////////////////////////////////////

// Compare 2 entries to sort in alphabetic order based on the keys and the years.
// Must take care of for example: ART, ARToolKit, Fua99, Fua99d, Fua00a, ...
bool compare_entries(const bibtex_entry & e1, const bibtex_entry & e2)
{
  string cut_key1, cut_key2;

  if (e1.key.length() <= 3)
    cut_key1 = e1.key;
  else if (islower(e1.key[e1.key.length()-1]) &&
	   isdigit(e1.key[e1.key.length()-2]) &&
	   isdigit(e1.key[e1.key.length()-3]))
    cut_key1 = e1.key.substr(0, e1.key.length()-3);
  else if (isdigit(e1.key[e1.key.length()-1]) &&
	   isdigit(e1.key[e1.key.length()-2]))
    cut_key1 = e1.key.substr(0, e1.key.length()-2);
  else
    cut_key1 = e1.key;

  if (e2.key.length() <= 3)
    cut_key2 = e2.key;
  else if (islower(e2.key[e2.key.length()-1]) &&
	   isdigit(e2.key[e2.key.length()-2]) &&
	   isdigit(e2.key[e2.key.length()-3]))
    cut_key2 = e2.key.substr(0, e2.key.length()-3);
  else if (isdigit(e2.key[e2.key.length()-1]) &&
	   isdigit(e2.key[e2.key.length()-2]))
    cut_key2 = e2.key.substr(0, e2.key.length()-2);
  else
    cut_key2 = e2.key;

  int comp = cut_key1.compare(cut_key2);

  if (comp == 0) {
    comp = e1.year.compare(e2.year);
    if (comp == 0) {
      return e1.key[e1.key.length()-1] < e2.key[e2.key.length()-1];
    } else return comp < 0;
  } else return comp < 0;
}

// Compare 2 string entries:
bool compare_string_entries(const bibtex_entry & e1, const bibtex_entry & e2)
{
  return e1.string_name.compare(e2.string_name) < 0;
}


////////////////////////////////////////////////////////////////////////////////

int main(int argc, char ** argv)
{
  string input_filename, output_filename;

  if (argc == 2) {
    input_filename = argv[1];
    output_filename = argv[1];
  } else if (argc == 3) {
    if (strcmp(argv[1], "--short") == 0 || strcmp(argv[1], "-s") == 0) {
      cerr << "Usage: bibfile_cleaner [--short | -s] input_file.bib [output_file.bib]" << endl;
      cerr << "Sorry, to use the --short option, you need to provide an output file different from the input file." << endl;
      return 0;
    } else {
      input_filename = argv[1];
      output_filename = argv[2];
    }
  } else if (argc == 4) {
    if (strcmp(argv[1], "--short") == 0 || strcmp(argv[1], "-s") == 0) {
      short_output = true;
      input_filename = argv[2];
      output_filename = argv[3];
      if (input_filename == output_filename) {
	cerr << "Usage: bibfile_cleaner [--short | -s] input_file.bib [output_file.bib]" << endl;
	cerr << "Sorry, to use the --short option, you need to provide an output file different from the input file." << endl;
	return 0;
      }
    } else {
      cerr << "Usage: bibfile_cleaner [--short | -s] input_file.bib [output_file.bib]" << endl;
      return 0;
    }
  } else {
    cerr << "Usage: bibfile_cleaner [--short | -s] input_file.bib [output_file.bib]" << endl;
    return 0;
  }

  // Parse the input file:
  parser.start_parsing(input_filename);

  vector<bibtex_entry> entries;
  vector<bibtex_entry> string_entries;
  do {
    bibtex_entry entry = read_entry(parser);
    
    if (!parser.reached_end_of_file) {
      if (entry.type.compare("string") == 0) {
	string_entries.push_back(entry);
	cout << "> reading string " << entry.string_name << "                         " << char(13) << flush;
      } else {
	entries.push_back(entry);
	cout << "> reading key " << entry.key << "                         " << char(13) << flush;
      }
    }

  } while(!parser.reached_end_of_file);
  parser.end_parsing();

  // Sort the entries:
  sort(entries.begin(), entries.end(), compare_entries);

  // Sort the strings:
  sort(string_entries.begin(), string_entries.end(), compare_string_entries);

  // Save the output file:
  output_file.open(output_filename.c_str());

  for(int i = 0; i < string_entries.size(); i++) {
    output_file << string_entries[i] << endl;
  }

  char letter = ' ';
  for(int i = 0; i < entries.size(); i++) {
    if (entries[i].key[0] != letter) {
      letter = entries[i].key[0];
      output_file << "%--" << endl;
      output_file << "%" << letter << endl;
      output_file << "%--" << endl;
      output_file << endl;
    }

    if (i > 0 && entries[i].key.compare(entries[i-1].key) == 0) {
      output_file << "% duplicated key:" << endl;
    }
    
    if (entries[i].author.length() > 0 &&
	(entries[i].type.compare("misc") != 0) &&
	(entries[i].author.find(".") == string::npos ||
	 entries[i].author.find(",") != string::npos) &&
	entries[i].author.find("Del Bue") == string::npos
	) {
      output_file << "% check authors names:" << endl;
    }

    output_file << entries[i] << endl;
  }
  output_file.close();


  // Print warnings for entries with uncorrect author names:
  bool all_good = true;
  
  for(int i = 0; i < entries.size(); i++) {
    if (entries[i].author.length() > 0 &&
	(entries[i].type.compare("misc") != 0) &&
	(entries[i].author.find(".") == string::npos ||
	 entries[i].author.find(",") != string::npos) &&
	entries[i].author.find("Del Bue") == string::npos
	) {
      if (all_good) {
	cerr << "The following entries should probably have their author fields corrected by hand:" << endl;
	cerr << endl;
	all_good = false;
      }
      cerr << " * " << entries[i].key << " (" << entries[i].author << ")" << endl;
    }
  }
  if (!all_good) cerr << endl;

  // Print warnings for entries with duplicated keys:
  all_good = true;
  for(int i = 1; i < entries.size(); i++) {
    if (entries[i].key.compare(entries[i-1].key) == 0) {
      if (all_good) {
	cerr << "The following entries have duplicated keys:" << endl;
	cerr << endl;
	cerr << " -> ";
	all_good = false;
      }
      cerr << entries[i].key << " ;  ";
    }
  }
  if (!all_good) {
    cerr << endl;
    cerr << endl;
  }

  // Print warnings for entries with weird years:
  all_good = true;
  for(int i = 1; i < entries.size(); i++) {
    if (entries[i].year.length() != 4 && entries[i].year.length() != 0) {
      if (all_good) {
	cerr << "The following entries have suspicious years:" << endl;
	cerr << endl;
	cerr << " -> ";
	all_good = false;
      }
      cerr << entries[i].key << " ;  ";
    }
  }
  if (!all_good) {
    cerr << endl;
    cerr << endl;
  }

  cout << "output saved in file " << output_filename << endl;

  return 0;
}
