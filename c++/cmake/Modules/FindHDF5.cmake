# version directly derived from the FindOmniORB.cmake

#
# Find the hdf5 include dir
#

# HDF5_INCLUDE_DIR  - Directories to include to use boost
# HDF5_FOUND        - When false, don't try to use boost
#
# HDF5_INCLUDE_DIR can be used to make it simpler to find the various include
# directories and compiled libraries when hdf5 was not installed in the
# usual/well-known directories (e.g. because you made an in tree-source
# compilation or because you installed it in an "unusual" directory).
# Just set HDF5_INCLUDE_DIR it to your specific installation directory
#

if (NOT MSVC)
find_path(HDF5_INCLUDE_DIR hdf5.h
  PATHS
  ${HDF5_DIR}/include
  ${HDF5_DIR}/
  /usr/include
  /usr/local/include
)
else (NOT MSVC)
find_path(HDF5_INCLUDE_DIR hdf5.h
  PATHS
  ${HDF5_DIR}/include
  ${HDF5_DIR}/
  "C:\\Program Files\\HDF5\\1.6.5\\include"
  "C:\\Program Files\\HDF5\\1.6.7\\include"
  "C:\\Program Files\\HDF5\\1.8.1\\include"
)  
endif (NOT MSVC)


mark_as_advanced (HDF5_INCLUDE_DIR)
  
if (HDF5_INCLUDE_DIR)
  set(HDF5_FOUND "YES")
endif (HDF5_INCLUDE_DIR)
          
if (NOT HDF5_FOUND)

  message("HDF5 installation was not found. "
          "Please provide a correct a HDF5_DIR variable.") 
  set(HDF5_DIR "" CACHE PATH "Root of hdf5 install tree.")

else (NOT HDF5_FOUND)

  # Searching for HDF5 libraries
  if (MSVC)
   find_library(HDF5_DLL_LIB hdf5dll
     PATHS ${HDF5_DIR}/dll
     "C:\\Program Files\\HDF5\\1.6.5\\dll"
     "C:\\Program Files\\HDF5\\1.6.7\\dll"
     "C:\\Program Files\\HDF5\\1.8.1\\dll"     
   )
   mark_as_advanced (HDF5_DLL_LIB)
  endif (MSVC)

  find_library(HDF5_LIB hdf5
    PATHS ${HDF5_DIR}/lib
    /opt/lib/
    /usr/lib/
     "C:\\Program Files\\HDF5\\1.6.5\\lib"
     "C:\\Program Files\\HDF5\\1.6.7\\lib"
     "C:\\Program Files\\HDF5\\1.8.1\\lib"     
  )
  mark_as_advanced (HDF5_LIB)
  
  if (MSVC)
   find_library(HDF5_DLL_DEBUG_LIB hdf5ddll
     PATHS ${HDF5_DIR}/dll
     "C:\\Program Files\\HDF5\\1.6.5\\dll"
     "C:\\Program Files\\HDF5\\1.6.7\\dll"
     "C:\\Program Files\\HDF5\\1.8.1\\dll"     
     PATHS ${HDF5_DIR}/lib
     "C:\\Program Files\\HDF5\\1.6.5\\lib"
     "C:\\Program Files\\HDF5\\1.6.7\\lib"
     "C:\\Program Files\\HDF5\\1.8.1\\lib"     
   )
   mark_as_advanced (HDF5_DLL_DEBUG_LIB)
  endif (MSVC)

  find_library(HDF5_DEBUG_LIB hdf5d
    PATHS ${HDF5_DIR}/lib
    /opt/lib/
    /usr/lib/
     "C:\\Program Files\\HDF5\\1.6.5\\lib"
     "C:\\Program Files\\HDF5\\1.6.7\\lib"
     "C:\\Program Files\\HDF5\\1.8.1\\lib"     
  )
  mark_as_advanced (HDF5_DEBUG_LIB)  

endif (NOT HDF5_FOUND)

