#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([rostam], [0.1], [])
AM_INIT_AUTOMAKE([foreign tar-ustar])
AC_CONFIG_SRCDIR([src/sipml/sipml-elect.cc])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIRS([m4])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

# Add picky CXXFLAGS
CXX11_FLAGS="-std=c++14 -pthread"
PICKY_CXXFLAGS="-pedantic -Wall -Wextra -Weffc++ -Werror"
AC_SUBST([CXX11_FLAGS])
AC_SUBST([PICKY_CXXFLAGS])
AC_LANG_PUSH(C++)

AC_ARG_ENABLE([debug],
  [AS_HELP_STRING([--enable-debug@<:@=no/asserts/sanitize@:>@],
     [Turn on asserts or santizers])],
  [case "$enableval" in
     no)
       NODEBUG_CXXFLAGS="-DNDEBUG"
       ;;
     '' | yes | sanitize)
       NODEBUG_CXXFLAGS="-fsanitize=address -fsanitize=undefined -fuse-ld=gold"
       ;;
     asserts)
       NODEBUG_CXXFLAGS=""
       ;;
     *)
       AC_MSG_ERROR([Unknown argument '$enableval' to --enable-debug])
       ;;
   esac],
  [NODEBUG_CXXFLAGS="-DNDEBUG"])

AC_SUBST(NODEBUG_CXXFLAGS)

AC_ARG_ENABLE([all-static],
  [AS_HELP_STRING([--enable-all-static], [Build statically linked binaries])],
  [case "$enableval" in
    '' | yes)
        STATIC_BUILD_FLAG="-all-static"
        ;;
    no)
        STATIC_BUILD_FLAG=""
        ;;
    *)
        AC_MSG_ERROR([Unknown argument '$enableval' to --enable-all-static])
        ;;
   esac],
  [STATIC_BUILD_FLAG=""])

AC_SUBST(STATIC_BUILD_FLAG)

# Checks for header files.
AC_PATH_X
AC_CHECK_HEADERS([limits.h unistd.h], [], AC_MSG_ERROR([Missing required header file]))

AC_ARG_WITH(gurobi,
[AS_HELP_STRING([--without-gurobi], [Do not use Gurobi])])
if test "x${with_gurobi}" == "xyes"; then
    AC_CHECK_HEADER([gurobi_c++.h], [AC_DEFINE(HAVE_GUROBI, 1, [Define to 1 if you have gurobi.])], AC_MSG_ERROR([Missing gurobi header file]))
    AC_CHECK_LIB([gurobi90], [GRBaddgenconstrIndicator], [], AC_MSG_ERROR([Missing gurobi90 package]), [])
    AC_CHECK_LIB([gurobi_c++], [GRBaddgenconstrIndicator], [], AC_MSG_ERROR([Missing gurobi_c++ package]), [])
	usegurobi=true
else
	usegurobi=false
fi
AM_CONDITIONAL([USEGUROBI], [test "$usegurobi" = true])



# Checks for programs.
AC_PROG_CXX
AC_PROG_AWK
AC_PROG_CC
AC_PROG_CPP
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_PROG_RANLIB

# Checks for libraries.
PKG_CHECK_MODULES([PROTOBUF], [protobuf])



# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T
AC_TYPE_UINT64_T
AC_TYPE_UINT32_T
AC_TYPE_UINT16_T
AC_TYPE_UINT8_T
AC_TYPE_INT64_T
AC_TYPE_INT32_T
AC_TYPE_INT16_T
AC_TYPE_INT8_T


# Checks for library functions.
AC_FUNC_MMAP
AC_CHECK_FUNCS([getcwd getpagesize memmove strerror strtol])

AC_CONFIG_FILES([Makefile
                 src/Makefile
                 src/utils/Makefile
                 src/compute/Makefile
                 src/executor/Makefile
                 src/interconnect/Makefile
                 src/placement/Makefile
                 src/graph_def/Makefile
                 src/sipml/Makefile])
AC_OUTPUT
