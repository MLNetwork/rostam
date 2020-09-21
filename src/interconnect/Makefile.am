AM_CPPFLAGS = -I$(srcdir)/../utils -I$(srcdir)/../compute -I$(srcdir)/../executor -isystem$(srcdir)/../graph_def $(CXX11_FLAGS)
AM_CXXFLAGS = $(PICKY_CXXFLAGS) $(NODEBUG_CXXFLAGS)

noinst_LIBRARIES = librostaminterconnect.a

librostaminterconnect_a_SOURCES = base_interconnect.cc electrical_switch.cc fullmesh.cc \
                                  mordia.cc ocs.cc ring.cc \
                                  transport.cc tm_estimator.cc 
