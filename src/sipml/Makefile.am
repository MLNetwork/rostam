AM_CPPFLAGS = -I$(srcdir)/../utils -isystem$(subdir)/../graph_def -I$(srcdir)/../compute -I$(srcdir)/../executor -I$(srcdir)/../interconnect -I$(srcdir)/../placement -isystem$(srcdir)/../graph_def $(PROTOBUF_CFLAGS) $(CXX11_FLAGS)
AM_CXXFLAGS = $(PICKY_CXXFLAGS) $(NODEBUG_CXXFLAGS)
AM_LDFLAGS = $(STATIC_BUILD_FLAG) 
BASE_LDADD = ../placement/librostamplacement.a ../interconnect/librostaminterconnect.a ../compute/librostamcompute.a ../executor/librostamexecutor.a ../graph_def/librostamgraphdef.a $(PROTOBUF_LIBS)

bin_PROGRAMS = sipml-elect sipml-fullmesh sipml-ocs sipml-ring

sipml_elect_SOURCES = sipml-elect.cc
sipml_elect_LDADD = $(BASE_LDADD)

sipml_fullmesh_SOURCES = sipml-fullmesh.cc
sipml_fullmesh_LDADD = $(BASE_LDADD)

sipml_ocs_SOURCES = sipml-ocs.cc
sipml_ocs_LDADD = $(BASE_LDADD)

sipml_ring_SOURCES = sipml-ring.cc
sipml_ring_LDADD = $(BASE_LDADD)
