AM_CPPFLAGS = -I$(srcdir)/../utils -I$(srcdir)/../compute -I$(srcdir)/../interconnect -isystem$(srcdir)/../graph_def $(PROTOBUF_CFLAGS) $(CXX11_FLAGS) 
AM_CXXFLAGS = $(PICKY_CXXFLAGS) $(NODEBUG_CXXFLAGS)

noinst_LIBRARIES = librostamexecutor.a

librostamexecutor_a_SOURCES = graph.cc op.cc session.cc 
