AM_CPPFLAGS = -I$(srcdir)/../utils -I$(srcdir)/../interconnect -I$(srcdir)/../compute -isystem$(srcdir)/../graph_def -I$(srcdir)/../executor $(ENALBLE_GUROBI) $(CXX11_FLAGS)
AM_CXXFLAGS = $(PICKY_CXXFLAGS) $(NODEBUG_CXXFLAGS)

noinst_LIBRARIES = librostamplacement.a

librostamplacement_a_SOURCES = base_placement.cc dp.cc mp.cc \
                               op_partitioner.cc strategy.cc 
