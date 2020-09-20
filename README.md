# rostam

## Build directions
### Gurobi Solver
We use [Gurobi](https://www.gurobi.com) for solving the Integer Linear Programs (ILP) for optical circuit decisions and wavelength schedulings. Gurobi can be installed from its official homepage (and it offers a free academic liscence). However, you may also continue to build without having Gurobi installed at a cost of having the solver replaced by suboptimal preset decisions in optical interconnects. Electrical and fullmesh interconnects have no dependency this package. To build without Gurobi installed, just run:
```
$ ./autogen.sh
$ ./configure
$ make -j$(nproc)
$ sudo make install
```
If you have Gurobi installed, just change ``./configure`` to  ``./configure --enable-gurobi`` in the commands above.

##SiPML
To simulate a tranining job on an electrical interconnect run:
```
sipml-elect --num_gpus 256 --bw_per_port_Gb 1000 --input_profile examples/resnet_v1_50_float32
```
