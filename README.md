# rostam

## Build directions
### Gurobi Solver
We use [Gurobi](https://www.gurobi.com) for solving the Integer Linear Programs (ILP) for optical circuit decisions and wavelength schedulings. Gurobi can be installed from its official homepage (and it offers a free academic liscence). However, you may also continue to build without having Gurobi installed at a cost of having the solver replaced by suboptimal preset decisions in optical interconnects. Electrical and fullmesh interconnects have no dependency this package. To build without Gurobi installed, just run:
```
$ ./autogen.sh
$ ./configure --without-gurobi
$ make -j$(nproc)
$ sudo make install
```
If you have Gurobi installed, just change the configure flag to  ``./configure --with-gurobi  LDFLAGS=-L/path/to/gurobi/lib CPPFLAGS=-I/path/to/gurobi/include`` in the commands above. We suggest using ``CPPFLAGS=-isystem/path/to/gurobi/include`` instead, to prevent warnings of the gurobi being treated as errors too.  

## SiPML
To simulate a tranining job on an electrical interconnect run:
```
sipml-elect --num_gpus 256 --bw_per_port_Gb 1000 --input_profile INPUT_PROFILE
```
Or 
```
sipml-ring -g 32 -w 400 -d 16 -s -b ILP -m 10 -i INPUT_PROFILE
```
to use a ring interconnect. Please check out ``sipml-ocs --help`` and ``sipml-fullmesh --help`` for more interconnects.
### Input profiles
Scripts for generating input profiles are available at [``src/scrips``](https://github.com/MLNetwork/rostam/tree/master/src/scripts). Recommended structure of input profiles is as follows:
```
+ profile_dir 
    ++ model_name_bs1.pb
    ++ model_name_bs2.pb
    .
    .
    .
    ++ model_name_bs512.pb
    ++ model_name_iter.prof
```
Using the above structure, the format of ``input_profile`` argument will be ``profile_dir/model_name``.

