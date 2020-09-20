# rostam

## Build directions
### Gurobi Solver
We use Gurobi for solving the Integer Linear Programs (ILP) in optical circuit decisions or wavelength scheduling. Gurobi solver can be installed from \url{} (it has a free academic liscence too). However, you may also compile without having Gurobi at the cost of having the solver replaced by suboptimal preset decisions in optical interconnects. To build without Gurobi installed, just run:
```
$ ./autogen.sh
$ ./configure
$ make -j$(nproc)
$ sudo make install
```
If you have Gurobi installed, just change ``./configure`` to  ``./configure --enable-gurobi`` in the commands above.
