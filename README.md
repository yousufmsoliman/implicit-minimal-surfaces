# Implicit Minimal Surfaces for Bijective Correspondences
![Toucan](https://www.yousufsoliman.com/img/implicit-minimal-surface-banner.png)

Official C++ implementation of the paper: 

[_Implicit Minimal Surfaces for Bijective Correspondences_](https://www.yousufsoliman.com/projects/download/ImplicitMinimalSurfaces.pdf)\
Etienne Corman, Yousuf Soliman, Robin Magnet, and Mark Gillespie\
_ACM Transactions on Graphics_ 45 (4) | **SIGGRAPH 2026**


The algorithm computes an implicit representation of bijective correspondences of genus zero surfaces. Our implicit representation is based on discrete complex line bundles over products of triangulated surfaces.

## Installation

```
gh repo clone yousufmsoliman/implicit-minimal-surfaces
cd implicit-minimal-surfaces
git submodule update --init --recursive
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j12
```

## Usage

A user interface to the algorithm can be launched using the command 
```
./bin/implicit-minimal-matcher [WORKING DIRECTORY] [SOURCE MESH] [TARGET MESH]
```

It will load the source mesh and the target mesh from the specified working directory. Landmark constraints are specified by including `[SOURCE MESH NAME].pinned` and `[TARGET MESH NAME].pinned` in the directory; if they exist, they will automatically be loaded. Otherwise, landmark points can be interactively added via the UI. 

The `Compute Correspondences` button will construct an initial section of the complex line bundle as described in Section 4.4 of the paper (this step can be run by pressing `Initialize`) and optimizing its Ginzburg-Landau functional (this step be run by pressing `Refine`). After computation, the output correspondences will be saved in `[WORKING DIRECTORY]/export/`. When initializing and refining separately, one must manually `Extract Vertex Correspondences` before exporting them. 

*Note: The user interface exposes slight variations on the discretization of the complex line bundle and its associated differential operators that were investigated throughout this research.* 

## Alternative Implementations
- **MATLAB:** https://github.com/etcorman/implicit-minimal-surfaces
- **Python:** https://github.com/RobinMagnet/implicit-minimal-surfaces

## Citation

If our work contributes to your academic work, please consider cite the following paper:

```bib
@article{Corman:2026:IMS,
  title={Implicit Minimal Surfaces for Bijective Correspondences},
  author={Corman, Etienne and Soliman, Yousuf and Magnet, Robin and Gillespie, Mark},
  journal={ACM Transactions on Graphics},
  volume={45},
  number={4},
  year={2026},
  publisher={ACM}
}
```
