# python-mesoscaler

MesoNet algorithms made modular.

## Requirements

- Python >3.7
- DeepLabCut
- The landmark-prediction network from
  [Xiao et al., 2019 Nat Commun](https://doi.org/10.1038/s41467-021-26255-2):
  download and store `atlas-DongshengXiao-2020-08-03`
  from [this OSF repository](https://osf.io/svztu/)

Note that the original [MesoNet library](https://github.com/bf777/MesoNet)
is _not_ necessary (but may be useful as a source of information about
how it works).

It is recommended that you set up the DeepLabCut environment
before installing this packaage.

## Installation

Currently, only installing from this repository is supported:

```bash
# recommended: set up DeepLabCut beforehand for your environment!
git clone git@github.com:BraiDyn-BC/python-ks-mesoscaler.git
cd python-ks-mesoscaler
pip install .  # add the `-e` switch in case you plan to modify the code
```

At this point, also specify the `MESONET_DLC_PROJECT_DIR`
environment variable so that the library can find the path to your
DeepLabCut project folder containing the landmark-inference network.
If the `config.yaml` file of the DeepLabCut project is found at
the path `D:\library\atlas-DongshengXiao-2020-08-03\config.yaml`,
Then register this path `D:\...(omitted)...\config.yaml` as
`MESONET_DLC_PROJECT_DIR`.

## License

(c) 2023 Keisuke Sehara, the MIT License

Note that the following files belong to:

 (c) 2019 Forys, Xiao, and Murphy lab, CC-BY 4.0

- `mesoscaler/landmarks/reference.py`
- all the binary files in the `mesoscaler/data` directory.
