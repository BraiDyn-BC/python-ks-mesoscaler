# python-mesoscaler

MesoNet algorithms made modular.

## Requirements

- Python >3.7
- DeepLabCut
- The landmark-prediction network from
  [Xiao et al., 2019 Nat Commun](https://doi.org/10.1038/s41467-021-26255-2)
  (see [below](#downloading-the-mesonet-landmark-prediction-model) for instructions)

Note that the original [MesoNet library](https://github.com/bf777/MesoNet)
is **_not necessary_** (but may be useful as a source of information about
how it works).

It is recommended that you set up the DeepLabCut environment
before installing this package.

## Downloading the MesoNet landmark-prediction model

By default, we use the landmark-prediction model provided for the original
MesoNet library. 
Follow the steps below to download and prepare the model (you can omit this step
in case you already have the original MesoNet up and running).

1. Open [this OSF repository](https://osf.io/svztu/)
2. Navigate in the "Files" pane: `MesoNet/OSF Storage/6_Landmark_estimation_model`
3. Follow the link at: `atlas-DongshengXiao-2020-08-03.zip`
4. In the opened link, you will find the file name (`atlas-DongshengXiao-2020-08-03.zip`)
   as the headline. Locate the tri-colon \[ &#x205D; \] button, and click it.
5. Click the 'Download' menu, and wait until downloading is done.
6. Extract the contents of the ZIP file, and locate it wherever permanent.

The 'DeepLabCut project directory' must contain the `config.yaml` file as the first-order
child (not as the child of any child directories). Make sure it is the case.

## Installation

> [!NOTE]
> We recommend setting up DeepLabCut beforehand for your environment

Currently, only installing from this repository is supported:

```bash
git clone git@github.com:BraiDyn-BC/python-ks-mesoscaler.git
cd python-ks-mesoscaler
pip install .  # add the `-e` switch in case you plan to modify the code
```

> [!CAUTION]
> In some cases, `pip` may refuse to install the executable to the non-user
> environments (and recommends to add `--user`), or compalains that the installation
> path is not included as `PATH`.
>
> Try running your terminal emulator (e.g. Anaconda Prompt) in the admin mode
> in such cases.

At this point, also specify the `MESONET_DLC_PROJECT_DIR`
environment variable so that the library can find the path to your
DeepLabCut project folder containing the landmark-inference network.

If the `config.yaml` file of the DeepLabCut project is found at
the path `D:\library\atlas-DongshengXiao-2020-08-03\config.yaml`,
Then register the path `D:\library\atlas-DongshengXiao-2020-08-03` as
`MESONET_DLC_PROJECT_DIR`.

## Usage

See our [HOWTO page](./HOWTO.md).

For the resulting file structure, refer to the [file structure page](./FILE_STRUCTURE.md).

## License

### Most source code and documentation

(c) 2023-2024 Keisuke Sehara, the MIT License

The DOI for reference will be obtained soon.

### Reference atlas data

The following files are attributable to:

 (c) 2019 Forys, Xiao, and Murphy lab, CC-BY 4.0

- `mesoscaler/landmarks/reference.py`
- all the binary files in the `mesoscaler/data` directory.

Cite [Xiao et al., 2019 Nat Commun](https://doi.org/10.1038/s41467-021-26255-2) paper.
