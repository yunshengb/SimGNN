# GraphEmbedding

This version is for the WSDM 2019 paper.


## Datasets

Get the datasets from https://drive.google.com/drive/folders/1lY3pqpnUAK0H9Tgjyh7tlMVYy0gYPthC?usp=sharing
and extract under `data/`:
* AIDS80nef
* AIDS700nef
* linux
* IMDBMulti

Get the pickle files (`/save`) from https://drive.google.com/drive/folders/1Eusvi4_iOKM0AsO1LhxQFkY62kDEtuMq?usp=sharing

Get the result files (`/result`)  https://drive.google.com/drive/folders/1UXEGozaThjjuC-hnt4C7jn06L6I2Ra1v?usp=sharing


## Dependencies

Install the following the tools and packages:

* `python3`: Assume `python3` by default (use `pip3` to install packages).
* `numpy`
* `pandas`
* `scipy`
* `scikit-learn`
* `tensorflow` (1.8.0 recommended)
* `networkx==1.10` (NOT `2.1`)
* `beautifulsoup4`
* `lxml`
* `matplotlib`
* `seaborn`
* `colour`
* `pytz`
* `pygraphviz`. The following is an example set of installation commands (tested on Ubuntu 16.04) 
    ```
    sudo apt-get install graphviz libgraphviz-dev pkg-config
    pip3 install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"
    ```
* Graph Edit Distance (GED):
    * `graph-matching-toolkit`
        * `cd src && git clone https://github.com/yunshengb/graph-matching-toolkit.git`
        * Follow the instructions on https://github.com/yunshengb/graph-matching-toolkit to compile
    * `java`
    
## Tips for PyCharm Users

* If you see red lines under `import`, mark `src` and `model/Siamese` as `Source Root`,
so that PyCharm can find those files.
* Mark `src/Siamese/logs` and `src/Siamese/exp` as `Excluded`, so that PyCharm won't spend time inspecting those logs.
